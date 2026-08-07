"""Conv-layer support: the unit is a CHANNEL (CLAUDE.md §5.5).

Setting 2 (label-shuffled CIFAR-10 with a small CNN) is a C2 extension, not a
portability exercise: whether the published definitions of "dead" survive the
generalisation to a unit whose activation is a feature map rather than a scalar
is part of the claim.

CLAUDE.md §5.5 says of the outgoing-slice indexing when the next layer is fully
connected: "getting that indexing wrong will silently zero the wrong columns.
Write the test before the implementation." `test_flatten_outgoing_slice_targets_
the_right_columns` is that test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src import interventions, probes
from src.interventions import Recycler, RecyclerConfig
from src.models import SmallCNN


# ---------------------------------------------------------------------------
# Folding (N, C, H, W) to (N*H*W, C)
# ---------------------------------------------------------------------------


def test_as_unit_matrix_puts_channels_on_the_column_axis():
    """A reshape without the permute would interleave units -- the silent
    failure that would corrupt every conv metric identically."""
    h = torch.zeros(2, 3, 4, 5)
    h[:, 1] = 7.0  # channel 1 hot everywhere
    m = probes.as_unit_matrix(h)
    assert m.shape == (2 * 4 * 5, 3)
    assert torch.all(m[:, 1] == 7.0)
    assert torch.all(m[:, 0] == 0.0) and torch.all(m[:, 2] == 0.0)


def test_as_unit_matrix_leaves_dense_activations_alone():
    h = torch.randn(8, 5)
    assert probes.as_unit_matrix(h) is h or torch.equal(probes.as_unit_matrix(h), h)


def test_as_unit_matrix_rejects_odd_ranks():
    with pytest.raises(ValueError, match="expected"):
        probes.as_unit_matrix(torch.zeros(2, 3, 4))


# ---------------------------------------------------------------------------
# Death definitions on channels
# ---------------------------------------------------------------------------


def _conv_acts():
    """(N, C, H, W) with one channel of each kind."""
    h = torch.zeros(4, 5, 3, 3)
    h[:, 0] = 0.0                      # 0: dead everywhere
    h[:, 1] = 1.0                      # 1: alive everywhere
    h[0, 2, 0, 0] = 2.0                # 2: fires on ONE position of ONE image
    h[:, 3] = 1e-8                     # 3: alive but vanishingly quiet
    h[:, 4, :, 0] = 3.0                # 4: alive on one column of every image
    return h


def test_dead_exact_requires_zero_at_every_position():
    """"A channel that fires on one pixel of one image is alive" -- §5.5."""
    m = probes.as_unit_matrix(_conv_acts())
    dead = probes.dead_exact_mask(m)
    assert dead.tolist() == [True, False, False, False, False]


def test_frac_inputs_active_counts_spatial_positions():
    p = probes.probe_layer(_conv_acts(), _conv_acts(), 0, probes.ProbeConfig())
    assert p.is_spatial and p.n_examples == 4
    assert p.n_inputs == 4 * 3 * 3  # positions, not examples
    # channel 2 fires at exactly 1 of 36 positions
    assert p.frac_inputs_active[2] == pytest.approx(1 / 36)
    assert p.frac_zero_positions[2] == pytest.approx(35 / 36)
    # channel 4 fires on 1 of 3 columns, every row, every image
    assert p.frac_inputs_active[4] == pytest.approx(1 / 3)


def test_the_conv_specific_gap_between_silent_and_dead():
    """A channel can be 97% spatially silent and still count as alive under
    Dohare et al.'s definition. That gap is the finding."""
    p = probes.probe_layer(_conv_acts(), _conv_acts(), 0, probes.ProbeConfig())
    assert not p.exact_zero[2]                      # alive by dead_exact
    assert p.frac_zero_positions[2] > 0.97          # ...and almost entirely silent
    assert p.dead_exact_frac == pytest.approx(1 / 5)
    assert p.frac_zero_positions.mean() > p.dead_exact_frac


def test_sokar_score_normalises_over_channels():
    """Sokar et al. specify the spatial mean for conv layers, so H^l = C."""
    p = probes.probe_layer(_conv_acts(), _conv_acts(), 0, probes.ProbeConfig())
    assert p.sokar_score.shape == (5,)
    assert float(p.sokar_score.mean()) == pytest.approx(1.0, rel=1e-12)
    assert p.sokar_score[0] == 0.0  # the dead channel


def test_dead_absolute_catches_the_quiet_channel_and_dormancy_does_not_alone():
    p = probes.probe_layer(_conv_acts(), _conv_acts(), 0, probes.ProbeConfig())
    assert p.mean_abs_act[3] == pytest.approx(1e-8)
    assert bool((p.mean_abs_act < 1e-6)[3])          # dead_absolute catches it
    assert not p.exact_zero[3]                       # dead_exact does not


# ---------------------------------------------------------------------------
# The dangerous indexing: conv -> flatten -> Linear
# ---------------------------------------------------------------------------


def test_flatten_outgoing_slice_targets_the_right_columns():
    """When the layer after a conv is fully connected, channel c's outgoing
    weights are the flattened columns [c*H*W, (c+1)*H*W).

    Written before the implementation, per CLAUDE.md §5.5: an off-by-H*W here
    zeroes a different channel's outgoing weights and nothing crashes.
    """
    c_out, h, w = 4, 3, 3
    n_next = 6
    w_next = torch.zeros(n_next, c_out * h * w)
    # Mark each column with the channel it belongs to.
    for c in range(c_out):
        w_next[:, c * h * w : (c + 1) * h * w] = float(c)

    cols = interventions.flattened_channel_columns(channel=2, spatial=h * w)
    assert cols == slice(18, 27)
    assert torch.all(w_next[:, cols] == 2.0)

    # And the fold used by the probe must agree with this column layout: a
    # feature map flattened by `.reshape(N, -1)` is channel-major.
    acts = torch.zeros(2, c_out, h, w)
    acts[:, 2] = 5.0
    flat = acts.reshape(2, -1)
    assert torch.all(flat[:, cols] == 5.0)
    assert flat[:, : 2 * h * w].sum() == 0


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@pytest.fixture
def cnn():
    g = torch.Generator().manual_seed(20260806)
    return SmallCNN(in_channels=3, image_size=32, out_features=10, generator=g)


def test_cnn_forward_shapes(cnn):
    x = torch.randn(4, 3, 32, 32)
    logits, pres, posts = cnn.forward_with_activations(x)
    assert logits.shape == (4, 10)
    assert len(posts) == cnn.n_hidden
    # three conv layers (4D) then one fully-connected hidden layer (2D)
    assert [p.dim() for p in posts] == [4, 4, 4, 2]
    assert posts[0].shape[1] == cnn.hidden_dims[0]


def test_cnn_probes_end_to_end(cnn):
    cnn.eval()
    x = torch.randn(8, 3, 32, 32)
    layers = probes.probe_model(cnn, x, probes.ProbeConfig())
    assert len(layers) == cnn.n_hidden
    for lp, units in zip(layers, cnn.hidden_dims):
        assert lp.n_neurons == units
        assert lp.n_examples == 8
        assert np.isfinite(lp.erank)
    assert layers[0].is_spatial and not layers[-1].is_spatial


# ---------------------------------------------------------------------------
# Recycling a channel
# ---------------------------------------------------------------------------


def test_recycling_a_conv_channel_zeroes_the_outgoing_filters(cnn):
    ch = [1, 5]
    with torch.no_grad():
        cnn.outgoing_module(0).weight.fill_(0.4)
    w_in_before = cnn.incoming_module(0).weight[ch].clone()

    interventions.recycle_neurons(cnn, 0, np.array(ch))

    # outgoing for a conv->conv boundary is W_next[:, c, :, :]
    assert torch.all(cnn.outgoing_module(0).weight[:, ch] == 0.0)
    untouched = [c for c in range(cnn.hidden_dims[0]) if c not in ch]
    assert torch.all(cnn.outgoing_module(0).weight[:, untouched] == 0.4)
    assert not torch.allclose(cnn.incoming_module(0).weight[ch], w_in_before)
    assert torch.all(cnn.incoming_module(0).bias[ch] == cnn.init_bias_value(0))


def test_recycling_the_last_conv_layer_zeroes_the_right_flat_columns(cnn):
    """The conv -> flatten -> Linear boundary, end to end."""
    last_conv = 2
    ch = 3
    spatial = cnn.flatten_spatial
    with torch.no_grad():
        cnn.outgoing_module(last_conv).weight.fill_(0.7)

    interventions.recycle_neurons(cnn, last_conv, np.array([ch]))

    w = cnn.outgoing_module(last_conv).weight
    cols = interventions.flattened_channel_columns(ch, spatial)
    assert torch.all(w[:, cols] == 0.0)
    # Every other channel's block is untouched -- the off-by-H*W check.
    for other in range(cnn.hidden_dims[last_conv]):
        if other == ch:
            continue
        assert torch.all(w[:, interventions.flattened_channel_columns(other, spatial)] == 0.7)


def test_redo_preserves_function_on_the_cnn(cnn):
    """CLAUDE.md §8.3 for Setting 2: with tau=0 only dead channels are recycled,
    so zeroing their outgoing weights must leave the function unchanged --
    including across the flatten boundary."""
    with torch.no_grad():
        for li in (0, 1, 2):
            cnn.incoming_module(li).bias[[0, 2]] = -1e6  # force channels dead
    cnn.eval()
    x = torch.randn(6, 3, 32, 32)
    before = cnn(x).clone()

    result = Recycler(RecyclerConfig(kind="redo", tau=0.0, freq=1), seed=1).run_event(
        cnn, optimizer=None, score_x=x, probe_x=x, step=1, task_idx=0
    )
    assert result.total_recycled >= 6
    for row in result.rows:
        assert row["k"] == row["n_dead_exact"]

    assert torch.allclose(before, cnn(x), rtol=1e-5, atol=1e-5), (
        "recycling changed the CNN's function at tau=0; the usual cause is a "
        "wrong outgoing slice across the flatten boundary (CLAUDE.md §5.5)"
    )


def test_gradient_tracker_handles_conv_gradients(cnn):
    """Regression: a Conv2d weight gradient is (C, C_in, kH, kW), so reducing
    only dim 1 leaves (C, kH, kW) and then broadcasts against the (C,) bias
    gradient. That crashed Setting 2 on its first real run."""
    tracker = probes.GradientTracker(list(cnn.linears), window=2)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    cnn.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(cnn(x), y).backward()
    tracker.update()

    for j, mod in enumerate(cnn.linears):
        per_neuron = tracker.neuron_mean(j)
        assert per_neuron.shape == (probes.out_units(mod),), (
            f"layer {j} ({type(mod).__name__}) gave {per_neuron.shape}"
        )
        assert np.all(np.isfinite(per_neuron))
        assert tracker.layer_mean(j) >= per_neuron.max() - 1e-6


def test_weight_norms_reduce_the_right_axes_for_conv(cnn):
    """Each of the three outgoing shapes -- Linear, Conv2d, and Linear-after-
    flatten -- puts the unit on a different axis."""
    for li in range(cnn.n_hidden):
        w_in = cnn.incoming_module(li).weight
        w_out = cnn.outgoing_module(li).weight
        n_in, n_out = probes.neuron_weight_norms(
            w_in, w_out, spatial=cnn.outgoing_spatial(li)
        )
        units = cnn.hidden_dims[li]
        assert n_in.shape == (units,), f"layer {li} incoming {n_in.shape}"
        assert n_out.shape == (units,), f"layer {li} outgoing {n_out.shape}"
        assert np.all(np.isfinite(n_in)) and np.all(np.isfinite(n_out))


def test_outgoing_norm_isolates_one_channel_across_the_flatten(cnn):
    """Zeroing one channel's outgoing block must zero exactly that channel's
    norm -- the off-by-H*W check, in norm form."""
    last_conv = cnn.n_conv - 1
    with torch.no_grad():
        cnn.outgoing_module(last_conv).weight.fill_(1.0)
        cnn.outgoing_module(last_conv).weight[
            :, interventions.flattened_channel_columns(2, cnn.flatten_spatial)
        ] = 0.0
    _, n_out = probes.neuron_weight_norms(
        cnn.incoming_module(last_conv).weight,
        cnn.outgoing_module(last_conv).weight,
        spatial=cnn.flatten_spatial,
    )
    assert n_out[2] == 0.0
    assert np.all(n_out[np.arange(len(n_out)) != 2] > 0)


def test_cnn_trains_end_to_end(tiny_cifar_root, tmp_path):
    """Setting 2 through the real training loop, on synthetic CIFAR.

    The conv unit tests all passed while the actual sweep crashed on its first
    optimizer step, because nothing exercised `Trainer` with a CNN. This does.
    """
    from src.train import Trainer

    cfg = {
        "run_id": "s2_smoke",
        "seed": 0,
        "device": "cpu",
        "data": {
            "name": "label_shuffled_cifar10",
            "root": str(tiny_cifar_root),
            "n_tasks": 2,
            "batch_size": 64,
            "flatten": False,
        },
        "model": {
            "architecture": "cnn",
            "channels": [4, 6],
            "fc_dims": [8],
            "image_size": 32,
        },
        "optim": {"name": "sgd", "lr": 0.01, "momentum": 0.9},
        "recycling": {"kind": "redo", "tau": 0.1, "freq": 2},
        "probe": {"n_probe": 32},
        "checkpoint": {"every_tasks": 1},
    }
    run_dir = Trainer(cfg, runs_root=tmp_path / "runs").run()

    import pyarrow.parquet as pq

    neurons = pq.read_table(run_dir / "neurons.parquet")
    # (2 tasks + init) x (4 + 6 + 8 units)
    assert neurons.num_rows == 3 * (4 + 6 + 8)
    cols = neurons.to_pydict()
    assert all(np.isfinite(cols["w_in_norm"]))
    assert all(np.isfinite(cols["w_out_norm"]))
    assert (run_dir / "recycling.parquet").exists()

    m = pq.read_table(run_dir / "metrics.parquet").to_pydict()
    assert any(m["is_spatial"]), "conv layers must be flagged spatial"
    assert not all(m["is_spatial"]), "the fc hidden layer must not be"


def test_conv_init_resampling_uses_the_layers_own_distribution(cnn):
    w = cnn.sample_incoming_weights(0, 64)
    spec = cnn.init_specs[0]
    assert w.shape == (64,) + tuple(cnn.incoming_module(0).weight.shape[1:])
    assert float(w.abs().max()) <= spec.bound + 1e-6


# ---------------------------------------------------------------------------
# Setting 3: smooth activations make dead_exact vanish by construction
# ---------------------------------------------------------------------------


def _smooth_model(act, bias, seed=7):
    """Every unit pinned to exactly `bias` pre-activation.

    Weights are zeroed so the pre-activation is the bias exactly, on every
    input. Without that, weight variance pushes some units past GELU's underflow
    point and others not, and the test measures the seed rather than the
    activation function.
    """
    from src.models import MLP

    g = torch.Generator().manual_seed(seed)
    model = MLP(
        in_features=16, hidden_dims=(24,), out_features=3, activation=act, generator=g
    )
    with torch.no_grad():
        model.incoming_linear(0).weight.zero_()
        model.incoming_linear(0).bias[:] = bias
    model.eval()
    return probes.probe_model(
        model, torch.rand(64, 16, generator=g), probes.ProbeConfig()
    )[0]


@pytest.mark.parametrize(
    "act, bias",
    [
        # Chosen just above each activation's float32 underflow point, so the
        # unit is functionally dead but not *exactly* zero. The two biases
        # differ by 4x because the underflow points do -- itself the point.
        ("gelu", -5.0),
        ("silu", -20.0),
    ],
)
def test_smooth_activations_hide_units_from_dead_exact(act, bias):
    """Setting 3's C2 result (CLAUDE.md §9): a functionally dead GELU/SiLU unit
    emits something tiny rather than zero, so Dohare et al.'s definition -- the
    one in the *Nature* paper -- reports nothing, while `dead_absolute` flags
    the same units as thoroughly dead."""
    p = _smooth_model(act, bias=bias)
    assert p.dead_exact_count == 0
    assert p.dead_absolute_count(1e-4) == p.n_neurons
    assert p.saturated is None  # unbounded: not applicable, never 0.0


def test_silu_essentially_cannot_produce_an_exact_zero():
    """SiLU needs a pre-activation near -90 before sigmoid underflows, which no
    trained network reaches. For SiLU the 'by construction' claim holds."""
    assert _smooth_model("silu", bias=-50.0).dead_exact_count == 0


def test_gelu_dead_exact_is_a_floating_point_artefact_not_a_property():
    """**Correction to CLAUDE.md §9.** `dead_exact` is NOT identically zero
    under GELU: in float32, ``1 + erf(x/sqrt(2))`` rounds to zero at
    x <= -5.55, so GELU underflows to exactly 0.0 there -- a pre-activation
    trained networks do reach.

    Worse for the definition, the threshold is dtype-dependent (-8.40 in
    float64), so the *same unit with the same weights on the same inputs* is
    "dead" in float32 and "alive" in float64. Dohare et al.'s definition is not
    merely activation-dependent; it is not dtype-independent either. That is a
    C2 finding in its own right and belongs in the paper rather than being
    engineered around.
    """
    gelu = torch.nn.GELU(approximate="none")
    x = torch.tensor([-7.0])
    assert float(gelu(x.float())) == 0.0
    assert float(gelu(x.double())) != 0.0

    # And it fires through the probe at a realistic magnitude.
    assert _smooth_model("gelu", bias=-7.0).dead_exact_count == 24
    assert _smooth_model("gelu", bias=-3.0).dead_exact_count == 0


def test_tanh_saturation_fires_where_dead_exact_cannot():
    """The other half of Setting 3: under tanh, `saturated` catches units that
    `dead_exact` structurally cannot."""
    from src.models import MLP

    g = torch.Generator().manual_seed(7)
    model = MLP(
        in_features=16, hidden_dims=(24,), out_features=3, activation="tanh", generator=g
    )
    with torch.no_grad():
        model.incoming_linear(0).bias[:] = -50.0  # pinned at -1
    model.eval()

    p = probes.probe_model(model, torch.rand(64, 16, generator=g), probes.ProbeConfig())[0]
    assert p.dead_exact_count == 0
    assert p.saturated_frac == pytest.approx(1.0)
