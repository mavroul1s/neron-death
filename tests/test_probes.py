"""Metric tests. Covers CLAUDE.md §8 items 1, 2 and 5.

The second test is the important one: it asserts that `dormant_tau` **fails** to
detect a uniformly shrunken layer. If that test ever starts passing in the other
direction, the layer-mean normalisation has been "fixed" and every C2 number is
wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src import probes
from src.models import MLP
from tests.conftest import kill_units, make_model


# ---------------------------------------------------------------------------
# CLAUDE.md §8.1 -- forced-dead network
# ---------------------------------------------------------------------------


def test_dead_exact_detects_forced_dead_units(gen):
    model = make_model(gen, hidden=(16, 16))
    dead_units = [1, 4, 9, 15]
    kill_units(model, 0, dead_units)
    model.eval()

    x = torch.rand(256, model.in_features, generator=gen)
    layers = probes.probe_model(model, x, probes.ProbeConfig())
    layer0 = layers[0]

    found = set(np.flatnonzero(layer0.exact_zero).tolist())
    assert found == set(dead_units)

    # The other three definitions must agree on units that are exactly zero.
    assert set(np.flatnonzero(layer0.mean_abs_act < 1e-6).tolist()) >= set(dead_units)
    assert set(layer0.dormant_indices(0.0).tolist()) == set(dead_units)
    # frac_inputs_active is the exact complement of exact_zero -- the identity
    # the C1 composition table depends on.
    assert np.array_equal(layer0.exact_zero, layer0.frac_inputs_active == 0.0)


def test_dead_exact_uses_no_tolerance(gen):
    """A neuron that is merely tiny is NOT dead_exact (Dohare et al. 2024)."""
    h = torch.full((64, 5), 1e-30)
    h[:, 2] = 0.0
    mask = probes.dead_exact_mask(h)
    assert mask.tolist() == [False, False, True, False, False]


def test_saturated_is_none_for_unbounded_activations(gen):
    relu_model = make_model(gen, activation="relu")
    tanh_model = make_model(gen, activation="tanh")
    x = torch.rand(64, relu_model.in_features, generator=gen)

    assert probes.probe_model(relu_model, x, probes.ProbeConfig())[0].saturated is None
    # Never 0.0 for unbounded -- "not applicable" must not read as "none found".
    assert probes.probe_model(relu_model, x, probes.ProbeConfig())[0].saturated_frac is None
    assert probes.probe_model(tanh_model, x, probes.ProbeConfig())[0].saturated is not None


def test_saturated_detects_pinned_tanh_units():
    h = torch.zeros(32, 4)
    h[:, 0] = 1.0 - 1e-9  # pinned at +1
    h[:, 1] = -1.0 + 1e-9  # pinned at -1
    h[:, 2] = 0.3  # healthy
    h[:, 3] = torch.linspace(-1.0, 1.0, 32)  # spans the range
    mask = probes.saturated_mask(h, extremes=(-1.0, 1.0), eps=1e-3)
    assert mask.tolist() == [True, True, False, False]


# ---------------------------------------------------------------------------
# CLAUDE.md §8.2 -- uniformly shrunken layer
#
# "assert dormant_tau reports near 0% (it should MISS this) while dead_absolute
#  catches it. If dormant_tau catches it, the layer-mean normalisation is
#  implemented wrongly."
# ---------------------------------------------------------------------------


def test_dormant_tau_is_blind_to_uniform_shrinkage_tensor_level():
    """The failure mode in its purest form: scaling every activation in a layer
    by 1e-8 leaves the Sokar score exactly unchanged, because the layer mean
    shrinks by the same factor."""
    h = torch.rand(256, 20) + 0.5  # every neuron healthy and comparable
    scores = probes.sokar_scores(probes.mean_abs_activation(h))
    assert (scores > 0.25).all(), "fixture must start with no dormant neurons"

    h_shrunk = h * 1e-8
    scores_shrunk = probes.sokar_scores(probes.mean_abs_activation(h_shrunk))

    # Scale invariance -- this is the bug, faithfully reproduced.
    assert torch.allclose(scores, scores_shrunk, rtol=1e-5)
    for tau in probes.DEFAULT_TAUS:
        assert probes.dormant_mask(scores_shrunk, tau).sum() == 0

    # dead_absolute, having no layer normalisation, catches all of them.
    mean_abs = probes.mean_abs_activation(h_shrunk)
    assert probes.dead_absolute_mask(mean_abs, 1e-6).all()
    # ...and they are not exactly zero, so dead_exact correctly misses them.
    assert probes.dead_exact_mask(h_shrunk).sum() == 0


def test_dormant_tau_is_blind_to_uniform_shrinkage_network_level(gen):
    """The same thing through a real forward pass.

    ReLU is positively homogeneous, so scaling a layer's W and b by s > 0 scales
    that layer's post-activations by exactly s.
    """
    model = make_model(gen, hidden=(16, 16), in_features=32)
    with torch.no_grad():
        # All-positive weights and inputs => every neuron has a comparable,
        # healthy activation, so nothing is dormant before we shrink.
        model.incoming_linear(0).weight.uniform_(0.5, 1.0, generator=gen)
        model.incoming_linear(0).bias.fill_(0.5)
    model.eval()
    x = torch.rand(256, model.in_features, generator=gen)
    cfg = probes.ProbeConfig()

    before = probes.probe_model(model, x, cfg)[0]
    for tau in cfg.taus:
        assert before.dormant_count(tau) == 0, f"fixture has dormant units at tau={tau}"
    assert before.dead_absolute_count(1e-6) == 0

    with torch.no_grad():
        model.incoming_linear(0).weight.mul_(1e-8)
        model.incoming_linear(0).bias.mul_(1e-8)

    after = probes.probe_model(model, x, cfg)[0]

    # THE assertion: tau-dormancy still reports nothing, at every tau.
    for tau in cfg.taus:
        assert after.dormant_count(tau) == 0, (
            f"dormant_tau caught a uniformly shrunken layer at tau={tau}; the "
            "layer-mean normalisation is implemented wrongly (CLAUDE.md §8.2)"
        )
    # dead_absolute catches the whole layer.
    assert after.dead_absolute_count(1e-6) == after.n_neurons
    # dead_exact misses it: the units are tiny, not zero.
    assert after.dead_exact_count == 0
    # And the Sokar scores are essentially unchanged, which is why it misses.
    assert np.allclose(before.sokar_score, after.sokar_score, rtol=1e-3)


# ---------------------------------------------------------------------------
# CLAUDE.md §8.5 -- effective rank
# ---------------------------------------------------------------------------


def _matrix_with_singular_values(sv, n_rows=64, n_cols=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    u, _ = torch.linalg.qr(torch.randn(n_rows, n_cols, generator=g, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(n_cols, n_cols, generator=g, dtype=torch.float64))
    s = torch.tensor(sv, dtype=torch.float64)
    return (u[:, : len(sv)] * s) @ v[: len(sv), :]


@pytest.mark.parametrize(
    "sv, expected",
    [
        ([1.0], 1.0),  # rank 1 -> erank exactly 1
        ([1.0, 1.0], 2.0),  # r equal singular values -> erank exactly r
        ([1.0, 1.0, 1.0], 3.0),
        ([1.0, 1.0, 1.0, 1.0, 1.0], 5.0),
        ([2.0, 2.0, 2.0], 3.0),  # scale invariant
        ([1.0, 1.0, 0.0, 0.0], 2.0),  # zero singular values contribute nothing
    ],
)
def test_effective_rank_on_known_spectra(sv, expected):
    m = _matrix_with_singular_values(sv)
    assert probes.effective_rank(m) == pytest.approx(expected, rel=1e-9)


def test_effective_rank_of_skewed_spectrum():
    """A dominant direction plus a weak one: erank must sit strictly between
    1 and 2 and match the closed-form entropy."""
    sv = [1.0, 0.1]
    p = np.array(sv) / np.sum(sv)
    expected = math.exp(-(p * np.log(p)).sum())
    assert 1.0 < expected < 2.0
    assert probes.effective_rank(_matrix_with_singular_values(sv)) == pytest.approx(
        expected, rel=1e-9
    )


def test_effective_rank_of_zero_matrix_is_zero():
    assert probes.effective_rank(torch.zeros(32, 8)) == 0.0


# ---------------------------------------------------------------------------
# Supporting definitions
# ---------------------------------------------------------------------------


def test_sokar_score_layer_mean_is_one():
    h = torch.rand(128, 32) + 0.1
    scores = probes.sokar_scores(probes.mean_abs_activation(h))
    assert float(scores.mean()) == pytest.approx(1.0, rel=1e-12)


def test_sokar_score_of_wholly_dead_layer_is_zero():
    """0/0 must resolve to 0 -- every neuron dormant at every tau, which is the
    semantically correct answer for a dead layer."""
    scores = probes.sokar_scores(probes.mean_abs_activation(torch.zeros(64, 8)))
    assert torch.equal(scores, torch.zeros(8, dtype=torch.float64))
    assert probes.dormant_mask(scores, 0.0).all()


def test_dormant_threshold_is_inclusive():
    """Sokar: tau-dormant iff s <= tau."""
    scores = torch.tensor([0.05, 0.1, 0.15], dtype=torch.float64)
    assert probes.dormant_mask(scores, 0.1).tolist() == [True, True, False]


def test_dead_absolute_threshold_is_strict():
    """CLAUDE.md §5.1: E|h| < a."""
    mean_abs = torch.tensor([0.5e-6, 1e-6, 2e-6], dtype=torch.float64)
    assert probes.dead_absolute_mask(mean_abs, 1e-6).tolist() == [True, False, False]


def test_sign_entropy_extremes():
    pre = torch.zeros(100, 3)
    pre[:, 0] = 1.0  # always positive -> 0 bits
    pre[:, 1] = -1.0  # always negative -> 0 bits
    pre[:50, 2] = 1.0
    pre[50:, 2] = -1.0  # balanced -> 1 bit
    per_neuron = probes.sign_entropy_per_neuron(pre).numpy()
    assert per_neuron == pytest.approx([0.0, 0.0, 1.0])
    assert probes.sign_entropy(pre) == pytest.approx(1.0 / 3.0)


def test_weight_stats():
    t = torch.tensor([[3.0, -4.0]])
    l2, mean_abs = probes.weight_stats([t])
    assert l2 == pytest.approx(5.0)
    assert mean_abs == pytest.approx(3.5)


def test_neuron_weight_norms_use_the_right_axis(gen):
    """w_in is a row of this layer's weight; w_out is a COLUMN of the next
    layer's. Getting the axis wrong here would silently mislabel every
    per-neuron weight column in the C4 dataset."""
    model = make_model(gen, hidden=(3, 5), in_features=4, out_features=2)
    with torch.no_grad():
        model.incoming_linear(0).weight.copy_(
            torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 6.0, 8.0], [1.0, 0.0, 0.0, 0.0]])
        )
        # Next layer is (5, 3): column 1 is neuron 1's outgoing weights.
        model.outgoing_linear(0).weight.zero_()
        model.outgoing_linear(0).weight[0, 1] = 3.0
        model.outgoing_linear(0).weight[1, 1] = 4.0
    w_in, w_out = probes.neuron_weight_norms(
        model.incoming_linear(0).weight, model.outgoing_linear(0).weight
    )
    assert w_in == pytest.approx([5.0, 10.0, 1.0])
    assert w_out[1] == pytest.approx(5.0)


def test_composition_partitions_the_recycled_set(gen):
    model = make_model(gen, hidden=(16, 16))
    kill_units(model, 0, [2, 5, 11])
    model.eval()
    x = torch.rand(128, model.in_features, generator=gen)
    layer0 = probes.probe_model(model, x, probes.ProbeConfig())[0]

    idx = np.arange(layer0.n_neurons)
    comp = probes.composition(layer0, idx)
    assert comp.k == layer0.n_neurons
    assert comp.n_dead_exact == 3
    # No neuron falls between the two categories, ever.
    assert comp.n_dead_exact + comp.n_alive_but_quiet == comp.k


def test_gradient_tracker_averages_the_window(gen):
    model = make_model(gen, hidden=(4,), in_features=3, out_features=2)
    tracker = probes.GradientTracker(list(model.linears), window=3)
    x = torch.rand(8, 3, generator=gen)
    y = torch.randint(0, 2, (8,), generator=gen)

    for _ in range(5):
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        tracker.update()

    assert tracker.n_steps_in_window == 3  # window saturates, does not grow
    per_neuron = tracker.neuron_mean(0)
    assert per_neuron.shape == (4,)
    # Layer norm is the norm of the whole gradient, i.e. >= any single neuron's.
    assert tracker.layer_mean(0) >= per_neuron.max() - 1e-9

    tracker.reset()
    assert tracker.n_steps_in_window == 0
    assert math.isnan(tracker.layer_mean(0))
