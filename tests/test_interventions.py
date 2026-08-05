"""Intervention tests. Covers CLAUDE.md §8 items 3 and 4.

`test_redo_preserves_function` is the guard against the specific
reimplementation bug CLAUDE.md §6 calls out: masking incoming weights but not
outgoing ones. That bug does not crash and does not look wrong in the loss
curve, so a test is the only thing that catches it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src import interventions, probes
from src.interventions import Recycler, RecyclerConfig
from tests.conftest import kill_units, make_model, quieten_units


def _dead_model(gen, dead_per_layer=((1, 4, 9), (0, 7))):
    model = make_model(gen, hidden=(16, 16), in_features=32, out_features=4)
    for layer_idx, units in enumerate(dead_per_layer):
        kill_units(model, layer_idx, units)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CLAUDE.md §8.3 -- ReDo function preservation at tau = 0
# ---------------------------------------------------------------------------


def test_redo_preserves_function(gen):
    """With tau = 0 only genuinely dead units are recycled, and zeroing their
    outgoing weights means the network computes exactly the same function."""
    model = _dead_model(gen)
    x = torch.rand(64, model.in_features, generator=gen)
    before = model(x).clone()

    recycler = Recycler(RecyclerConfig(kind="redo", tau=0.0, freq=1), seed=7)
    result = recycler.run_event(
        model,
        optimizer=None,
        score_x=x,
        probe_x=x,
        step=1,
        task_idx=0,
        ref_x=None,
    )
    # The units this test killed are all recycled. The second hidden layer also
    # contains units that died on their own -- which is the real population ReDo
    # targets, so it belongs in the fixture rather than being engineered away.
    assert set(result.recycled[0].tolist()) >= {1, 4, 9}
    assert set(result.recycled[1].tolist()) >= {0, 7}
    for row in result.rows:
        assert row["k"] == row["n_dead_exact"], (
            "at tau=0 only exactly-dead units may be recycled"
        )

    after = model(x)
    assert torch.allclose(before, after, rtol=1e-6, atol=1e-6), (
        "ReDo changed the network's function at tau=0. The usual cause is "
        "failing to zero the OUTGOING weights (CLAUDE.md §6)."
    )


def test_redo_zeroes_outgoing_and_resamples_incoming(gen):
    """The two halves of Algorithm 1, checked separately."""
    model = _dead_model(gen)
    units = [1, 4, 9]
    w_in_before = model.incoming_linear(0).weight[units].clone()

    with torch.no_grad():  # make sure the outgoing weights start non-zero
        model.outgoing_linear(0).weight.fill_(0.3)

    interventions.recycle_neurons(model, 0, np.array(units), weight_generator=gen)

    assert torch.all(model.outgoing_linear(0).weight[:, units] == 0.0)
    assert not torch.allclose(model.incoming_linear(0).weight[units], w_in_before)
    # Untouched neurons keep their outgoing weights.
    untouched = [i for i in range(16) if i not in units]
    assert torch.all(model.outgoing_linear(0).weight[:, untouched] == 0.3)
    # Bias returns to its initialisation value.
    assert torch.all(model.incoming_linear(0).bias[units] == model.init_bias_value(0))


def test_redo_touches_the_output_layer_for_the_last_hidden_layer(gen):
    """The last hidden layer's 'outgoing' weights live in the output layer --
    the case a partial implementation forgets."""
    model = _dead_model(gen)
    last = model.n_hidden - 1
    with torch.no_grad():
        model.outgoing_linear(last).weight.fill_(0.5)
    assert model.outgoing_linear(last) is model.linears[-1]

    interventions.recycle_neurons(model, last, np.array([0, 7]), weight_generator=gen)
    assert torch.all(model.linears[-1].weight[:, [0, 7]] == 0.0)


def test_resampled_weights_come_from_the_original_init_distribution(gen):
    """Sampled from the layer's stored InitSpec, not from PyTorch's default."""
    model = make_model(gen, hidden=(2048,), in_features=64, out_features=4)
    spec = model.init_specs[0]
    w = model.sample_incoming_weights(0, 2048, gen)
    assert w.shape == (2048, 64)
    assert float(w.abs().max()) <= spec.bound + 1e-6
    # U(-b, b) has std b/sqrt(3); 2048x64 samples pin it tightly.
    assert float(w.std()) == pytest.approx(spec.bound / np.sqrt(3.0), rel=0.05)


def test_optimizer_state_is_reset_for_recycled_slices_only(gen):
    model = _dead_model(gen)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x = torch.rand(32, model.in_features, generator=gen)
    y = torch.randint(0, 4, (32,), generator=gen)
    torch.nn.functional.cross_entropy(model(x), y).backward()
    opt.step()  # populates momentum buffers

    buf = opt.state[model.incoming_linear(0).weight]["momentum_buffer"]
    with torch.no_grad():
        buf.fill_(1.0)
    interventions.recycle_neurons(
        model, 0, np.array([1, 4]), weight_generator=gen, optimizer=opt
    )
    assert torch.all(buf[[1, 4]] == 0.0)
    assert torch.all(buf[[0, 2, 3, 5]] == 1.0)


# ---------------------------------------------------------------------------
# CLAUDE.md §8.4 -- random-matched cardinality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["redo", "random_matched", "inverse_matched"])
@pytest.mark.parametrize("tau", [0.0, 0.05, 0.1, 0.25])
def test_every_arm_recycles_exactly_k_equals_dormant_count(gen, kind, tau):
    """k must equal |tau-dormant set|, per layer, per event -- recomputed every
    time, never a fixed fraction or a schedule (CLAUDE.md §6)."""
    model = _dead_model(gen)
    x = torch.rand(64, model.in_features, generator=gen)

    # Ground truth from the probe module, independently of the recycler.
    layers = probes.probe_model(model, x, probes.ProbeConfig())
    expected_k = [int((lp.sokar_score <= tau).sum()) for lp in layers]

    recycler = Recycler(RecyclerConfig(kind=kind, tau=tau, freq=1), seed=3)
    result = recycler.run_event(
        model, optimizer=None, score_x=x, probe_x=x, step=1, task_idx=0, ref_x=None
    )

    for row in result.rows:
        li = row["layer_idx"]
        assert row["k"] == expected_k[li] == row["n_dormant"], (
            f"arm={kind} tau={tau} layer={li}: recycled {row['k']} neurons but "
            f"the dormant set has {expected_k[li]}"
        )
        assert result.recycled[li].size == expected_k[li]


def test_k_is_recomputed_at_every_event(gen):
    """Killing more units between events must change k."""
    model = _dead_model(gen, dead_per_layer=((1,), ()))
    x = torch.rand(64, model.in_features, generator=gen)
    recycler = Recycler(RecyclerConfig(kind="random_matched", tau=0.0, freq=1), seed=1)

    k1 = recycler.run_event(
        model, None, score_x=x, probe_x=x, step=1, task_idx=0
    ).rows[0]["k"]
    kill_units(model, 0, [2, 3, 5])
    k2 = recycler.run_event(
        model, None, score_x=x, probe_x=x, step=2, task_idx=0
    ).rows[0]["k"]
    assert k1 == 1 and k2 == 4


def test_inverse_matched_selects_the_highest_scoring_neurons(gen):
    scores = np.array([0.0, 0.0, 0.5, 2.0, 3.0, 1.0])
    rng = np.random.default_rng(0)
    selected, dormant = interventions.select_recycle_indices(
        "inverse_matched", scores, tau=0.0, rng=rng
    )
    assert dormant.tolist() == [0, 1]
    assert selected.tolist() == [3, 4]  # the two highest


def test_random_matched_draws_from_the_whole_layer(gen):
    """Uniform over all neurons, not over the non-dormant complement."""
    scores = np.array([0.0] + [1.2] * 99)
    rng = np.random.default_rng(0)
    picks = [
        int(interventions.select_recycle_indices("random_matched", scores, 0.0, rng)[0][0])
        for _ in range(400)
    ]
    assert 0 in picks, "the dormant neuron must be eligible for the random draw"
    assert len(set(picks)) > 50, "draw does not look uniform over the layer"


def test_no_dormant_units_means_no_recycling(gen):
    scores = np.array([1.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    for kind in ("redo", "random_matched", "inverse_matched"):
        selected, dormant = interventions.select_recycle_indices(kind, scores, 0.0, rng)
        assert selected.size == 0 and dormant.size == 0


# ---------------------------------------------------------------------------
# Composition logging -- the C1 headline table
# ---------------------------------------------------------------------------


def test_composition_row_splits_dead_from_quiet(gen):
    """At a tau large enough to catch living units, the recycled set contains
    both genuinely dead and merely quiet neurons, and the counts add up to k.
    This is the number the paper's main figure is built from."""
    model = _dead_model(gen, dead_per_layer=((1, 4, 9), ()))
    quieten_units(model, 0, [2, 5], value=0.01)  # alive, but very quiet
    x = torch.rand(512, model.in_features, generator=gen)

    recycler = Recycler(RecyclerConfig(kind="redo", tau=0.5, freq=1), seed=5, run_id="t")
    rows = recycler.run_event(
        model, None, score_x=x, probe_x=x, step=1, task_idx=0, ref_x=x
    ).rows

    row0 = rows[0]
    assert row0["n_dead_exact"] == 3
    assert row0["n_alive_but_quiet"] >= 2, (
        "at tau=0.5 the recycled set should include living, quiet neurons"
    )
    assert row0["n_dead_exact"] + row0["n_alive_but_quiet"] == row0["k"]
    assert set(row0) >= {
        "run_id", "step", "layer_idx", "tau", "k",
        "n_dead_exact", "n_alive_but_quiet", "mean_sokar_score",
    }
    assert "n_dead_exact_ref" in row0  # reference-batch composition logged too


def test_events_fire_on_the_frequency(gen):
    r = Recycler(RecyclerConfig(kind="redo", tau=0.0, freq=1000), seed=0)
    assert not r.due(0), "an event at step 0 would recycle the initialisation"
    assert not r.due(999)
    assert r.due(1000) and r.due(2000)
    assert not Recycler(RecyclerConfig(kind="none"), seed=0).due(1000)


def test_recycler_state_round_trips(gen):
    """Recycling must be reproducible across a checkpoint."""
    cfg = RecyclerConfig(kind="random_matched", tau=0.5, freq=1)
    model_a = _dead_model(gen)
    x = torch.rand(64, model_a.in_features, generator=gen)

    r1 = Recycler(cfg, seed=11)
    r1.run_event(model_a, None, score_x=x, probe_x=x, step=1, task_idx=0)
    state = r1.state_dict()
    picks_1 = r1.run_event(
        model_a, None, score_x=x, probe_x=x, step=2, task_idx=0
    ).recycled[0]

    r2 = Recycler(cfg, seed=11)
    r2.load_state_dict(state)
    model_b = _dead_model(torch.Generator().manual_seed(20260805))
    picks_2 = r2.run_event(
        model_b, None, score_x=x, probe_x=x, step=2, task_idx=0
    ).recycled[0]

    assert np.array_equal(picks_1, picks_2)
    assert r2.event_idx == r1.event_idx


# ---------------------------------------------------------------------------
# L2 and shrink-and-perturb
# ---------------------------------------------------------------------------


def test_l2_penalty_gradient_matches_sgd_weight_decay(gen):
    """The explicit penalty must be exactly equivalent to weight_decay=lambda,
    so the two never silently stack."""
    lam = 0.01
    model = make_model(gen, hidden=(8,), in_features=6, out_features=3)
    x = torch.rand(16, 6, generator=gen)
    y = torch.randint(0, 3, (16,), generator=gen)

    model.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    (loss + lam * interventions.l2_penalty(model)).backward()
    with_penalty = [p.grad.clone() for p in model.parameters()]

    model.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(x), y).backward()
    manual = [p.grad + lam * p.detach() for p in model.parameters()]

    for a, b in zip(with_penalty, manual):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7)


def test_shrink_and_perturb_shrinks_toward_zero(gen):
    model = make_model(gen, hidden=(64,), in_features=32, out_features=4)
    before = model.linears[0].weight.abs().mean().item()
    interventions.shrink_and_perturb(model, shrink=0.5, perturb=0.0, generator=gen)
    after = model.linears[0].weight.abs().mean().item()
    assert after == pytest.approx(0.5 * before, rel=1e-5)


def test_shrink_and_perturb_adds_init_scale_noise(gen):
    model = make_model(gen, hidden=(1024,), in_features=64, out_features=4)
    with torch.no_grad():
        model.linears[0].weight.zero_()
    interventions.shrink_and_perturb(model, shrink=0.0, perturb=1.0, generator=gen)
    spec = model.init_specs[0]
    assert float(model.linears[0].weight.detach().std()) == pytest.approx(
        spec.bound / np.sqrt(3.0), rel=0.05
    )
