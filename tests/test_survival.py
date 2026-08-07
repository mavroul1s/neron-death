"""The C4 survival instrument, checked against trajectories worked by hand.

CLAUDE.md §10: when a result looks surprising, suspect the instrument first.
Everything in `src/analysis/survival.py` is an indexing argument -- which row is
"the boundary before", which column is "the same unit" -- and every one of those
mistakes produces a plausible number rather than a crash. So each test fixes a
tiny trajectory whose answer can be read off by eye.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import survival


def make_frame(
    dead,
    recycled=None,
    times=None,
    layers=None,
    mean_abs_act=None,
    run_id="r",
    dead_ref=None,
):
    """Long-format per-neuron log from (T, U) matrices.

    `layers` assigns each unit column to a layer; neuron_idx is the unit's
    position within its layer, matching how `probes` writes the real log.
    """
    dead = np.asarray(dead, dtype=bool)
    T, U = dead.shape
    recycled = np.zeros_like(dead) if recycled is None else np.asarray(recycled, bool)
    times = np.arange(T) if times is None else np.asarray(times)
    layers = np.zeros(U, int) if layers is None else np.asarray(layers)
    act = np.zeros((T, U)) if mean_abs_act is None else np.asarray(mean_abs_act, float)

    neuron = np.concatenate(
        [np.arange(int((layers == l).sum())) for l in sorted(set(layers.tolist()))]
    )
    rows = []
    for i in range(T):
        for j in range(U):
            row = dict(
                run_id=run_id,
                task_idx=int(times[i]),
                layer_idx=int(layers[j]),
                neuron_idx=int(neuron[j]),
                exact_zero_flag=bool(dead[i, j]),
                was_recycled_this_task=bool(recycled[i, j]),
                mean_abs_act=float(act[i, j]),
            )
            if dead_ref is not None:
                row["exact_zero_flag_ref"] = bool(np.asarray(dead_ref, bool)[i, j])
            rows.append(row)
    return pd.DataFrame(rows)


# -- the pivot ----------------------------------------------------------------


def test_panel_survives_row_order():
    """A unit's history must be assembled by identity, not by file order.

    A misaligned pivot would attribute one neuron's deaths to another and every
    downstream number would still look reasonable.
    """
    rng = np.random.default_rng(0)
    dead = rng.random((6, 4)) < 0.4
    rec = rng.random((6, 4)) < 0.2
    df = make_frame(dead, rec, layers=[0, 0, 1, 1])

    ordered = survival.panel_from_frame(df)
    shuffled = survival.panel_from_frame(df.sample(frac=1.0, random_state=1))

    assert np.array_equal(ordered.dead, shuffled.dead)
    assert np.array_equal(ordered.recycled, shuffled.recycled)
    assert np.array_equal(ordered.layer, shuffled.layer)
    assert np.array_equal(ordered.dead, dead)


def test_panel_rejects_a_ragged_log():
    df = make_frame(np.zeros((4, 3), bool))
    with pytest.raises(ValueError, match="ragged"):
        survival.panel_from_frame(df.iloc[:-1])


def test_panel_rejects_duplicated_cells():
    """Right row count, wrong cells: the check must be on coverage, not length."""
    df = make_frame(np.zeros((3, 2), bool))
    df = pd.concat([df.iloc[:1], df.iloc[:-1]], ignore_index=True)  # dup one, drop one
    with pytest.raises(ValueError, match="duplicate or missing"):
        survival.panel_from_frame(df)


def test_init_row_is_recognised_and_excluded_from_the_trained_window():
    df = make_frame(np.zeros((3, 2), bool), times=[-1, 0, 1])
    panel = survival.panel_from_frame(df)
    assert panel.has_init
    assert panel.trained_times.tolist() == [0, 1]
    assert panel.trained(panel.dead).shape[0] == 2


# -- Kaplan-Meier -------------------------------------------------------------


def test_kaplan_meier_hand_computed_with_administrative_censoring():
    """Four units, two deaths, two survivors censored at the horizon."""
    km = survival.kaplan_meier(
        death_task=np.array([1, 2, 3, 3]),
        event=np.array([True, True, False, False]),
        horizon=3,
    )
    assert km.survival.tolist() == pytest.approx([1.0, 0.75, 0.5, 0.5])
    assert km.at_risk.tolist() == [4, 4, 3, 2]
    assert km.n_events.tolist() == [0, 1, 1, 0]


def test_kaplan_meier_differs_from_empirical_survival_when_censoring_is_early():
    """A unit censored mid-run must leave the risk set, not count as a survivor.

    Empirical survival here would be 2/4 = 0.5; the product-limit answer is
    0.375. If a run is ever truncated, this is the difference between an honest
    curve and one biased upward exactly where the deaths are.
    """
    km = survival.kaplan_meier(
        death_task=np.array([1, 2, 1, 3]),
        event=np.array([True, True, False, False]),
        horizon=3,
    )
    assert km.survival.iloc[-1] == pytest.approx(0.375)


def test_units_dead_at_init_are_prevalent_not_incident():
    """Never observed alive means no time-to-event; counting them as dying at
    task 0 would invent an event training did not cause."""
    # Unit 0 is dead from the init probe onward; unit 1 is alive at init and at
    # task 0, and dies at task 1.
    dead = np.array([[True, False], [True, False], [True, True]])
    panel = survival.panel_from_frame(make_frame(dead, times=[-1, 0, 1]))
    fd = survival.first_death(panel)
    assert fd.prevalent.tolist() == [True, False]
    assert fd.event.tolist() == [False, True]
    assert fd.death_task.tolist() == [0, 1]  # unit 0's value is not an event


# -- the two probe batches ----------------------------------------------------


def test_recovery_is_read_off_whichever_probe_batch_is_asked_for():
    """The confound this separates is the point of logging two batches.

    Here the unit revives on the current task's inputs and stays silent on the
    fixed reference batch -- i.e. the permutation moved, the unit did not. A
    version that ignored `probe` would report a spontaneous recovery.
    """
    dead = np.array([[False], [True], [False]])
    dead_ref = np.array([[False], [True], [True]])
    panel = survival.panel_from_frame(make_frame(dead, dead_ref=dead_ref))

    cur = survival.transitions(panel, per_layer=False, probe="current").iloc[0]
    ref = survival.transitions(panel, per_layer=False, probe="reference").iloc[0]
    assert cur.n_dead_alive == 1 and cur.p_recover == pytest.approx(1.0)
    assert ref.n_dead_alive == 0 and ref.p_recover == pytest.approx(0.0)
    assert ref.n_dead_dead == 1


def test_reference_probe_is_refused_when_the_log_lacks_it():
    """Silently falling back to the current batch would answer a different
    question under the same name."""
    panel = survival.panel_from_frame(make_frame(np.zeros((3, 1), bool)))
    with pytest.raises(ValueError, match="exact_zero_flag_ref"):
        survival.transitions(panel, probe="reference")


def test_analyse_reports_both_probes_when_both_are_logged(tmp_path):
    dead = np.array([[False], [True], [False]])
    dead_ref = np.array([[False], [True], [True]])
    path = tmp_path / "neurons.parquet"
    make_frame(dead, dead_ref=dead_ref).to_parquet(path)

    tables = survival.analyse(path)
    assert set(tables["transitions"]["probe"]) == {"current", "reference"}
    assert set(tables["episodes"]["probe"]) == {"current", "reference"}
    assert "survival_matrix_reference" in tables


# -- transitions --------------------------------------------------------------


def test_transitions_on_a_hand_worked_trajectory():
    dead = np.array(
        [
            [False, False],
            [True, False],
            [True, True],
            [False, True],
        ]
    )
    panel = survival.panel_from_frame(make_frame(dead))
    tr = survival.transitions(panel, per_layer=False).iloc[0]
    assert tr.n_alive_dead == 2
    assert tr.n_alive_alive == 1
    assert tr.n_dead_alive == 1
    assert tr.n_dead_dead == 2
    assert tr.p_recover == pytest.approx(1 / 3)
    assert tr.p_die == pytest.approx(2 / 3)


def test_revival_after_a_recycle_is_not_spontaneous_recovery():
    """ReDo reinitialises incoming weights, so a recycled unit is alive by
    construction. Pooling those with genuine recoveries would manufacture the
    finding that death is reversible."""
    dead = np.array([[False], [True], [True], [False]])
    rec = np.array([[False], [False], [False], [True]])
    panel = survival.panel_from_frame(make_frame(dead, rec))
    tr = survival.transitions(panel, per_layer=False).iloc[0]
    assert tr.n_dead_alive == 0
    assert tr.n_dead_alive_after_recycle == 1
    assert tr.p_recover == pytest.approx(0.0)


def test_transitions_are_reported_per_layer():
    dead = np.array([[False, False], [True, False], [True, False]])
    panel = survival.panel_from_frame(make_frame(dead, layers=[0, 1]))
    tr = survival.transitions(panel).set_index("layer_idx")
    assert tr.loc[0, "n_alive_dead"] == 1
    assert tr.loc[1, "n_alive_dead"] == 0
    assert tr.loc[1, "n_alive_alive"] == 2


# -- episodes -----------------------------------------------------------------


def test_death_episodes_lengths_and_censoring():
    dead = np.array(
        [
            [False, False],
            [True, False],
            [True, True],
            [False, True],
        ]
    )
    panel = survival.panel_from_frame(make_frame(dead))
    ep = survival.death_episodes(panel).sort_values(["neuron_idx", "start_task"])
    assert len(ep) == 2
    assert ep.start_task.tolist() == [1, 2]
    assert ep.length_tasks.tolist() == [2, 2]
    # Unit 1 is still dead at the last boundary: its episode has no end.
    assert ep.censored.tolist() == [False, True]


def test_death_episodes_splits_a_unit_that_dies_twice():
    dead = np.array([[True], [False], [True], [True]])
    panel = survival.panel_from_frame(make_frame(dead))
    ep = survival.death_episodes(panel).sort_values("start_task")
    assert ep.start_task.tolist() == [0, 2]
    assert ep.length_tasks.tolist() == [1, 2]
    assert ep.censored.tolist() == [False, True]


def test_death_episodes_flag_recycling_inside_the_episode_only():
    """The prefix-sum shortcut must span [start, end-1] -- one row too many and
    every episode looks like it was ended by the intervention."""
    dead = np.array([[False], [True], [True], [False], [False]])
    rec = np.array([[False], [False], [False], [True], [False]])
    panel = survival.panel_from_frame(make_frame(dead, rec))
    ep = survival.death_episodes(panel)
    assert len(ep) == 1
    # The recycle lands at task 3, after the last dead boundary (task 2).
    assert not bool(ep.recycled_during.iloc[0])

    rec_inside = np.array([[False], [False], [True], [False], [False]])
    panel2 = survival.panel_from_frame(make_frame(dead, rec_inside))
    assert bool(survival.death_episodes(panel2).recycled_during.iloc[0])


# -- recycling recurrence -----------------------------------------------------


def test_recurrence_reads_state_at_the_boundary_before_the_event():
    """The crux of the whole module.

    A unit recycled during task t is alive at t's boundary *because* it was
    recycled. Reading the state at t would report every recycled unit as alive
    and the C1 claim would be a tautology; reading t-1 is the real question.
    """
    dead = np.array([[True], [False], [False]])  # init: dead, then alive
    rec = np.array([[False], [True], [False]])  # recycled during task 0
    panel = survival.panel_from_frame(make_frame(dead, rec, times=[-1, 0, 1]))
    r = survival.recycling_recurrence(panel).iloc[0]
    assert r.n_recycled == 1
    assert r.n_recycled_while_dead == 1
    assert r.n_recycled_while_alive == 0


def test_recurrence_counts_a_living_unit_as_living():
    dead = np.array([[False], [False], [False]])
    rec = np.array([[False], [True], [True]])
    panel = survival.panel_from_frame(make_frame(dead, rec, times=[-1, 0, 1]))
    r = survival.recycling_recurrence(panel).iloc[0]
    assert r.n_recycled == 2
    assert r.n_recycled_while_alive == 2
    assert bool(r.ever_dead) is False


def test_recurrence_without_an_init_row_drops_the_first_task():
    """No prior boundary exists for task 0, so its events are not guessed at."""
    dead = np.array([[False], [False]])
    rec = np.array([[True], [True]])
    panel = survival.panel_from_frame(make_frame(dead, rec))
    r = survival.recycling_recurrence(panel).iloc[0]
    assert r.n_recycled == 2  # the raw count still sees both
    assert r.n_recycled_while_alive == 1  # only task 1 has a prior state


# -- concentration and the null ----------------------------------------------


def test_gini_is_zero_when_every_unit_is_recycled_equally():
    c = survival.concentration(np.array([3, 3, 3, 3]))
    assert c["gini"] == pytest.approx(0.0)
    assert c["frac_never"] == pytest.approx(0.0)
    assert c["mean_count"] == pytest.approx(3.0)


def test_gini_is_near_one_when_one_unit_takes_everything():
    counts = np.zeros(100, dtype=int)
    counts[0] = 50
    c = survival.concentration(counts)
    assert c["gini"] == pytest.approx(0.99, abs=0.01)
    assert c["frac_never"] == pytest.approx(0.99)
    assert c["top_decile_share"] == pytest.approx(1.0)


def test_null_holds_k_fixed_per_task_and_per_layer():
    """The null exists to separate *which* units from *how many*; if it does not
    reproduce the observed dose per layer it answers neither question."""
    rng = np.random.default_rng(0)
    rec = np.array(
        [
            [True, False, False, True, False, False],
            [True, True, False, False, False, False],
        ]
    )
    layers = np.array([0, 0, 0, 1, 1, 1])
    panel = survival.panel_from_frame(
        make_frame(np.zeros_like(rec), rec, layers=layers)
    )
    for _ in range(20):
        counts = survival.null_counts(panel, rng)
        assert counts[layers == 0].sum() == rec[:, layers == 0].sum()
        assert counts[layers == 1].sum() == rec[:, layers == 1].sum()
        assert counts.max() <= 2  # two tasks, sampled without replacement


def test_persistence_enrichment_is_about_one_for_independent_choices():
    """With units drawn afresh each task the previous choice carries no
    information, so the conditional rate must land on the marginal."""
    rng = np.random.default_rng(3)
    T, U, k = 400, 100, 20
    rec = np.zeros((T, U), bool)
    for t in range(T):
        rec[t, rng.choice(U, size=k, replace=False)] = True
    panel = survival.panel_from_frame(make_frame(np.zeros((T, U), bool), rec))
    p = survival.recycling_persistence(panel).iloc[0]
    assert p.p_recycled_marginal == pytest.approx(k / U, abs=0.01)
    assert p.enrichment == pytest.approx(1.0, abs=0.06)


# -- against the real logs ----------------------------------------------------

_REAL_RUNS = Path(__file__).resolve().parents[1] / "runs"


def _a_run_with_a_per_neuron_log():
    for d in sorted(p for p in _REAL_RUNS.glob("*") if p.is_dir()):
        if (d / "neurons.parquet").exists() and (d / "metrics.parquet").exists():
            return d
    return None


@pytest.mark.skipif(
    _a_run_with_a_per_neuron_log() is None,
    reason="no run outputs on this machine (runs/ is gitignored)",
)
def test_panel_prevalence_reproduces_metrics_parquet():
    """The strongest check available: two independent code paths, same number.

    `metrics.parquet` gets `dead_exact_frac` from `probes` aggregating over a
    layer at training time. The panel gets it by pivoting the per-neuron log and
    averaging a boolean matrix. They share no code. If the pivot misassigned
    units to layers or boundaries, this is where it shows up -- and nowhere in
    the survival output would it look wrong.
    """
    run_dir = _a_run_with_a_per_neuron_log()
    panel = next(survival.iter_panels(_REAL_RUNS, [run_dir.name]))
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    metrics = metrics[metrics.probe_point == "task_end"]

    for layer in panel.layers:
        mask = panel.layer_mask(layer)
        trained = panel.times >= 0
        got = panel.dead[np.ix_(trained, mask)].mean()
        want = metrics[metrics.layer_idx == layer].dead_exact_frac.mean()
        assert got == pytest.approx(want, abs=1e-9), f"layer {layer}"

        got_ref = panel.dead_ref[np.ix_(trained, mask)].mean()
        want_ref = metrics[metrics.layer_idx == layer].dead_exact_frac_ref.mean()
        assert got_ref == pytest.approx(want_ref, abs=1e-9), f"layer {layer} ref"


def test_persistence_detects_a_perfectly_sticky_choice():
    """The same 20 units every task: conditional probability 1, enrichment 1/rate."""
    T, U, k = 50, 100, 20
    rec = np.zeros((T, U), bool)
    rec[:, :k] = True
    panel = survival.panel_from_frame(make_frame(np.zeros((T, U), bool), rec))
    p = survival.recycling_persistence(panel).iloc[0]
    assert p.p_recycled_given_prev == pytest.approx(1.0)
    assert p.enrichment == pytest.approx(U / k)
