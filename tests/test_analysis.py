"""Analysis-side tests: the frozen plan and the gate evaluator.

These assert *machinery*, never a scientific outcome -- the fixtures train on
noise, so any claim about accuracy or dead units here would be meaningless.
What is worth asserting is that the windows convert correctly, that the plan
cannot be bypassed, and that the dissociation stop-rule actually fires.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from src.analysis import gate as gate_mod
from src.analysis.load import load_runs
from src.analysis.stats import Estimate
from src.train import Trainer
from tests.test_train import _cfg


def test_frozen_plan_is_self_consistent():
    plan = gate_mod.load_plan()
    assert plan["frozen_before_any_data"] is True

    # The protocol's 1-based windows and the code's 0-based ones must agree.
    for name, (lo1, hi1) in {
        "early": (1, 10),
        "late": (151, 200),
        "trend": (1, 200),
    }.items():
        w = plan["windows"][name]
        assert w["protocol_tasks"] == [lo1, hi1]
        assert w["task_idx"] == [lo1 - 1, hi1 - 1]
        assert w["n_tasks"] == hi1 - lo1 + 1
        assert gate_mod._window(plan, name) == list(range(lo1 - 1, hi1))

    # Decision thresholds are the protocol's, unaltered.
    assert plan["decision_thresholds"]["approx_equal"]["abs_delta_pp"] == 1.0
    assert plan["decision_thresholds"]["much_greater"]["delta_pp"] == 2.0
    assert plan["decision_thresholds"]["response_to_inconclusive"].startswith("add seeds")
    assert plan["seeds"] == {
        "gate": 5, "tau_sweep": 10, "c3_anomaly": 5, "c5_optimizer": 5, "eps_sweep": 5,
        "_note": plan["seeds"]["_note"],
    }
    assert plan["gate"]["accuracy_criterion"]["min_drop_pp"] == 3.0
    assert plan["primary_experiment"]["taus"] == [0.0, 0.01, 0.025, 0.05, 0.1, 0.25]
    assert plan["statistics"]["estimator"] == "IQM"
    assert plan["statistics"]["confidence"] == 0.95

    # The init probe must never leak into an analysis window.
    assert plan["task_indexing"]["init_probe_task_idx"] == -1
    assert plan["task_indexing"]["init_probe_excluded_from_all_windows"] is True


def test_gate_refuses_an_unfrozen_plan(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"frozen_before_any_data": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not marked as frozen"):
        gate_mod.load_plan(p)


def _short_plan(n_tasks: int, n_seeds: int) -> dict:
    """The real plan with the windows shrunk to fit a test-sized run."""
    plan = copy.deepcopy(gate_mod.load_plan())
    plan["windows"] = {
        "early": {"protocol_tasks": [1, 2], "task_idx": [0, 1], "n_tasks": 2},
        "late": {"protocol_tasks": [n_tasks - 1, n_tasks], "task_idx": [n_tasks - 2, n_tasks - 1], "n_tasks": 2},
        "trend": {"protocol_tasks": [1, n_tasks], "task_idx": [0, n_tasks - 1], "n_tasks": n_tasks},
    }
    plan["seeds"]["gate"] = n_seeds
    plan["statistics"]["n_bootstrap"] = 200
    return plan


@pytest.fixture(scope="module")
def _gate_runs(tmp_path_factory):
    """Three short runs sharing a learning rate, differing only in seed."""
    import numpy as _np  # noqa: F401  (keep the fixture import-light)

    root = tmp_path_factory.mktemp("gate")
    data = root / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    np.savez(
        data / "mnist.npz",
        x_train=rng.integers(0, 256, size=(512, 28, 28), dtype=np.uint8),
        y_train=rng.integers(0, 10, size=512, dtype=np.uint8),
        x_test=rng.integers(0, 256, size=(128, 28, 28), dtype=np.uint8),
        y_test=rng.integers(0, 10, size=128, dtype=np.uint8),
    )
    runs_root = root / "runs"
    for seed in range(3):
        cfg = _cfg(data, f"gate_test_lr0p05_s{seed}")
        cfg["seed"] = seed
        cfg["data"]["n_tasks"] = 6
        cfg["recycling"] = {"kind": "none"}
        Trainer(cfg, runs_root=runs_root).run()
    return runs_root


def test_gate_evaluator_runs_end_to_end(_gate_runs):
    plan = _short_plan(n_tasks=6, n_seeds=3)
    runs = load_runs(_gate_runs, "gate_*")
    assert len(runs) == 3

    groups = gate_mod.group_by_learning_rate(runs)
    assert list(groups) == [0.05]

    result = gate_mod.evaluate_gate(groups[0.05], plan, 0.05)
    assert result.n_seeds == 3
    assert isinstance(result.passed, bool)
    assert isinstance(result.dissociation, bool)
    # A CI must bracket its own point estimate.
    for est in (result.accuracy_drop_pp, result.early_accuracy, result.dead_rho):
        assert est.lo <= est.point <= est.hi
    assert -1.0 <= result.dead_rho.point <= 1.0
    assert 0.0 <= result.dead_early <= 1.0 and 0.0 <= result.dead_late <= 1.0

    text = gate_mod.report([result], plan)
    assert "reproduction gate" in text
    assert plan["frozen_at"] in text


def test_pooled_dead_fraction_is_neuron_weighted(_gate_runs):
    """Pooled over neurons, not a mean of per-layer fractions."""
    runs = load_runs(_gate_runs, "gate_*")
    series = gate_mod.pooled_dead_fraction(runs[0], "current")
    assert series.index.min() == 0  # the init probe is excluded
    assert series.between(0.0, 1.0).all()

    metrics = runs[0].table("metrics")
    metrics = metrics[(metrics["probe_point"] == "task_end") & (metrics["task_idx"] == 0)]
    expected = metrics["dead_exact_count"].sum() / metrics["n_neurons"].sum()
    assert series.loc[0] == pytest.approx(expected)


def test_dissociation_stop_rule_fires():
    """Accuracy collapses, dead units do not rise -> the run must be flagged,
    because protocol §A.4 says that outcome changes the paper."""
    r = gate_mod.GateResult(
        learning_rate=0.01,
        n_seeds=5,
        accuracy_drop_pp=Estimate(5.0, 4.0, 6.0, 5, 1, 100, 0.95),
        early_accuracy=Estimate(0.9, 0.89, 0.91, 5, 10, 100, 0.95),
        late_accuracy=Estimate(0.85, 0.84, 0.86, 5, 50, 100, 0.95),
        dead_rho=Estimate(0.01, -0.2, 0.2, 5, 1, 100, 0.95),
        dead_early=0.02,
        dead_late=0.02,
        accuracy_pass=True,
        dead_pass=False,
    )
    assert r.dissociation and not r.passed
    text = gate_mod.report([r], gate_mod.load_plan())
    assert "STOP" in text and "changes" in text


def test_failing_gate_reports_the_prespecified_response():
    r = gate_mod.GateResult(
        learning_rate=0.001,
        n_seeds=5,
        accuracy_drop_pp=Estimate(0.2, -0.5, 0.9, 5, 1, 100, 0.95),
        early_accuracy=Estimate(0.9, 0.89, 0.91, 5, 10, 100, 0.95),
        late_accuracy=Estimate(0.9, 0.89, 0.91, 5, 50, 100, 0.95),
        dead_rho=Estimate(0.0, -0.2, 0.2, 5, 1, 100, 0.95),
        dead_early=0.0,
        dead_late=0.0,
        accuracy_pass=False,
        dead_pass=False,
    )
    text = gate_mod.report([r], gate_mod.load_plan())
    assert "raise the learning rate" in text
    assert "Forbidden: weakening the criterion" in text
