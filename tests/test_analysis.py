"""Analysis-side tests: the frozen plan and the gate evaluator.

These assert *machinery*, never a scientific outcome -- the fixtures train on
noise, so any claim about accuracy or dead units here would be meaningless.
What is worth asserting is that the windows convert correctly, that the plan
cannot be bypassed, and that the dissociation stop-rule actually fires.
"""

from __future__ import annotations

import copy
import json

import pandas as pd
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


# -- the collapse health check (advisory, not part of the frozen criterion) ----


def _fake_run(root, run_id, *, accuracy, loss, grad_norm, n_tasks=6):
    """A minimal run tree: enough for `load_run` and `collapse_diagnosis`."""
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"run_id": run_id, "seed": 0, "optim": {"lr": 0.03, "name": "sgd"},
           "data": {"name": "label_shuffled_cifar10", "n_tasks": n_tasks}}
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    pd.DataFrame([
        {"run_id": run_id, "task_idx": t, "probe_point": "task_end",
         "online_accuracy": accuracy, "mean_loss": loss}
        for t in range(n_tasks)
    ]).to_parquet(d / "tasks.parquet")
    pd.DataFrame([
        {"run_id": run_id, "task_idx": t, "probe_point": "task_end", "layer_idx": l,
         "n_neurons": 10, "dead_exact_count": 5, "dead_exact_count_ref": 5,
         "grad_norm_layer": grad_norm}
        for t in range(n_tasks) for l in range(2)
    ]).to_parquet(d / "metrics.parquet")
    return d


def test_collapse_check_catches_a_network_that_stopped_training(tmp_path):
    """Chance accuracy + ln(10) loss + zero gradient = the network died.

    The frozen criterion reads this as the most emphatic possible PASS, because
    accuracy fell as far as it can and every unit ended up dead. The health
    check is what keeps that from being accepted as the phenomenon.
    """
    plan = _short_plan(n_tasks=6, n_seeds=1)
    root = tmp_path / "runs"
    _fake_run(root, "s2gate_collapsed_s0", accuracy=0.0991,
              loss=float(np.log(10)), grad_norm=0.0)
    runs = load_runs(root, "s2gate_*")
    msg = gate_mod.collapse_diagnosis(runs, plan)
    assert msg is not None
    assert "chance" in msg and "ln(10)" in msg


def test_collapse_check_passes_a_healthy_run(tmp_path):
    """Setting 1 at lr=0.1 drops 4.5 pp and stays far above chance; flagging it
    would make the check useless."""
    plan = _short_plan(n_tasks=6, n_seeds=1)
    root = tmp_path / "runs"
    _fake_run(root, "s2gate_healthy_s0", accuracy=0.877, loss=0.47, grad_norm=0.9)
    runs = load_runs(root, "s2gate_*")
    assert gate_mod.collapse_diagnosis(runs, plan) is None


def test_collapse_does_not_alter_the_frozen_verdict(tmp_path):
    """`passed` must keep reporting the frozen criterion exactly; only `usable`
    combines it with the health check."""
    plan = _short_plan(n_tasks=6, n_seeds=1)
    root = tmp_path / "runs"
    _fake_run(root, "s2gate_collapsed_s0", accuracy=0.0991,
              loss=float(np.log(10)), grad_norm=0.0)
    runs = load_runs(root, "s2gate_*")
    r = gate_mod.evaluate_gate(runs, plan, 0.03)
    assert r.collapse is not None
    assert r.passed == (r.accuracy_pass and r.dead_pass)
    assert r.usable is False
    assert "COLLAPSED" in gate_mod.report([r], plan)
