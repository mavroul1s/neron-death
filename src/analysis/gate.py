"""Evaluate the Week-1 reproduction gate against the frozen analysis plan.

    python -m src.analysis.gate --runs-root runs --pattern "gate_*"

Every threshold and window is read from ``configs/analysis_plan.json``. Nothing
here has a default that could quietly substitute for a missing plan entry: if
the plan does not specify something, this raises. That is the point of freezing
the plan -- an analysis that can fall back on a default is not pre-registered.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .load import Run, load_runs, score_matrix, task_curve
from .stats import Estimate, iqm, stratified_bootstrap

PLAN_PATH = Path(__file__).resolve().parents[2] / "configs" / "analysis_plan.json"


def load_plan(path=PLAN_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    if not plan.get("frozen_before_any_data"):
        raise RuntimeError(f"{path} is not marked as frozen; refusing to use it")
    return plan


def _window(plan: dict, name: str) -> List[int]:
    """0-based task indices for a named window.

    The plan carries both the protocol's 1-based numbers and the 0-based
    equivalents precisely so this conversion never happens by hand.
    """
    lo, hi = plan["windows"][name]["task_idx"]
    window = list(range(lo, hi + 1))
    expected = plan["windows"][name]["n_tasks"]
    if len(window) != expected:
        raise RuntimeError(
            f"plan window {name!r} says n_tasks={expected} but spans {len(window)}"
        )
    return window


# ---------------------------------------------------------------------------
# Dead-unit trend
# ---------------------------------------------------------------------------


def pooled_dead_fraction(run: Run, probe_batch: str) -> pd.Series:
    """Network-level dead_exact fraction per task, pooled over hidden layers.

    Pooled by neuron count, not by averaging the per-layer fractions: the layers
    happen to be equal width here, but a mean-of-fractions would silently
    become the wrong statistic the moment they are not.
    """
    col = "dead_exact_count" if probe_batch == "current" else "dead_exact_count_ref"
    df = run.table("metrics")
    df = df[df["probe_point"] == "task_end"]
    if col not in df.columns:
        raise RuntimeError(f"{run.run_id}: metrics.parquet has no column {col!r}")
    grouped = df.groupby("task_idx")
    return (grouped[col].sum() / grouped["n_neurons"].sum()).sort_index()


def spearman_per_seed(runs: Sequence[Run], window: Sequence[int], probe_batch: str) -> np.ndarray:
    """Spearman rho between task index and pooled dead fraction, one per run."""
    rhos = []
    for run in runs:
        series = pooled_dead_fraction(run, probe_batch)
        missing = [t for t in window if t not in series.index]
        if missing:
            raise RuntimeError(f"{run.run_id} is missing tasks {missing[:5]} from the trend window")
        y = series.loc[list(window)].to_numpy(dtype=np.float64)
        if np.all(y == y[0]):
            # A perfectly flat curve has no monotone trend. Spearman returns NaN
            # here; 0.0 is the correct answer and must not be silently dropped.
            rhos.append(0.0)
            continue
        rhos.append(float(sp_stats.spearmanr(np.asarray(window, dtype=np.float64), y).statistic))
    return np.asarray(rhos, dtype=np.float64)[:, None]


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    learning_rate: float
    n_seeds: int
    accuracy_drop_pp: Estimate
    early_accuracy: Estimate
    late_accuracy: Estimate
    dead_rho: Estimate
    dead_early: float
    dead_late: float
    accuracy_pass: bool
    dead_pass: bool

    @property
    def passed(self) -> bool:
        return self.accuracy_pass and self.dead_pass

    @property
    def dissociation(self) -> bool:
        """Accuracy collapsed but dead units did not rise.

        Protocol §A.4: if this happens, STOP and report. It is itself a result
        and it changes the paper.
        """
        return self.accuracy_pass and not self.dead_pass

    def as_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "n_seeds": self.n_seeds,
            "accuracy_drop_pp": self.accuracy_drop_pp.as_dict(),
            "early_accuracy": self.early_accuracy.as_dict(),
            "late_accuracy": self.late_accuracy.as_dict(),
            "dead_exact_spearman_rho": self.dead_rho.as_dict(),
            "dead_exact_frac_early": self.dead_early,
            "dead_exact_frac_late": self.dead_late,
            "accuracy_pass": self.accuracy_pass,
            "dead_pass": self.dead_pass,
            "passed": self.passed,
            "dissociation": self.dissociation,
        }


def evaluate_gate(runs: Sequence[Run], plan: dict, learning_rate: float) -> GateResult:
    gate = plan["gate"]
    stats_cfg = plan["statistics"]
    acc_cfg = gate["accuracy_criterion"]
    dead_cfg = gate["dead_unit_criterion"]

    expected_seeds = plan["seeds"]["gate"]
    if len(runs) != expected_seeds:
        print(
            f"  warning: plan specifies {expected_seeds} seeds, found {len(runs)}. "
            "A missing seed changes the bootstrap CI."
        )

    boot = dict(
        n_bootstrap=stats_cfg["n_bootstrap"],
        confidence=stats_cfg["confidence"],
        seed=stats_cfg["bootstrap_seed"],
    )
    measure = plan["outcome_measure"]["primary"]
    early_w, late_w = _window(plan, "early"), _window(plan, "late")

    early = score_matrix(runs, early_w, measure)
    late = score_matrix(runs, late_w, measure)

    # Per-seed drop: each seed contributes one number, so the bootstrap
    # resamples seeds rather than pretending 10 and 50 tasks are commensurate.
    drop = (early.mean(axis=1) - late.mean(axis=1))[:, None]
    drop_est = stratified_bootstrap(drop, iqm, **boot)
    drop_pp = Estimate(
        drop_est.point * 100, drop_est.lo * 100, drop_est.hi * 100,
        drop_est.n_runs, drop_est.n_tasks, drop_est.n_bootstrap, drop_est.confidence,
    )

    accuracy_pass = (
        drop_pp.point >= acc_cfg["min_drop_pp"]
        and (drop_pp.excludes_zero if acc_cfg["require_ci_excludes_zero"] else True)
    )

    rhos = spearman_per_seed(runs, _window(plan, "trend"), dead_cfg["probe_batch"])
    rho_est = stratified_bootstrap(rhos, iqm, **boot)
    dead_pass = (
        (rho_est.point > 0.0 if dead_cfg["require_rho_positive"] else True)
        and (rho_est.excludes_zero if dead_cfg["require_ci_excludes_zero"] else True)
    )

    dead_curves = np.vstack(
        [pooled_dead_fraction(r, dead_cfg["probe_batch"]).to_numpy() for r in runs]
    )
    return GateResult(
        learning_rate=learning_rate,
        n_seeds=len(runs),
        accuracy_drop_pp=drop_pp,
        early_accuracy=stratified_bootstrap(early, iqm, **boot),
        late_accuracy=stratified_bootstrap(late, iqm, **boot),
        dead_rho=rho_est,
        dead_early=float(iqm(dead_curves[:, early_w])),
        dead_late=float(iqm(dead_curves[:, late_w])),
        accuracy_pass=accuracy_pass,
        dead_pass=dead_pass,
    )


def group_by_learning_rate(runs: Sequence[Run]) -> Dict[float, List[Run]]:
    out = defaultdict(list)
    for run in runs:
        out[float(run.config["optim"]["lr"])].append(run)
    return dict(sorted(out.items(), reverse=True))


def report(results: Sequence[GateResult], plan: dict) -> str:
    gate = plan["gate"]
    lines = [
        "Week-1 reproduction gate (protocol §A.4)",
        f"  plan frozen at {plan['frozen_at']}",
        f"  criterion: drop >= {gate['accuracy_criterion']['min_drop_pp']} pp, "
        f"95% CI excluding zero, and dead_exact rising",
        f"  dead_exact rise operationalised as: {gate['dead_unit_criterion']['statistic']}",
        "",
    ]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        lines += [
            f"lr = {r.learning_rate:g}  ({r.n_seeds} seeds)   [{mark}]",
            f"  online accuracy  early {r.early_accuracy}   late {r.late_accuracy}",
            f"  drop (pp)        {r.accuracy_drop_pp}"
            f"   -> {'pass' if r.accuracy_pass else 'FAIL'}",
            f"  dead_exact frac  early {r.dead_early:.4f}   late {r.dead_late:.4f}",
            f"  dead_exact rho   {r.dead_rho}"
            f"   -> {'pass' if r.dead_pass else 'FAIL'}",
        ]
        if r.dissociation:
            lines += [
                "",
                "  *** STOP: accuracy collapsed but dead units did not rise. ***",
                "  Protocol §A.4: this dissociation is itself a result and changes",
                "  the paper. Report it before running anything in §B.",
            ]
        lines.append("")

    if not any(r.passed for r in results):
        lines += [
            "No learning rate passed. Pre-specified response, in order:",
        ] + [f"  {i + 1}. {s}" for i, s in enumerate(gate["on_failure"]["order"])] + [
            f"Forbidden: {gate['on_failure']['forbidden']}.",
            "",
        ]
    else:
        best = max((r for r in results if r.passed), key=lambda r: r.accuracy_drop_pp.point)
        lines.append(f"Best passing learning rate: {best.learning_rate:g}. Use it for §B.1.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--pattern", default="gate_*")
    ap.add_argument("--plan", default=str(PLAN_PATH))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    runs = load_runs(args.runs_root, args.pattern)
    if not runs:
        print(f"no complete runs matching {args.pattern!r} under {args.runs_root}")
        return 1

    results = [
        evaluate_gate(group, plan, lr)
        for lr, group in group_by_learning_rate(runs).items()
    ]
    print(report(results, plan))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {"plan_frozen_at": plan["frozen_at"], "results": [r.as_dict() for r in results]},
                f,
                indent=2,
            )
    return 0 if any(r.passed for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
