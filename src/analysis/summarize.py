"""In-session summary of an intervention sweep.

Post-hoc only. Prints the two numbers that decide the paper, using the frozen
plan's windows, estimator and thresholds:

1. **Does the arm beat baseline?** IQM of late-window online accuracy per arm,
   and the paired difference against the no-intervention baseline. For the
   tau-sweep this answers the standing untested assumption -- ReDo was designed
   for RL, and Sokar et al. themselves found it gives no gain on sufficiently
   over-parameterised networks (their Table 6). If ReDo does not beat baseline
   here, C1 is vacuous and the paper needs reshaping.

2. **What was actually recycled?** The genuinely-dead share of every recycled
   set, from the composition table. This is the C1 headline quantity.

    python -m src.analysis.summarize --runs-root runs --pattern "tau_*"

Recomputes no metric: every column read here was written by `src.probes` at
training time.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gate import PLAN_PATH, _window, load_plan
from .load import Run, load_runs, recycling_composition, score_matrix
from .stats import classify_difference, iqm, stratified_bootstrap_difference, stratified_bootstrap


def arm_key(run: Run) -> Tuple[str, Optional[float]]:
    """(arm, tau) from the run's own config, never from its name.

    Parsing run_ids would couple analysis to a naming convention; the config is
    the ground truth and is stored beside the outputs for exactly this reason.
    """
    rec = run.config.get("recycling", {}) or {}
    kind = rec.get("kind", "none")
    return (kind, None if kind == "none" else float(rec.get("tau", 0.0)))


def _label(arm: str, tau: Optional[float]) -> str:
    return arm if tau is None else f"{arm}(tau={tau:g})"


def summarize(
    runs_root: str = "runs",
    pattern: str = "*",
    plan_path=PLAN_PATH,
    baseline_arm: str = "none",
) -> int:
    plan = load_plan(plan_path)
    late = _window(plan, "late")
    stats = plan["statistics"]
    thr = plan["decision_thresholds"]
    n_boot = int(stats["n_bootstrap"])
    conf = float(stats["confidence"])
    seed = int(stats["bootstrap_seed"])
    measure = plan["outcome_measure"]["primary"]

    runs = load_runs(runs_root, pattern)
    if not runs:
        print(f"no complete runs matching {pattern!r} under {runs_root}")
        return 1

    groups: Dict[Tuple[str, Optional[float]], List[Run]] = defaultdict(list)
    for r in runs:
        groups[arm_key(r)].append(r)

    print(f"Sweep summary  ({len(runs)} runs matching {pattern!r})")
    print(f"  plan frozen at {plan['frozen_at']}")
    print(f"  outcome: IQM of {measure} over tasks "
          f"{plan['windows']['late']['protocol_tasks']}, "
          f"{int(conf*100)}% stratified-bootstrap CI")
    print(f"  thresholds: ~= |delta| < {thr['approx_equal']['abs_delta_pp']} pp with CI "
          f"containing zero;  >> delta > {thr['much_greater']['delta_pp']} pp with CI "
          "excluding zero\n")

    baseline = [g for k, g in groups.items() if k[0] == baseline_arm]
    base_runs = baseline[0] if baseline else []
    base_scores = score_matrix(base_runs, late, measure) if base_runs else None
    if base_scores is not None:
        est = stratified_bootstrap(base_scores, iqm, n_boot, conf, seed)
        print(f"  baseline {baseline_arm:<22s} n={len(base_runs):2d}  {est}\n")
    else:
        print(f"  (no {baseline_arm!r} baseline among these runs; "
              "differences are not computed)\n")

    print(f"  {'arm':<26s} {'n':>2s}  {'accuracy':<26s} {'vs baseline (pp)':<28s} verdict")
    for key in sorted(groups, key=lambda k: (k[0], -1 if k[1] is None else k[1])):
        arm, tau = key
        g = groups[key]
        scores = score_matrix(g, late, measure)
        est = stratified_bootstrap(scores, iqm, n_boot, conf, seed)
        line = f"  {_label(arm, tau):<26s} {len(g):>2d}  {str(est):<26s}"
        if base_scores is not None and arm != baseline_arm:
            paired = base_scores.shape == scores.shape
            diff = stratified_bootstrap_difference(
                scores, base_scores, iqm, n_boot, conf, seed, paired=paired
            )
            verdict = classify_difference(
                diff,
                float(thr["approx_equal"]["abs_delta_pp"]),
                float(thr["much_greater"]["delta_pp"]),
            )
            d = f"{diff.point*100:+.2f} [{diff.lo*100:+.2f}, {diff.hi*100:+.2f}]"
            line += f" {d:<28s} {verdict}"
            if not paired:
                line += "  (unpaired: seed counts differ)"
        print(line)

    # -- the C1 headline ----------------------------------------------------
    comp = recycling_composition(runs)
    if comp.empty:
        print("\n  no recycling events logged (no-intervention runs only)")
        return 0

    print("\n  Recycled-set composition -- of the units ReDo recycled, how many "
          "were genuinely dead?")
    print(f"  {'arm':<26s} {'events':>7s} {'mean k':>8s} {'% of layer':>11s} "
          f"{'% truly dead':>13s} {'% alive':>9s}")
    comp["arm_label"] = [
        _label(a, None if a == "none" else t)
        for a, t in zip(comp["arm"], comp["tau"])
    ]
    for label, g in comp.groupby("arm_label"):
        fired = g[g.k > 0]
        if fired.empty:
            print(f"  {label:<26s} {len(g):>7d} {0:>8.1f} "
                  f"{'-':>11s} {'-':>13s} {'-':>9s}   (nothing recycled)")
            continue
        print(
            f"  {label:<26s} {len(g):>7d} {fired.k.mean():>8.1f} "
            f"{(fired.k / fired.n_neurons).mean()*100:>10.1f}% "
            f"{fired.dead_fraction.mean()*100:>12.1f}% "
            f"{fired.alive_fraction.mean()*100:>8.1f}%"
        )
    print(
        "\n  'truly dead' = dead_exact on the 2048-example probe batch, measured\n"
        "  at the event before any weights moved. If accuracy improves with tau\n"
        "  while that column falls, C1 is confirmed: ReDo is not a resurrection\n"
        "  method (protocol §B.1)."
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarize an intervention sweep.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--plan", default=str(PLAN_PATH))
    ap.add_argument("--baseline", default="none")
    args = ap.parse_args(argv)
    return summarize(args.runs_root, args.pattern, args.plan, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
