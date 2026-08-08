"""Recompute every number the paper cites, straight from `runs/_extracts/`.

PAPER_BRIEF.md §9: "Numbers in the paper must trace to runs/LEDGER.md or the
parquet outputs -- do not restate a number from memory of an earlier
conversation without checking it against the committed results."

This script is that check. It writes `paper/numbers.json`, and the LaTeX is
written against that file rather than against any prose summary. Statistics are
the frozen ones (IQM + stratified bootstrap, `configs/analysis_plan.json`) via
`src.analysis.stats`, so the paper and the figures cannot drift apart.

    python paper/collect_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.figures import Extracts  # noqa: E402
from src.analysis.stats import (  # noqa: E402
    classify_difference,
    iqm,
    stratified_bootstrap,
    stratified_bootstrap_difference,
)

EX = Extracts(ROOT / "runs" / "_extracts")
PLAN = EX.plan
LATE = EX.window_late
EARLY = EX.window_early
APPROX_PP = PLAN["decision_thresholds"]["approx_equal"]["abs_delta_pp"]
MUCH_PP = PLAN["decision_thresholds"]["much_greater"]["delta_pp"]
NBOOT = PLAN["statistics"]["n_bootstrap"]
SEED = PLAN["statistics"]["bootstrap_seed"]

TASKS = EX.table("tasks")
TASKS = TASKS[TASKS.probe_point == "task_end"]
METRICS = EX.table("metrics")
RECYC = EX.table("recycling")


def matrix(run_ids, window, column="online_accuracy"):
    """(n_seeds, n_tasks) score matrix, rows ordered by seed."""
    df = TASKS[TASKS.run_id.isin(run_ids) & TASKS.task_idx.isin(window)]
    wide = df.pivot_table(index="run_id", columns="task_idx", values=column)
    wide = wide.reindex(sorted(run_ids))
    assert wide.notna().all().all(), f"missing tasks for {sorted(run_ids)[:3]}"
    return wide[sorted(window)].to_numpy(dtype=np.float64)


def est(run_ids, window=None, column="online_accuracy"):
    m = matrix(run_ids, window or LATE, column)
    e = stratified_bootstrap(m, n_bootstrap=NBOOT, seed=SEED)
    return {"pct": 100 * e.point, "lo": 100 * e.lo, "hi": 100 * e.hi, "n_seeds": m.shape[0]}


def diff(a_ids, b_ids, window=None, column="online_accuracy"):
    """Paired IQM difference in percentage points, plus the frozen verdict."""
    w = window or LATE
    a, b = matrix(a_ids, w, column), matrix(b_ids, w, column)
    e = stratified_bootstrap_difference(a, b, n_bootstrap=NBOOT, seed=SEED)
    return {
        "pp": 100 * e.point,
        "lo": 100 * e.lo,
        "hi": 100 * e.hi,
        "excludes_zero": e.excludes_zero,
        "verdict": classify_difference(e, APPROX_PP, MUCH_PP),
        "n_seeds": a.shape[0],
    }


def ids(**filters):
    df = EX.runs
    for k, v in filters.items():
        df = df[df[k] == v] if not isinstance(v, (list, tuple)) else df[df[k].isin(v)]
    return sorted(df.run_id)


def prefixed(prefix, **filters):
    return sorted(r for r in ids(**filters) if r.startswith(prefix))


def late_metric(run_ids, column, pooled=True):
    """Late-window mean of a per-layer metric, network-pooled by neuron count."""
    df = METRICS[
        METRICS.run_id.isin(run_ids)
        & METRICS.task_idx.isin(LATE)
        & (METRICS.probe_point == "task_end")
    ]
    if df.empty:
        return None
    if pooled:
        per = df.groupby(["run_id", "task_idx"]).apply(
            lambda g: np.average(g[column], weights=g["n_neurons"]), include_groups=False
        )
        return 100 * float(iqm(per.groupby("run_id").mean().to_numpy()))
    return {
        int(lyr): 100 * float(iqm(g.groupby("run_id")[column].mean().to_numpy()))
        for lyr, g in df.groupby("layer_idx")
    }


def late_erank(run_ids):
    df = METRICS[
        METRICS.run_id.isin(run_ids)
        & METRICS.task_idx.isin(LATE)
        & (METRICS.probe_point == "task_end")
    ]
    return float(iqm(df.groupby("run_id")["erank"].mean().to_numpy()))


OUT: dict = {
    "_generated_from": "runs/_extracts",
    "_plan_frozen_at": PLAN["frozen_at"],
    "_thresholds": {"approx_pp": APPROX_PP, "much_greater_pp": MUCH_PP},
    "_windows": {"early": [EARLY[0], EARLY[-1]], "late": [LATE[0], LATE[-1]]},
    "_n_bootstrap": NBOOT,
}

# ---------------------------------------------------------------- C1: tau sweep
TAUS = PLAN["primary_experiment"]["taus"]
none_ids = prefixed("tau_none", arm="none")
OUT["c1"] = {"taus": TAUS, "none": est(none_ids), "arms": {}}

for arm, pfx in [
    ("redo", "tau_redo"),
    ("random_matched", "tau_random"),
    ("inverse_matched", "tau_inverse"),
]:
    OUT["c1"]["arms"][arm] = {}
    for tau in TAUS:
        run_ids = prefixed(pfx, arm=arm, tau=tau)
        if not run_ids:
            continue
        comp = RECYC[RECYC.run_id.isin(run_ids)]
        k_tot, dead_tot = comp["k"].sum(), comp["n_dead_exact"].sum()
        # composition on the reference batch too -- the two probes disagree, and
        # the plan's headline quantity is the current-task one.
        row = {
            "acc": est(run_ids),
            "vs_none": diff(run_ids, none_ids),
            "dead_share_pct": 100 * float(dead_tot / k_tot),
            "dead_share_ref_pct": 100 * float(comp["n_dead_exact_ref"].sum() / k_tot),
            "recycled_pct_of_net": 100
            * float(comp["k"].sum() / comp["n_neurons"].sum()),
            "mean_k_per_event_layer": float(comp["k"].mean()),
            "mean_sokar_score": float(
                np.average(comp["mean_sokar_score"], weights=comp["k"])
            ),
            "n_events": int(comp.event_idx.nunique()),
            "n_event_rows": int(len(comp)),
        }
        if arm != "redo":
            row["vs_redo"] = diff(prefixed("tau_redo", arm="redo", tau=tau), run_ids)
        OUT["c1"]["arms"][arm][str(tau)] = row

# tau invariance: the highest tau against the lowest, within the ReDo arm
OUT["c1"]["redo_tau_span"] = diff(
    prefixed("tau_redo", arm="redo", tau=0.25), prefixed("tau_redo", arm="redo", tau=0.0)
)
# how much of ReDo's benefit random-matched recovers, per tau
OUT["c1"]["random_share_of_redo_pct"] = {
    str(t): 100
    * OUT["c1"]["arms"]["random_matched"][str(t)]["vs_none"]["pp"]
    / OUT["c1"]["arms"]["redo"][str(t)]["vs_none"]["pp"]
    for t in TAUS
    if str(t) in OUT["c1"]["arms"].get("random_matched", {})
}
OUT["c1"]["inverse_share_of_redo_pct"] = {
    str(t): 100
    * OUT["c1"]["arms"]["inverse_matched"][str(t)]["vs_none"]["pp"]
    / OUT["c1"]["arms"]["redo"][str(t)]["vs_none"]["pp"]
    for t in TAUS
    if str(t) in OUT["c1"]["arms"].get("inverse_matched", {})
}
# the dose confound: k per event, ReDo vs each control
OUT["c1"]["dose"] = {
    arm: {
        str(t): OUT["c1"]["arms"][arm][str(t)]["mean_k_per_event_layer"]
        for t in TAUS
        if str(t) in OUT["c1"]["arms"].get(arm, {})
    }
    for arm in ("redo", "random_matched", "inverse_matched")
}
# dead_exact and erank under each arm, late window
OUT["c1"]["health"] = {
    "none": {
        "dead_exact_pct": late_metric(none_ids, "dead_exact_frac"),
        "erank": late_erank(none_ids),
    }
}
for arm, pfx in [
    ("redo", "tau_redo"),
    ("random_matched", "tau_random"),
    ("inverse_matched", "tau_inverse"),
]:
    OUT["c1"]["health"][arm] = {
        str(t): {
            "dead_exact_pct": late_metric(prefixed(pfx, arm=arm, tau=t), "dead_exact_frac"),
            "erank": late_erank(prefixed(pfx, arm=arm, tau=t)),
        }
        for t in TAUS
        if prefixed(pfx, arm=arm, tau=t)
    }

# composition stability across training (C1 robustness)
redo_t0 = RECYC[RECYC.run_id.isin(prefixed("tau_redo", arm="redo", tau=0.0))]
OUT["c1"]["composition_drift_tau0"] = {
    "tasks_0_24": 100
    * float(
        redo_t0[redo_t0.task_idx < 25]["n_dead_exact"].sum()
        / redo_t0[redo_t0.task_idx < 25]["k"].sum()
    ),
    "tasks_125_199": 100
    * float(
        redo_t0[redo_t0.task_idx >= 125]["n_dead_exact"].sum()
        / redo_t0[redo_t0.task_idx >= 125]["k"].sum()
    ),
}

# --------------------------------------------------------------- C2 / the gate
gate_ids = prefixed("gate", lr=0.1) + prefixed("gatehi", lr=0.1)
gate_ids = sorted(set(gate_ids))
OUT["gate"] = {}
for lr in sorted(EX.runs[EX.runs.run_id.str.startswith(("gate_", "gatehi_"))].lr.unique()):
    rid = sorted(r for r in ids(lr=lr) if r.startswith(("gate_", "gatehi_")))
    OUT["gate"][str(lr)] = {
        "early": est(rid, EARLY),
        "late": est(rid, LATE),
        "drop_pp": diff(rid, rid, EARLY)["pp"],  # placeholder, replaced below
        "n_seeds": len(rid),
    }
    a, b = matrix(rid, EARLY), matrix(rid, LATE)
    e = stratified_bootstrap(
        np.hstack([a.mean(1, keepdims=True) - b.mean(1, keepdims=True)]),
        n_bootstrap=NBOOT,
        seed=SEED,
    )
    OUT["gate"][str(lr)]["drop_pp"] = 100 * e.point
    OUT["gate"][str(lr)]["drop_lo"] = 100 * e.lo
    OUT["gate"][str(lr)]["drop_hi"] = 100 * e.hi
    OUT["gate"][str(lr)]["dead_exact_early_pct"] = 100 * float(
        iqm(
            METRICS[METRICS.run_id.isin(rid) & METRICS.task_idx.isin(EARLY)]
            .groupby("run_id")
            .apply(lambda g: np.average(g.dead_exact_frac, weights=g.n_neurons),
                   include_groups=False)
            .to_numpy()
        )
    )
    OUT["gate"][str(lr)]["dead_exact_late_pct"] = 100 * float(
        iqm(
            METRICS[METRICS.run_id.isin(rid) & METRICS.task_idx.isin(LATE)]
            .groupby("run_id")
            .apply(lambda g: np.average(g.dead_exact_frac, weights=g.n_neurons),
                   include_groups=False)
            .to_numpy()
        )
    )

# C2 headline: every death definition on the same activations, lr=0.1 baseline
gate01 = prefixed("gatehi", lr=0.1)
DEFS = {
    "dead_exact": "dead_exact_frac",
    "dead_abs_1e-6": "dead_abs_frac_1em06",
    "dead_abs_1e-4": "dead_abs_frac_1em04",
    "dead_abs_1e-2": "dead_abs_frac_1em02",
    "dormant_tau_0": "dormant_frac_tau_0",
    "dormant_tau_0.01": "dormant_frac_tau_0p01",
    "dormant_tau_0.025": "dormant_frac_tau_0p025",
    "dormant_tau_0.05": "dormant_frac_tau_0p05",
    "dormant_tau_0.1": "dormant_frac_tau_0p1",
    "dormant_tau_0.25": "dormant_frac_tau_0p25",
}
OUT["c2"] = {
    "definitions_lr0p1": {k: late_metric(gate01, v) for k, v in DEFS.items()},
    "definitions_lr0p1_per_layer": {
        k: late_metric(gate01, v, pooled=False) for k, v in DEFS.items()
    },
    "erank_lr0p1": late_erank(gate01),
}
# reference-batch asymmetry, per layer
OUT["c2"]["reference_asymmetry"] = {
    "current": late_metric(gate01, "dead_exact_frac", pooled=False),
    "reference": late_metric(gate01, "dead_exact_frac_ref", pooled=False),
    "current_pooled": late_metric(gate01, "dead_exact_frac"),
    "reference_pooled": late_metric(gate01, "dead_exact_frac_ref"),
}

# Setting 3: activation sweep
OUT["setting3"] = {}
for act in sorted(EX.runs[EX.runs.run_id.str.startswith("s3_")].activation.unique()):
    rid = prefixed("s3_", activation=act)
    if not rid:
        continue
    a, b = matrix(rid, EARLY), matrix(rid, LATE)
    OUT["setting3"][act] = {
        "lr": float(EX.runs[EX.runs.run_id.isin(rid)].lr.iloc[0]),
        "early": est(rid, EARLY),
        "late": est(rid, LATE),
        "drop_pp": 100 * (iqm(a) - iqm(b)),
        "dead_exact_pct": late_metric(rid, "dead_exact_frac"),
        "dormant_0p1_pct": late_metric(rid, "dormant_frac_tau_0p1"),
        "dead_abs_1em02_pct": late_metric(rid, "dead_abs_frac_1em02"),
        "saturated_pct": late_metric(rid, "saturated_frac"),
        "erank": late_erank(rid),
        "n_seeds": len(rid),
    }

# the tanh learning-rate calibration
OUT["setting3_tanh_gate"] = {}
for lr in sorted(EX.runs[EX.runs.run_id.str.startswith("s3tanh")].lr.unique()):
    rid = prefixed("s3tanh", lr=lr)
    a, b = matrix(rid, EARLY), matrix(rid, LATE)
    OUT["setting3_tanh_gate"][str(lr)] = {
        "early": est(rid, EARLY),
        "late": est(rid, LATE),
        "drop_pp": 100 * (iqm(a) - iqm(b)),
        "dead_exact_pct": late_metric(rid, "dead_exact_frac"),
        "dormant_0p1_pct": late_metric(rid, "dormant_frac_tau_0p1"),
        "dead_abs_1em02_pct": late_metric(rid, "dead_abs_frac_1em02"),
        "saturated_pct": late_metric(rid, "saturated_frac"),
        "erank": late_erank(rid),
        "n_seeds": len(rid),
    }

# ------------------------------------------------------------------- C3 / C5
def sweep_table(prefix, key_fn, baseline_key):
    rows, base_ids = {}, None
    sub = EX.runs[EX.runs.run_id.str.startswith(prefix)]
    for _, r in sub.iterrows():
        rows.setdefault(key_fn(r), []).append(r.run_id)
    if baseline_key in rows:
        base_ids = sorted(rows[baseline_key])
    out = {}
    for key, rid in rows.items():
        rid = sorted(rid)
        a, b = matrix(rid, EARLY), matrix(rid, LATE)
        out[key] = {
            "late": est(rid),
            "early": est(rid, EARLY),
            "drop_pp": 100 * (iqm(a) - iqm(b)),
            "dead_exact_pct": late_metric(rid, "dead_exact_frac"),
            "dead_abs_1em02_pct": late_metric(rid, "dead_abs_frac_1em02"),
            "dormant_0p1_pct": late_metric(rid, "dormant_frac_tau_0p1"),
            "erank": late_erank(rid),
            "lr": float(sub[sub.run_id.isin(rid)].lr.iloc[0]),
            "n_seeds": len(rid),
        }
        if base_ids and key != baseline_key:
            out[key]["vs_baseline"] = diff(rid, base_ids)
    return out


def c3_key(r):
    if r.norm and r.norm != "none":
        return f"norm_{r.norm}"
    if r.l2 > 0:
        return f"l2_{r.l2:g}"
    return r.run_id.split("_lr")[0].replace("c3_", "")


OUT["c3"] = sweep_table("c3_", c3_key, "backprop")
OUT["c5"] = sweep_table("c5_", lambda r: r.run_id.split("_lr")[0].replace("c5_", ""), "sgd")

# ------------------------------------------------------------------- compute
OUT["compute"] = {
    "runs_with_local_extracts": int(EX.runs.run_id.nunique()),
    "per_extract": {k: int(v) for k, v in EX.runs.extract.value_counts().items()},
}

with open(ROOT / "paper" / "numbers.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))
