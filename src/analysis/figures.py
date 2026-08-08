"""The paper's figures. Post-hoc only; never imported by training code.

Every number plotted here is read from a run's own parquet output and its
`config.json`. Nothing is typed in from a results table, including tables in
`CLAUDE.md` -- a figure that cannot be regenerated from `runs/` is a figure whose
provenance the paper cannot defend.

    python -m src.analysis.figures --extracts runs/_extracts --out figures

**Arms that have no extract on this machine are omitted and named in the caption
line printed to stdout**, rather than being silently dropped. A four-arm figure
that quietly becomes a three-arm figure is the kind of thing that survives all
the way into a submission.

Style follows the project's data-viz reference palette (light surface, the first
three categorical slots, which are the validated all-pairs subset). The baseline
arm is deliberately *not* a categorical slot: it is a reference level, drawn in
muted ink so the three intervention arms carry the colour.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gate import PLAN_PATH, load_plan
from .stats import iqm, stratified_bootstrap
from .survival import concentration

# -- palette (data-viz reference instance, light surface) ---------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
#: Categorical slots in fixed order. Only the first three are used for series
#: that appear together, which is the subset validated on the all-pairs list.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
#: One-hue sequential ramp (blue), for ordered parameters such as tau.
BLUES = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95"]

ARM_COLOR = {
    "none": MUTED,
    "redo": SLOT[0],
    "random_matched": SLOT[1],
    "inverse_matched": SLOT[2],
}
ARM_LABEL = {
    "none": "no intervention",
    "redo": "ReDo",
    "random_matched": "random-matched",
    "inverse_matched": "inverse-matched",
}
ARM_ORDER = ["none", "redo", "random_matched", "inverse_matched"]


def use_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "figure.dpi": 160,
        }
    )


def _grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    png = out / f"{name}.png"
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return png


# -- loading ------------------------------------------------------------------


class Extracts:
    """Every analysis extract found under a root, concatenated and indexed.

    An "extract" is a directory holding `runs.json` plus any of `tasks`,
    `metrics`, `recycling`, `intra_task` parquet files -- what
    `scripts/make_analysis_extract.py` writes. Several may be present, one per
    Kaggle session.
    """

    def __init__(self, root, plan_path=PLAN_PATH):
        self.root = Path(root)
        self.plan = load_plan(plan_path)
        frames: Dict[str, List[pd.DataFrame]] = {}
        meta: List[dict] = []
        self.sources: List[str] = []

        for d in sorted(p for p in self.root.glob("*") if p.is_dir()):
            manifest = d / "runs.json"
            if not manifest.exists():
                continue
            self.sources.append(d.name)
            with open(manifest, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    cfg = entry["config"]
                    rec = cfg.get("recycling", {}) or {}
                    meta.append(
                        {
                            "run_id": entry["run_id"],
                            "extract": d.name,
                            "arm": rec.get("kind", "none"),
                            "tau": float(rec.get("tau", 0.0)),
                            "lr": float(cfg["optim"]["lr"]),
                            "optimizer": cfg["optim"]["name"],
                            "activation": cfg["model"]["activation"],
                            "norm": cfg["model"].get("norm", "none"),
                            "l2": float((cfg.get("l2") or {}).get("lambda", 0.0)),
                            "seed": int(cfg["seed"]),
                            "dataset": cfg["data"]["name"],
                            "status": entry.get("summary", {}).get("status"),
                        }
                    )
            for name in ("tasks", "metrics", "recycling", "intra_task"):
                path = d / f"{name}.parquet"
                if path.exists():
                    frames.setdefault(name, []).append(pd.read_parquet(path))

        self.runs = pd.DataFrame(meta)
        self._tables = {
            k: pd.concat(v, ignore_index=True) for k, v in frames.items()
        }

    def table(self, name: str) -> pd.DataFrame:
        """A table joined to the run metadata, so every figure can filter by arm.

        Columns the table already carries win. `recycling.parquet` writes its own
        `arm` and `tau` -- the values in force at the event -- and pandas would
        otherwise suffix them to `tau_x`/`tau_y`, which every downstream groupby
        then gets wrong in a way that only shows up as a KeyError if you are
        lucky.
        """
        df = self._tables.get(name)
        if df is None:
            return pd.DataFrame()
        extra = [c for c in self.runs.columns if c == "run_id" or c not in df.columns]
        return df.merge(self.runs[extra], on="run_id", how="left")

    @property
    def window_late(self) -> List[int]:
        return list(
            range(
                self.plan["windows"]["late"]["task_idx"][0],
                self.plan["windows"]["late"]["task_idx"][1] + 1,
            )
        )

    @property
    def window_early(self) -> List[int]:
        return list(
            range(
                self.plan["windows"]["early"]["task_idx"][0],
                self.plan["windows"]["early"]["task_idx"][1] + 1,
            )
        )

    def arms_present(self) -> List[str]:
        if self.runs.empty:
            return []
        return [a for a in ARM_ORDER if a in set(self.runs.arm)]

    def select(
        self, by: str = "seeds", run_prefix: Optional[str] = None, **filters
    ) -> pd.DataFrame:
        """Matching runs, from **one** extract rather than pooled across them.

        Extracts overlap: the reproduction gate and the tau sweep both contain a
        no-intervention arm at lr=0.1, seeds 0-4, and the configs differ only in
        fields added to the schema afterwards. Concatenating them would count
        those five seeds twice and hand the bootstrap a correlation it assumes
        away. Picking one sweep also keeps `n` constant along an axis.

        ``by="seeds"`` takes the extract contributing the most runs (for a
        single-condition figure); ``by="levels"`` takes the one spanning the most
        learning rates (for a dose-response, which needs the whole ladder).

        **Filter on everything that has to match, including `activation`.**
        "arm == none, lr == 0.1, permuted MNIST" also describes all 25 runs of
        the activation sweep, whose baseline pools five different activations
        and a diverged tanh arm. That selection once silently supplied the
        tau-sweep figure's baseline, and it did not look wrong -- the IQM trims
        the outer 25%, so tanh's 10% was discarded and the number moved by less
        than a percentage point. ``run_prefix`` pins the comparison to a single
        sweep, which is the reliable guard.
        """
        runs = self.runs
        if run_prefix is not None:
            runs = runs[runs.run_id.str.startswith(run_prefix)]
        for key, value in filters.items():
            if value is None:
                continue
            runs = runs[runs[key] == value]
        if runs.empty or runs.extract.nunique() == 1:
            return runs
        score = (
            runs.groupby("extract").lr.nunique()
            if by == "levels"
            else runs.groupby("extract").size()
        )
        return runs[runs.extract == score.idxmax()]


def _score_matrix(tasks: pd.DataFrame, run_ids: Sequence[str], window: Sequence[int]) -> np.ndarray:
    """(n_runs, n_tasks) online accuracy, one row per seed -- the plan's shape."""
    sub = tasks[
        (tasks.run_id.isin(run_ids))
        & (tasks.probe_point == "task_end")
        & (tasks.task_idx.isin(window))
    ]
    wide = sub.pivot_table(
        index="run_id", columns="task_idx", values="online_accuracy"
    ).reindex(columns=window)
    if wide.isna().any().any():
        missing = wide.isna().any(axis=1)
        raise ValueError(f"missing window tasks for {list(wide.index[missing])[:3]}")
    return wide.to_numpy(dtype=np.float64)


def _estimate(scores: np.ndarray, plan: dict) -> Tuple[float, float, float]:
    st = plan["statistics"]
    est = stratified_bootstrap(
        scores,
        iqm,
        int(st["n_bootstrap"]),
        float(st["confidence"]),
        int(st["bootstrap_seed"]),
    )
    return est.point, est.lo, est.hi


# -- Figure 1: the tau sweep, the paper's headline ---------------------------


def fig_tau_sweep(ex: Extracts, out: Path) -> Optional[Path]:
    """Accuracy against tau, over the dead share of what each arm recycled.

    The whole C1 argument in one figure: as tau rises the recycled set gets
    *less* dead (lower panel) while accuracy gets *better* (upper panel). If
    recycling worked by resurrection those two panels would move together.
    """
    tasks, rec = ex.table("tasks"), ex.table("recycling")
    if tasks.empty:
        return None
    # Pinned to the tau sweep: every arm here, baseline included, must come from
    # the sweep whose seeds are paired by construction.
    runs = pd.concat(
        [
            ex.select(run_prefix="tau_", arm=a, lr=0.1, activation="relu",
                      dataset="permuted_mnist")
            for a in ex.arms_present()
        ],
        ignore_index=True,
    )
    interventions = [a for a in ARM_ORDER if a != "none" and a in set(runs.arm)]
    if not interventions:
        return None

    fig, (ax_acc, ax_dead) = plt.subplots(
        2, 1, figsize=(5.4, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.16},
    )

    # Baseline: one horizontal reference level, not a series.
    base = runs[runs.arm == "none"]
    if not base.empty:
        p, lo, hi = _estimate(_score_matrix(tasks, base.run_id, ex.window_late), ex.plan)
        ax_acc.axhspan(lo * 100, hi * 100, color=MUTED, alpha=0.18, lw=0, zorder=1)
        ax_acc.axhline(p * 100, color=MUTED, lw=1.4, ls=(0, (4, 2)), zorder=2)
        ax_acc.annotate(
            f"{ARM_LABEL['none']}  {p*100:.2f}%",
            xy=(0.985, p * 100), xycoords=("axes fraction", "data"),
            ha="right", va="bottom", fontsize=7.5, color=INK_2,
        )

    for arm in interventions:
        g = runs[runs.arm == arm]
        taus = sorted(g.tau.unique())
        xs, pts, los, his = [], [], [], []
        for t in taus:
            ids = g[g.tau == t].run_id
            p, lo, hi = _estimate(_score_matrix(tasks, ids, ex.window_late), ex.plan)
            xs.append(t); pts.append(p * 100); los.append(lo * 100); his.append(hi * 100)
        c = ARM_COLOR[arm]
        ax_acc.fill_between(xs, los, his, color=c, alpha=0.20, lw=0, zorder=3)
        ax_acc.plot(xs, pts, color=c, marker="o", zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.0)
        ax_acc.annotate(
            ARM_LABEL[arm], xy=(xs[-1], pts[-1]), xytext=(4, 0),
            textcoords="offset points", va="center", fontsize=8,
            color=INK_2, fontweight="semibold",
        )

        if not rec.empty:
            r = rec[rec.run_id.isin(g.run_id) & (rec.k > 0)]
            if not r.empty:
                # Composition on the fixed reference batch: the current-task
                # batch conflates a dead unit with one that is merely silent on
                # the permutation in force at that moment (CLAUDE.md §5).
                r = r.assign(dead_frac=r.n_dead_exact_ref / r.k)
                by_tau = r.groupby("tau").dead_frac.mean() * 100
                ax_dead.plot(by_tau.index, by_tau.values, color=c, marker="o",
                             markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
                ax_dead.annotate(
                    ARM_LABEL[arm], xy=(by_tau.index[-1], by_tau.values[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK_2, fontweight="semibold",
                )

    ax_acc.set_ylabel("late-window online accuracy (%)")
    ax_acc.set_title("Recycling helps more as its target set gets less dead", loc="left")
    ax_dead.set_ylabel("of the recycled set,\n% genuinely dead")
    ax_dead.set_xlabel(r"dormancy threshold $\tau$")
    ax_dead.set_ylim(bottom=0)
    for a in (ax_acc, ax_dead):
        _grid(a)
        # Room on the right for the direct labels, which sit outside the data.
        a.set_xlim(-0.012, 0.335)
    return _save(fig, out, "fig1_tau_sweep")


# -- Figure 2: C2, the definitions disagree -----------------------------------

#: (label, colour, columns) per death definition. The `dormant_tau` and
#: `dead_absolute` families are plotted as their headline parameter with a band
#: spanning the rest, because the spread *within* a definition is part of C2.
DEFINITIONS = [
    ("dead_exact  (Dohare et al.)", SLOT[0], ["dead_exact_frac_ref"], None),
    (
        r"dormant $\tau$  (Sokar et al.)",
        SLOT[1],
        ["dormant_frac_tau_0p1_ref"],
        ["dormant_frac_tau_0_ref", "dormant_frac_tau_0p25_ref"],
    ),
    # No family band for dead_absolute: its a=1e-6 member lands exactly on
    # dead_exact (which is the sanity check that both are implemented right, and
    # is reported in the text), so drawing it here just doubles the blue line.
    ("dead_absolute  (ours)", SLOT[2], ["dead_abs_frac_1em02_ref"], None),
]


def fig_c2_definitions(
    ex: Extracts, out: Path, arm: str = "none", lr: float = 0.1
) -> Optional[Path]:
    """The same activations, four definitions, wildly different answers."""
    metrics = ex.table("metrics")
    if metrics.empty:
        return None
    chosen = ex.select(arm=arm, lr=lr, activation="relu", dataset="permuted_mnist")
    m = metrics[
        metrics.run_id.isin(chosen.run_id) & (metrics.probe_point == "task_end")
    ]
    if m.empty:
        return None
    layers = sorted(m.layer_idx.unique())

    fig, axes = plt.subplots(1, len(layers), figsize=(2.35 * len(layers) + 0.6, 2.7),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lyr in zip(axes, layers):
        ml = m[m.layer_idx == lyr]
        for label, colour, main, band in DEFINITIONS:
            series = ml.groupby("task_idx")[main[0]].mean() * 100
            if band:
                # Hairlines, not a fill. Both families span most of the axis, so
                # three translucent regions stacked on each other reduce the
                # panel to mud -- and the point of the figure is that the lines
                # are far apart.
                for col in band:
                    ax.plot(ml.groupby("task_idx")[col].mean().index,
                            ml.groupby("task_idx")[col].mean().values * 100,
                            color=colour, lw=0.7, ls=(0, (2, 2)), alpha=0.75)
            ax.plot(series.index, series.values, color=colour, lw=1.8,
                    label=label if lyr == layers[0] else None)
        ax.set_title(f"hidden layer {lyr}", loc="left", color=INK_2, fontsize=8.5)
        ax.set_xlabel("task")
        _grid(ax)
    axes[0].set_ylabel("% of units flagged")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.10))
    fig.suptitle(
        "One network, one set of activations, three published definitions",
        x=0.0, ha="left", y=1.20, fontsize=9.5, fontweight="semibold", color=INK,
    )
    return _save(fig, out, "fig2_c2_definitions")


# -- Figure 3: C4, mortality is reversible ------------------------------------


def _no_intervention_runs(
    surv: Dict[str, pd.DataFrame], ex: Optional[Extracts], lr: Optional[float]
) -> List[str]:
    """Run ids in `surv` where nothing was ever recycled, at one learning rate.

    Panel (b) is captioned "recovery without intervention", so it has to come
    from runs that had none -- a recycled unit is alive again by construction and
    would be counted as a spontaneous recovery. The learning-rate filter matters
    for the same kind of reason: the gate sweep spans two decades of step size
    and pooling them would draw one curve through five different regimes.
    """
    rc = surv.get("recurrence")
    if rc is None or rc.empty:
        return []
    totals = rc.groupby("run_id").n_recycled.sum()
    ids = [str(r) for r in totals[totals == 0].index]
    if ex is not None and lr is not None and not ex.runs.empty:
        by_lr = ex.runs.set_index("run_id").lr.to_dict()
        ids = [r for r in ids if by_lr.get(r) == lr]
    return ids


def fig_c4_survival(
    surv: Dict[str, pd.DataFrame],
    out: Path,
    ex: Optional[Extracts] = None,
    lr: Optional[float] = 0.1,
) -> Optional[Path]:
    """Death is a state units churn through, not a fate they arrive at.

    Left: the share of units never yet silenced, against task. Middle: the
    probability a dead unit is alive again one task later, on both probe
    batches -- the gap between them is how much of "recovery" is really the
    input distribution moving. Right: how long an episode of silence lasts.
    """
    keep = _no_intervention_runs(surv, ex, lr)
    if not keep:
        return None
    tr, ep = surv.get("transitions"), surv.get("episodes")
    km_cur, km_ref = surv.get("survival_matrix_current"), surv.get("survival_matrix_reference")
    if tr is None or tr.empty or km_ref is None:
        return None
    tr = tr[tr.run_id.isin(keep)]
    ep = None if ep is None else ep[ep.run_id.isin(keep)]
    km_ref = km_ref.loc[km_ref.index.isin(keep)]
    km_cur = None if km_cur is None else km_cur.loc[km_cur.index.isin(keep)]
    if tr.empty or km_ref.empty:
        return None

    fig, (ax_km, ax_rec, ax_ep) = plt.subplots(1, 3, figsize=(8.6, 2.8))

    # -- (a) Kaplan-Meier, reference batch
    grid = np.asarray([int(c) for c in km_ref.columns])
    curves = km_ref.to_numpy() * 100
    med = np.median(curves, axis=0)
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    ax_km.fill_between(grid, lo, hi, color=SLOT[0], alpha=0.20, lw=0)
    ax_km.plot(grid, med, color=SLOT[0])
    if km_cur is not None:
        ax_km.plot(grid, np.median(km_cur.to_numpy() * 100, axis=0),
                   color=MUTED, lw=1.4, ls=(0, (4, 2)))
        ax_km.annotate("current-task batch", xy=(grid[-1], np.median(km_cur.to_numpy()*100, axis=0)[-1]),
                       xytext=(-4, 6), textcoords="offset points", ha="right",
                       fontsize=7.5, color=MUTED)
    ax_km.annotate("fixed reference batch", xy=(grid[-1], med[-1]), xytext=(-4, 8),
                   textcoords="offset points", ha="right", fontsize=7.5,
                   color=SLOT[0], fontweight="semibold")
    ax_km.set_xlabel("task"); ax_km.set_ylabel("% of units never yet silent")
    ax_km.set_ylim(0, 100)
    ax_km.set_title("(a) time to first death", loc="left", fontsize=8.5, color=INK_2)
    _grid(ax_km)

    # -- (b) recovery probability per layer, both probes
    g = tr.groupby(["probe", "layer_idx"])[["n_dead_alive", "n_dead_dead"]].sum()
    g["p"] = 100 * g.n_dead_alive / (g.n_dead_alive + g.n_dead_dead)
    layers = sorted(tr.layer_idx.unique())
    width = 0.38
    xs = np.arange(len(layers))
    for i, (probe, colour, label) in enumerate(
        [("current", MUTED, "current-task batch"), ("reference", SLOT[0], "fixed reference batch")]
    ):
        if probe not in g.index.get_level_values(0):
            continue
        vals = [g.loc[(probe, l), "p"] for l in layers]
        # 2px surface gap between adjacent bars, per the mark spec.
        ax_rec.bar(xs + (i - 0.5) * (width + 0.03), vals, width, color=colour,
                   label=label, zorder=3, edgecolor=SURFACE, linewidth=1.0)
        for x, v in zip(xs + (i - 0.5) * (width + 0.03), vals):
            ax_rec.annotate(f"{v:.0f}", (x, v), xytext=(0, 2),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=INK_2)
    ax_rec.set_xticks(xs, [f"layer {l}" for l in layers])
    ax_rec.set_ylabel("% of dead units alive\none task later")
    ax_rec.set_title("(b) recovery without intervention", loc="left", fontsize=8.5, color=INK_2)
    # Above the plot area: inside, it lands on the tallest bar.
    ax_rec.legend(loc="lower left", bbox_to_anchor=(0, 1.10), ncol=2,
                  columnspacing=1.2, handlelength=1.4)
    ax_rec.set_ylim(0, max(g.p) * 1.18)
    _grid(ax_rec)

    # -- (c) episode length ECDF, reference batch
    if ep is not None and not ep.empty:
        e = ep[(ep.probe == "reference") & (~ep.censored)]
        # Depth is ordered, so the layers take steps of one hue rather than
        # categorical slots -- but widely separated steps, or three near-
        # identical blues make the panel unreadable.
        for lyr, colour, y_at in zip(layers, (BLUES[0], BLUES[2], BLUES[5]), (32, 55, 78)):
            v = np.sort(e[e.layer_idx == lyr].length_tasks.to_numpy())
            if not v.size:
                continue
            frac = 100 * np.arange(1, v.size + 1) / v.size
            ax_ep.step(v, frac, where="post", color=colour, lw=1.8)
            # Anchor each label on its own curve at a different height, so three
            # curves that converge at the right do not stack their labels.
            x_at = v[np.searchsorted(frac, y_at)] if frac[-1] >= y_at else v[-1]
            ax_ep.annotate(f"layer {lyr}", xy=(x_at, y_at), xytext=(5, -3),
                           textcoords="offset points", fontsize=7.5,
                           color=colour, fontweight="semibold")
        ax_ep.set_xscale("log")
        ax_ep.set_xlabel("episode length (tasks)")
        ax_ep.set_ylabel("% of episodes at most this long")
        ax_ep.set_ylim(0, 100)
        ax_ep.set_title("(c) how long silence lasts", loc="left", fontsize=8.5, color=INK_2)
        _grid(ax_ep)

    fig.suptitle(
        "Dead units come back on their own",
        x=0.0, ha="left", y=1.06, fontsize=9.5, fontweight="semibold", color=INK,
    )
    fig.tight_layout()
    return _save(fig, out, "fig3_c4_survival")


# -- Figure 4: C4, the same living units, over and over -----------------------


def fig_c4_recurrence(
    survs: Dict[str, Dict[str, pd.DataFrame]], out: Path
) -> Optional[Path]:
    """Which units recycling keeps choosing, and what state they were in.

    Takes one survival directory **per arm**, because the comparison is the
    argument. `random_matched` picks uniformly by construction, so it is the
    negative control: if the concentration statistics do not read exactly the
    same as the null on that arm, they are measuring something other than
    targeting and nothing else here can be trusted.
    """
    arms = {
        name: s for name, s in survs.items()
        if s.get("recurrence") is not None
        and not s["recurrence"].empty
        and s["recurrence"].n_recycled.any()
    }
    if not arms:
        return None
    order = [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]

    fig, (ax_hist, ax_state) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    # -- (a) how unevenly the recycling slots were spread over units
    labels, ginis, nulls, colours = [], [], [], []
    for arm in order:
        s = arms[arm]
        r = s["recurrence"][s["recurrence"].probe == "reference"]
        obs = r.groupby(["run_id", "layer_idx", "neuron_idx"]).n_recycled.first().to_numpy()
        colour = ARM_COLOR.get(arm, SLOT[0])
        ax_hist.hist(obs, bins=np.arange(0, obs.max() + 3) - 0.5, density=True,
                     histtype="step", lw=1.8, color=colour, zorder=3)
        labels.append(ARM_LABEL.get(arm, arm))
        colours.append(colour)
        ginis.append(concentration(obs)["gini"])
        null = s.get("null")
        nulls.append(float(null.gini.mean()) if null is not None and not null.empty
                     else float("nan"))
    ax_hist.set_xlabel("times a unit was recycled")
    ax_hist.set_ylabel("density of units")
    ax_hist.set_title("(a) how concentrated the choice is", loc="left",
                      fontsize=8.5, color=INK_2)
    rows = [
        f"{lab}: Gini {g:.2f}" + ("" if np.isnan(n) else f"  (null {n:.2f})")
        for lab, g, n in zip(labels, ginis, nulls)
    ]
    ax_hist.annotate(
        "\n".join(rows), xy=(0.97, 0.93), xycoords="axes fraction", ha="right",
        va="top", fontsize=7.5, color=INK_2, linespacing=1.5,
        bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.94, pad=3.0),
    )
    _grid(ax_hist)

    # -- (b) alive or dead when chosen, per arm
    ys = np.arange(len(order))[::-1]
    for y, arm, colour in zip(ys, order, colours):
        r = arms[arm]["recurrence"]
        r = r[r.probe == "reference"]
        alive = r.n_recycled_while_alive.sum()
        dead = r.n_recycled_while_dead.sum()
        pct = 100 * alive / (alive + dead)
        ax_state.barh([y], [pct], color=colour, height=0.42, zorder=3,
                      edgecolor=SURFACE, linewidth=1.0)
        ax_state.annotate(f"{pct:.1f}%", (pct, y), xytext=(4, 0),
                          textcoords="offset points", va="center",
                          fontsize=8.5, color=INK, fontweight="semibold")
    ax_state.set_yticks(ys, [ARM_LABEL.get(a, a) for a in order])
    ax_state.set_xlim(0, 112)
    ax_state.set_xlabel("% of recycling slots where the unit was ALIVE")
    ax_state.set_title("(b) state at the boundary before", loc="left",
                       fontsize=8.5, color=INK_2)
    _grid(ax_state, axis="x")

    fig.suptitle("Recycling the living", x=0.0, ha="left", y=1.05,
                 fontsize=9.5, fontweight="semibold", color=INK)
    fig.tight_layout()
    return _save(fig, out, "fig4_c4_recurrence")


# -- Figure 5: the reproduction gate as a dose-response -----------------------


def fig_gate_dose_response(ex: Extracts, out: Path) -> Optional[Path]:
    """Plasticity loss and dead units both scale with step size, monotonically."""
    tasks, metrics = ex.table("tasks"), ex.table("metrics")
    if tasks.empty:
        return None
    base = ex.select(by="levels", arm="none", activation="relu",
                     dataset="permuted_mnist")
    lrs = sorted(base.lr.unique())
    if len(lrs) < 3:
        return None

    fig, (ax_drop, ax_dead) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    drops, deads = [], []
    for lr in lrs:
        ids = base[base.lr == lr].run_id
        early = _score_matrix(tasks, ids, ex.window_early)
        late = _score_matrix(tasks, ids, ex.window_late)
        drops.append((iqm(early) - iqm(late)) * 100)
        m = metrics[
            (metrics.run_id.isin(ids))
            & (metrics.probe_point == "task_end")
            & (metrics.task_idx.isin(ex.window_late))
        ]
        deads.append(m.dead_exact_frac_ref.mean() * 100)

    for ax, vals, ylab, colour, title in [
        (ax_drop, drops, "accuracy drop, early → late (pp)", SLOT[0],
         "(a) plasticity loss"),
        (ax_dead, deads, "% dead_exact, late window", SLOT[1], "(b) dead units"),
    ]:
        ax.plot(lrs, vals, color=colour, marker="o", markeredgecolor=SURFACE,
                markeredgewidth=1.0, zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK_2)
        _grid(ax)
    ax_drop.axhline(0, color=AXIS, lw=0.8, zorder=2)
    fig.suptitle(
        "The phenomenon is present and dose-dependent, not absent",
        x=0.0, ha="left", y=1.04, fontsize=9.5, fontweight="semibold", color=INK,
    )
    fig.tight_layout()
    return _save(fig, out, "fig5_gate_dose_response")


# -- Figure 6: death is partly distribution-relative --------------------------


def fig_reference_asymmetry(
    ex: Extracts, out: Path, arm: str = "none", lr: float = 0.1
) -> Optional[Path]:
    """`dead_exact` on the current task against the fixed reference batch.

    None of the four source papers can see this: none of them probes a second,
    fixed distribution. A unit that is alive for the permutation being trained on
    and silent on everything else is not "dead" in the sense the literature means.
    """
    metrics = ex.table("metrics")
    if metrics.empty:
        return None
    chosen = ex.select(arm=arm, lr=lr, activation="relu", dataset="permuted_mnist")
    m = metrics[
        metrics.run_id.isin(chosen.run_id) & (metrics.probe_point == "task_end")
    ]
    if m.empty:
        return None
    layers = sorted(m.layer_idx.unique())

    fig, axes = plt.subplots(1, len(layers), figsize=(2.35 * len(layers) + 0.6, 2.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lyr in zip(axes, layers):
        ml = m[m.layer_idx == lyr]
        cur = ml.groupby("task_idx").dead_exact_frac.mean() * 100
        ref = ml.groupby("task_idx").dead_exact_frac_ref.mean() * 100
        ax.fill_between(cur.index, cur.values, ref.values,
                        color=SLOT[1], alpha=0.20, lw=0)
        ax.plot(cur.index, cur.values, color=MUTED, lw=1.6, label="current task")
        ax.plot(ref.index, ref.values, color=SLOT[1], lw=1.8, label="fixed reference")
        ax.set_title(f"hidden layer {lyr}", loc="left", color=INK_2, fontsize=8.5)
        ax.set_xlabel("task")
        _grid(ax)
    axes[0].set_ylabel("% dead_exact")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        '"Dead" is partly a statement about the input distribution',
        x=0.0, ha="left", y=1.06, fontsize=9.5, fontweight="semibold", color=INK,
    )
    fig.tight_layout()
    return _save(fig, out, "fig6_reference_asymmetry")


# -- Figure 7: C3, the anomalies ---------------------------------------------


def _arm_from_run_id(run_id: str, prefix: str) -> str:
    """Arm name out of a run_id like `c3_l2_1em3_lr0p1_s2`.

    C3 and C5 vary the arm through fields that are not one config key -- an L2
    lambda, an optimizer's eps and beta2, a shrink-and-perturb flag -- so unlike
    the tau sweep there is no single column to group by. The run_id is
    constructed by `scripts/make_configs.py` and is the only place the arm is
    named as a unit.
    """
    stem = run_id[len(prefix):] if run_id.startswith(prefix) else run_id
    # The seed suffix is always present; the `_lr...` segment is not -- C3 writes
    # `c3_l2_1em3_lr0p1_s2` but C5 writes `c5_adam_lyle_s0`, because the C5 arms
    # differ in optimizer hyperparameters rather than in learning rate. Strip the
    # seed first, then the learning rate if there is one.
    stem = re.sub(r"_s\d+$", "", stem)
    return re.sub(r"_lr[0-9p.eE+-]+$", "", stem)


def _arm_points(
    ex: Extracts, runs: pd.DataFrame, prefix: str
) -> List[Tuple[str, float, float, float, float]]:
    """(arm, accuracy, ci_lo, ci_hi, dead_exact) per arm, late window."""
    tasks, metrics = ex.table("tasks"), ex.table("metrics")
    out = []
    runs = runs.assign(_arm=[_arm_from_run_id(r, prefix) for r in runs.run_id])
    for arm, g in runs.groupby("_arm"):
        acc, lo, hi = _estimate(_score_matrix(tasks, g.run_id, ex.window_late), ex.plan)
        m = metrics[
            metrics.run_id.isin(g.run_id)
            & (metrics.probe_point == "task_end")
            & (metrics.task_idx.isin(ex.window_late))
        ]
        out.append((arm, acc * 100, lo * 100, hi * 100, m.dead_exact_frac_ref.mean() * 100))
    return sorted(out, key=lambda r: r[4])


def fig_c3_anomaly(ex: Extracts, out: Path, lr: float = 0.1) -> Optional[Path]:
    """C3: plasticity and dead units move *independently*.

    Plotted as accuracy against dead-unit count, one point per arm, because the
    claim is a dissociation and a dissociation is a two-dimensional statement.
    Two bar charts side by side would let a reader match arms up by eye and get
    the same information, but the point -- that arms sit up and to the *right*,
    better and deader -- only becomes obvious in the plane.

    Arms are direct-labelled rather than coloured: a scatter needs the all-pairs
    palette gate, which caps categorical hues at three, and there are seven arms.
    """
    # By run_id prefix, not `select`: the C3 arms differ in fields that are not
    # one config key (an L2 lambda, a shrink-and-perturb flag), so `arm` is
    # "none" for all seven and a metadata filter cannot separate them from the
    # gate's no-intervention runs.
    runs = ex.runs[ex.runs.run_id.str.startswith("c3_") & (ex.runs.lr == lr)]
    if runs.empty:
        return None
    pts = _arm_points(ex, runs, "c3_")
    baseline = next((p for p in pts if "backprop" in p[0]), None)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for arm, acc, lo, hi, dead in pts:
        is_base = baseline is not None and arm == baseline[0]
        colour = MUTED if is_base else SLOT[0]
        ax.errorbar(dead, acc, yerr=[[acc - lo], [hi - acc]], fmt="o", color=colour,
                    ecolor=colour, elinewidth=1.2, capsize=2.5, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.0)
        ax.annotate(arm.replace("_", " "), xy=(dead, acc), xytext=(6, 3),
                    textcoords="offset points", fontsize=7.5,
                    color=INK_2 if not is_base else MUTED,
                    fontweight="semibold" if not is_base else "normal")
    if baseline is not None:
        ax.axhline(baseline[1], color=MUTED, lw=1.0, ls=(0, (4, 2)), zorder=2)
        ax.axvline(baseline[4], color=MUTED, lw=1.0, ls=(0, (4, 2)), zorder=2)
        ax.annotate("plain backprop", xy=(0.015, baseline[1]),
                    xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", fontsize=7.5, color=MUTED)
    ax.set_xlabel("% dead_exact, late window (reference batch)")
    ax.set_ylabel("late-window online accuracy (%)")
    ax.set_title("Anything up and to the right is better AND deader", loc="left")
    _grid(ax, axis="both")
    ax.margins(0.14)
    return _save(fig, out, "fig7_c3_anomaly")


# -- Figure 8: C5, optimizers -------------------------------------------------


def fig_c5_optimizers(ex: Extracts, out: Path, lr: Optional[float] = None) -> Optional[Path]:
    """C5: Adam's death accumulates *across* tasks, not in a spike after each one.

    Left panel is the pre-registered prediction, drawn on the grid it was
    predicted on (steps since the task switch). Right panel is what actually
    carries the effect. Reporting only the right panel would hide a failed
    prediction; reporting only the left would hide a real result.
    """
    intra, metrics = ex.table("intra_task"), ex.table("metrics")
    runs = ex.runs[ex.runs.run_id.str.startswith("c5_")]
    if runs.empty:
        return None
    runs = runs.assign(_arm=[_arm_from_run_id(r, "c5_") for r in runs.run_id])
    arms = sorted(runs._arm.unique())
    colours = {a: c for a, c in zip(arms, [MUTED] + SLOT)}

    # The within-task panel needs `intra_task.parquet`, which the first C5
    # extract omitted (the `EXTRACT_TABLES` bug, since fixed and pinned by
    # `test_train.py::test_intra_task_table_is_in_the_analysis_extract`). Draw
    # the cross-task panel alone rather than nothing, and say which panel is
    # absent -- the missing one is the *pre-registered prediction*, so silently
    # shipping only the panel that worked would be exactly the wrong failure.
    has_intra = not intra.empty
    if not has_intra:
        print("    fig_c5_optimizers: no intra_task.parquet in this extract, so "
              "panel (a) -- the pre-registered within-task spike -- is NOT drawn. "
              "Re-extract from the archival runs.zip to restore it.")
    fig, axes = plt.subplots(1, 2 if has_intra else 1,
                             figsize=(7.2 if has_intra else 4.0, 2.9))
    axes = np.atleast_1d(axes)
    ax_task = axes[-1]

    if has_intra:
        ax_step = axes[0]
        it = intra.merge(runs[["run_id", "_arm"]], on="run_id", how="inner")
        # Tasks 50+ only: the first tasks are still in the initial transient,
        # where every arm's death count is moving for reasons that have nothing
        # to do with distance from a switch.
        late = it[it.task_idx >= 50]
        for arm in arms:
            g = late[late._arm == arm].groupby("step_in_task").dead_exact_frac.mean() * 100
            if g.empty:
                continue
            ax_step.plot(g.index, g.values, color=colours[arm], lw=1.8)
            ax_step.annotate(arm.replace("_", " "), xy=(g.index[-1], g.values[-1]),
                             xytext=(4, 0), textcoords="offset points", va="center",
                             fontsize=7.5, color=colours[arm], fontweight="semibold")
        ax_step.set_xlabel("optimizer steps since the task switch")
        ax_step.set_ylabel("% dead_exact")
        ax_step.set_title("(a) within a task: the predicted spike", loc="left",
                          fontsize=8.5, color=INK_2)
        _grid(ax_step)
        ax_step.margins(x=0.22)

    m = metrics[metrics.probe_point == "task_end"].merge(
        runs[["run_id", "_arm"]], on="run_id", how="inner"
    )
    for arm in arms:
        g = m[m._arm == arm].groupby("task_idx").dead_exact_frac_ref.mean() * 100
        if g.empty:
            continue
        ax_task.plot(g.index, g.values, color=colours[arm], lw=1.8)
        ax_task.annotate(arm.replace("_", " "), xy=(g.index[-1], g.values[-1]),
                         xytext=(4, 0), textcoords="offset points", va="center",
                         fontsize=7.5, color=colours[arm], fontweight="semibold")
    ax_task.set_xlabel("task")
    ax_task.set_ylabel("% dead_exact")
    ax_task.set_title(
        ("(b) " if has_intra else "") + "across tasks: where it actually accumulates",
        loc="left", fontsize=8.5, color=INK_2,
    )
    _grid(ax_task)
    ax_task.margins(x=0.22)
    if not has_intra:
        ax_task.annotate(
            "the within-task panel needs intra_task.parquet,\n"
            "absent from this extract",
            xy=(0.5, -0.34), xycoords="axes fraction", ha="center",
            fontsize=7, color=MUTED,
        )
    fig.tight_layout()
    return _save(fig, out, "fig8_c5_optimizers")


# -- Figure 9: Setting 3, the definitions depend on the activation ------------


def fig_setting3_activations(
    ex: Extracts,
    out: Path,
    lr: float = 0.1,
    tanh_lr: Optional[float] = None,
    chance_margin_pp: float = 5.0,
) -> Optional[Path]:
    """The strongest single C2 demonstration: change the activation, and the
    field's standard definition stops finding anything while the others do not.

    ``tanh_lr`` swaps the tanh row for the calibrated runs from
    `configs/setting3_tanh_gate` once that gate has been decided. Until it is
    passed, a tanh arm that diverged is **dropped, not drawn** -- at lr=0.1 tanh
    reached 10.05%, which is chance for ten classes, and its death metrics are
    meaningless. Plotting it would put a failed arm in a table as though it were
    a result (CLAUDE.md §11 says explicitly not to).
    """
    metrics, tasks = ex.table("metrics"), ex.table("tasks")
    runs = ex.runs[ex.runs.run_id.str.startswith("s3_") & (ex.runs.lr == lr)]
    if tanh_lr is not None:
        gate_runs = ex.runs[
            ex.runs.run_id.str.startswith("s3tanh_") & (ex.runs.lr == tanh_lr)
        ]
        runs = pd.concat([runs[runs.activation != "tanh"], gate_runs], ignore_index=True)
    if runs.empty or metrics.empty:
        return None

    chance_pp = 100.0 / 10  # ten classes
    kept, dropped = [], []
    for act, g in runs.groupby("activation"):
        acc, _, _ = _estimate(_score_matrix(tasks, g.run_id, ex.window_late), ex.plan)
        (kept if acc * 100 > chance_pp + chance_margin_pp else dropped).append(act)
    if dropped:
        print(f"    fig_setting3_activations: dropped {', '.join(sorted(dropped))} "
              f"-- at or near chance ({chance_pp:.0f}%), so the death metrics are "
              "meaningless; calibrate its learning rate and pass tanh_lr")
    runs = runs[runs.activation.isin(kept)]
    if runs.empty:
        return None

    # `ex.table` has already joined the run metadata, so `activation` is present;
    # merging it again would produce activation_x / activation_y.
    m = metrics[
        metrics.run_id.isin(runs.run_id) & (metrics.probe_point == "task_end")
        & (metrics.task_idx.isin(ex.window_late))
    ]
    if m.empty:
        return None

    cols = [
        ("dead_exact  (Dohare et al.)", SLOT[0], "dead_exact_frac_ref"),
        (r"dormant $\tau$=0.1  (Sokar et al.)", SLOT[1], "dormant_frac_tau_0p1_ref"),
        ("dead_absolute a=1e-2  (ours)", SLOT[2], "dead_abs_frac_1em02_ref"),
        # Defined only for bounded activations, so it is absent everywhere but
        # tanh -- and on tanh it is the ONLY definition that fires at all.
        # Omitting it made the tanh group read as "nothing is wrong here", which
        # is the opposite of what those runs show.
        ("saturated  (bounded only)", SLOT[3], "saturated_frac_ref"),
    ]
    acts = [a for a in ("relu", "leaky_relu", "gelu", "silu", "tanh")
            if a in set(m.activation)]
    xs = np.arange(len(acts))
    width = 0.20

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, (label, colour, col) in enumerate(cols):
        vals = []
        for a in acts:
            sub = m[m.activation == a]
            vals.append(
                sub[col].mean() * 100
                if col in sub and sub[col].notna().any() else np.nan
            )
        pos = xs + (i - 1.5) * (width + 0.02)
        ax.bar(pos, vals, width, color=colour, label=label, zorder=3,
               edgecolor=SURFACE, linewidth=1.0)
        for x, v in zip(pos, vals):
            if np.isnan(v):
                continue  # undefined for this activation; a 0 here would be a lie
            # A zero bar is invisible, and "exactly zero" is the whole finding.
            ax.annotate("0" if v == 0 else f"{v:.0f}", (x, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7,
                        color=INK if v == 0 else INK_2,
                        fontweight="semibold" if v == 0 else "normal")
    ax.set_xticks(xs, [a.replace("_", "-") for a in acts])
    ax.set_ylabel("% of units flagged, late window")
    ax.set_title("Change the activation and the definitions stop agreeing",
                 loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=4,
              columnspacing=1.0, handlelength=1.3)
    _grid(ax)
    return _save(fig, out, "fig9_setting3_activations")


# -- Figure 10: Setting 2, the unit is a channel ------------------------------


def fig_setting2_channels(ex: Extracts, out: Path) -> Optional[Path]:
    """In a conv layer `dead_exact` finds ~nothing while most of the feature map
    is silent -- the failure CLAUDE.md §5.5 predicted, measured.
    """
    metrics = ex.table("metrics")
    runs = ex.runs[ex.runs.dataset == "label_shuffled_cifar10"]
    if runs.empty or metrics.empty:
        return None
    base = runs[runs.arm == "none"]
    m = metrics[
        metrics.run_id.isin(base.run_id) & (metrics.probe_point == "task_end")
    ]
    if m.empty:
        return None
    # The late window is a 200-task window; Setting 2 may be shorter, so use its
    # own last fifth rather than silently comparing different points in training.
    cutoff = m.task_idx.max() - max(1, (m.task_idx.max() + 1) // 5)
    m = m[m.task_idx > cutoff]
    layers = sorted(m.layer_idx.unique())

    cols = [
        ("dead_exact (whole channel)", SLOT[0], "dead_exact_frac_ref"),
        ("spatially silent positions", SLOT[1], "mean_frac_zero_positions_ref"),
        (r"dormant $\tau$=0.1", SLOT[2], "dormant_frac_tau_0p1_ref"),
    ]
    xs = np.arange(len(layers))
    width = 0.26
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for i, (label, colour, col) in enumerate(cols):
        vals = [m[m.layer_idx == l][col].mean() * 100 for l in layers]
        pos = xs + (i - 1) * (width + 0.02)
        ax.bar(pos, vals, width, color=colour, label=label, zorder=3,
               edgecolor=SURFACE, linewidth=1.0)
        for x, v in zip(pos, vals):
            ax.annotate(f"{v:.0f}" if v >= 0.5 else f"{v:.2f}", (x, v),
                        xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=7, color=INK_2)
    spatial = m.groupby("layer_idx").is_spatial.first()
    ax.set_xticks(xs, [f"layer {l}\n{'conv' if spatial.get(l, False) else 'fc'}"
                       for l in layers])
    ax.set_ylabel("% flagged, last fifth of training")
    ax.set_title("A channel can be 88% silent and still count as alive", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=3,
              columnspacing=1.2, handlelength=1.4)
    _grid(ax)
    return _save(fig, out, "fig10_setting2_channels")


# -- driver -------------------------------------------------------------------


def build_all(
    extracts_root="runs/_extracts",
    survival_dirs: Optional[Sequence[str]] = None,
    out="figures",
    tanh_lr: Optional[float] = None,
) -> List[Path]:
    use_paper_style()
    out = Path(out)
    ex = Extracts(extracts_root)
    made: List[Path] = []

    print(f"extracts: {', '.join(ex.sources) or '(none)'}")
    print(f"runs:     {len(ex.runs)}   arms: {', '.join(ex.arms_present()) or '(none)'}")
    absent = [a for a in ARM_ORDER if a not in ex.arms_present()]
    if absent:
        print(f"MISSING ARMS, figures below are drawn without them: {', '.join(absent)}")

    for fn in (
        fig_tau_sweep,
        fig_c2_definitions,
        fig_gate_dose_response,
        fig_reference_asymmetry,
        fig_c3_anomaly,
        fig_c5_optimizers,
        fig_setting3_activations,
        fig_setting2_channels,
    ):
        kwargs = {"tanh_lr": tanh_lr} if fn is fig_setting3_activations else {}
        path = fn(ex, out, **kwargs)
        print(f"  {'wrote' if path else 'skipped'} {fn.__name__}"
              + (f" -> {path.name}" if path else " (no extract for it yet)"))
        if path:
            made.append(path)

    # Several survival directories may be given: the C4 mortality figure needs
    # runs with no intervention, the recurrence figure compares arms that had
    # one, and no single sweep supplies both.
    loaded: Dict[str, Dict[str, pd.DataFrame]] = {}
    for d in survival_dirs or []:
        surv = {p.stem: pd.read_parquet(p) for p in sorted(Path(d).glob("*.parquet"))}
        loaded[_arm_of_survival(surv, Path(d).name)] = surv
        path = fig_c4_survival(surv, out, ex=ex)
        if path:
            print(f"  wrote fig_c4_survival -> {path.name}   [{Path(d).name}]")
            made.append(path)

    if loaded:
        path = fig_c4_recurrence(loaded, out)
        if path:
            print(f"  wrote fig_c4_recurrence -> {path.name}   "
                  f"[{', '.join(sorted(loaded))}]")
            made.append(path)
    return made


def _arm_of_survival(surv: Dict[str, pd.DataFrame], fallback: str) -> str:
    """Name the arm from the run_ids inside, not from the directory name.

    A directory called `surv_random` is a convention; `tau_random_matched_*` in
    the data is the fact. Mislabelling an arm in this figure would invert its
    argument, so it is read from the runs.
    """
    rc = surv.get("recurrence")
    if rc is None or rc.empty:
        return fallback
    ids = " ".join(rc.run_id.unique()[:5])
    for arm in ("random_matched", "inverse_matched", "redo"):
        if arm in ids:
            return arm
    return fallback


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the paper's figures.")
    ap.add_argument("--extracts", default="runs/_extracts")
    ap.add_argument("--survival", nargs="*", default=[],
                    help="directories written by `python -m src.analysis.survival --out`")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--tanh-lr", type=float, default=None,
                    help="learning rate that passed configs/setting3_tanh_gate; "
                         "swaps the diverged tanh row for the calibrated runs")
    args = ap.parse_args(argv)
    build_all(args.extracts, args.survival, out=args.out, tanh_lr=args.tanh_lr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
