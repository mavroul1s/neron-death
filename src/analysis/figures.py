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

    def select(self, by: str = "seeds", **filters) -> pd.DataFrame:
        """Matching runs, from **one** extract rather than pooled across them.

        Extracts overlap: the reproduction gate and the tau sweep both contain a
        no-intervention arm at lr=0.1, seeds 0-4, and the configs differ only in
        fields added to the schema afterwards. Concatenating them would count
        those five seeds twice and hand the bootstrap a correlation it assumes
        away. Picking one sweep also keeps `n` constant along an axis.

        ``by="seeds"`` takes the extract contributing the most runs (for a
        single-condition figure); ``by="levels"`` takes the one spanning the most
        learning rates (for a dose-response, which needs the whole ladder).
        """
        runs = self.runs
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
    runs = pd.concat(
        [ex.select(arm=a, lr=0.1, dataset="permuted_mnist") for a in ex.arms_present()],
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
    (
        "dead_absolute  (ours)",
        SLOT[2],
        ["dead_abs_frac_1em02_ref"],
        ["dead_abs_frac_1em06_ref", "dead_abs_frac_1em02_ref"],
    ),
]


def fig_c2_definitions(
    ex: Extracts, out: Path, arm: str = "none", lr: float = 0.1
) -> Optional[Path]:
    """The same activations, four definitions, wildly different answers."""
    metrics = ex.table("metrics")
    if metrics.empty:
        return None
    chosen = ex.select(arm=arm, lr=lr, dataset="permuted_mnist")
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


def fig_c4_recurrence(surv: Dict[str, pd.DataFrame], out: Path) -> Optional[Path]:
    """How concentrated recycling is on particular units, against a same-dose null."""
    rc, null = surv.get("recurrence"), surv.get("null")
    if rc is None or rc.empty or not rc.n_recycled.any():
        return None
    r = rc[rc.probe == "reference"]

    fig, (ax_hist, ax_state) = plt.subplots(1, 2, figsize=(6.6, 2.8))

    # -- (a) distribution of per-unit recycle counts, observed vs null
    obs = r.groupby(["run_id", "layer_idx", "neuron_idx"]).n_recycled.first().to_numpy()
    bins = np.arange(0, obs.max() + 3) - 0.5
    ax_hist.hist(obs, bins=bins, color=SLOT[0], alpha=0.85, zorder=3, density=True)
    ax_hist.axvline(obs.mean(), color=INK_2, lw=1.2, ls=(0, (4, 2)), zorder=4)
    ax_hist.annotate(
        f"mean {obs.mean():.0f}", xy=(obs.mean(), 0.99), xycoords=("data", "axes fraction"),
        xytext=(4, -2), textcoords="offset points", fontsize=7.5, color=INK_2, va="top",
    )
    # The null holds the per-task, per-layer dose fixed and redraws only *which*
    # units, so any excess concentration is targeting, not dose.
    never = 100 * (obs == 0).mean()
    lines = [f"{never:.1f}% of units never recycled once"]
    if null is not None and not null.empty:
        obs_gini = concentration(obs)["gini"]
        q = np.percentile(null.gini.to_numpy(), [2.5, 97.5])
        lines += [
            f"Gini {obs_gini:.2f}",
            f"same-dose null: {null.gini.mean():.2f} [{q[0]:.2f}, {q[1]:.2f}],",
            "and it leaves no unit untouched",
        ]
    ax_hist.annotate(
        "\n".join(lines), xy=(0.30, 0.86), xycoords="axes fraction",
        fontsize=7.5, color=INK_2, va="top", linespacing=1.45,
    )
    ax_hist.set_xlabel("times a unit was recycled")
    ax_hist.set_ylabel("density of units")
    ax_hist.set_title("(a) recycling concentrates on some units", loc="left",
                      fontsize=8.5, color=INK_2)
    _grid(ax_hist)

    # -- (b) what state were they in?
    tot_dead = r.n_recycled_while_dead.sum()
    tot_alive = r.n_recycled_while_alive.sum()
    share = 100 * np.array([tot_alive, tot_dead]) / (tot_alive + tot_dead)
    ax_state.barh([1, 0], share, color=[SLOT[1], SLOT[0]], height=0.34,
                  zorder=3, edgecolor=SURFACE, linewidth=1.0)
    ax_state.set_yticks([1, 0], ["alive when chosen", "genuinely dead"])
    for y, v in zip([1, 0], share):
        ax_state.annotate(f"{v:.1f}%", (v, y), xytext=(4, 0),
                          textcoords="offset points", va="center",
                          fontsize=8.5, color=INK, fontweight="semibold")
    ax_state.set_xlim(0, 108)
    ax_state.set_xlabel("% of all (unit, event) recycling slots")
    ax_state.set_title("(b) state at the boundary before", loc="left",
                       fontsize=8.5, color=INK_2)
    _grid(ax_state, axis="x")

    fig.tight_layout()
    return _save(fig, out, "fig4_c4_recurrence")


# -- Figure 5: the reproduction gate as a dose-response -----------------------


def fig_gate_dose_response(ex: Extracts, out: Path) -> Optional[Path]:
    """Plasticity loss and dead units both scale with step size, monotonically."""
    tasks, metrics = ex.table("tasks"), ex.table("metrics")
    if tasks.empty:
        return None
    base = ex.select(by="levels", arm="none", dataset="permuted_mnist")
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
    chosen = ex.select(arm=arm, lr=lr, dataset="permuted_mnist")
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


# -- driver -------------------------------------------------------------------


def build_all(
    extracts_root="runs/_extracts",
    survival_dirs: Optional[Sequence[str]] = None,
    out="figures",
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

    for fn, args in [
        (fig_tau_sweep, (ex, out)),
        (fig_c2_definitions, (ex, out)),
        (fig_gate_dose_response, (ex, out)),
        (fig_reference_asymmetry, (ex, out)),
    ]:
        path = fn(*args)
        print(f"  {'wrote' if path else 'skipped'} {fn.__name__}"
              + (f" -> {path.name}" if path else " (no data)"))
        if path:
            made.append(path)

    # Several survival directories may be given: the C4 mortality figure needs
    # runs with no intervention, the recurrence figure needs runs with one, and
    # no single sweep supplies both.
    for d in survival_dirs or []:
        surv = {p.stem: pd.read_parquet(p) for p in sorted(Path(d).glob("*.parquet"))}
        for fn, kwargs in ((fig_c4_survival, {"ex": ex}), (fig_c4_recurrence, {})):
            path = fn(surv, out, **kwargs)
            if path:
                print(f"  wrote {fn.__name__} -> {path.name}   [{Path(d).name}]")
                made.append(path)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the paper's figures.")
    ap.add_argument("--extracts", default="runs/_extracts")
    ap.add_argument("--survival", nargs="*", default=[],
                    help="directories written by `python -m src.analysis.survival --out`")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)
    build_all(args.extracts, args.survival, out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
