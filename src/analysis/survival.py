"""Per-neuron mortality dynamics -- claim C4, and the per-unit evidence for C1.

Post-hoc only. Never imported by training code (CLAUDE.md §4).

The per-neuron log (CLAUDE.md §5.4) is the only table in the project that
carries **identity across time**: `metrics.parquet` says 20% of layer 0 was dead
at task 150, but it cannot say whether that is the same 20% that was dead at
task 149. `recycling.parquet` says a recycling event took 141 units of which 36
were dead, but it does not say *which* 141, so it cannot say whether the same
units are being recycled over and over. Both questions need this file.

Four analyses live here, in increasing order of how much they depend on identity:

1. **Time to first death** -- Kaplan-Meier over task index, right-censored at the
   end of the run. Answers "when do units go silent", the literal C4 question.
2. **Transitions** -- alive->dead, dead->alive between consecutive boundaries.
   Answers "is death absorbing?", which the literature assumes without measuring.
   If units revive on their own, "dead neuron" is a state, not a fate, and a
   method that reinitialises them is not restoring anything that was lost.
3. **Death episodes** -- run-length encoding of the dead state per unit. Turns
   the transition rates into a distribution of how long silence lasts.
4. **Recycling recurrence** -- how often each individual unit is recycled, and
   what state it was in beforehand, against a null that keeps the per-task,
   per-layer count `k` fixed and reshuffles *which* units were chosen. This is
   the paper's title measured directly: if the same living units are recycled
   again and again, the intervention is not resurrection under any reading.

**A note on when the state is read.** `exact_zero_flag` at task `t` is probed at
the end of task `t`, i.e. *after* any recycling event inside that task -- and a
recycled unit has fresh incoming weights, so it is alive by construction. The
state a unit was in *when it was chosen* is therefore the boundary before,
`t - 1`, and every function here that asks "what was recycled" uses that. For
`t = 0` the answer comes from the `init` probe row (`task_idx == -1`), which is
why the extract keeps it.

Boundary resolution is a real limit and is not papered over: several recycling
events can fall inside one task (`recycling.freq = 1000` steps against ~469
steps per task), so `was_recycled_this_task` is an OR over the events in that
task, and a unit that died and revived *within* a task is invisible here. The
per-event, per-layer composition in `recycling.parquet` is the finer-grained
companion; it just cannot name units.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: What a panel needs. `exact_zero_flag_ref` is optional: it is present in both
#: the full per-neuron log and the C4 extract, but the reference-batch analysis
#: is skipped rather than failed if some future extract drops it.
PANEL_COLUMNS = (
    "task_idx",
    "layer_idx",
    "neuron_idx",
    "exact_zero_flag",
    "was_recycled_this_task",
    "mean_abs_act",
)
OPTIONAL_COLUMNS = ("exact_zero_flag_ref",)

#: The probe row written before task 0 (`probe_point == "init"`).
INIT_TASK_IDX = -1


@dataclass
class NeuronPanel:
    """One run's per-neuron log, reshaped to (n_boundaries, n_units) matrices.

    Everything downstream is vectorised over these, which keeps the analyses
    short enough to check by eye -- the alternative, groupby over 300k rows per
    run, hides indexing mistakes that would silently answer a different question.

    ``times`` is ascending and includes ``-1`` (the init probe) when the log has
    it. ``layer`` and ``neuron`` identify the columns; a unit is a *column*, and
    its identity is stable down the whole matrix, which is the entire point.
    """

    run_id: str
    times: np.ndarray  # (T,) int, ascending, -1 first if present
    layer: np.ndarray  # (U,) int
    neuron: np.ndarray  # (U,) int
    dead: np.ndarray  # (T, U) bool -- exact_zero_flag, current-task probe batch
    recycled: np.ndarray  # (T, U) bool
    mean_abs_act: np.ndarray  # (T, U) float64
    dead_ref: Optional[np.ndarray] = None  # (T, U) bool, fixed reference batch

    @property
    def n_units(self) -> int:
        return int(self.layer.size)

    @property
    def has_init(self) -> bool:
        return bool(self.times.size) and int(self.times[0]) == INIT_TASK_IDX

    @property
    def trained_times(self) -> np.ndarray:
        """Task indices excluding the init probe."""
        return self.times[self.times >= 0]

    def trained(self, matrix: np.ndarray) -> np.ndarray:
        """Rows of `matrix` for trained tasks only."""
        return matrix[self.times >= 0]

    def layer_mask(self, layer_idx: int) -> np.ndarray:
        return self.layer == layer_idx

    @property
    def layers(self) -> List[int]:
        return sorted(set(int(x) for x in self.layer))

    def dead_matrix(self, probe: str = "current") -> np.ndarray:
        """The death flag on one of the two probe batches (CLAUDE.md §5).

        ``"current"`` is the current task's distribution, ``"reference"`` the
        fixed batch that never changes across the run.

        This choice is not cosmetic and every recurrent-event result here has to
        be read against it. The task distribution changes at every boundary, so
        a unit that is silent on task t's inputs and firing on task t+1's may
        never have changed at all -- only the inputs did. A dead->alive
        transition on the *reference* batch cannot be explained that way, which
        is the whole reason a second probe batch is logged.
        """
        if probe == "current":
            return self.dead
        if probe == "reference":
            if self.dead_ref is None:
                raise ValueError(
                    f"{self.run_id}: no exact_zero_flag_ref in this log, so the "
                    "distribution-relative confound cannot be separated"
                )
            return self.dead_ref
        raise ValueError(f"unknown probe {probe!r}; use 'current' or 'reference'")


def panel_from_frame(df: pd.DataFrame, run_id: str = "") -> NeuronPanel:
    """Reshape one run's rows into a panel.

    The pivot is done with explicit index arithmetic rather than
    ``DataFrame.pivot``: the log is written in a known order but nothing
    guarantees it, an unsorted pivot would misalign a unit's history against
    itself, and that failure is invisible in the output.
    """
    missing = [c for c in PANEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"per-neuron log is missing {missing}; got {list(df.columns)}")
    if not run_id:
        run_id = str(df["run_id"].iloc[0]) if "run_id" in df.columns else ""

    times = np.unique(df["task_idx"].to_numpy())
    units = np.unique(
        np.stack([df["layer_idx"].to_numpy(), df["neuron_idx"].to_numpy()], axis=1),
        axis=0,
    )
    T, U = times.size, units.shape[0]
    if len(df) != T * U:
        raise ValueError(
            f"{run_id}: per-neuron log is ragged -- {len(df)} rows for "
            f"{T} boundaries x {U} units = {T * U}. A partial task would bias "
            "every survival curve here, so this is an error, not a warning."
        )

    row = np.searchsorted(times, df["task_idx"].to_numpy())
    # Units are (layer, neuron) pairs; encode to a single sortable key so the
    # column index is found the same way the rows are.
    width = int(units[:, 1].max()) + 1
    key = units[:, 0] * width + units[:, 1]
    col = np.searchsorted(key, df["layer_idx"].to_numpy() * width + df["neuron_idx"].to_numpy())

    # The row count matching T*U does not by itself prove every cell is filled
    # once -- a duplicated (task, unit) row plus a missing one would pass it and
    # leave uninitialised memory in the matrix. Check the cells, not the total.
    if np.unique(row.astype(np.int64) * U + col).size != T * U:
        raise ValueError(
            f"{run_id}: per-neuron log has duplicate or missing (task, unit) cells"
        )

    def grid(column: str, dtype) -> np.ndarray:
        out = np.empty((T, U), dtype=dtype)
        out[row, col] = df[column].to_numpy()
        return out

    dead_ref = (
        grid("exact_zero_flag_ref", bool) if "exact_zero_flag_ref" in df.columns else None
    )
    return NeuronPanel(
        run_id=run_id,
        times=times,
        layer=units[:, 0],
        neuron=units[:, 1],
        dead=grid("exact_zero_flag", bool),
        recycled=grid("was_recycled_this_task", bool),
        mean_abs_act=grid("mean_abs_act", np.float64),
        dead_ref=dead_ref,
    )


def iter_panels(
    source, run_ids: Optional[Sequence[str]] = None
) -> Iterator[NeuronPanel]:
    """Yield one panel per run, from a concatenated C4 extract or a runs/ tree.

    Streams run by run. The tau-sweep extract is 18 million rows; materialising
    it whole would work on this machine and stop working on the next one, and
    every analysis here is per-run anyway.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    source = Path(source)
    if source.is_dir() and not (source / "neurons_c4.parquet").exists():
        # A runs/ tree: one neurons.parquet per run directory.
        for run_dir in sorted(p for p in source.iterdir() if p.is_dir()):
            path = run_dir / "neurons.parquet"
            if not path.exists():
                continue
            if run_ids is not None and run_dir.name not in run_ids:
                continue
            df = pd.read_parquet(path)
            if "probe_point" in df.columns:
                # The full log tags rows; the C4 extract encodes init as -1.
                df = df.drop(columns=["probe_point"])
            yield panel_from_frame(df, run_dir.name)
        return

    path = source / "neurons_c4.parquet" if source.is_dir() else source
    dataset = pads.dataset(path)
    columns = [c for c in PANEL_COLUMNS] + [
        c for c in OPTIONAL_COLUMNS if c in dataset.schema.names
    ]
    all_ids = (
        pc.unique(dataset.to_table(columns=["run_id"]).column("run_id"))
        .to_pylist()
    )
    for rid in sorted(all_ids):
        if run_ids is not None and rid not in run_ids:
            continue
        table = dataset.to_table(columns=columns, filter=pc.field("run_id") == rid)
        yield panel_from_frame(table.to_pandas(), rid)


# -- 1. time to first death ---------------------------------------------------


def first_death(panel: NeuronPanel, probe: str = "current") -> pd.DataFrame:
    """Per unit: the task it first went silent, or right-censoring at the end.

    Units already dead at the init probe are reported with ``prevalent=True`` and
    excluded from the Kaplan-Meier risk set: they were never observed alive, so
    they have no time-to-event, and folding them in as "died at task 0" would
    invent an event that training did not cause.
    """
    flag = panel.dead_matrix(probe)
    dead = panel.trained(flag)
    times = panel.trained_times
    ever = dead.any(axis=0)
    first_row = np.argmax(dead, axis=0)  # 0 where never dead; masked by `ever`
    horizon = int(times[-1])

    prevalent = flag[0] if panel.has_init else np.zeros(panel.n_units, dtype=bool)
    return pd.DataFrame(
        {
            "run_id": panel.run_id,
            "layer_idx": panel.layer,
            "neuron_idx": panel.neuron,
            "probe": probe,
            "prevalent": prevalent,
            "event": ever & ~prevalent,
            # Task index of first death, or the horizon for a censored unit.
            "death_task": np.where(ever, times[first_row], horizon),
            "horizon": horizon,
        }
    )


def kaplan_meier(
    death_task: np.ndarray, event: np.ndarray, horizon: int
) -> pd.DataFrame:
    """Product-limit survival on the discrete task grid, times 0..horizon.

    With complete runs the only censoring is administrative and lands at the
    horizon, so this equals the empirical survival function. The product-limit
    form is kept anyway: a run that stopped early would otherwise contribute its
    survivors as if they had been followed to the end, which biases the curve
    upward exactly where the interesting part is.
    """
    death_task = np.asarray(death_task)
    event = np.asarray(event, dtype=bool)
    n = death_task.size
    grid = np.arange(0, horizon + 1)
    surv = np.ones(grid.size, dtype=np.float64)
    at_risk = np.zeros(grid.size, dtype=np.int64)
    events = np.zeros(grid.size, dtype=np.int64)

    n_at_risk = n
    s = 1.0
    for i, t in enumerate(grid):
        at_risk[i] = n_at_risk
        d = int(np.count_nonzero((death_task == t) & event))
        c = int(np.count_nonzero((death_task == t) & ~event))
        events[i] = d
        if n_at_risk > 0 and d > 0:
            s *= 1.0 - d / n_at_risk
        surv[i] = s
        n_at_risk -= d + c

    return pd.DataFrame(
        {"task_idx": grid, "at_risk": at_risk, "n_events": events, "survival": surv}
    )


def survival_matrix(
    panels: Sequence[NeuronPanel],
    layer_idx: Optional[int] = None,
    probe: str = "current",
) -> Tuple[np.ndarray, np.ndarray]:
    """(n_runs, n_tasks) survival curves, one row per run.

    Shaped for ``stats.stratified_bootstrap``, which is the project's only
    sanctioned way to put an interval on a cross-seed quantity (CLAUDE.md §7).
    """
    rows, grid = [], None
    for p in panels:
        fd = first_death(p, probe)
        if layer_idx is not None:
            fd = fd[fd.layer_idx == layer_idx]
        alive_at_entry = fd[~fd.prevalent]
        km = kaplan_meier(
            alive_at_entry.death_task.to_numpy(),
            alive_at_entry.event.to_numpy(),
            int(fd.horizon.iloc[0]),
        )
        if grid is None:
            grid = km.task_idx.to_numpy()
        rows.append(km.survival.to_numpy())
    return np.vstack(rows), grid


# -- 2. transitions: is death absorbing? --------------------------------------


def transitions(
    panel: NeuronPanel, per_layer: bool = True, probe: str = "current"
) -> pd.DataFrame:
    """Counts of the four state changes between consecutive task boundaries.

    ``recovered`` is the number of dead->alive transitions. The literature's
    framing -- that dead units are lost capacity a method must restore -- is
    only coherent if this is near zero without any intervention. Measuring it is
    the point.

    Transitions into a task where the unit was recycled are counted separately
    (``*_after_recycle``): a unit that comes back because ReDo reinitialised it
    is not evidence of spontaneous recovery, and pooling the two would
    manufacture the result.

    Read ``p_recover`` on both probes before believing it. On the current-task
    batch a recovery may be the permutation changing rather than the unit; on
    the reference batch it cannot be. See ``NeuronPanel.dead_matrix``.
    """
    dead = panel.trained(panel.dead_matrix(probe))
    rec = panel.trained(panel.recycled)
    prev, curr = dead[:-1], dead[1:]
    # Recycling in the *destination* task is what could have caused the change.
    touched = rec[1:]

    frames = []
    groups = panel.layers if per_layer else [None]
    for lyr in groups:
        m = np.ones(panel.n_units, bool) if lyr is None else panel.layer_mask(lyr)
        p, c, tch = prev[:, m], curr[:, m], touched[:, m]
        untouched = ~tch
        frames.append(
            {
                "run_id": panel.run_id,
                "probe": probe,
                "layer_idx": -1 if lyr is None else lyr,
                "n_alive_dead": int(np.count_nonzero(~p & c & untouched)),
                "n_alive_alive": int(np.count_nonzero(~p & ~c & untouched)),
                "n_dead_alive": int(np.count_nonzero(p & ~c & untouched)),
                "n_dead_dead": int(np.count_nonzero(p & c & untouched)),
                "n_dead_alive_after_recycle": int(np.count_nonzero(p & ~c & tch)),
                "n_dead_dead_after_recycle": int(np.count_nonzero(p & c & tch)),
                "n_recycled_transitions": int(np.count_nonzero(tch)),
            }
        )
    out = pd.DataFrame(frames)
    dead_at_risk = out.n_dead_alive + out.n_dead_dead
    alive_at_risk = out.n_alive_dead + out.n_alive_alive
    # NaN, not 0, where nothing was at risk: an empty risk set has no rate.
    out["p_recover"] = np.where(
        dead_at_risk > 0, out.n_dead_alive / dead_at_risk.where(dead_at_risk > 0, 1), np.nan
    )
    out["p_die"] = np.where(
        alive_at_risk > 0, out.n_alive_dead / alive_at_risk.where(alive_at_risk > 0, 1), np.nan
    )
    return out


# -- 3. death episodes --------------------------------------------------------


def death_episodes(panel: NeuronPanel, probe: str = "current") -> pd.DataFrame:
    """One row per maximal run of consecutive boundaries a unit was dead for.

    ``censored`` marks an episode still open at the last task -- those are the
    permanent deaths, and averaging their length in with the closed ones would
    understate how long silence lasts.
    """
    dead = panel.trained(panel.dead_matrix(probe))
    rec = panel.trained(panel.recycled)
    times = panel.trained_times
    T = dead.shape[0]

    # Pad with False so every episode has a start edge and an end edge.
    padded = np.vstack([np.zeros((1, panel.n_units), bool), dead, np.zeros((1, panel.n_units), bool)])
    diff = padded[1:].astype(np.int8) - padded[:-1].astype(np.int8)
    start_row, start_col = np.nonzero(diff == 1)
    end_row, end_col = np.nonzero(diff == -1)
    order_s = np.lexsort((start_row, start_col))
    order_e = np.lexsort((end_row, end_col))
    start_row, start_col = start_row[order_s], start_col[order_s]
    end_row = end_row[order_e]

    # diff == -1 at row e means the last dead boundary was e - 1, so the episode
    # occupies rows [start_row, end_row - 1] and its length is end_row - start_row.
    length = end_row - start_row
    censored = end_row >= T
    # Was the unit recycled during the episode, i.e. did the intervention end it?
    # Prefix sums rather than a slice per episode: there are tens of thousands of
    # episodes per run and the loop was the whole cost of this function.
    cum = np.vstack([np.zeros((1, panel.n_units), np.int64), np.cumsum(rec, axis=0)])
    ended_by_recycle = (
        cum[end_row, start_col] - cum[start_row, start_col]
    ) > 0

    return pd.DataFrame(
        {
            "run_id": panel.run_id,
            "probe": probe,
            "layer_idx": panel.layer[start_col],
            "neuron_idx": panel.neuron[start_col],
            "start_task": times[start_row],
            "length_tasks": length,
            "censored": censored,
            "recycled_during": ended_by_recycle,
        }
    )


# -- 4. recycling recurrence: the paper's title -------------------------------


def recycling_recurrence(panel: NeuronPanel, probe: str = "current") -> pd.DataFrame:
    """Per unit: how often it was recycled, and how often it was ever dead.

    ``n_recycled_while_alive`` uses the state at the boundary *before* the task
    the recycling happened in, because the boundary probe of that task is taken
    after the reinitialisation (see the module docstring). ``ever_dead`` spans
    every boundary in the run, so a unit with ``n_recycled > 0`` and
    ``ever_dead == False`` was chosen by the intervention despite never once
    being silent anywhere in 200 tasks.
    """
    dead_all = panel.dead_matrix(probe)
    rec_trained = panel.trained(panel.recycled)
    # State one boundary earlier. With the init probe present this is defined for
    # every trained task; without it, task 0's events have no prior state and are
    # dropped from the "while alive" counts rather than guessed at.
    if panel.has_init:
        prior = dead_all[:-1]
        rec = rec_trained
    else:
        prior = dead_all[:-1]
        rec = rec_trained[1:]

    was_dead_when_chosen = rec & prior
    was_alive_when_chosen = rec & ~prior

    return pd.DataFrame(
        {
            "run_id": panel.run_id,
            "layer_idx": panel.layer,
            "neuron_idx": panel.neuron,
            "n_recycled": rec_trained.sum(axis=0).astype(np.int64),
            "n_recycled_while_dead": was_dead_when_chosen.sum(axis=0).astype(np.int64),
            "n_recycled_while_alive": was_alive_when_chosen.sum(axis=0).astype(np.int64),
            "n_dead_boundaries": panel.trained(panel.dead).sum(axis=0).astype(np.int64),
            "ever_dead": panel.trained(panel.dead).any(axis=0),
            "mean_abs_act": panel.trained(panel.mean_abs_act).mean(axis=0),
        }
    )


def recurrence_null(
    panel: NeuronPanel, n_draws: int = 200, seed: int = 0
) -> pd.DataFrame:
    """The same per-unit recycle counts under "same k, different units".

    At every task and layer the observed number of recycled units is held fixed
    and the *identity* is redrawn uniformly. That isolates targeting persistence
    from dose: a method that recycles 141 units per event will produce repeat
    hits by chance alone, and the question is whether it produces more than
    chance.

    Returns one row per draw with the concentration statistics, so the observed
    value can be read against a distribution rather than a point.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(n_draws):
        rows.append({"draw": draw, **concentration(null_counts(panel, rng))})
    return pd.DataFrame(rows)


def null_counts(panel: NeuronPanel, rng: np.random.Generator) -> np.ndarray:
    """One draw of per-unit recycle counts with `k` held fixed per task and layer."""
    rec = panel.trained(panel.recycled)
    counts = np.zeros(panel.n_units, dtype=np.int64)
    for lyr in panel.layers:
        mask = panel.layer_mask(lyr)
        idx = np.flatnonzero(mask)
        per_task = rec[:, mask].sum(axis=1)
        for k in per_task:
            if k:
                counts[rng.choice(idx, size=int(k), replace=False)] += 1
    return counts


def concentration(counts: np.ndarray) -> Dict[str, float]:
    """How unevenly the recycling slots were spread over units.

    ``gini`` of the per-unit count, ``top_decile_share`` of all slots taken by
    the busiest 10% of units, and ``frac_never`` -- the share of the network the
    intervention never touched at all, which is the complement of "the same
    units over and over" and the easiest number to state in a sentence.
    """
    c = np.sort(np.asarray(counts, dtype=np.float64))
    n = c.size
    total = c.sum()
    if n == 0 or total == 0:
        return {
            "gini": float("nan"),
            "top_decile_share": float("nan"),
            "frac_never": 1.0 if n else float("nan"),
            "max_count": 0.0,
            "mean_count": 0.0,
        }
    index = np.arange(1, n + 1)
    gini = float((2.0 * (index * c).sum()) / (n * total) - (n + 1.0) / n)
    top = max(1, int(round(0.1 * n)))
    return {
        "gini": gini,
        "top_decile_share": float(c[-top:].sum() / total),
        "frac_never": float(np.count_nonzero(c == 0) / n),
        "max_count": float(c[-1]),
        "mean_count": float(total / n),
    }


def recycling_persistence(panel: NeuronPanel) -> pd.DataFrame:
    """P(recycled at t+1 | recycled at t) against the marginal rate.

    The enrichment ratio is the sharpest single number for "the same units get
    picked again": 1.0 means the last choice says nothing about the next one.

    Taken over consecutive *active* tasks, not consecutive tasks. Recycling
    fires every ``recycling.freq`` optimizer steps (1000) against ~469 steps per
    task, so events land in roughly every other task and adjacent-task pairs are
    almost never both active -- pairing on the raw task index silently returns
    nothing. The quantity of interest is "recycled at the last event, recycled
    at the next", which is what the active subsequence gives.
    """
    rec = panel.trained(panel.recycled)
    rows = []
    for lyr in panel.layers:
        m = panel.layer_mask(lyr)
        r = rec[:, m]
        r = r[r.any(axis=1)]  # drop tasks with no event
        if r.shape[0] < 2:
            continue
        prev, curr = r[:-1], r[1:]
        n_prev = int(prev.sum())
        both = int((prev & curr).sum())
        marginal = float(curr.mean())
        cond = both / n_prev if n_prev else float("nan")
        rows.append(
            {
                "run_id": panel.run_id,
                "layer_idx": lyr,
                "n_event_pairs": int(prev.shape[0]),
                "p_recycled_marginal": marginal,
                "p_recycled_given_prev": cond,
                "enrichment": cond / marginal if marginal > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# -- assembly -----------------------------------------------------------------


def analyse(
    source,
    run_ids: Optional[Sequence[str]] = None,
    null_draws: int = 0,
    null_seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Every table above, concatenated over the runs in `source`."""
    out: Dict[str, List[pd.DataFrame]] = {
        "first_death": [],
        "transitions": [],
        "episodes": [],
        "recurrence": [],
        "persistence": [],
        "null": [],
    }
    panels: List[NeuronPanel] = []
    for panel in iter_panels(source, run_ids):
        panels.append(panel)
        out["first_death"].append(first_death(panel))
        out["transitions"].append(transitions(panel))
        out["episodes"].append(death_episodes(panel))
        out["recurrence"].append(recycling_recurrence(panel))
        out["persistence"].append(recycling_persistence(panel))
        if null_draws and panel.recycled.any():
            n = recurrence_null(panel, null_draws, null_seed)
            n.insert(0, "run_id", panel.run_id)
            out["null"].append(n)

    tables = {
        k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
        for k, v in out.items()
    }
    surv, grid = survival_matrix(panels)
    tables["survival_matrix"] = pd.DataFrame(surv, columns=grid, index=[p.run_id for p in panels])
    return tables


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C4 survival analysis (post-hoc).")
    ap.add_argument("source", help="neurons_c4.parquet, its directory, or a runs/ tree")
    ap.add_argument("--out", default=None, help="directory to write the tables to")
    ap.add_argument("--null-draws", type=int, default=0)
    ap.add_argument("--null-seed", type=int, default=0)
    args = ap.parse_args(argv)

    tables = analyse(args.source, null_draws=args.null_draws, null_seed=args.null_seed)
    for name, df in tables.items():
        print(f"{name:<16s} {len(df):>9,d} rows")
    if args.out:
        dest = Path(args.out)
        dest.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_parquet(dest / f"{name}.parquet")
        print(f"wrote {len(tables)} tables to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
