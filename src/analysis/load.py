"""Reading runs back off disk.

Post-hoc only. Recomputes nothing: every column here was written by
``src.probes`` at training time. If an analysis needs a quantity that is not in
the parquet files, the fix is to add it to ``probes.py`` and re-run -- not to
re-derive it here, where it would silently drift from the logged version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

TABLES = ("tasks", "metrics", "neurons", "recycling")


@dataclass
class Run:
    run_id: str
    path: Path
    config: dict
    summary: dict

    @property
    def seed(self) -> int:
        return int(self.config["seed"])

    @property
    def complete(self) -> bool:
        return self.summary.get("status") == "complete"

    def table(self, name: str) -> pd.DataFrame:
        if name not in TABLES:
            raise ValueError(f"unknown table {name!r}; known: {TABLES}")
        path = self.path / f"{name}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)


def load_run(run_dir) -> Run:
    run_dir = Path(run_dir)
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    summary_path = run_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    return Run(config["run_id"], run_dir, config, summary)


def load_runs(runs_root="runs", pattern: str = "*", require_complete: bool = True) -> List[Run]:
    """All runs under `runs_root` whose run_id matches `pattern`.

    Incomplete runs are dropped by default and *reported*, never silently
    skipped -- a missing seed changes a bootstrap CI.
    """
    root = Path(runs_root)
    runs, incomplete = [], []
    for d in sorted(root.glob(pattern)):
        if not (d / "config.json").exists():
            continue
        run = load_run(d)
        (runs if run.complete else incomplete).append(run)
    if incomplete and require_complete:
        print(
            f"warning: skipping {len(incomplete)} incomplete run(s): "
            + ", ".join(r.run_id for r in incomplete)
        )
        return runs
    return runs + incomplete


def task_curve(run: Run, column: str = "online_accuracy") -> pd.Series:
    """One run's per-task series, indexed by task_idx, excluding the init probe."""
    df = run.table("tasks")
    df = df[df["probe_point"] == "task_end"]
    return df.set_index("task_idx")[column].sort_index()


def score_matrix(
    runs: Sequence[Run],
    window: Sequence[int],
    column: str = "online_accuracy",
) -> np.ndarray:
    """(n_runs, n_tasks) matrix for the bootstrap, one row per seed.

    ``window`` is a list of **0-based** task indices. The frozen analysis plan
    states its windows in 1-based task numbers and carries the 0-based
    equivalents alongside; use those, and never convert by hand here.
    """
    window = list(window)
    rows = []
    for run in runs:
        curve = task_curve(run, column)
        missing = [t for t in window if t not in curve.index]
        if missing:
            raise ValueError(
                f"{run.run_id} is missing tasks {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} from the analysis window"
            )
        rows.append(curve.loc[window].to_numpy(dtype=np.float64))
    if not rows:
        raise ValueError("no runs supplied")
    return np.vstack(rows)


def recycling_composition(runs: Iterable[Run]) -> pd.DataFrame:
    """Every recycling event across runs, with the dead fraction of each
    recycled set -- the C1 headline quantity.

    ``dead_fraction`` is ``n_dead_exact / k``; NaN where nothing was recycled,
    which is meaningfully different from 0 and must not be filled.
    """
    frames = []
    for run in runs:
        df = run.table("recycling")
        if df.empty:
            continue
        df = df.copy()
        df["seed"] = run.seed
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["dead_fraction"] = np.where(out["k"] > 0, out["n_dead_exact"] / out["k"], np.nan)
    out["alive_fraction"] = np.where(
        out["k"] > 0, out["n_alive_but_quiet"] / out["k"], np.nan
    )
    return out


def death_metric_columns(metrics: pd.DataFrame) -> Dict[str, List[str]]:
    """Group metrics.parquet columns by death definition, for the C2 figure."""
    cols = list(metrics.columns)
    return {
        "dead_exact": [c for c in cols if c.startswith("dead_exact_frac")],
        "dormant_tau": [c for c in cols if c.startswith("dormant_frac_tau_")],
        "dead_absolute": [c for c in cols if c.startswith("dead_abs_frac_")],
        "saturated": [c for c in cols if c.startswith("saturated_frac")],
    }
