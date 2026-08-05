"""Crash-tolerant parquet logging.

Kaggle cuts the session at 12 hours, abruptly and without warning (CLAUDE.md
§3). A single parquet file written at the end of a run is therefore the wrong
shape: a killed run leaves a zero-byte file and the per-neuron dataset -- which
is explicitly *not recoverable retrospectively* -- is gone.

So every log is written as a sequence of shards, flushed at each checkpoint.
A killed run leaves every shard up to its last checkpoint fully readable, and
``finalize()`` concatenates them into the single ``<name>.parquet`` that
CLAUDE.md §4 names. On resume, ``rewind_to()`` drops shards that ran ahead of
the checkpoint so no task is logged twice.

Ordering rule the caller must respect: **flush shards, then write the
checkpoint.** A shard beyond the checkpoint is redundant and gets dropped on
resume; a checkpoint beyond the shards would silently lose a task's rows.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .probes import NEURON_COLUMNS

SHARD_DIRNAME = "_shards"
COMPRESSION = "zstd"


def _neuron_schema() -> pa.Schema:
    """Explicit schema for neurons.parquet.

    Explicit rather than inferred because two columns are nullable
    (``saturated_flag`` is None for unbounded activations, and
    ``exact_zero_flag_ref`` is None if no reference batch was probed). Type
    inference on an all-None first shard would produce a null-typed column and
    then fail to concatenate with later shards.
    """
    f = {
        "run_id": pa.string(),
        "task_idx": pa.int32(),
        "layer_idx": pa.int32(),
        "neuron_idx": pa.int32(),
        "exact_zero_flag": pa.bool_(),
        "was_recycled_this_task": pa.bool_(),
        "exact_zero_flag_ref": pa.bool_(),
        "saturated_flag": pa.bool_(),
        "probe_point": pa.string(),
    }
    return pa.schema([pa.field(c, f.get(c, pa.float64())) for c in NEURON_COLUMNS])


NEURON_SCHEMA = _neuron_schema()


class ShardedParquetLog:
    """Append-only columnar log, flushed to numbered shards.

    ``add_rows`` takes a list of dicts (one row each); ``add_columns`` takes a
    dict of equal-length arrays. Both buffer in memory until ``flush()``.
    """

    def __init__(
        self,
        run_dir,
        name: str,
        schema: Optional[pa.Schema] = None,
        type_overrides: Optional[Dict[str, pa.DataType]] = None,
    ):
        self.run_dir = Path(run_dir)
        self.name = name
        self.shard_dir = self.run_dir / SHARD_DIRNAME
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._type_overrides = dict(type_overrides or {})
        self._buffer: List[pa.Table] = []
        self._buffered_rows = 0
        self._min_task: Optional[int] = None
        self._max_task: Optional[int] = None

    # -- ingestion ------------------------------------------------------------

    def _note_tasks(self, tasks: Sequence[int]) -> None:
        if not len(tasks):
            return
        lo, hi = int(np.min(tasks)), int(np.max(tasks))
        self._min_task = lo if self._min_task is None else min(self._min_task, lo)
        self._max_task = hi if self._max_task is None else max(self._max_task, hi)

    def _to_table(self, data, columnar: bool) -> pa.Table:
        if self._schema is not None:
            return (
                pa.Table.from_pydict(data, schema=self._schema)
                if columnar
                else pa.Table.from_pylist(data, schema=self._schema)
            )
        table = (
            pa.Table.from_pydict(data) if columnar else pa.Table.from_pylist(data)
        )
        if self._type_overrides:
            fields = [
                pa.field(f.name, self._type_overrides.get(f.name, f.type))
                for f in table.schema
            ]
            table = table.cast(pa.schema(fields))
        # Lock the schema in from the first batch so later shards must match.
        self._schema = table.schema
        return table

    def add_rows(self, rows: List[dict]) -> None:
        if not rows:
            return
        self._note_tasks([r["task_idx"] for r in rows if "task_idx" in r])
        self._buffer.append(self._to_table(rows, columnar=False))
        self._buffered_rows += len(rows)

    def add_columns(self, cols: Dict[str, np.ndarray]) -> None:
        n = len(next(iter(cols.values())))
        if n == 0:
            return
        if "task_idx" in cols:
            self._note_tasks(np.asarray(cols["task_idx"]))
        self._buffer.append(self._to_table(cols, columnar=True))
        self._buffered_rows += n

    # -- shards ---------------------------------------------------------------

    def _shard_path(self, lo: int, hi: int) -> Path:
        # Signed-friendly formatting: the init probe uses task_idx = -1.
        return self.shard_dir / f"{self.name}.part_{lo:+06d}_{hi:+06d}.parquet"

    def flush(self) -> Optional[Path]:
        """Write the buffer as one shard covering its task range."""
        if not self._buffer:
            return None
        lo = self._min_task if self._min_task is not None else 0
        hi = self._max_task if self._max_task is not None else 0
        table = pa.concat_tables(self._buffer)
        path = self._shard_path(lo, hi)
        pq.write_table(table, path, compression=COMPRESSION)
        self._buffer.clear()
        self._buffered_rows = 0
        self._min_task = self._max_task = None
        return path

    @staticmethod
    def _shard_range(path: Path) -> tuple:
        lo, hi = path.stem.split(".part_")[1].split("_")
        return int(lo), int(hi)

    def _shards(self) -> List[Path]:
        # Sorted by parsed task range, not by filename: the init probe's shard
        # starts at task -1, and "-00001" sorts *after* "+00000" as a string.
        return sorted(
            self.shard_dir.glob(f"{self.name}.part_*.parquet"), key=self._shard_range
        )

    def rewind_to(self, last_task: int) -> List[Path]:
        """Delete shards containing tasks after `last_task` (resume path)."""
        dropped = []
        for p in self._shards():
            _, hi = self._shard_range(p)
            if hi > last_task:
                p.unlink()
                dropped.append(p)
        self._buffer.clear()
        self._buffered_rows = 0
        self._min_task = self._max_task = None
        return dropped

    def finalize(self, keep_shards: bool = False) -> Optional[Path]:
        """Concatenate shards into ``<run_dir>/<name>.parquet``."""
        self.flush()
        shards = self._shards()
        if not shards:
            return None
        table = pa.concat_tables([pq.read_table(p) for p in shards])
        out = self.run_dir / f"{self.name}.parquet"
        pq.write_table(table, out, compression=COMPRESSION)
        if not keep_shards:
            for p in shards:
                p.unlink()
        return out

    @property
    def n_buffered(self) -> int:
        return self._buffered_rows


def cleanup_shard_dir(run_dir) -> None:
    d = Path(run_dir) / SHARD_DIRNAME
    if d.exists() and not any(d.iterdir()):
        shutil.rmtree(d)
