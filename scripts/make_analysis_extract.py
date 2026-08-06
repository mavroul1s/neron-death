"""Build a small analysis extract from a runs/ tree.

Answers the practical question "must I download the whole output every time?"
No. Per 200-task run the tables weigh:

    neurons.parquet    27.0 MB      per-neuron time series (C4)
    metrics.parquet     0.09 MB     per task x layer -- C2, C3, effective rank
    tasks.parquet       0.02 MB     per task -- the outcome measure, the gate
    recycling.parquet   ~0.05 MB    per event x layer -- the C1 composition table

So `neurons.parquet` is 99.6% of the bytes and none of the headline figures.
This script concatenates everything *except* the per-neuron log into one file
per table, adding a `run_id` column, giving a few MB that covers the gate, C1,
C2, C3 and C5 analyses.

    python scripts/make_analysis_extract.py --runs-root runs --out dist/extract

**The per-neuron log is not optional and is not deleted by this.** It is the C4
dataset, it cannot be reconstructed retrospectively (CLAUDE.md §5.4), and it
must stay in the archival Kaggle Dataset. This only changes what you *download*,
never what you *keep*.

Intended use on Kaggle: run this at the end of a session and download the
extract, while `runs.zip` goes to the versioned Dataset and stays there.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

#: Everything except the per-neuron log. Adding `neurons` here would defeat the
#: entire point of the script.
EXTRACT_TABLES = ("tasks", "metrics", "recycling")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="dist/extract")
    ap.add_argument("--pattern", default="*", help="glob over run_ids")
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip")
    args = ap.parse_args(argv)

    root = Path(args.runs_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(d for d in root.glob(args.pattern) if (d / "config.json").exists())
    if not run_dirs:
        raise SystemExit(f"no runs matching {args.pattern!r} under {root}")

    summaries, skipped = [], []
    for name in EXTRACT_TABLES:
        tables = []
        for d in run_dirs:
            path = d / f"{name}.parquet"
            if not path.exists():
                continue  # e.g. recycling.parquet is absent for no-intervention runs
            t = pq.read_table(path)
            if "run_id" not in t.column_names:
                t = t.append_column(
                    "run_id", pa.array([d.name] * t.num_rows, pa.string())
                )
            tables.append(t)
        if not tables:
            skipped.append(name)
            continue
        combined = pa.concat_tables(tables, promote_options="permissive")
        dest = out / f"{name}.parquet"
        pq.write_table(combined, dest, compression="zstd")
        print(f"  {name:12s} {combined.num_rows:>9,d} rows  {dest.stat().st_size/1e6:6.2f} MB")

    # Configs and summaries are tiny and make the extract self-describing: the
    # analysis can recover every hyperparameter without the full runs tree.
    for d in run_dirs:
        rec = {"run_id": d.name, "config": json.loads((d / "config.json").read_text())}
        s = d / "summary.json"
        if s.exists():
            rec["summary"] = json.loads(s.read_text())
        summaries.append(rec)
    (out / "runs.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"{len(run_dirs)} runs -> {out}  ({total/1e6:.2f} MB)")
    if skipped:
        print(f"  (no data for: {', '.join(skipped)})")

    if args.zip:
        z = Path(str(out) + ".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.iterdir()):
                zf.write(p, p.name)
        print(f"wrote {z} ({z.stat().st_size/1e6:.2f} MB)")

    print(
        "\nneurons.parquet is deliberately excluded: it is 99.6% of the bytes and\n"
        "none of the headline figures. Keep it in the archival Kaggle Dataset --\n"
        "it is the C4 dataset and cannot be rebuilt after the fact."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
