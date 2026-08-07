"""Write a CPU-only Kaggle notebook that rebuilds `c4.zip` from an archival run tree.

The C4 per-neuron log is written by every run and archived inside `runs.zip`
(~27 MB/run), but the small `c4.zip` is only produced if the session that
trained the runs asked for it. Two sweeps -- `none`+`redo` and `random_matched`
-- were run before that was routine, so their per-neuron logs exist but have
never been extracted.

**Nothing needs re-training.** `neurons.parquet` is already in the archival
Dataset; this notebook attaches it, runs `make_analysis_extract.py --with-c4`,
and emits the few-MB file. It requests **no accelerator**, so it does not touch
the weekly GPU budget at all.

    python scripts/make_c4_notebook.py --out kaggle_upload

Then on Kaggle: New Notebook -> import the .ipynb -> attach the code Dataset AND
the archival runs Dataset -> Accelerator **None** -> Run All -> download c4.zip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE = r'''
# C4 extraction from an archival run tree. No accelerator required.
#
# Attach two Datasets:
#   1. the neuron-death code Dataset (the one every other notebook uses)
#   2. whichever versioned Dataset holds the archived runs/ tree for the sweep
#      you want -- the one `runs.zip` was pushed to.
#
# Both are found by searching /kaggle/input, so their names and mount paths do
# not matter.

import glob, os, subprocess, sys, zipfile

hits = sorted(glob.glob("/kaggle/input/**/src/train.py", recursive=True))
assert hits, "code Dataset not attached (no src/train.py under /kaggle/input)"
REPO = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, REPO)
print("code:", REPO)

# An archived run directory is one holding neurons.parquet. Zipped archives are
# unpacked first: `runs.zip` is normally attached as a Dataset that Kaggle has
# already expanded, but a hand-uploaded zip shows up unexpanded.
for z in sorted(glob.glob("/kaggle/input/**/runs*.zip", recursive=True)):
    dest = "/kaggle/working/unpacked/" + os.path.splitext(os.path.basename(z))[0]
    if not os.path.isdir(dest):
        print("unpacking", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)

roots = set()
for p in glob.glob("/kaggle/input/**/neurons.parquet", recursive=True):
    roots.add(os.path.dirname(os.path.dirname(p)))
for p in glob.glob("/kaggle/working/unpacked/**/neurons.parquet", recursive=True):
    roots.add(os.path.dirname(os.path.dirname(p)))
assert roots, (
    "no neurons.parquet found. Attach the Dataset that runs.zip was pushed to; "
    "extract.zip does NOT contain the per-neuron log."
)

for root in sorted(roots):
    runs = sorted(
        d for d in glob.glob(os.path.join(root, "*"))
        if os.path.exists(os.path.join(d, "neurons.parquet"))
    )
    print(f"\n{root}: {len(runs)} run(s) with a per-neuron log")
    for d in runs[:3]:
        print("   ", os.path.basename(d))
    if len(runs) > 3:
        print("    ...")

    name = os.path.basename(root.rstrip("/")) or "archive"
    subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "make_analysis_extract.py"),
         "--runs-root", root, "--out", f"/kaggle/working/c4_{name}",
         "--with-c4", "--zip"],
        check=True, cwd=REPO,
    )

print("\nOutputs (download the c4_*.zip files):")
for f in sorted(glob.glob("/kaggle/working/*.zip")):
    print(f"  {os.path.basename(f):40s} {os.path.getsize(f)/1e6:8.1f} MB")
'''

MARKDOWN = """# Rebuild `c4.zip` from archived runs — **no GPU**

The C4 survival analysis needs `neurons.parquet`, the per-neuron time series.
It is written by every run and lives in the archival `runs.zip`, but the small
`c4.zip` was only emitted by sessions that asked for it — the `none`+`redo` and
`random_matched` sweeps predate that.

This notebook re-derives it. It **retrains nothing** and needs **no accelerator**:
set Accelerator to *None* so it costs zero GPU-hours.

**Attach:** the code Dataset, plus the versioned Dataset holding the archived
`runs/` tree. `extract.zip` will not work — it deliberately excludes the
per-neuron log, which is 99.6% of the bytes.

**Download:** the `c4_*.zip` files this produces.
"""


def notebook() -> dict:
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": MARKDOWN.splitlines(True)},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": CODE.strip().splitlines(True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            # No accelerator: this is pure I/O and a parquet rewrite.
            "kaggle": {"accelerator": "none", "dataSources": [], "isInternetEnabled": False},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="kaggle_upload")
    args = ap.parse_args(argv)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "11_c4_from_archive.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook(), f, indent=1)
    print(f"wrote {path}")
    print("Attach the code Dataset + the archival runs Dataset, Accelerator=None, Run All.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
