"""Build a clean zip of the repository for upload as a Kaggle code Dataset.

Uploading the working tree raw fails twice over: Kaggle rejects `__pycache__`
directories under its reserved-name rule ("uses a reserved naming pattern:
__name__"), and the cached MNIST idx files push the tree past the 1000-file
limit. Both were hit on 2026-08-05.

    python scripts/package_for_kaggle.py

Writes `dist/neuron-death-code.zip` containing only what a run needs, with the
repository contents at the **root** of the archive so the notebook's dataset
discovery finds `src/train.py` one level down rather than three.

Excluded: `__pycache__`, `.git`, caches, `runs/` (results live in their own
versioned Dataset), `data/` (uploaded separately -- it is 55 MB of MNIST and
changes never).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "dist",
    "runs",
    "data",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".zip", ".parquet", ".pt"}

#: Kept despite their directories being excluded.
#:
#: `runs/LEDGER.md` is the compute appendix and is a few KB.
#:
#: `data/mnist.npz` is 11 MB and never changes, so bundling it costs almost
#: nothing and removes a standing failure mode: the notebook locates MNIST by
#: searching for `mnist.npz` under /kaggle/input, and with it in here the run
#: works whether or not a separate data Dataset happens to be attached. The raw
#: idx-ubyte files stay excluded -- they are the same bytes again, and they are
#: what pushed the upload past Kaggle's 1000-file limit.
FORCE_INCLUDE = {Path("runs/LEDGER.md"), Path("data/mnist.npz")}


def _wanted(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel in FORCE_INCLUDE:
        return True
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "dist" / "neuron-death-code.zip"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and _wanted(p))
    if not any(p.name == "train.py" for p in files):
        raise SystemExit("src/train.py not in the archive; refusing to write")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(files)} files, {size_mb:.1f} MB)")
    if len(files) > 1000:
        print(f"WARNING: {len(files)} files exceeds Kaggle's 1000-file limit")
    print("Upload this single .zip as the code Dataset; Kaggle unzips it on upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
