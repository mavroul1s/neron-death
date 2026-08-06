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
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


UPLOAD_README = """\
KAGGLE UPLOAD -- everything you need, nothing you don't
=======================================================

Two files in this folder. They go to two DIFFERENT places on Kaggle.


1_DATASET_neuron-death-code.zip
    -> https://www.kaggle.com/datasets   ("New Dataset", or "New Version" of
       your existing neuron-death-code dataset)
    Upload the .zip as-is. Kaggle unzips it for you.
    It already contains mnist.npz, so this is the ONLY dataset you need to
    attach. You do not need a separate MNIST dataset.


2_NOTEBOOK_neuron-death.ipynb
    -> https://www.kaggle.com/code   ("New Notebook" -> File -> Import Notebook)


THEN, in the notebook:
    - Add Input -> attach the dataset from step 1
    - Accelerator -> GPU T4 x2
    - Set EXPERIMENT in the "Pick the experiment" cell (see below)
    - Run All


PRE-FLIGHT CHECK (saves hours)
    The "Pick the experiment" cell prints the config count. Confirm it matches
    what you expect before letting the sweep run. If it errors with
    "no configs in .../configs/<name>", the dataset version did not update --
    re-upload rather than letting it run.


AT THE END
    extract.zip   -> DOWNLOAD this one. A few MB. All the analysis needs.
    runs.zip      -> push to a versioned Dataset. Hundreds of MB. Do not
                     download; it is the archival copy of the per-neuron log.
"""


def _write_upload_folder(zip_path: Path, notebook: Path) -> Path:
    """Stage the two upload artifacts in one folder with instructions.

    A Dataset and a Notebook are separate uploads on Kaggle and cannot be
    merged, so the next best thing is one folder, numbered in upload order,
    with a README that says which goes where.
    """
    folder = ROOT / "kaggle_upload"
    folder.mkdir(exist_ok=True)
    for stale in folder.iterdir():
        if stale.is_file():
            stale.unlink()

    dataset_dest = folder / "1_DATASET_neuron-death-code.zip"
    notebook_dest = folder / "2_NOTEBOOK_neuron-death.ipynb"
    dataset_dest.write_bytes(zip_path.read_bytes())
    notebook_dest.write_bytes(notebook.read_bytes())
    (folder / "README.txt").write_text(UPLOAD_README, encoding="utf-8")
    return folder


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "dist" / "neuron-death-code.zip"))
    ap.add_argument(
        "--notebook",
        default=str(ROOT / "notebooks" / "kaggle_week1_gate.ipynb"),
        help="notebook staged alongside the dataset in kaggle_upload/",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and _wanted(p))
    if not any(p.name == "train.py" for p in files):
        raise SystemExit("src/train.py not in the archive; refusing to write")
    if not any(p.name == "mnist.npz" for p in files):
        raise SystemExit(
            "data/mnist.npz not in the archive. Run "
            "`python scripts/prepare_data.py --dataset mnist --root data` first, "
            "or the Kaggle run will have no data and no way to fetch it."
        )

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(files)} files, {size_mb:.1f} MB)")
    if len(files) > 1000:
        print(f"WARNING: {len(files)} files exceeds Kaggle's 1000-file limit")

    folder = _write_upload_folder(out, Path(args.notebook))

    # One ready-to-run notebook per independent job, so parallel sessions never
    # need an edit (and never accidentally re-run the same sweep).
    import make_kaggle_notebooks  # noqa: E402  -- same directory

    print()
    make_kaggle_notebooks.main(["--out", str(folder)])

    print(f"\nUpload folder ready: {folder}")
    for p in sorted(folder.iterdir()):
        print(f"  {p.name:38s} {p.stat().st_size/1e6:7.2f} MB")
    print("\nOpen kaggle_upload/README.txt -- it says which file goes where.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
