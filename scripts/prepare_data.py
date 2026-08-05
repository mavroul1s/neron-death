"""One-off, setup-time dataset cache builder. **Never called at runtime.**

CLAUDE.md §3 forbids download-at-runtime code paths, because Kaggle sessions
have no internet and a silent fallback would be worse than a crash. This script
exists so the cache can be built once, on a machine that does have internet, and
uploaded as a Kaggle Dataset.

    python scripts/prepare_data.py --dataset mnist --root data

Produces ``<root>/mnist.npz`` with keys x_train, y_train, x_test, y_test --
exactly what ``src/data.py`` reads. Nothing in ``src/`` imports this file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import pickle
import struct
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

MNIST_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}

MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _download(filename: str, dest: Path, mirrors) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  have {dest.name}")
        return dest
    last = None
    for base in mirrors:
        url = base + filename
        try:
            print(f"  fetching {url}")
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - report and try the next mirror
            last = exc
            print(f"    failed: {exc}")
    raise RuntimeError(f"could not download {filename}: {last}")


def _read_idx_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = f.read(4)
        ndim = magic[3]
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def prepare_mnist(root: Path) -> Path:
    raw = root / "raw"
    arrays = {
        key: _read_idx_gz(_download(fn, raw / fn, MNIST_MIRRORS))
        for key, fn in MNIST_FILES.items()
    }
    assert arrays["x_train"].shape == (60000, 28, 28), arrays["x_train"].shape
    assert arrays["x_test"].shape == (10000, 28, 28), arrays["x_test"].shape
    out = root / "mnist.npz"
    np.savez_compressed(out, **arrays)
    return out


def prepare_cifar10(root: Path) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tar = raw / "cifar-10-python.tar.gz"
    if not tar.exists():
        print(f"  fetching {CIFAR10_URL}")
        urllib.request.urlretrieve(CIFAR10_URL, tar)

    xs, ys = [], []
    with tarfile.open(tar, "r:gz") as tf:
        for i in range(1, 6):
            member = tf.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            batch = pickle.load(member, encoding="bytes")
            xs.append(batch[b"data"])
            ys.append(np.array(batch[b"labels"]))
        member = tf.extractfile("cifar-10-batches-py/test_batch")
        test = pickle.load(member, encoding="bytes")

    def _reshape(flat):
        # CIFAR's python pickle stores (N, 3072) as channel-major.
        return flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    out = root / "cifar10.npz"
    np.savez_compressed(
        out,
        x_train=_reshape(np.concatenate(xs)).astype(np.uint8),
        y_train=np.concatenate(ys).astype(np.uint8),
        x_test=_reshape(test[b"data"]).astype(np.uint8),
        y_test=np.array(test[b"labels"], dtype=np.uint8),
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument("--root", default="data")
    args = ap.parse_args(argv)

    root = Path(args.root)
    out = prepare_mnist(root) if args.dataset == "mnist" else prepare_cifar10(root)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, sha256:{digest})")
    print("Upload this folder as a Kaggle Dataset and point data.root at it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
