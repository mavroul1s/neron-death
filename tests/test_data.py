"""Dataset loading, including every CIFAR-10 layout we might be handed.

CIFAR-10 is distributed in several shapes and public Kaggle datasets use all of
them. Rather than force one, `load_cifar10` reads whichever is present -- so the
sweep can attach an existing Kaggle dataset instead of re-uploading 163 MB.

The point of these tests is that all three paths produce **identical arrays**:
if the pickle branch forgot the channel-major transpose, images would arrive
scrambled and every conv activation statistic would silently be measuring noise.
"""

from __future__ import annotations

import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

from src.data import load_cifar10


def _fake_cifar_arrays(seed=0):
    rng = np.random.default_rng(seed)
    # 2 images per train batch x 5 batches, 3 test images.
    x_train = rng.integers(0, 256, size=(10, 32, 32, 3), dtype=np.uint8)
    y_train = rng.integers(0, 10, size=10, dtype=np.uint8)
    x_test = rng.integers(0, 256, size=(3, 32, 32, 3), dtype=np.uint8)
    y_test = rng.integers(0, 10, size=3, dtype=np.uint8)
    return x_train, y_train, x_test, y_test


def _to_flat(images):
    """(N, 32, 32, 3) -> (N, 3072) channel-major, the official pickle layout."""
    return images.transpose(0, 3, 1, 2).reshape(len(images), -1)


def _write_batches(dest: Path, arrays):
    x_train, y_train, x_test, y_test = arrays
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        sl = slice(i * 2, (i + 1) * 2)
        with open(dest / f"data_batch_{i + 1}", "wb") as f:
            pickle.dump(
                {b"data": _to_flat(x_train[sl]), b"labels": list(y_train[sl])}, f
            )
    with open(dest / "test_batch", "wb") as f:
        pickle.dump({b"data": _to_flat(x_test), b"labels": list(y_test)}, f)


@pytest.fixture
def arrays():
    return _fake_cifar_arrays()


def _assert_matches(got, arrays):
    x_train, y_train, x_test, y_test = arrays
    assert np.array_equal(got[0], x_train)
    assert np.array_equal(got[1], y_train)
    assert np.array_equal(got[2], x_test)
    assert np.array_equal(got[3], y_test)


def test_loads_npz(tmp_path, arrays):
    x_train, y_train, x_test, y_test = arrays
    np.savez(
        tmp_path / "cifar10.npz",
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    )
    _assert_matches(load_cifar10(tmp_path), arrays)


def test_loads_extracted_pickle_batches(tmp_path, arrays):
    """The official `cifar-10-batches-py/` folder, as most Kaggle datasets ship."""
    _write_batches(tmp_path / "cifar-10-batches-py", arrays)
    _assert_matches(load_cifar10(tmp_path), arrays)


def test_loads_nested_pickle_batches(tmp_path, arrays):
    """Kaggle often nests the folder one level deeper than you expect."""
    _write_batches(tmp_path / "cifar-10-python" / "cifar-10-batches-py", arrays)
    _assert_matches(load_cifar10(tmp_path), arrays)


def test_loads_tarball_without_extracting(tmp_path, arrays):
    staging = tmp_path / "staging" / "cifar-10-batches-py"
    _write_batches(staging, arrays)
    with tarfile.open(tmp_path / "cifar-10-python.tar.gz", "w:gz") as tf:
        for p in sorted(staging.iterdir()):
            tf.add(p, arcname=f"cifar-10-batches-py/{p.name}")
    _assert_matches(load_cifar10(tmp_path), arrays)


def test_every_layout_yields_identical_arrays(tmp_path, arrays):
    """The layouts must be interchangeable. A missing channel-major transpose in
    the pickle branch would scramble images and quietly corrupt every conv
    activation statistic downstream."""
    npz_dir, pkl_dir = tmp_path / "a", tmp_path / "b"
    npz_dir.mkdir()
    x_train, y_train, x_test, y_test = arrays
    np.savez(
        npz_dir / "cifar10.npz",
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    )
    _write_batches(pkl_dir / "cifar-10-batches-py", arrays)

    for a, b in zip(load_cifar10(npz_dir), load_cifar10(pkl_dir)):
        assert np.array_equal(a, b)


def test_missing_cifar_names_every_accepted_layout(tmp_path):
    """The error has to be actionable: it is the only thing a user sees when the
    Kaggle dataset they attached is in a shape we do not read."""
    with pytest.raises(FileNotFoundError) as e:
        load_cifar10(tmp_path)
    msg = str(e.value)
    assert "cifar10.npz" in msg
    assert "cifar-10-batches-py" in msg
    assert "cifar-10-python.tar.gz" in msg
    assert "no download-at-runtime" in msg
