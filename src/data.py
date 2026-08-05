"""Non-stationary datasets: online permuted MNIST, label-shuffled CIFAR-10.

**There is no download-at-runtime path here, by design** (CLAUDE.md §3). Kaggle
sessions have no internet; the data must already be on disk, cached as a Kaggle
Dataset. If it is missing, this module raises with instructions rather than
silently falling back to anything.

Two decisions worth stating explicitly, because they shape every measurement:

*Probe batches come from the test split.* The 2048 probe examples are never
trained on, so "this neuron is dead" cannot be confounded with "this neuron
memorised the probe batch".

*The reference distribution is the identity permutation.* Task 0 gets its own
random permutation like every other task, so the identity permutation is a real
MNIST distribution that the network is never trained on and that never moves --
which is exactly what is needed to separate "the neuron is dead" from "the input
distribution moved" (CLAUDE.md §5).
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import torch

#: Salts keep the several independent random streams from colliding when they
#: are all derived from one integer `seed`.
_SALT_PERMUTATION = 0x50E4
_SALT_PROBE = 0x9C3B
_SALT_STREAM = 0x1D77
_SALT_SCORE = 0x4A11


# ---------------------------------------------------------------------------
# Raw file loading
# ---------------------------------------------------------------------------


def _read_idx(path: Path) -> np.ndarray:
    """Minimal IDX (idx1/idx3-ubyte) reader, transparently gunzipping.

    Written out rather than pulled from torchvision so that the data path has
    no dependency that could change preprocessing between versions.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic = f.read(4)
        if len(magic) != 4 or magic[0] != 0 or magic[1] != 0:
            raise ValueError(f"{path} is not an IDX file (bad magic {magic!r})")
        dtype_code, ndim = magic[2], magic[3]
        if dtype_code != 0x08:
            raise ValueError(f"{path}: only uint8 IDX files are supported")
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8)
    expected = int(np.prod(dims))
    if arr.size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {arr.size}")
    return arr.reshape(dims)


_MNIST_IDX = {
    "x_train": "train-images-idx3-ubyte",
    "y_train": "train-labels-idx1-ubyte",
    "x_test": "t10k-images-idx3-ubyte",
    "y_test": "t10k-labels-idx1-ubyte",
}


def _missing_data_error(name: str, root: Path, expected: Sequence[str]) -> FileNotFoundError:
    return FileNotFoundError(
        f"{name} not found under {root}.\n"
        f"Expected one of:\n"
        f"  - {root / (name.lower() + '.npz')} "
        f"(keys: x_train, y_train, x_test, y_test)\n"
        + "".join(f"  - {root / e} (or {e}.gz)\n" for e in expected)
        + "\nThere is deliberately no download-at-runtime path (CLAUDE.md §3).\n"
        f"Run `python scripts/prepare_data.py --dataset {name.lower()} --root {root}` "
        "once on a machine with internet, then upload the folder as a Kaggle Dataset."
    )


def load_mnist(root) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x_train, y_train, x_test, y_test) as raw uint8 arrays.

    Accepts either ``<root>/mnist.npz`` or the standard idx-ubyte files, in
    ``<root>`` or ``<root>/MNIST/raw`` (torchvision's layout).
    """
    root = Path(root)
    npz = root / "mnist.npz"
    if npz.exists():
        with np.load(npz) as z:
            return z["x_train"], z["y_train"], z["x_test"], z["y_test"]

    for sub in (root, root / "MNIST" / "raw", root / "mnist"):
        if (sub / _MNIST_IDX["x_train"]).exists() or (
            sub / (_MNIST_IDX["x_train"] + ".gz")
        ).exists():
            out = []
            for key in ("x_train", "y_train", "x_test", "y_test"):
                base = sub / _MNIST_IDX[key]
                path = base if base.exists() else Path(str(base) + ".gz")
                out.append(_read_idx(path))
            return tuple(out)  # type: ignore[return-value]

    raise _missing_data_error("MNIST", root, list(_MNIST_IDX.values()))


def load_cifar10(root) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x_train, y_train, x_test, y_test); images uint8 (N, 32, 32, 3)."""
    root = Path(root)
    npz = root / "cifar10.npz"
    if npz.exists():
        with np.load(npz) as z:
            return z["x_train"], z["y_train"], z["x_test"], z["y_test"]
    raise _missing_data_error("CIFAR10", root, ["cifar10.npz"])


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ContinualDataset:
    """Common machinery for a stream of tasks over a fixed image set.

    Subclasses define what changes between tasks: the input permutation
    (permuted MNIST) or the label map (label-shuffled CIFAR-10).

    The whole training set is held on the compute device. At 60000x784 float32
    that is 188 MB, comfortably inside a T4, and it removes the DataLoader --
    which removes worker-ordering as a source of nondeterminism.
    """

    input_dim: int
    n_classes: int

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        n_tasks: int,
        seed: int,
        device: torch.device,
        batch_size: int = 128,
        n_probe: int = 2048,
    ):
        self.n_tasks = int(n_tasks)
        self.seed = int(seed)
        self.device = device
        self.batch_size = int(batch_size)
        self.n_probe = int(n_probe)

        self.x_train = torch.from_numpy(np.ascontiguousarray(x_train)).to(device)
        self.y_train = torch.from_numpy(np.ascontiguousarray(y_train)).to(device)
        self.x_test = torch.from_numpy(np.ascontiguousarray(x_test)).to(device)
        self.y_test = torch.from_numpy(np.ascontiguousarray(y_test)).to(device)

        self.n_train = int(self.x_train.shape[0])
        if self.n_probe > int(self.x_test.shape[0]):
            raise ValueError(
                f"n_probe={self.n_probe} exceeds test split size {self.x_test.shape[0]}"
            )

        # Fixed probe indices, drawn once from the *test* split and never
        # resampled for the life of the run.
        probe_rng = np.random.default_rng([self.seed, _SALT_PROBE])
        self.probe_idx = torch.from_numpy(
            np.sort(probe_rng.choice(int(self.x_test.shape[0]), self.n_probe, replace=False))
        ).to(device)

    # -- to be provided by subclasses ----------------------------------------

    def _apply_task(self, x: torch.Tensor, y: torch.Tensor, task_idx: int):
        raise NotImplementedError

    def _apply_reference(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    # -- stream ---------------------------------------------------------------

    @property
    def steps_per_task(self) -> int:
        """One pass over the training set per task (protocol §A.4, 'online').

        The final short batch is kept -- dropping it would silently discard
        ~0.16% of every task's data for no benefit.
        """
        return (self.n_train + self.batch_size - 1) // self.batch_size

    def _stream_order(self, task_idx: int) -> torch.Tensor:
        """Deterministic example order for a task.

        Derived from (seed, task_idx) rather than from the global RNG so that a
        run resumed at task 137 sees exactly the stream it would have seen
        uninterrupted, no matter what else consumed random numbers.
        """
        rng = np.random.default_rng([self.seed, _SALT_STREAM, int(task_idx)])
        return torch.from_numpy(rng.permutation(self.n_train)).to(self.device)

    def task_batches(self, task_idx: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        order = self._stream_order(task_idx)
        for start in range(0, self.n_train, self.batch_size):
            idx = order[start : start + self.batch_size]
            x, y = self._apply_task(self.x_train[idx], self.y_train[idx], task_idx)
            yield x, y

    def score_batch(
        self, task_idx: int, event_idx: int, n: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A batch for computing ReDo's Sokar scores.

        Sokar et al. 2023 use 64 examples from the current training
        distribution; they report 32-1024 gives near-identical results. Drawn
        deterministically from (seed, task, event) so recycling is reproducible.
        """
        rng = np.random.default_rng([self.seed, _SALT_SCORE, int(task_idx), int(event_idx)])
        idx = torch.from_numpy(rng.choice(self.n_train, int(n), replace=False)).to(self.device)
        return self._apply_task(self.x_train[idx], self.y_train[idx], task_idx)

    # -- probes ---------------------------------------------------------------

    def probe_batch(self, task_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed 2048 examples under the *current* task's transformation."""
        x, y = self.x_test[self.probe_idx], self.y_test[self.probe_idx]
        return self._apply_task(x, y, task_idx)

    def reference_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The same 2048 examples under a transformation that never changes."""
        x, y = self.x_test[self.probe_idx], self.y_test[self.probe_idx]
        return self._apply_reference(x, y)


# ---------------------------------------------------------------------------
# Permuted MNIST
# ---------------------------------------------------------------------------


class PermutedMNIST(ContinualDataset):
    """Online permuted MNIST: 200 tasks, one pass over 60k images per task.

    Each task applies a fixed random permutation of the 784 input pixels.
    Pixels are scaled to [0, 1] by dividing by 255 and nothing else -- per-pixel
    standardisation would be meaningless once pixels are permuted, and Dohare
    et al.'s reference implementation does the same.
    """

    input_dim = 784
    n_classes = 10

    def __init__(
        self,
        root,
        n_tasks: int,
        seed: int,
        device: torch.device,
        batch_size: int = 128,
        n_probe: int = 2048,
        reference: str = "identity",
    ):
        x_tr, y_tr, x_te, y_te = load_mnist(root)
        x_tr = (x_tr.reshape(len(x_tr), -1).astype(np.float32) / 255.0)
        x_te = (x_te.reshape(len(x_te), -1).astype(np.float32) / 255.0)
        y_tr = y_tr.astype(np.int64)
        y_te = y_te.astype(np.int64)
        super().__init__(x_tr, y_tr, x_te, y_te, n_tasks, seed, device, batch_size, n_probe)

        d = self.x_train.shape[1]
        if d != self.input_dim:
            raise ValueError(f"expected {self.input_dim} input features, got {d}")

        perm_rng = np.random.default_rng([self.seed, _SALT_PERMUTATION])
        perms = np.stack([perm_rng.permutation(d) for _ in range(self.n_tasks)])
        self.permutations = torch.from_numpy(perms).to(device)

        if reference not in ("identity", "task0"):
            raise ValueError(f"unknown reference {reference!r}")
        self.reference = reference
        self._ref_perm = (
            torch.arange(d, device=device)
            if reference == "identity"
            else self.permutations[0]
        )

    def _apply_task(self, x, y, task_idx: int):
        return x[:, self.permutations[task_idx]], y

    def _apply_reference(self, x, y):
        return x[:, self._ref_perm], y


# ---------------------------------------------------------------------------
# Label-shuffled CIFAR-10
# ---------------------------------------------------------------------------


class LabelShuffledCIFAR10(ContinualDataset):
    """CIFAR-10 with a fresh random label permutation each task.

    Not used by the Weeks 1-2 protocol (which is permuted MNIST throughout);
    present because CLAUDE.md §4 puts it in this module. The reference
    "distribution" is the identity label map -- note that for this dataset the
    reference and current probe batches have identical *inputs* and differ only
    in labels, so every activation metric is by construction identical between
    them. That is correct, not a bug: label shuffling does not move the input
    distribution.
    """

    input_dim = 3072
    n_classes = 10

    def __init__(
        self,
        root,
        n_tasks: int,
        seed: int,
        device: torch.device,
        batch_size: int = 128,
        n_probe: int = 2048,
        reference: str = "identity",
    ):
        x_tr, y_tr, x_te, y_te = load_cifar10(root)
        x_tr = x_tr.reshape(len(x_tr), -1).astype(np.float32) / 255.0
        x_te = x_te.reshape(len(x_te), -1).astype(np.float32) / 255.0
        y_tr = y_tr.astype(np.int64).ravel()
        y_te = y_te.astype(np.int64).ravel()
        super().__init__(x_tr, y_tr, x_te, y_te, n_tasks, seed, device, batch_size, n_probe)

        rng = np.random.default_rng([self.seed, _SALT_PERMUTATION])
        maps = np.stack([rng.permutation(self.n_classes) for _ in range(self.n_tasks)])
        self.label_maps = torch.from_numpy(maps).to(device)
        self.reference = reference

    def _apply_task(self, x, y, task_idx: int):
        return x, self.label_maps[task_idx][y]

    def _apply_reference(self, x, y):
        return x, y


# ---------------------------------------------------------------------------


def build_dataset(cfg: dict, device: torch.device) -> ContinualDataset:
    """Construct the dataset from the ``data`` block of a run config."""
    cfg = dict(cfg or {})
    name = cfg.get("name", "permuted_mnist")
    kwargs = dict(
        root=cfg["root"],
        n_tasks=int(cfg["n_tasks"]),
        seed=int(cfg["seed"]),
        device=device,
        batch_size=int(cfg.get("batch_size", 128)),
        n_probe=int(cfg.get("n_probe", 2048)),
        reference=cfg.get("reference", "identity"),
    )
    if name == "permuted_mnist":
        return PermutedMNIST(**kwargs)
    if name == "label_shuffled_cifar10":
        return LabelShuffledCIFAR10(**kwargs)
    raise ValueError(f"unknown dataset {name!r}")
