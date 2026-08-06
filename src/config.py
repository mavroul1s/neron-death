"""Run configuration: defaults, resolution, hashing, determinism.

CLAUDE.md §7: *never silently change a hyperparameter.* A run is fully
described by its config file; if a run needs a different learning rate that is a
new file with a new ``run_id``, not an edit. To make that enforceable rather
than aspirational:

* the *resolved* config (defaults merged in) is written beside the outputs and
  hashed, so the hash pins every value the run actually used, including ones the
  author never typed;
* ``assert_config_unchanged`` refuses to reuse a run directory whose stored
  config hash differs from the one being launched.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

DEFAULT_CONFIG: Dict[str, Any] = {
    "run_id": None,  # required
    "seed": 0,
    "device": "auto",  # 'auto' | 'cuda' | 'cuda:0' | 'cpu'
    "notes": "",
    "data": {
        "name": "permuted_mnist",
        "root": "data",
        "n_tasks": 200,
        "batch_size": 128,
        "n_probe": 2048,
        "reference": "identity",
    },
    "model": {
        "in_features": 784,
        "hidden_dims": [500, 500, 500],
        "out_features": 10,
        "activation": "relu",
        "activation_param": 0.01,
        "norm": "none",
        "dropout": 0.0,
        "init": "kaiming_uniform",
        "output_nonlinearity": None,
        "bias_init": 0.0,
    },
    "optim": {
        "name": "sgd",  # sgd | adam | adamw
        "lr": 0.01,
        "momentum": 0.9,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        # Only meaningful for AdamW (decoupled decay). Coupled L2 goes in the
        # `l2` block so it appears explicitly in the loss.
        "weight_decay": 0.0,
    },
    "l2": {
        "lambda": 0.0,
        "include_bias": True,
    },
    "recycling": {
        "kind": "none",  # none | redo | random_matched | inverse_matched
        "tau": 0.0,
        "freq": 1000,
        "score_batch_size": 64,
        "reset_optimizer_state": True,
        "composition_on_reference": True,
    },
    "shrink_perturb": {
        "enabled": False,
        "shrink": 0.5,
        "perturb": 0.01,
        "every_tasks": 1,
    },
    "probe": {
        "taus": [0.0, 0.01, 0.025, 0.05, 0.1, 0.25],
        "abs_thresholds": [1e-6, 1e-4, 1e-2],
        "saturation_eps": 1e-3,
        "n_probe": 2048,
        "grad_window": 100,
        # Intra-task probing (CLAUDE.md §5.6). null disables and costs nothing;
        # the C5 arm must set it, because boundary-only sampling cannot see a
        # death spike that happens in the first few hundred steps after a task
        # switch, and the data is not recoverable retrospectively.
        "intra_task_probe_every": None,
        "intra_task_probe_reference": True,
    },
    "checkpoint": {
        "every_tasks": 10,
        "keep_last": 2,
    },
    "determinism": {
        # strict=True calls torch.use_deterministic_algorithms(True). If an op
        # has no deterministic kernel the run fails loudly instead of producing
        # irreproducible activation statistics.
        "strict": True,
        "warn_only": False,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_config(raw: dict) -> dict:
    cfg = deep_merge(DEFAULT_CONFIG, raw)
    if not cfg.get("run_id"):
        raise ValueError("config must set a non-empty run_id")
    # The dataset needs the run seed; keep exactly one source of truth for it.
    cfg["data"]["seed"] = int(cfg["seed"])
    cfg["data"]["n_probe"] = int(cfg["probe"]["n_probe"])
    unknown = set(cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    return cfg


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return resolve_config(json.load(f))


def canonical_json(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)


#: Config entries that describe *where the bytes live*, not what the experiment
#: is. Excluded from the hash so that the same run is recognisably the same run
#: whether it is executing on Kaggle (`/kaggle/input/<slug>/...`) or locally
#: (`data/`). Without this, moving a run between machines makes it un-resumable
#: and makes the ledger show two different hashes for one experiment.
_ENVIRONMENTAL_KEYS = (("data", "root"),)


def hashable_config(cfg: dict) -> dict:
    """The resolved config with environmental paths stripped."""
    out = copy.deepcopy(cfg)
    for section, key in _ENVIRONMENTAL_KEYS:
        out.get(section, {}).pop(key, None)
    return out


def config_hash(cfg: dict) -> str:
    """SHA-256 of the *resolved* config, first 16 hex chars.

    Resolved rather than raw so that a value inherited from DEFAULT_CONFIG is
    still pinned by the hash. Consequence, stated so it is never a surprise:
    changing a default changes every hash. That is the correct behaviour --
    those runs really were run under different settings.

    ``data.root`` is deliberately excluded (see `_ENVIRONMENTAL_KEYS`); the path
    a dataset happens to be mounted at is not a property of the experiment. The
    full config, paths included, is still written to `config.json` beside the
    outputs, so nothing is lost.
    """
    return hashlib.sha256(
        canonical_json(hashable_config(cfg)).encode("utf-8")
    ).hexdigest()[:16]


def save_config(cfg: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")


def assert_config_unchanged(cfg: dict, run_dir) -> None:
    """Refuse to resume into a run directory written by a different config."""
    stored = Path(run_dir) / "config.json"
    if not stored.exists():
        return
    with open(stored, "r", encoding="utf-8") as f:
        old = json.load(f)
    if config_hash(old) != config_hash(cfg):
        raise RuntimeError(
            f"{run_dir} was produced by a different config "
            f"(stored hash {config_hash(old)}, new hash {config_hash(cfg)}).\n"
            "Hyperparameters are never edited in place (CLAUDE.md §7): give the "
            "new settings a new run_id."
        )


# ---------------------------------------------------------------------------
# Determinism (CLAUDE.md §7)
# ---------------------------------------------------------------------------


def set_determinism(seed: int, strict: bool = True, warn_only: bool = False) -> dict:
    """Seed every RNG and pin the deterministic backends.

    Must be called before any CUDA tensor is created: cuBLAS reads
    ``CUBLAS_WORKSPACE_CONFIG`` when its handle is first created, and setting it
    afterwards silently has no effect.

    Returns a report that is saved into the run directory, so that any
    nondeterminism we could not remove is documented rather than ignored.
    """
    report: Dict[str, Any] = {"seed": int(seed), "strict": bool(strict)}

    if strict and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    report["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if strict:
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
    report["use_deterministic_algorithms"] = bool(strict)
    report["warn_only"] = bool(warn_only)
    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["cuda_version"] = torch.version.cuda
    return report


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)
