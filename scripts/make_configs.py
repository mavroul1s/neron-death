"""Generate run configs from the protocol's experiment definitions.

One JSON per run (CLAUDE.md §4), because a run is only reproducible if the exact
settings it used sit on disk beside its outputs. Generating them rather than
hand-writing 190 files keeps the sweep definition in one auditable place; the
generated files are still the ground truth once written, and are never edited
afterwards -- a changed hyperparameter is a new run_id.

    python scripts/make_configs.py --experiment gate
    python scripts/make_configs.py --experiment tau_sweep --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_hash, resolve_config  # noqa: E402

TAUS = [0.0, 0.01, 0.025, 0.05, 0.1, 0.25]
DATA_ROOT = "data"


def _base(run_id: str, seed: int, lr: float, n_tasks: int = 200) -> dict:
    """Protocol §A.4: online permuted MNIST, 200 tasks, 784-500-500-500-10,
    ReLU, Kaiming init, SGD + momentum 0.9, batch 128."""
    return {
        "run_id": run_id,
        "seed": seed,
        "device": "auto",
        "data": {
            "name": "permuted_mnist",
            "root": DATA_ROOT,
            "n_tasks": n_tasks,
            "batch_size": 128,
        },
        "model": {
            "hidden_dims": [500, 500, 500],
            "activation": "relu",
            "init": "kaiming_uniform",
        },
        "optim": {"name": "sgd", "lr": lr, "momentum": 0.9},
    }


def gate() -> list:
    """§A.4 reproduction gate: 3 learning rates x 5 seeds."""
    out = []
    for lr in (0.01, 0.003, 0.001):
        for seed in range(5):
            # Dots become 'p' throughout: run_ids end up as directory names,
            # dataset slugs and filenames, and a bare '.' in those is asking for
            # a suffix-stripping bug somewhere downstream.
            rid = f"gate_pmnist_w500_sgd_lr{lr:g}_s{seed}".replace(".", "p")
            cfg = _base(rid, seed, lr)
            cfg["notes"] = "Week 1 reproduction gate (protocol A.4)."
            out.append(cfg)
    return out


def tau_sweep(lr: float) -> list:
    """§B.1 primary experiment: 4 arms x 6 taus x 10 seeds.

    'None' is a single baseline shared across taus, not one per tau: with no
    intervention, tau does nothing. That makes 3*6 + 1 = 19 configurations,
    matching the protocol's count.
    """
    out = []
    for seed in range(10):
        rid = f"tau_none_lr{lr:g}_s{seed}".replace(".", "p")
        cfg = _base(rid, seed, lr)
        cfg["notes"] = "tau-sweep baseline, no intervention (protocol B.1)."
        out.append(cfg)
        for arm in ("redo", "random_matched", "inverse_matched"):
            for tau in TAUS:
                rid = f"tau_{arm}_t{tau:g}_lr{lr:g}_s{seed}".replace(".", "p")
                cfg = _base(rid, seed, lr)
                cfg["recycling"] = {
                    "kind": arm,
                    "tau": tau,
                    "freq": 1000,
                    "score_batch_size": 64,
                }
                cfg["notes"] = f"tau-sweep arm {arm} at tau={tau} (protocol B.1)."
                out.append(cfg)
    return out


def c3_anomaly(lr: float) -> list:
    """§B.2 replication of Dohare et al. Fig. 4b with the C2 decomposition.

    The `online-norm` arm is intentionally absent: `src/models.py` raises rather
    than approximate Online Normalization (Chiley et al. 2019). Implement it
    faithfully, then add it here.
    """
    out = []
    for seed in range(5):
        variants = {
            "backprop": {},
            "l2_1em4": {"l2": {"lambda": 1e-4}},
            "l2_1em3": {"l2": {"lambda": 1e-3}},
            "l2_1em2": {"l2": {"lambda": 1e-2}},
            "sp": {"shrink_perturb": {"enabled": True, "shrink": 0.5, "perturb": 0.01}},
            "dropout01": {"model": {"dropout": 0.1}},
        }
        for name, over in variants.items():
            cfg = _base(f"c3_{name}_lr{lr:g}_s{seed}".replace(".", "p"), seed, lr)
            for k, v in over.items():
                cfg.setdefault(k, {}).update(v)
            cfg["notes"] = f"C3 anomaly replication, arm {name} (protocol B.2)."
            out.append(cfg)
    return out


def c5_optimizer(lr: float) -> list:
    """§B.3 optimizer arms. Lyle-tuned Adam is eps=1e-3, beta2=0.9."""
    out = []
    arms = {
        "sgd": {"name": "sgd", "lr": lr, "momentum": 0.9},
        "adam_default": {"name": "adam", "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8},
        "adam_lyle": {"name": "adam", "lr": 1e-3, "betas": [0.9, 0.9], "eps": 1e-3},
        "adamw": {"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8,
                  "weight_decay": 1e-2},
    }
    for seed in range(5):
        for name, optim in arms.items():
            cfg = _base(f"c5_{name}_s{seed}", seed, lr)
            cfg["optim"] = optim
            cfg["notes"] = f"C5 optimizer arm {name} (protocol B.3)."
            out.append(cfg)
    return out


def eps_sweep(lr: float) -> list:
    """§B.4 demoted epsilon sweep: ReLU vs LeakyReLU dose-response."""
    out = []
    for seed in range(5):
        cfg = _base(f"eps_relu_lr{lr:g}_s{seed}".replace(".", "p"), seed, lr)
        cfg["notes"] = "eps-sweep control, plain ReLU (protocol B.4)."
        out.append(cfg)
        for eps in (1e-4, 1e-3, 1e-2, 1e-1):
            rid = f"eps_leaky_{eps:g}_s{seed}".replace(".", "p")
            cfg = _base(rid, seed, lr)
            cfg["model"]["activation"] = "leaky_relu"
            cfg["model"]["activation_param"] = eps
            cfg["notes"] = f"eps-sweep, LeakyReLU eps={eps} (protocol B.4)."
            out.append(cfg)
    return out


EXPERIMENTS = {
    "gate": lambda lr: gate(),
    "tau_sweep": tau_sweep,
    "c3": c3_anomaly,
    "c5": c5_optimizer,
    "eps": eps_sweep,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    ap.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="best learning rate from the gate; ignored for --experiment gate",
    )
    ap.add_argument("--out", default="configs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    configs = EXPERIMENTS[args.experiment](args.lr)
    out_dir = Path(args.out) / args.experiment
    ids = [c["run_id"] for c in configs]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate run_ids generated; refusing to write")

    print(f"{args.experiment}: {len(configs)} configs -> {out_dir}")
    if args.dry_run:
        for c in configs[:5]:
            print(f"  {c['run_id']}  hash={config_hash(resolve_config(c))}")
        print(f"  ... ({len(configs)} total)")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        resolve_config(cfg)  # fail now, not eleven hours into a session
        path = out_dir / f"{cfg['run_id']}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != cfg:
                raise SystemExit(
                    f"{path} exists with different contents. Configs are never "
                    "edited in place (CLAUDE.md §7); use a new run_id."
                )
            continue
        path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(configs)} configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
