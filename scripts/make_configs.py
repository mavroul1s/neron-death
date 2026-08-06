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


def gate_raised_lr() -> list:
    """§A.4 failure response, step 1: 'raise the learning rate first (Dohare's
    largest step size shows the strongest effect)'.

    Why these values. The first gate (lr 0.001/0.003/0.01, 5 seeds, run
    2026-08-05) failed the >=3 pp criterion at every rate, but the drop moved
    monotonically with the step size -- -1.18 pp at 0.001, -0.25 pp at 0.003,
    +1.30 pp at 0.01 -- so the phenomenon is present and under-driven, not
    absent. These extend the ladder by two half-decades in the same direction.

    NOTE for whoever reads the results: the optimizer keeps momentum 0.9 per
    protocol §A.4, so the effective step is roughly 10x the nominal lr. lr=0.1
    with momentum may be unstable. A diverged run is itself an answer and gets
    reported as one -- it must not be quietly dropped from the sweep.

    New run_ids; the original gate configs are untouched (CLAUDE.md §7).
    """
    out = []
    for lr in (0.03, 0.1):
        for seed in range(5):
            rid = f"gatehi_pmnist_w500_sgd_lr{lr:g}_s{seed}".replace(".", "p")
            cfg = _base(rid, seed, lr)
            cfg["notes"] = (
                "Week 1 gate, raised learning rate "
                "(protocol A.4 failure response step 1)."
            )
            out.append(cfg)
    return out


def gate_long() -> list:
    """§A.4 failure response, step 2: 'extend to 400 tasks'.

    Only run this if step 1 (gate_hi) also fails; the pre-specified order is not
    optional. At ~4.2 s/task these are ~28 min per run, so 10 runs is ~4.7
    GPU-hours -- the most expensive of the three responses, which is why it is
    second rather than first.
    """
    out = []
    for lr in (0.01, 0.03):
        for seed in range(5):
            rid = f"gatelong_pmnist_w500_sgd_lr{lr:g}_s{seed}".replace(".", "p")
            cfg = _base(rid, seed, lr, n_tasks=400)
            cfg["notes"] = (
                "Week 1 gate, 400 tasks (protocol A.4 failure response step 2)."
            )
            out.append(cfg)
    return out


def gate_narrow() -> list:
    """§A.4 failure response, step 3: 'narrow to width 200'.

    Dohare et al. report plasticity loss is most pronounced at smaller widths
    (their Fig. 2b, middle panel), which is the same reasoning that put us at
    500 rather than 2000 in the first place.
    """
    out = []
    for lr in (0.01, 0.03):
        for seed in range(5):
            rid = f"gatenarrow_pmnist_w200_sgd_lr{lr:g}_s{seed}".replace(".", "p")
            cfg = _base(rid, seed, lr)
            cfg["model"]["hidden_dims"] = [200, 200, 200]
            cfg["notes"] = (
                "Week 1 gate, width 200 (protocol A.4 failure response step 3)."
            )
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


#: Intra-task probe spacing for the C5 arm (CLAUDE.md §5.6).
#:
#: 25 steps out of ~469 per task gives ~19 samples per task, which resolves a
#: spike confined to "the first few hundred steps after a task switch" -- the
#: thing Lyle et al.'s Fig. 1 mechanism predicts and that nobody has shown
#: explicitly. Boundary-only sampling cannot see it at all.
#:
#: The reference batch is switched OFF here, halving the probe cost: C5 asks
#: *when* units die under the distribution being trained on, and the boundary
#: log still captures the reference batch once per task. Forward-only probing at
#: this spacing costs roughly 20% on top of training.
C5_INTRA_TASK_EVERY = 25


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
            cfg["probe"] = {
                "intra_task_probe_every": C5_INTRA_TASK_EVERY,
                "intra_task_probe_reference": False,
            }
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


def setting2_cifar_cnn(lr: float) -> list:
    """Setting 2 (CLAUDE.md §9): label-shuffled CIFAR-10 with a small CNN.

    Replicates C1 and C2 only, and is a C2 *extension* rather than a repeat
    because the unit becomes a channel: "dead" now has to mean zero across all
    inputs AND all spatial positions, and whether the published definitions
    survive that generalisation is part of the claim.

    Chosen over a CNN on MNIST deliberately -- it varies dataset, architecture
    and kind of non-stationarity at once, where a CNN on MNIST would vary
    architecture alone and be much weaker evidence.

    Arms: baseline, ReDo and the size-matched random control at Sokar's best
    tau. Not a full tau-sweep: Setting 2 exists to show the Setting 1 result is
    not an artefact of MLPs on permuted MNIST, and ~6 GPU-h is the budget.
    """
    out = []
    for seed in range(5):
        arms = {
            "none": {"kind": "none"},
            "redo_t0p1": {"kind": "redo", "tau": 0.1, "freq": 1000},
            "random_t0p1": {"kind": "random_matched", "tau": 0.1, "freq": 1000},
        }
        for name, recycling in arms.items():
            rid = f"s2_cifar_cnn_{name}_lr{lr:g}_s{seed}".replace(".", "p")
            cfg = _base(rid, seed, lr, n_tasks=50)
            cfg["data"] = {
                "name": "label_shuffled_cifar10",
                "root": DATA_ROOT,
                "n_tasks": 50,
                "batch_size": 128,
                "flatten": False,  # keep NCHW for the conv stack
            }
            cfg["model"] = {
                "architecture": "cnn",
                "in_channels": 3,
                "image_size": 32,
                "channels": [32, 64, 64],
                "fc_dims": [128],
                "out_features": 10,
                "activation": "relu",
            }
            cfg["recycling"] = recycling
            cfg["notes"] = (
                f"Setting 2, label-shuffled CIFAR-10 + small CNN, arm {name} "
                "(CLAUDE.md §9). Unit is a channel."
            )
            out.append(cfg)
    return out


def setting3_activations(lr: float) -> list:
    """Setting 3 (CLAUDE.md §9): the activation sweep, one config field.

    The strongest single demonstration that the field's definitions are not
    measuring one underlying thing:

    * under GELU and SiLU a unit never emits exactly zero, so `dead_exact` --
      Dohare et al.'s definition, the one in the *Nature* paper -- is
      identically zero **by construction**, while `dormant_tau` and
      `dead_absolute` keep flagging units normally;
    * under tanh, `saturated` fires where `dead_exact` cannot.
    """
    out = []
    for seed in range(5):
        for act in ("relu", "leaky_relu", "gelu", "silu", "tanh"):
            rid = f"s3_act_{act}_lr{lr:g}_s{seed}".replace(".", "p")
            cfg = _base(rid, seed, lr)
            cfg["model"]["activation"] = act
            cfg["notes"] = f"Setting 3 activation sweep, {act} (CLAUDE.md §9)."
            out.append(cfg)
    return out


EXPERIMENTS = {
    "gate": lambda lr: gate(),
    # Protocol §A.4 failure responses, in the order they must be tried.
    "gate_hi": lambda lr: gate_raised_lr(),
    "gate_long": lambda lr: gate_long(),
    "gate_narrow": lambda lr: gate_narrow(),
    "tau_sweep": tau_sweep,
    "c3": c3_anomaly,
    "c5": c5_optimizer,
    "eps": eps_sweep,
    # CLAUDE.md §9 replacements for the cancelled transformer arm.
    "setting2": setting2_cifar_cnn,
    "setting3": setting3_activations,
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
