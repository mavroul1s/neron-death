"""Stamp out one ready-to-run Kaggle notebook per job.

Every remaining experiment is independent -- none reads another's output, and
every learning rate is already fixed by the gate -- so they can all run
concurrently on separate Kaggle sessions. The only thing stopping that was that
the template notebook has `EXPERIMENT` hardcoded, which is how `gate_hi` came to
be run three times.

So: one notebook per job, each with its parameter block pre-filled and nothing
to edit. Open, attach the dataset, Run All.

    python scripts/make_kaggle_notebooks.py

Notebooks are named `NN_<job>.ipynb` in upload order and land in
`kaggle_upload/` beside the dataset zip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: One entry per independently-runnable Kaggle session.
#:
#: `runs` and `hours` are for planning only; hours assume the measured
#: 12.6 min/run on 2xT4 unless noted. Ordering reflects scientific priority,
#: not dependency -- there are no dependencies.
JOBS = [
    {
        "name": "tau_a_none_redo",
        "experiment": "tau_sweep",
        "globs": ["tau_none_*.json", "tau_redo_*.json"],
        "pattern": "tau_*",
        "is_gate": False,
        "runs": 70,
        "hours": 7.6,
        "why": (
            "The decisive run. Answers the untested assumption -- does ReDo beat "
            "baseline at all in continual supervised learning? -- and produces "
            "the C1 recycled-set composition table. If ReDo does not beat "
            "`none`, C1 is vacuous and the paper reshapes, so this is the one "
            "to run first if you only run one."
        ),
    },
    {
        "name": "tau_b_random_matched",
        "experiment": "tau_sweep",
        "globs": ["tau_random_*.json"],
        "pattern": "tau_random_*",
        "is_gate": False,
        "runs": 60,
        "hours": 6.5,
        "why": (
            "The size-matched random control -- the actual C1 comparison, and "
            "the thing Sokar et al. did not run. Independent of tau_a; only the "
            "*decision* to bother was ever sequential."
        ),
    },
    {
        "name": "tau_c_inverse_matched",
        "experiment": "tau_sweep",
        "globs": ["tau_inverse_*.json"],
        "pattern": "tau_inverse_*",
        "is_gate": False,
        "runs": 60,
        "hours": 6.5,
        "why": (
            "Sanity check replicating Sokar et al. Fig. 15: recycling the "
            "highest-scoring units should collapse performance. Lowest priority "
            "of the three tau slices -- if compute is short, cut this to 1-2 tau "
            "values and record the trim in configs/DEVIATIONS.md."
        ),
    },
    {
        "name": "setting3_activations",
        "experiment": "setting3",
        "globs": ["*.json"],
        "pattern": "s3_*",
        "is_gate": False,
        "runs": 25,
        "hours": 2.8,
        "why": (
            "Cheapest strong C2 result: ReLU / LeakyReLU / GELU / SiLU / tanh on "
            "Setting 1. Under SiLU `dead_exact` cannot fire; under GELU it fires "
            "only as a float32 underflow artefact; under tanh `saturated` fires "
            "where `dead_exact` structurally cannot."
        ),
    },
    {
        "name": "c5_optimizers",
        "experiment": "c5",
        "globs": ["*.json"],
        "pattern": "c5_*",
        "is_gate": False,
        "runs": 20,
        "hours": 3.2,
        "why": (
            "SGD / Adam / Lyle-tuned Adam / AdamW, with intra-task probing every "
            "25 steps to catch the post-switch death spike. ~20% slower per run "
            "than the others because of the extra probes."
        ),
    },
    {
        "name": "setting2_cifar_cnn",
        "experiment": "setting2",
        "globs": ["*.json"],
        "pattern": "s2_*",
        "is_gate": False,
        "runs": 15,
        "hours": None,  # unmeasured; conv cost is not the MLP cost
        "why": (
            "C1 + C2 with a CONV net on label-shuffled CIFAR-10, where the unit "
            "is a channel. BLOCKED until cifar10.npz is built and packaged: run "
            "`python scripts/prepare_data.py --dataset cifar10 --root data`. "
            "Per-run cost is unmeasured -- read the smoke-test cell's time and "
            "multiply before letting the sweep run."
        ),
    },
]


def _find_cell(nb: dict, needle: str) -> dict:
    for cell in nb["cells"]:
        if needle in "".join(cell["source"]):
            return cell
    raise SystemExit(f"template has no cell containing {needle!r}")


def _as_source(text: str) -> list:
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def build(job: dict, template: dict) -> dict:
    nb = json.loads(json.dumps(template))  # deep copy

    globs = ", ".join(f'"{g}"' for g in job["globs"])
    _find_cell(nb, "EXPERIMENT =")["source"] = _as_source(
        f'''from src.config import load_config
from src.train import Trainer

# --- pre-filled for this job; nothing to edit ------------------------------
EXPERIMENT = "{job['experiment']}"
CONFIG_GLOBS = [{globs}]
RUN_PATTERN = "{job['pattern']}"
IS_GATE = {job['is_gate']}
EXPECTED_RUNS = {job['runs']}
# ---------------------------------------------------------------------------

configs = sorted(
    {{p for g in CONFIG_GLOBS for p in glob.glob(f"{{REPO}}/configs/{{EXPERIMENT}}/{{g}}")}}
)
assert configs, f"no configs matching {{CONFIG_GLOBS}} in {{REPO}}/configs/{{EXPERIMENT}}"
assert len(configs) == EXPECTED_RUNS, (
    f"expected {{EXPECTED_RUNS}} configs, found {{len(configs)}}. The code Dataset "
    "is probably an older version -- re-upload before running the sweep."
)
print(f"{{EXPERIMENT}}: {{len(configs)}} configs -- as expected")

cfg = load_config(configs[0])
cfg["data"]["root"] = DATA
print(Trainer(cfg, runs_root=RUNS).run())'''
    )

    hours = "unmeasured" if job["hours"] is None else f"~{job['hours']:.1f} h"
    nb["cells"][0]["source"] = _as_source(
        f"""# {job['name']}

**{job['runs']} runs · {hours} wall clock on 2×T4 · nothing to edit.**

{job['why']}

## Before you run
1. Attach the code Dataset (the one built by `scripts/package_for_kaggle.py`).
   It bundles `mnist.npz`, so it is the only Dataset you need.
2. Accelerator → **GPU T4 × 2**.
3. Run All.

The config-count assertion in the "Pick the experiment" cell fails fast if the
Dataset is a stale version, rather than silently running the wrong sweep.

## When it finishes
- **`extract.zip`** — download this. A few MB; everything the analysis needs.
- **`runs.zip`** — push to a versioned Dataset, do not download. It holds the
  per-neuron log, which cannot be rebuilt after the fact (CLAUDE.md §5.4).

This notebook contains no logic: it imports from `src/` and calls one function."""
    )
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", default=str(ROOT / "notebooks" / "kaggle_week1_gate.ipynb"))
    ap.add_argument("--out", default=str(ROOT / "kaggle_upload"))
    args = ap.parse_args(argv)

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.ipynb"):
        stale.unlink()

    print(f"{'notebook':42s} {'runs':>5s} {'hours':>7s}")
    for i, job in enumerate(JOBS, start=2):  # 1_ is the dataset zip
        path = out / f"{i}_{job['name']}.ipynb"
        path.write_text(json.dumps(build(job, template), indent=1), encoding="utf-8")
        hrs = "?" if job["hours"] is None else f"{job['hours']:.1f}"
        print(f"  {path.name:40s} {job['runs']:>5d} {hrs:>7s}")

    total = sum(j["runs"] for j in JOBS)
    known = sum(j["hours"] for j in JOBS if j["hours"])
    print(f"\n{len(JOBS)} notebooks, {total} runs, {known:.1f} h known + setting2 unmeasured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
