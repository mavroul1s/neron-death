# Recycling the Living

**Do dormant-neuron interventions work by reviving dead units? We have evidence they don't.**

Research code for a single analysis paper on dead and dormant neurons in continual
learning. This is not a method paper — we are not proposing a new algorithm, and
correctness of measurement matters more here than performance or speed.

---

## The question

When a network trained on non-stationary data loses its ability to learn, is the
accumulation of inactive neurons a **cause** of that loss, or merely a **correlate**?

The literature assumes cause. Roughly fifteen mitigation methods have been built on
that assumption — most prominently ReDo (Sokar et al., 2023), which every ~1000 steps
re-initialises neurons whose activation magnitude falls below a threshold τ.

Four claims under test:

| | Claim |
|---|---|
| **C1** | ReDo's benefit does not come from reviving *dead* neurons. It comes from perturbing low-magnitude *living* ones. |
| **C2** | The published definitions of "dead" disagree materially on the same network. |
| **C3** | Interventions that *help* plasticity can *increase* dead units (L2); interventions designed to *prevent* dead units can *increase* them (online norm). |
| **C5** | Adam-induced neuron death is a distinct, identifiable mechanism, separable from plasticity loss generally. |

C4 (survival analysis of neuron mortality) needs no separate experiment — only the
per-neuron logging, which runs at every task boundary.

The design that makes C1 testable is a **size-matched random control**: at every
recycling event we compute the τ-dormant set, take `k = |dormant set|`, and recycle
`k` neurons chosen *uniformly at random* from the same layer. `k` is recomputed per
layer, per event. Sokar et al.'s own random baseline used a fixed percentage on a
cosine schedule, which confounds *which* neurons are recycled with *how many*.

---

## Status

**Gate passed. The τ-sweep is complete. ~290 runs, ~57 GPU-hours.**

The reproduction gate (protocol §A.4) passed at lr=0.1 — a 4.54 pp [4.45, 4.69]
accuracy drop from tasks 1–10 to 151–200, with `dead_exact` rising 4.8% → 20.5%
(Spearman ρ = 0.90). It failed at every smaller step size, monotonically, which is
Dohare et al.'s own finding reproduced as a dose–response rather than at a point.

### C1 — ReDo is not a resurrection method

The full τ-sweep: 4 arms × 6 τ × 10 seeds. Late-window online accuracy, baseline
`none` = 87.683%.

| arm | what it recycles | dead share of it | Δ vs baseline |
|---|---|---:|---:|
| **ReDo** | the τ-dormant set | 12–31% | **+4.76 … +4.92 pp** |
| **Random-matched** | k units at random | 8–13% | +4.32 … +4.39 pp |
| **Inverse-matched** | the k *highest*-scoring | ~4% | +3.01 … +3.31 pp |

Random selection recovers ~90% of ReDo's benefit. Recycling the units *least* likely
to be dead still buys +3 pp. In both controls the dose confound runs the wrong way —
they recycle **more** units and do **worse** — so the ordering is not explained by
perturbation volume.

The pre-registered C1 prediction fired exactly: accuracy rises with τ while the
truly-dead fraction of the recycled set falls (31.4% → 12.3%).

**The strong form is refuted.** Targeting contributes ~10% (vs random) to ~35% (vs
inverse). Report that, not "targeting is irrelevant".

A finding about the published method: even at **τ=0** — supposedly "only provably dead
units" — ReDo recycles 20% of the network, and **69% of that is alive**, because the
score uses Sokar et al.'s 64-example default while `dead_exact` is measured on 2048.

### C2 — the definitions disagree, and one of them vanishes

Four independent demonstrations:

| setting | result |
|---|---|
| permuted MNIST, lr=0.1 | `dead_exact` 20.5% vs `dormant τ=0.1` **64.8%** — 3.2× on identical activations |
| GELU / SiLU / LeakyReLU | `dead_exact` is **exactly 0.00%** while the others flag 1–20%. GELU has the **largest** plasticity drop (4.84 pp) — loss with no dead neurons by the *Nature* metric |
| conv channels (CIFAR CNN) | `dead_exact` **0.00%** in conv layers while 81–88% of spatial positions are silent; the fc layer in the same net says 26% |
| numerics | whether a GELU unit counts as dead depends on **float32 vs float64** |

Also: `dead_exact` is *higher* on the fixed reference batch than on the current task's,
and the gap is localised to the last hidden layer (20.8% vs 37.4%). Death is partly
distribution-relative — a measurement none of the four source papers can make, because
none uses a second probe batch.

### C5 — Adam-induced death is real; its predicted timing is not

| optimizer | accuracy | drop | `dead_exact` |
|---|---:|---:|---:|
| SGD (lr 0.1) | 87.69% | 4.56 pp | 20.5% |
| Adam (default) | 90.27% | 3.18 pp | **58.8%** |
| Adam (Lyle-tuned ε=1e-3, β₂=0.9) | 90.45% | 2.27 pp | **24.6%** |
| AdamW | 91.17% | 2.28 pp | 58.0% |

Lyle et al.'s two-hyperparameter fix cuts dead units 2.4× at identical learning rate.
But the predicted post-task-switch death spike **is not there** — SGD has one (+2.44 pp
at step 25), Adam declines monotonically. Adam's death accumulates *across* tasks
(25.5% → 64.9%), not within them. Reported as a negative result, with two limitations
stated: a 25-step probe grid could hide a faster spike, and the reference batch was
disabled for those runs.

### The thread running through all of it

Three independent dissociations between dead units and plasticity:

1. At lr=0.001, dead units nearly tripled while accuracy **improved**
2. Adam carries 3× SGD's dead units and a **smaller** plasticity drop
3. GELU has the largest plasticity drop and **zero** dead units by the standard metric

### Open

**C3** has no data yet — configs exist for all seven arms including Online Normalization
(implemented against Chiley et al. 2019, backward control processes and all).
**C4** has its data collected but no analysis written. **Setting 2** hit a calibration
failure — baseline accuracy *rose* 45% → 57% over 50 tasks, so there was no plasticity
loss for recycling to fix; its own gate is queued. The **tanh** arm of Setting 3
diverged at lr=0.1 and needs recalibrating before it can be reported.

---

## What's here

```
configs/
  analysis_plan.json     FROZEN pre-registration — see below
  gate/ gate_hi/ ...     one JSON per run, never edited in place
src/
  probes.py              ALL metric definitions. Nothing else computes a metric.
  models.py              MLP; activation and norm are config fields
  data.py                permuted MNIST, label-shuffled CIFAR-10
  interventions.py       ReDo, random-matched, inverse-matched, L2, shrink-and-perturb
  train.py               task loop, sharded parquet logging, checkpoint/resume
  analysis/              post-hoc only; never imported by training code
tests/                   68 tests, all on synthetic data, ~11 s
runs/LEDGER.md           run ledger / compute appendix
```

Four parquet tables per run: `tasks` (per task), `metrics` (per task × layer),
`neurons` (per task × layer × neuron), `recycling` (per event × layer). Every metric
is logged on **two** probe batches — the current task's distribution and a fixed
reference distribution that never changes — which is what separates "this neuron is
dead" from "the input distribution moved".

## Design decisions worth knowing

- **`configs/analysis_plan.json` is frozen.** Outcome measure, task windows, decision
  thresholds (`|Δ| < 1 pp` for "≈"; `Δ > 2 pp` with CI excluding zero for "≫"), seed
  counts and the gate criterion were committed before any data existed. If a result is
  ambiguous the pre-specified response is *add seeds*, never adjust a threshold.
- **Metrics are implemented exactly as published, including their flaws.** The Sokar
  score normalises by the layer mean, so it is blind to a layer whose activations all
  shrink uniformly. That blind spot is the point of C2 and must not be "fixed" —
  `tests/test_probes.py` asserts that `dormant_tau` *misses* a uniformly shrunken layer
  while `dead_absolute` catches it.
- **ReDo zeroes outgoing weights.** Re-initialising incoming weights without zeroing
  outgoing ones turns ReDo into a far more destructive intervention while still
  producing plausible curves. A test asserts the network's function is unchanged after
  a τ=0 recycling event.
- **Statistics are IQM with stratified bootstrap 95% CIs** (Agarwal et al., 2021),
  never bare means over seeds.
- **Determinism is required**: seeded torch/numpy/python, `cudnn.deterministic`,
  `use_deterministic_algorithms`, weight resampling on CPU so recycling is identical
  across hardware. Runs are resumable by `run_id` alone and checkpoint every 10 tasks,
  because the compute target (Kaggle) kills sessions at 12 hours without warning.
- **`norm="online"` deliberately raises `NotImplementedError`.** Online Normalization
  (Chiley et al., 2019) was not reconstructed from memory — a wrong backward-control
  approximation would still train and still produce a plausible dead-unit curve,
  silently invalidating C3. It must be implemented against the paper before that arm
  runs.

## Running it

```bash
python -m pytest tests/
```

```bash
python scripts/prepare_data.py --dataset mnist --root data
```

```bash
python -m src.train --config configs/gate/gate_pmnist_w500_sgd_lr0p01_s0.json
```

Two configs run in parallel, one per GPU:

```bash
python scripts/launch_pair.py configs/gate_hi/*.json --data-root data --budget-hours 10.5
```

```bash
python -m src.analysis.gate --pattern "gate_*"
```

On Kaggle, build the code dataset with `python scripts/package_for_kaggle.py` and run
`notebooks/kaggle_week1_gate.ipynb`, which locates its datasets by content rather than
by name. There is no download-at-runtime path anywhere in `src/` — Kaggle sessions have
no internet, and a silent fallback would be worse than a crash.

## Scope

Deliberately excluded: deep RL, Mixture-of-Experts, sparse autoencoders, scaling-law
sweeps, models above ~50M parameters, distributed training, and **any new reset or
recycling method of our own**. The contribution is measurement, not method.

## References

Sokar et al. 2023, *The Dormant Neuron Phenomenon in Deep RL* ·
Dohare et al. 2024, *Loss of plasticity in deep continual learning* (Nature) ·
Lyle et al. 2023/2024 · Roy & Vetterli 2007 (effective rank) ·
Agarwal et al. 2021 (IQM, stratified bootstrap) · Ash & Adams 2020 (shrink-and-perturb)
