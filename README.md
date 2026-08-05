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

**Week 1 complete. The reproduction gate FAILED. Week 2 has not started.**

The gate (protocol §A.4) requires a ≥3 pp accuracy drop from tasks 1–10 to 151–200 on
200-task online permuted MNIST, plus a rise in dead units. 15 runs, 3 learning rates,
5 seeds, 3.43 GPU-hours:

| lr | accuracy drop | `dead_exact` (early → late) | Spearman ρ | verdict |
|---|---:|---|---:|---|
| 0.01 | **+1.33 pp** (need ≥ 3) | 0.28% → 7.63% | 0.96 | FAIL |
| 0.003 | −0.25 pp | 0.98% → 3.76% | 0.90 | FAIL |
| 0.001 | −1.18 pp | 1.80% → 4.76% | 0.91 | FAIL |

The dead-unit half of the gate passed at every learning rate; the accuracy half failed
at every learning rate. The drop moves monotonically with step size, so the phenomenon
is **present and under-driven, not absent**. Currently running the pre-specified
failure response: raise the learning rate to {0.03, 0.1} (`configs/gate_hi/`).

The τ-sweep will not start until the gate passes. With no plasticity loss there is no
ReDo benefit to decompose.

### Preliminary, from the gate runs

**C2 already looks large.** Same networks, same activations, tasks 151–200, lr=0.01:

| definition | flagged "dead" |
|---|---:|
| `dead_exact` — output exactly 0 on all 2048 probe inputs | 7.6% |
| `dead_absolute` a=1e-2 | 47.4% |
| `dormant τ=0.1` — Sokar's best-reported τ, what ReDo uses | **50.0%** |
| `dormant τ=0.25` | 59.4% |

If ReDo(τ=0.1) ran on these networks it would recycle about half of all units, of which
roughly 15% are genuinely dead. That is C1's headline claim visible before the
intervention has been run — but it is a prior, not evidence. The composition table from
the τ-sweep is the actual measurement.

Also unexpected: `dead_exact` is *higher* on the fixed reference batch than on the
current task's batch (12.7% vs 7.6%). Units stay alive for the permutation being
trained on and go silent on a distribution the network has not seen recently, so
"dead" is partly distribution-relative.

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
