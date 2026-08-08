# Setting changes

CLAUDE.md §3 splits the experiment in two. **`analysis_plan.json` is frozen**:
outcome measure, task windows, decision thresholds, seed counts, the gate
criterion. **The setting is calibrated**: batch size, width, depth, number of
tasks, learning rate, dataset. Calibrating so that the phenomenon under study is
present at all is a manipulation check, not p-hacking — the published work being
replicated calibrated its own setting the same way.

The rule that keeps that honest is this file. **Every setting change is recorded
here before the run, with a date and a reason** — never by editing a frozen
field, and never after seeing which setting produces the preferred answer.

> **Entries dated 2026-08-08 marked `[reconstructed]` were written after the
> fact.** This file should have existed from the first session and did not; it
> was created on 2026-08-08. Those entries are reconstructed from
> `runs/LEDGER.md`, the run configs and CLAUDE.md §11, all of which are
> contemporaneous with the runs. They are recorded as reconstructions rather
> than backdated, because a deviations log that cannot be told apart from one
> kept properly is worth less than no log at all. Everything from
> `2026-08-08 setting 2` onward was written before its run.

---

## 2026-08-05 — batch size stays at 128, against the protocol's own arithmetic
`[reconstructed]`

**Change:** none. Recorded because the *absence* of a change is the deviation.

Protocol §A.4 specifies `batch_size: 128`. Dohare et al.'s Online Permuted MNIST
is genuinely online — batch size 1, one example at a time, single pass. Their
setup performs 800 × 60,000 = 48M updates; ours performs 200 × (60,000/128)
≈ 94k, a factor of **512** fewer, and plasticity loss accumulates per update
rather than per task boundary.

**Reason for not changing it:** batch size 1 at 200 tasks does not fit the
11-hour job ceiling. The learning-rate ladder below was the affordable response.
CLAUDE.md §3 records the standing instruction that **batch size should be
reduced before any other knob is escalated**, so if the phenomenon ever needs
strengthening again, that is the first move and not a further lr increase.

## 2026-08-05 → 2026-08-06 — learning rate 0.001 → 0.1 (Setting 1)
`[reconstructed]`

**Change:** `optim.lr` 0.001 → 0.003 → 0.01 (`configs/gate/`), then 0.03 → 0.1
(`configs/gate_hi/`). Every subsequent Setting 1 experiment uses **lr = 0.1**.

**Reason:** the reproduction gate failed its accuracy criterion at all three of
the originally planned learning rates (drop of +1.33 pp at lr=0.01 against a
required 3 pp), while the dead-unit criterion passed at every one. Protocol
§A.4's failure response step 1 is "raise the learning rate", which was followed
in order. The drop is perfectly monotonic in step size across two decades
(−1.18 → +4.54 pp), which is Dohare et al.'s own finding — largest step size,
strongest effect — so the phenomenon was **present and under-driven, not
absent**, and escalating the step size was the indicated move rather than a
search for a congenial number.

**Checked before accepting the pass**, so that a diverged run could not be
mistaken for a strong effect: no NaN, max mean-task-loss 0.472, minimum online
accuracy 0.8505, peak 0.9304 at task 1, held-out probe accuracy falling 95.88%
→ 93.77% independently of the online measure, and no wholly dead layer
(max 27.9%). Recorded in CLAUDE.md §11.

## 2026-08-05 — Setting 3 (activation sweep) inherits lr = 0.1
`[reconstructed]`

**Change:** `configs/setting3/` generated at lr = 0.1 for all five activations.

**Reason:** it is the calibrated Setting 1 value and the sweep is meant to vary
one field. **This was wrong for tanh** and is the deviation worth recording:
tanh reached 10.05%, chance for ten classes, so it diverged rather than trained
and its death metrics are meaningless. `configs/setting3_tanh_gate/` (lr ∈
{0.003, 0.01, 0.03}) calibrates it separately. The failed row is excluded from
every table and figure until then — `fig_setting3_activations` drops any arm
within 5 pp of chance and says so, rather than trusting anyone to remember.

## 2026-08-05 — Setting 2 (CIFAR-10 + CNN) at lr = 0.01, 50 tasks
`[reconstructed]`

**Change:** `configs/setting2/` generated at lr = 0.01, `n_tasks = 50`.

**Reason:** a guess. It is recorded because it was a guess and because it
failed: baseline online accuracy **rises** 45.10% (tasks 0–4) → 56.63% (tasks
40–49), so there is no plasticity loss, nothing for recycling to fix, and the
C1 comparison is void. The gate calibrated 0.1 for an MLP on permuted MNIST and
that does not transfer to a CNN on CIFAR-10. The C2-on-channels measurement from
those runs is still valid, because it is a statement about definitions rather
than about plasticity loss.

---

## 2026-08-08 — Setting 2 gate result: BOTH rungs collapse. Notebook 10 is blocked.

**Result, not a change.** `configs/setting2_gate/` (lr ∈ {0.03, 0.1}, 200 tasks,
baseline only, 5 seeds each) ran as notebook 9. **The frozen criterion returns
PASS at both learning rates, and both passes are spurious.**

| lr | early acc | late acc | drop | frozen verdict | health check |
|---|---:|---:|---:|---|---|
| 0.03 | 46.68% | **9.91%** | 36.75 pp | PASS | **COLLAPSED** |
| 0.1 | 29.22% | **9.95%** | 20.11 pp | PASS | **COLLAPSED** |

Late accuracy is chance for ten classes at both. The supporting evidence is
unambiguous and consistent across every seed:

- **mean loss = 2.303 = ln(10)** exactly — the loss of a uniform predictor;
- **gradient norm = 0.0 in every layer** — no parameter is moving any more;
- **weight L2 frozen** from task ~30 (lr=0.03) and task ~20 (lr=0.1);
- the **fully-connected layer is 99.8% / 100% `dead_exact`**, effective rank
  0.20 / 0.00.

The FC layer died completely, so no gradient reaches anything upstream and
training stops permanently. That is a dying-ReLU collapse of the whole network,
**not** loss of plasticity: accuracy fell because training stopped, not because
the network became unable to fit new labels.

**Why the frozen criterion cannot see it.** Its two conditions are "accuracy
fell ≥ 3 pp with CI excluding zero" and "`dead_exact` rose". A network that dies
completely satisfies both *maximally* — accuracy falls as far as it possibly
can, and every unit ends up dead — so a total collapse produces the most
emphatic possible PASS. This is a real limitation of the frozen gate. **It is
being reported, not fixed**: the plan is not amended, `GateResult.passed` still
reports the criterion verbatim, and the health check is a separate advisory
(`gate.collapse_diagnosis`, added 2026-08-08) that sets `usable` alongside it.
The same check was already performed by hand before the Setting 1 gate was
accepted (CLAUDE.md §11, "lr=0.1 is a healthy regime, not a diverged one"); it
is now written down so it cannot be skipped.

**Consequence: do NOT run notebook 10 at either learning rate.** Both would
produce 15 runs of a dead network.

**What we now know bounds the answer.** lr=0.01 over 50 tasks gave *rising*
accuracy (45.10% → 56.63%, no plasticity loss); lr=0.03 over 200 tasks gives
total collapse by task 30. The usable setting, if it exists, is between them —
and note lr=0.01 has **never been run at 200 tasks**, so its 50-task rise may
simply be the pre-decline phase. That is the cheapest thing to test.

**Proposed next rung — PENDING, needs a decision:** extend
`configs/setting2_gate/` with lr ∈ {0.01, 0.02} at 200 tasks, 5 seeds, ~1.1
GPU-h. CLAUDE.md §3 also ranks *reducing batch size* ahead of any further
learning-rate move, which applies here too and is untried in Setting 2.

## 2026-08-08 — Setting 2 horizon 50 → 200 tasks

**Change:** `scripts/make_configs.py:setting2_cifar_cnn` hardcodes
`n_tasks = 50`; proposed change to 200.

**Reason:** not cosmetic. The frozen plan's late window is tasks **151–200**, so
at 50 tasks Setting 2 has no late window and **cannot be evaluated with the
frozen estimator at all** — the first Setting 2 run was read with ad-hoc windows
(tasks 0–4 vs 40–49), which is exactly the freedom the frozen plan exists to
remove. `configs/setting2_gate/` already runs 200 tasks, so the gate and the arm
it calibrates currently disagree about the horizon. At 200 tasks the arm lands
inside the frozen windows and needs no bespoke analysis.

**Cost:** ~0.4 → ~1.6 GPU-h. `n_tasks` is a calibrated field under §3.

**Status: PENDING — moot until a learning rate survives the gate.**
