"""All metric definitions for the neuron-death study.

CLAUDE.md §4: **every metric lives in this file and nowhere else.** If a metric
is computed inline in ``train.py`` or re-derived in ``src/analysis/`` it will
drift from the version used here, and the C2 comparison (which is entirely about
definitions disagreeing) becomes meaningless.

Design rules followed throughout:

* Statistics that feed a threshold are accumulated in ``float64``. The smallest
  ``dead_absolute`` threshold is 1e-6, which is within a factor of ~10 of the
  float32 accumulation error of a 2048-term sum of activations, so a float32
  mean would make that threshold partly an artefact of summation order.
* ``dead_exact`` uses an exact ``!= 0`` comparison on the raw activations. No
  tolerance (CLAUDE.md §5.1, Dohare et al. 2024).
* Nothing here mutates the model, and every entry point runs under
  ``torch.no_grad()`` with the model in ``eval()`` mode (set by the caller).

Model contract (duck-typed, to keep this module independent of ``models.py``):
a probed model must expose

    forward_with_activations(x) -> (logits, pre_activations, post_activations)
    activation_extremes -> tuple[float, ...] | None

where ``pre_activations[i]`` is the input to the nonlinearity of hidden layer
``i`` (i.e. after any normalisation layer) and ``post_activations[i]`` is its
output (before dropout).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Protocol defaults (protocol_weeks_1_2_v2.md §A.3, CLAUDE.md §5)
# ---------------------------------------------------------------------------

#: Sokar et al. 2023 dormancy thresholds. tau=0 is "exactly dead by the Sokar
#: score"; tau=0.1 is the value Sokar et al. report as best.
DEFAULT_TAUS: Tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.1, 0.25)

#: Our fixed absolute thresholds. These are the control that exposes the
#: layer-mean normalisation blind spot in `dormant_tau` (C2).
DEFAULT_ABS_THRESHOLDS: Tuple[float, ...] = (1e-6, 1e-4, 1e-2)

#: Saturation tolerance for bounded activations (Rakitianskaia & Engelbrecht
#: 2015). Not pinned by the protocol; recorded in every run config so it is
#: never an implicit constant.
DEFAULT_SATURATION_EPS: float = 1e-3

#: Probe batch size. CLAUDE.md §5 says 2048; Dohare et al. use 2000. We use
#: 2048 per CLAUDE.md and log the number actually used with every run.
PROBE_BATCH_SIZE: int = 2048

#: "gradient norm averaged over the last 100 steps of the task" (CLAUDE.md §5.3).
GRAD_WINDOW: int = 100


# ---------------------------------------------------------------------------
# Column-name helpers
#
# Parameterised metrics (dormant_tau[tau], dead_absolute[a]) become flat parquet
# columns. The mapping lives here so `src/analysis/` can regenerate the exact
# names instead of hard-coding them.
# ---------------------------------------------------------------------------


def tau_key(tau: float) -> str:
    """0.025 -> '0p025', 0.0 -> '0', 0.1 -> '0p1'."""
    return f"{tau:g}".replace(".", "p").replace("-", "m")


def abs_key(a: float) -> str:
    """1e-6 -> '1em06', 1e-2 -> '1em02'."""
    return f"{a:.0e}".replace("-", "m").replace("+", "p")


def dormant_col(tau: float) -> str:
    return f"dormant_frac_tau_{tau_key(tau)}"


def dead_abs_col(a: float) -> str:
    return f"dead_abs_frac_{abs_key(a)}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    """Everything that parameterises a measurement. Serialised into run configs."""

    taus: Tuple[float, ...] = DEFAULT_TAUS
    abs_thresholds: Tuple[float, ...] = DEFAULT_ABS_THRESHOLDS
    saturation_eps: float = DEFAULT_SATURATION_EPS
    n_probe: int = PROBE_BATCH_SIZE
    grad_window: int = GRAD_WINDOW

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ProbeConfig":
        d = dict(d or {})
        return cls(
            taus=tuple(float(t) for t in d.get("taus", DEFAULT_TAUS)),
            abs_thresholds=tuple(
                float(a) for a in d.get("abs_thresholds", DEFAULT_ABS_THRESHOLDS)
            ),
            saturation_eps=float(d.get("saturation_eps", DEFAULT_SATURATION_EPS)),
            n_probe=int(d.get("n_probe", PROBE_BATCH_SIZE)),
            grad_window=int(d.get("grad_window", GRAD_WINDOW)),
        )

    def to_dict(self) -> dict:
        return {
            "taus": list(self.taus),
            "abs_thresholds": list(self.abs_thresholds),
            "saturation_eps": self.saturation_eps,
            "n_probe": self.n_probe,
            "grad_window": self.grad_window,
        }


# ---------------------------------------------------------------------------
# Primitive metrics -- each one is the single definition used everywhere
# ---------------------------------------------------------------------------


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float64)


def mean_abs_activation(post: torch.Tensor) -> torch.Tensor:
    """E_x |h_i(x)| per neuron. Shape (N, H) -> (H,), float64.

    The abs() is exact in float32; only the sum is promoted, which is the part
    that matters for the 1e-6 threshold.
    """
    return _f64(post).abs().mean(dim=0)


def active_mask(post: torch.Tensor) -> torch.Tensor:
    """(N, H) bool: h_i(x) != 0, exactly. No tolerance."""
    return post.detach() != 0


def frac_inputs_active(post: torch.Tensor) -> torch.Tensor:
    """Fraction of probe inputs on which the neuron emits a non-zero value.

    This is the exact complement of `dead_exact`, by construction:
    ``dead_exact_mask(h) == (frac_inputs_active(h) == 0)``. The recycled-set
    composition table (CLAUDE.md §6) relies on that identity -- it splits every
    recycled neuron into "genuinely dead" and "alive but quiet" with no gap.

    For unbounded-below activations (tanh, sigmoid) an exact zero essentially
    never occurs, so this is ~1.0 everywhere. That is correct and expected:
    `dead_exact` is a ReLU-family notion (Dohare et al. 2024).
    """
    return active_mask(post).to(torch.float64).mean(dim=0)


def dead_exact_mask(post: torch.Tensor) -> torch.Tensor:
    """`dead_exact` (Dohare et al. 2024): h(x) == 0 for ALL probe inputs.

    Exact comparison to zero -- do NOT introduce a tolerance (CLAUDE.md §5.1).
    """
    return ~active_mask(post).any(dim=0)


def sokar_scores(mean_abs: torch.Tensor) -> torch.Tensor:
    """Sokar et al. 2023 dormancy score.

        s_i^l = E_x|h_i^l(x)| / ( (1/H^l) * sum_k E_x|h_k^l(x)| )

    Note the LAYER-MEAN normalisation: this is a *relative* measure and is
    invariant to any uniform rescaling of the whole layer. That invariance is
    the blind spot C2 exists to demonstrate -- do not "fix" it (CLAUDE.md §5.1).

    Degenerate case: if every neuron in the layer has E|h| == 0 the ratio is
    0/0. We define the score as 0, which makes every neuron tau-dormant for
    every tau >= 0 -- the semantically correct answer for a wholly dead layer.
    """
    layer_mean = mean_abs.mean()
    if float(layer_mean) <= 0.0:
        return torch.zeros_like(mean_abs)
    return mean_abs / layer_mean


def dormant_mask(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """tau-dormant iff s_i^l <= tau (Sokar et al. 2023). Inclusive."""
    return scores <= tau


def dead_absolute_mask(mean_abs: torch.Tensor, a: float) -> torch.Tensor:
    """`dead_absolute[a]` (ours): E_x|h(x)| < a for a FIXED threshold a.

    Strict inequality, per CLAUDE.md §5.1. No layer normalisation -- that
    absence is the whole point.
    """
    return mean_abs < a


def saturated_mask(
    post: torch.Tensor,
    extremes: Optional[Sequence[float]],
    eps: float = DEFAULT_SATURATION_EPS,
) -> Optional[torch.Tensor]:
    """`saturated` (Rakitianskaia & Engelbrecht 2015), bounded activations only.

    A neuron is saturated if, for *every* probe input, its output lies within
    eps of one of the activation's extreme values (tanh: -1/+1, sigmoid: 0/1).

    Returns ``None`` for unbounded activations -- never 0, so that "not
    applicable" is never silently read as "none saturated" (CLAUDE.md §5.1).
    """
    if extremes is None:
        return None
    h = post.detach()
    dist = torch.stack([(h - float(e)).abs() for e in extremes], dim=0).amin(dim=0)
    return (dist < eps).all(dim=0)


def effective_rank(post: torch.Tensor) -> float:
    """Entropy-based effective rank, Roy & Vetterli (2007), as used by Dohare
    et al. 2024 -- NOT the srank_delta of Kumar et al. 2021 (CLAUDE.md §5.2).

        p_k = sigma_k / ||sigma||_1
        erank = exp( -sum_k p_k log p_k )      with 0 log 0 := 0

    Computed on the raw (un-centred) activation matrix, matching Dohare et al.
    SVD is done in float64: the singular-value tail is what the entropy is most
    sensitive to, and float32 truncates it.
    """
    m = _f64(post)
    sv = torch.linalg.svdvals(m)
    total = sv.sum()
    if float(total) <= 0.0:
        # Wholly dead layer: no directions of variation at all.
        return 0.0
    p = sv / total
    zero = torch.zeros((), dtype=p.dtype, device=p.device)
    plogp = torch.where(p > 0, p * torch.log(p), zero)
    return float(torch.exp(-plogp.sum()))


def sign_entropy_per_neuron(pre: torch.Tensor) -> torch.Tensor:
    """Binary entropy of the pre-activation sign, per neuron, in BITS.

    Lewandowski et al. 2024. Let q_i = P_x(z_i(x) > 0); the neuron's sign
    entropy is H(q_i) = -q log2 q - (1-q) log2 (1-q), so 0 means the neuron
    always has the same sign and 1 means it flips half the time.

    Base 2 is our choice (the protocol does not pin it) so the value is
    normalised to [0, 1]; recorded here so analysis code does not have to guess.
    Exactly-zero pre-activations count as non-positive.
    """
    q = (pre.detach() > 0).to(torch.float64).mean(dim=0)
    zero = torch.zeros((), dtype=q.dtype, device=q.device)
    t1 = torch.where(q > 0, q * torch.log2(q), zero)
    t2 = torch.where(q < 1, (1.0 - q) * torch.log2(1.0 - q), zero)
    return -(t1 + t2)


def sign_entropy(pre: torch.Tensor) -> float:
    """Layer-average sign entropy of pre-activations (CLAUDE.md §5.3)."""
    return float(sign_entropy_per_neuron(pre).mean())


def weight_stats(tensors: Iterable[torch.Tensor]) -> Tuple[float, float]:
    """(||theta||_2, mean |w|) over the given tensors, accumulated in float64.

    Dohare et al. report mean absolute weight; CLAUDE.md §5.3 asks for both.
    """
    sq = 0.0
    abs_sum = 0.0
    n = 0
    for t in tensors:
        x = _f64(t)
        sq += float(x.pow(2).sum())
        abs_sum += float(x.abs().sum())
        n += x.numel()
    l2 = float(np.sqrt(sq))
    mean_abs = float(abs_sum / n) if n else float("nan")
    return l2, mean_abs


def neuron_weight_norms(
    w_in: torch.Tensor, w_out: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-neuron incoming and outgoing weight norms for one hidden layer.

    ``w_in`` is (H, fan_in) -- the layer's own nn.Linear weight, so row i is
    neuron i's incoming weights. ``w_out`` is (H_next, H) -- the *next*
    layer's weight, so column i is neuron i's outgoing weights.
    """
    w_in_norm = _f64(w_in).pow(2).sum(dim=1).sqrt().cpu().numpy()
    w_out_norm = _f64(w_out).pow(2).sum(dim=0).sqrt().cpu().numpy()
    return w_in_norm, w_out_norm


# ---------------------------------------------------------------------------
# Per-layer probe result
# ---------------------------------------------------------------------------


@dataclass
class LayerProbe:
    """Every metric for one hidden layer, on one probe batch.

    Per-neuron fields are float64 numpy arrays of length ``n_neurons``; keeping
    them as arrays (not just aggregate counts) is what makes the C4 per-neuron
    log and the C1 composition table derivable from the same measurement.
    """

    layer_idx: int
    n_neurons: int
    n_inputs: int

    mean_abs_act: np.ndarray
    frac_inputs_active: np.ndarray
    exact_zero: np.ndarray  # bool
    sokar_score: np.ndarray
    preact_mean: np.ndarray
    preact_std: np.ndarray
    saturated: Optional[np.ndarray]  # bool, or None for unbounded activations

    erank: float
    sign_entropy: float

    taus: Tuple[float, ...]
    abs_thresholds: Tuple[float, ...]

    # ---- derived aggregates -------------------------------------------------

    @property
    def dead_exact_count(self) -> int:
        return int(self.exact_zero.sum())

    @property
    def dead_exact_frac(self) -> float:
        return float(self.exact_zero.mean())

    def dormant_count(self, tau: float) -> int:
        return int((self.sokar_score <= tau).sum())

    def dormant_frac(self, tau: float) -> float:
        return float((self.sokar_score <= tau).mean())

    def dormant_indices(self, tau: float) -> np.ndarray:
        return np.flatnonzero(self.sokar_score <= tau)

    def dead_absolute_count(self, a: float) -> int:
        return int((self.mean_abs_act < a).sum())

    def dead_absolute_frac(self, a: float) -> float:
        return float((self.mean_abs_act < a).mean())

    @property
    def saturated_frac(self) -> Optional[float]:
        # None, never 0.0, for unbounded activations (CLAUDE.md §5.1).
        return None if self.saturated is None else float(self.saturated.mean())


def probe_layer(
    pre: torch.Tensor,
    post: torch.Tensor,
    layer_idx: int,
    cfg: ProbeConfig,
    extremes: Optional[Sequence[float]] = None,
    compute_erank: bool = True,
) -> LayerProbe:
    """Compute every death metric for one layer from one probe batch.

    All four death definitions are computed from the *same* activation tensor in
    a single pass -- that simultaneity is the C2 measurement (CLAUDE.md §5.1).

    ``compute_erank=False`` skips only the SVD and records ``erank=nan``. It is
    used at recycling events, which need the death metrics for the composition
    table but not the effective rank; the SVD is by far the most expensive part
    of a probe and running it ~100 extra times per run buys nothing. Task-
    boundary probes always compute it.
    """
    if pre.shape != post.shape:
        raise ValueError(f"pre {tuple(pre.shape)} and post {tuple(post.shape)} differ")
    if post.dim() != 2:
        raise ValueError(f"expected (N, H) activations, got {tuple(post.shape)}")

    mean_abs = mean_abs_activation(post)
    scores = sokar_scores(mean_abs)
    sat = saturated_mask(post, extremes, cfg.saturation_eps)
    pre64 = _f64(pre)

    return LayerProbe(
        layer_idx=layer_idx,
        n_neurons=int(post.shape[1]),
        n_inputs=int(post.shape[0]),
        mean_abs_act=mean_abs.cpu().numpy(),
        frac_inputs_active=frac_inputs_active(post).cpu().numpy(),
        exact_zero=dead_exact_mask(post).cpu().numpy(),
        sokar_score=scores.cpu().numpy(),
        preact_mean=pre64.mean(dim=0).cpu().numpy(),
        # correction=0 (population std): deterministic and independent of N.
        preact_std=pre64.std(dim=0, correction=0).cpu().numpy(),
        saturated=None if sat is None else sat.cpu().numpy(),
        erank=effective_rank(post) if compute_erank else float("nan"),
        sign_entropy=sign_entropy(pre),
        taus=cfg.taus,
        abs_thresholds=cfg.abs_thresholds,
    )


@torch.no_grad()
def probe_model_and_logits(
    model,
    x: torch.Tensor,
    cfg: ProbeConfig,
    compute_erank: bool = True,
) -> Tuple[List[LayerProbe], torch.Tensor]:
    """Run the probe batch through the model and measure every hidden layer.

    The caller is responsible for ``model.eval()``. Probing in eval mode is
    deliberate: dropout and batch-norm training behaviour would make the
    activation statistics a property of the sampling noise rather than of the
    network.

    Logits are returned as well so that held-out probe accuracy comes from the
    same forward pass as the activation statistics -- one pass, one network
    state, no possibility of the two disagreeing.
    """
    logits, pres, posts = model.forward_with_activations(x)
    extremes = getattr(model, "activation_extremes", None)
    layers = [
        probe_layer(pre, post, i, cfg, extremes, compute_erank=compute_erank)
        for i, (pre, post) in enumerate(zip(pres, posts))
    ]
    return layers, logits


def probe_model(
    model,
    x: torch.Tensor,
    cfg: ProbeConfig,
    compute_erank: bool = True,
) -> List[LayerProbe]:
    """As `probe_model_and_logits`, discarding the logits."""
    return probe_model_and_logits(model, x, cfg, compute_erank)[0]


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy as a fraction. Ties in the argmax break toward the lowest
    class index, which is torch's behaviour and is not worth overriding."""
    if logits.shape[0] == 0:
        return float("nan")
    return float((logits.argmax(dim=1) == targets).to(torch.float64).mean())


# ---------------------------------------------------------------------------
# Recycled-set composition (the C1 headline measurement, CLAUDE.md §6)
# ---------------------------------------------------------------------------


@dataclass
class Composition:
    """What a recycled set was actually made of."""

    k: int
    n_dead_exact: int
    n_alive_but_quiet: int
    mean_sokar_score: float
    mean_abs_act: float

    def as_dict(self, suffix: str = "") -> dict:
        return {
            f"n_dead_exact{suffix}": self.n_dead_exact,
            f"n_alive_but_quiet{suffix}": self.n_alive_but_quiet,
            f"mean_sokar_score{suffix}": self.mean_sokar_score,
            f"mean_abs_act{suffix}": self.mean_abs_act,
        }


def composition(probe: LayerProbe, indices: np.ndarray) -> Composition:
    """Decompose a recycled set into genuinely-dead vs alive-but-quiet.

    ``n_dead_exact + n_alive_but_quiet == k`` always holds, because
    ``frac_inputs_active > 0`` is the exact complement of ``exact_zero``.

    NOTE on which batch this is measured against: ReDo *selects* neurons using
    a 64-example batch (Sokar et al.'s Algorithm 1 default). Composition is
    measured on the full fixed 2048-example probe batch instead, because
    "was this neuron genuinely dead" is a claim about the network, not about
    64 draws. The two batches are deliberately different; see interventions.py.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return Composition(0, 0, 0, float("nan"), float("nan"))
    dead = probe.exact_zero[idx]
    quiet = probe.frac_inputs_active[idx] > 0.0
    return Composition(
        k=int(idx.size),
        n_dead_exact=int(dead.sum()),
        n_alive_but_quiet=int(quiet.sum()),
        mean_sokar_score=float(probe.sokar_score[idx].mean()),
        mean_abs_act=float(probe.mean_abs_act[idx].mean()),
    )


# ---------------------------------------------------------------------------
# Gradient tracking
# ---------------------------------------------------------------------------


class GradientTracker:
    """Rolling window of gradient norms over the last `window` optimizer steps.

    CLAUDE.md §5.3 asks for the gradient norm averaged over the last 100 steps
    of the task; §5.4 asks for a per-neuron gradient norm. Both come from the
    same buffer so they cannot disagree.

    Per-neuron norm for neuron i of layer l is
        sqrt( ||dL/dW_l[i, :]||^2 + (dL/db_l[i])^2 )
    i.e. the gradient of everything flowing *into* that neuron.

    Call ``update()`` after ``loss.backward()`` and before ``optimizer.step()``.
    Everything stays on-device; there is no host sync until ``layer_mean`` /
    ``neuron_mean`` is called at the task boundary.
    """

    def __init__(
        self,
        linears: Sequence[torch.nn.Linear],
        window: int = GRAD_WINDOW,
        device: Optional[torch.device] = None,
    ):
        self.linears = list(linears)
        self.window = int(window)
        dev = device if device is not None else self.linears[0].weight.device
        self._layer_buf = [
            torch.zeros(self.window, device=dev) for _ in self.linears
        ]
        self._neuron_buf = [
            torch.zeros(self.window, lin.out_features, device=dev)
            for lin in self.linears
        ]
        self._ptr = 0
        self._count = 0

    def reset(self) -> None:
        """Clear the window. Called at each task start so that 'the last 100
        steps of the task' cannot bleed across a task boundary."""
        for b in self._layer_buf:
            b.zero_()
        for b in self._neuron_buf:
            b.zero_()
        self._ptr = 0
        self._count = 0

    @torch.no_grad()
    def update(self) -> None:
        i = self._ptr
        for j, lin in enumerate(self.linears):
            gw = lin.weight.grad
            if gw is None:
                self._neuron_buf[j][i].zero_()
                self._layer_buf[j][i].zero_()
                continue
            per_neuron_sq = gw.pow(2).sum(dim=1)
            gb = lin.bias.grad if lin.bias is not None else None
            if gb is not None:
                per_neuron_sq = per_neuron_sq + gb.pow(2)
            self._neuron_buf[j][i] = per_neuron_sq.sqrt()
            # Layer norm is the norm of the whole (W, b) gradient, which is the
            # sqrt of the sum of the per-neuron squared norms -- not the mean of
            # the per-neuron norms.
            self._layer_buf[j][i] = per_neuron_sq.sum().sqrt()
        self._ptr = (i + 1) % self.window
        self._count = min(self._count + 1, self.window)

    @property
    def n_steps_in_window(self) -> int:
        return self._count

    def layer_mean(self, j: int) -> float:
        if self._count == 0:
            return float("nan")
        return float(self._layer_buf[j][: self._count].to(torch.float64).mean())

    def neuron_mean(self, j: int) -> np.ndarray:
        if self._count == 0:
            return np.full(self.linears[j].out_features, np.nan, dtype=np.float64)
        buf = self._neuron_buf[j][: self._count].to(torch.float64)
        return buf.mean(dim=0).cpu().numpy()


# ---------------------------------------------------------------------------
# Row builders -- the on-disk schema
#
# These live here (not in train.py) because the column names encode metric
# parameterisation; keeping them beside the definitions is what stops the log
# schema and the metric set drifting apart.
#
# Convention for the two probe batches (CLAUDE.md §5, "compute every metric on
# both and log both"): unsuffixed columns are the CURRENT task's probe batch;
# ``*_ref`` columns are the fixed reference batch that never changes across the
# run. Weight- and gradient-derived columns are batch-independent and appear
# once.
# ---------------------------------------------------------------------------


def layer_metric_row(
    run_id: str,
    task_idx: int,
    probe_point: str,
    layer_idx: int,
    cur: LayerProbe,
    ref: Optional[LayerProbe],
    weight_l2: float,
    weight_mean_abs: float,
    grad_norm_layer: float,
    grad_window_steps: int,
) -> dict:
    """One row of metrics.parquet: (run_id, task_idx, layer_idx)."""
    row: Dict[str, object] = {
        "run_id": run_id,
        "task_idx": task_idx,
        "probe_point": probe_point,
        "layer_idx": layer_idx,
        "n_neurons": cur.n_neurons,
        "n_probe": cur.n_inputs,
        "weight_l2": weight_l2,
        "weight_mean_abs": weight_mean_abs,
        "grad_norm_layer": grad_norm_layer,
        "grad_window_steps": grad_window_steps,
    }

    def _fill(p: LayerProbe, suffix: str) -> None:
        row[f"dead_exact_frac{suffix}"] = p.dead_exact_frac
        row[f"dead_exact_count{suffix}"] = p.dead_exact_count
        for tau in p.taus:
            row[f"{dormant_col(tau)}{suffix}"] = p.dormant_frac(tau)
        for a in p.abs_thresholds:
            row[f"{dead_abs_col(a)}{suffix}"] = p.dead_absolute_frac(a)
        row[f"saturated_frac{suffix}"] = p.saturated_frac  # None -> null in parquet
        row[f"erank{suffix}"] = p.erank
        row[f"sign_entropy{suffix}"] = p.sign_entropy
        row[f"mean_abs_act_layer{suffix}"] = float(p.mean_abs_act.mean())

    _fill(cur, "")
    if ref is not None:
        _fill(ref, "_ref")
    return row


#: Exact column order of neurons.parquet. The first block is the schema
#: mandated by CLAUDE.md §5.4 verbatim; the ``_ref`` block is the same
#: measurement on the reference probe batch (CLAUDE.md §5: "log both").
NEURON_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "task_idx",
    "layer_idx",
    "neuron_idx",
    "mean_abs_act",
    "frac_inputs_active",
    "exact_zero_flag",
    "sokar_score",
    "preact_mean",
    "preact_std",
    "w_in_norm",
    "w_out_norm",
    "bias",
    "grad_norm_neuron",
    "was_recycled_this_task",
    # -- reference probe batch --
    "mean_abs_act_ref",
    "frac_inputs_active_ref",
    "exact_zero_flag_ref",
    "sokar_score_ref",
    "preact_mean_ref",
    "preact_std_ref",
    # -- bookkeeping --
    "saturated_flag",
    "probe_point",
)


def neuron_rows(
    run_id: str,
    task_idx: int,
    probe_point: str,
    layer_idx: int,
    cur: LayerProbe,
    ref: Optional[LayerProbe],
    w_in_norm: np.ndarray,
    w_out_norm: np.ndarray,
    bias: np.ndarray,
    grad_norm_neuron: np.ndarray,
    was_recycled: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Columnar block of neurons.parquet rows for one (task, layer).

    One row per (run, task, layer, neuron), per CLAUDE.md §5.4. This dataset is
    not recoverable retrospectively -- a run that completes without it is a
    wasted run, so treat any failure here as fatal, never a warning.
    """
    h = cur.n_neurons
    for name, arr in (
        ("w_in_norm", w_in_norm),
        ("w_out_norm", w_out_norm),
        ("bias", bias),
        ("grad_norm_neuron", grad_norm_neuron),
        ("was_recycled", was_recycled),
    ):
        if len(arr) != h:
            raise ValueError(f"{name} has length {len(arr)}, expected {h}")

    cols: Dict[str, np.ndarray] = {
        "run_id": np.full(h, run_id, dtype=object),
        "task_idx": np.full(h, task_idx, dtype=np.int32),
        "layer_idx": np.full(h, layer_idx, dtype=np.int32),
        "neuron_idx": np.arange(h, dtype=np.int32),
        "mean_abs_act": cur.mean_abs_act.astype(np.float64),
        "frac_inputs_active": cur.frac_inputs_active.astype(np.float64),
        "exact_zero_flag": cur.exact_zero.astype(bool),
        "sokar_score": cur.sokar_score.astype(np.float64),
        "preact_mean": cur.preact_mean.astype(np.float64),
        "preact_std": cur.preact_std.astype(np.float64),
        "w_in_norm": np.asarray(w_in_norm, dtype=np.float64),
        "w_out_norm": np.asarray(w_out_norm, dtype=np.float64),
        "bias": np.asarray(bias, dtype=np.float64),
        "grad_norm_neuron": np.asarray(grad_norm_neuron, dtype=np.float64),
        "was_recycled_this_task": np.asarray(was_recycled, dtype=bool),
        "saturated_flag": (
            np.full(h, None, dtype=object)
            if cur.saturated is None
            else cur.saturated.astype(bool)
        ),
        "probe_point": np.full(h, probe_point, dtype=object),
    }
    if ref is not None:
        cols.update(
            {
                "mean_abs_act_ref": ref.mean_abs_act.astype(np.float64),
                "frac_inputs_active_ref": ref.frac_inputs_active.astype(np.float64),
                "exact_zero_flag_ref": ref.exact_zero.astype(bool),
                "sokar_score_ref": ref.sokar_score.astype(np.float64),
                "preact_mean_ref": ref.preact_mean.astype(np.float64),
                "preact_std_ref": ref.preact_std.astype(np.float64),
            }
        )
    else:
        nan = np.full(h, np.nan, dtype=np.float64)
        cols.update(
            {
                "mean_abs_act_ref": nan.copy(),
                "frac_inputs_active_ref": nan.copy(),
                "exact_zero_flag_ref": np.full(h, None, dtype=object),
                "sokar_score_ref": nan.copy(),
                "preact_mean_ref": nan.copy(),
                "preact_std_ref": nan.copy(),
            }
        )
    return {name: cols[name] for name in NEURON_COLUMNS}
