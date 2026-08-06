"""Recycling interventions and regularisers.

Contains ReDo (Sokar et al. 2023), our size-matched random control, the
inverse-matched sanity check, L2, and shrink-and-perturb.

Two things in here are load-bearing for the paper:

**Zeroing the outgoing weights.** ReDo re-initialises a recycled neuron's
incoming weights and sets its *outgoing* weights to zero; that is what makes the
event function-preserving for a genuinely dead unit. Applying the mask to
incoming weights only is a known reimplementation bug that turns ReDo into a far
more destructive intervention while still producing plausible curves
(CLAUDE.md §6). ``tests/test_interventions.py::test_redo_preserves_function``
exists solely to catch it.

**Per-event, per-layer cardinality matching.** ``k`` is recomputed at every
event for every layer. Sokar et al.'s own random baseline used a fixed
percentage on a cosine schedule, which confounds *which* neurons are recycled
with *how many*. Removing that confound is the crux of C1, so ``k`` must never
become a schedule or a fixed fraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import probes
from .probes import LayerProbe, ProbeConfig

#: Recycling arms of the primary experiment (protocol §B.1).
RECYCLE_KINDS = ("none", "redo", "random_matched", "inverse_matched")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_recycle_indices(
    kind: str,
    scores: np.ndarray,
    tau: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose which neurons of one layer to recycle at one event.

    Returns ``(selected_indices, dormant_indices)``; both sorted ascending.
    Every arm recycles exactly ``k = |dormant set|`` neurons, so the arms differ
    only in *which* neurons, never in *how many*.
    """
    scores = np.asarray(scores, dtype=np.float64)
    h = scores.size
    dormant = np.flatnonzero(scores <= tau)  # Sokar: tau-dormant iff s <= tau
    k = int(dormant.size)

    if kind == "none" or k == 0:
        return np.empty(0, dtype=np.int64), dormant
    if kind == "redo":
        return dormant.astype(np.int64), dormant
    if kind == "random_matched":
        # Uniformly at random from the whole layer -- not from the complement of
        # the dormant set. The control asks "does recycling k arbitrary neurons
        # do as well as recycling the k quietest", so the draw must be over all
        # neurons.
        return np.sort(rng.choice(h, size=k, replace=False)).astype(np.int64), dormant
    if kind == "inverse_matched":
        # k highest-scoring neurons. Stable sort so ties break by index and the
        # arm is reproducible.
        order = np.argsort(-scores, kind="stable")
        return np.sort(order[:k]).astype(np.int64), dormant
    raise ValueError(f"unknown recycling kind {kind!r}; known: {RECYCLE_KINDS}")


# ---------------------------------------------------------------------------
# Applying a recycle
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reset_optimizer_slice(
    optimizer: Optional[torch.optim.Optimizer],
    param: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    spatial: int = 1,
) -> None:
    """Zero the optimizer's per-parameter state for the recycled slice.

    Sokar et al.'s Algorithm 1 resets the optimizer state of the recycled
    weights; without it, a freshly re-initialised neuron inherits the momentum
    (or Adam moments) of the dead neuron it replaced and is immediately dragged
    back down.

    Adam's ``step`` counter is a per-parameter scalar and cannot be reset for a
    slice without also resetting it for the untouched weights of the same
    tensor, which would change their bias correction. It is therefore left
    alone -- a deliberate, documented deviation.
    """
    if optimizer is None:
        return
    state = optimizer.state.get(param)
    if not state:
        return
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != param.shape:
            continue  # 'step' scalars and anything else non-conformant
        if dim == 0:
            value[index] = 0
        elif dim == 1:
            if spatial == 1:
                value[:, index] = 0
            else:
                # conv -> flatten -> Linear: mirror the weight slicing exactly,
                # or the moments of the wrong columns get cleared.
                for c in index.tolist():
                    value[:, flattened_channel_columns(int(c), spatial)] = 0
        else:
            raise ValueError(f"unsupported dim {dim}")


def flattened_channel_columns(channel: int, spatial: int) -> slice:
    """Columns of a post-flatten Linear that belong to one conv channel.

    A feature map flattened with ``.reshape(N, -1)`` is channel-major, so
    channel ``c`` owns ``[c*spatial, (c+1)*spatial)``.

    Isolated into its own function because CLAUDE.md §5.5 singles this out:
    "getting that indexing wrong will silently zero the wrong columns". An
    off-by-``spatial`` here recycles one channel and blanks another's outgoing
    weights, and nothing raises.
    """
    return slice(channel * spatial, (channel + 1) * spatial)


@torch.no_grad()
def _zero_outgoing(module, idx_t: torch.Tensor, spatial: int) -> None:
    """Zero the outgoing weights of the given units of the previous layer."""
    if spatial == 1:
        # Linear (H_next, H) or Conv2d (C_next, C, kH, kW): unit is axis 1.
        module.weight[:, idx_t] = 0.0
        return
    # conv -> flatten -> Linear: each unit owns `spatial` contiguous columns.
    for c in idx_t.tolist():
        module.weight[:, flattened_channel_columns(int(c), spatial)] = 0.0


@torch.no_grad()
def recycle_neurons(
    model,
    layer_idx: int,
    indices: np.ndarray,
    weight_generator: Optional[torch.Generator] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    reset_optimizer_state: bool = True,
) -> int:
    """Re-initialise the given units of one hidden layer, in place.

    Sokar et al. 2023, Algorithm 1:
      * incoming weights re-sampled from the layer's *original* init
        distribution, bias reset to its init value;
      * outgoing weights set to **zero**.

    Works for a fully-connected neuron and for a conv channel, where "incoming
    weights" is the whole filter bank ``W[c, :, :, :]`` and the outgoing slice
    may span every spatial position belonging to that channel (CLAUDE.md §5.5).

    Returns the number of units recycled.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return 0

    mod_in = model.incoming_linear(layer_idx)
    mod_out = model.outgoing_linear(layer_idx)
    spatial = getattr(model, "outgoing_spatial", lambda _: 1)(layer_idx)
    device = mod_in.weight.device
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)

    # --- incoming ---------------------------------------------------------
    new_w = model.sample_incoming_weights(layer_idx, int(idx.size), weight_generator)
    mod_in.weight[idx_t] = new_w.to(device=device, dtype=mod_in.weight.dtype)
    if mod_in.bias is not None:
        mod_in.bias[idx_t] = model.init_bias_value(layer_idx)

    # --- outgoing: THE half that gets forgotten ---------------------------
    _zero_outgoing(mod_out, idx_t, spatial)
    # The outgoing layer's *bias* is untouched: it is not a property of this
    # unit, and zeroing it would change the function for every other unit.

    if reset_optimizer_state:
        _reset_optimizer_slice(optimizer, mod_in.weight, idx_t, dim=0)
        if mod_in.bias is not None:
            _reset_optimizer_slice(optimizer, mod_in.bias, idx_t, dim=0)
        _reset_optimizer_slice(
            optimizer, mod_out.weight, idx_t, dim=1, spatial=spatial
        )

    return int(idx.size)


# ---------------------------------------------------------------------------
# The recycler
# ---------------------------------------------------------------------------


@dataclass
class RecyclerConfig:
    kind: str = "none"
    tau: float = 0.0
    freq: int = 1000  # F = 1000 optimizer steps (Sokar et al. 2023)
    score_batch_size: int = 64  # Sokar et al.'s default
    reset_optimizer_state: bool = True
    composition_on_reference: bool = True

    def __post_init__(self):
        if self.kind not in RECYCLE_KINDS:
            raise ValueError(f"unknown recycling kind {self.kind!r}; known: {RECYCLE_KINDS}")
        if self.freq <= 0:
            raise ValueError("recycling freq must be positive")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RecyclerConfig":
        d = dict(d or {})
        return cls(
            kind=d.get("kind", "none"),
            tau=float(d.get("tau", 0.0)),
            freq=int(d.get("freq", 1000)),
            score_batch_size=int(d.get("score_batch_size", 64)),
            reset_optimizer_state=bool(d.get("reset_optimizer_state", True)),
            composition_on_reference=bool(d.get("composition_on_reference", True)),
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "tau": self.tau,
            "freq": self.freq,
            "score_batch_size": self.score_batch_size,
            "reset_optimizer_state": self.reset_optimizer_state,
            "composition_on_reference": self.composition_on_reference,
        }


@dataclass
class EventResult:
    """Outcome of one recycling event, across all hidden layers."""

    event_idx: int
    step: int
    task_idx: int
    rows: List[dict] = field(default_factory=list)
    recycled: Dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def total_recycled(self) -> int:
        return int(sum(v.size for v in self.recycled.values()))


class Recycler:
    """Runs a recycling event every F optimizer steps.

    The intervention and its measurement use deliberately different batches:

    * **selection** uses a fresh 64-example batch from the current task, exactly
      as Sokar et al.'s Algorithm 1 specifies;
    * **composition logging** uses the fixed 2048-example probe batch, because
      "was this neuron genuinely dead" is a claim about the network, not about
      64 draws. Sixty-four samples would systematically over-report
      ``dead_exact`` and inflate exactly the number C1 is about.

    Both random streams (weight resampling, random-matched selection) are
    dedicated generators, checkpointed with the run, so recycling is
    reproducible independently of anything else that consumes randomness.
    """

    def __init__(
        self,
        cfg: RecyclerConfig,
        seed: int,
        probe_cfg: Optional[ProbeConfig] = None,
        run_id: str = "",
    ):
        self.cfg = cfg
        self.run_id = run_id
        self.probe_cfg = probe_cfg or ProbeConfig()
        # CPU generators: recycling must be bit-identical whether a resumed run
        # lands on a T4 or a laptop.
        self._weight_gen = torch.Generator()
        self._weight_gen.manual_seed(int(seed) ^ 0x5EED_1)
        self._select_rng = np.random.default_rng([int(seed), 0x5EED_2])
        self.event_idx = 0

    @property
    def enabled(self) -> bool:
        return self.cfg.kind != "none"

    def due(self, step: int) -> bool:
        """True on steps 1000, 2000, ... (never on step 0: an event before any
        training would recycle the initialisation itself)."""
        return self.enabled and step > 0 and step % self.cfg.freq == 0

    @torch.no_grad()
    def run_event(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer],
        score_x: torch.Tensor,
        probe_x: torch.Tensor,
        step: int,
        task_idx: int,
        ref_x: Optional[torch.Tensor] = None,
    ) -> EventResult:
        was_training = model.training
        model.eval()
        try:
            # 1. Scores, from the 64-example batch (Sokar Algorithm 1).
            _, _, score_posts = model.forward_with_activations(score_x)
            # as_unit_matrix folds a conv layer's (N, C, H, W) to (N*H*W, C) so
            # the unit is the channel. Without it a conv layer yields C*H*W
            # "units" and every index downstream is meaningless.
            layer_scores = [
                probes.sokar_scores(
                    probes.mean_abs_activation(probes.as_unit_matrix(p))
                )
                .cpu()
                .numpy()
                for p in score_posts
            ]

            # 2. Composition, from the full probe batch, BEFORE any weights move.
            # compute_erank=False: the composition table does not use effective
            # rank, and the SVD dominates probe cost.
            cur_probes = probes.probe_model(
                model, probe_x, self.probe_cfg, compute_erank=False
            )
            ref_probes = (
                probes.probe_model(model, ref_x, self.probe_cfg, compute_erank=False)
                if (ref_x is not None and self.cfg.composition_on_reference)
                else None
            )

            result = EventResult(event_idx=self.event_idx, step=step, task_idx=task_idx)

            for layer_idx, scores in enumerate(layer_scores):
                selected, dormant = select_recycle_indices(
                    self.cfg.kind, scores, self.cfg.tau, self._select_rng
                )
                comp = probes.composition(cur_probes[layer_idx], selected)
                row = {
                    "run_id": self.run_id,
                    "event_idx": self.event_idx,
                    "step": step,
                    "task_idx": task_idx,
                    "arm": self.cfg.kind,
                    "layer_idx": layer_idx,
                    "tau": self.cfg.tau,
                    "k": comp.k,
                    "n_neurons": int(scores.size),
                    "n_dormant": int(dormant.size),
                    "score_batch_size": int(score_x.shape[0]),
                    # Mean Sokar score over the *dormant* set on the selection
                    # batch, for reference; the headline number is the mean
                    # score of the neurons actually recycled, below.
                    "mean_sokar_score_dormant_scorebatch": (
                        float(scores[dormant].mean()) if dormant.size else float("nan")
                    ),
                }
                row.update(comp.as_dict())
                if ref_probes is not None:
                    row.update(
                        probes.composition(ref_probes[layer_idx], selected).as_dict("_ref")
                    )
                result.rows.append(row)

                # 3. Apply.
                recycle_neurons(
                    model,
                    layer_idx,
                    selected,
                    weight_generator=self._weight_gen,
                    optimizer=optimizer,
                    reset_optimizer_state=self.cfg.reset_optimizer_state,
                )
                result.recycled[layer_idx] = selected

            self.event_idx += 1
            return result
        finally:
            if was_training:
                model.train()

    # -- checkpointing --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "event_idx": self.event_idx,
            "weight_gen": self._weight_gen.get_state(),
            "select_rng": self._select_rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.event_idx = int(state["event_idx"])
        self._weight_gen.set_state(state["weight_gen"])
        self._select_rng.bit_generator.state = state["select_rng"]


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------


def l2_penalty(model, include_bias: bool = True) -> torch.Tensor:
    """0.5 * sum(theta^2) over the model's parameters.

    Added to the loss as ``lambda * l2_penalty(model)``, whose gradient is
    ``lambda * theta`` -- identical to passing ``weight_decay=lambda`` to SGD,
    but written out so it is visible in the loss and cannot be confused with
    AdamW's *decoupled* decay. That distinction matters for the C5 AdamW arm
    (protocol §B.3).
    """
    total = None
    for name, p in model.named_parameters():
        if not include_bias and name.endswith("bias"):
            continue
        s = p.pow(2).sum()
        total = s if total is None else total + s
    if total is None:
        raise ValueError("model has no parameters")
    return 0.5 * total


# ---------------------------------------------------------------------------
# Shrink and perturb
# ---------------------------------------------------------------------------


@dataclass
class ShrinkPerturbConfig:
    """Ash & Adams 2020, applied continually (Dohare et al. Fig. 4b arm).

    ``theta <- shrink * theta + perturb * nu``, ``nu`` drawn from the layer's
    original initialisation distribution.
    """

    enabled: bool = False
    shrink: float = 0.5
    perturb: float = 0.01
    every_tasks: int = 1  # applied at task boundaries

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ShrinkPerturbConfig":
        d = dict(d or {})
        return cls(
            enabled=bool(d.get("enabled", False)),
            shrink=float(d.get("shrink", 0.5)),
            perturb=float(d.get("perturb", 0.01)),
            every_tasks=int(d.get("every_tasks", 1)),
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "shrink": self.shrink,
            "perturb": self.perturb,
            "every_tasks": self.every_tasks,
        }


@torch.no_grad()
def shrink_and_perturb(
    model, shrink: float, perturb: float, generator: Optional[torch.Generator] = None
) -> None:
    """Apply shrink-and-perturb to every Linear in the model, in place.

    Biases are shrunk but not perturbed: the bias initialisation distribution is
    the constant 0, so "perturb with noise from the init distribution" adds
    nothing for them.
    """
    for lin, spec in zip(model.linears, model.init_specs):
        noise = spec.sample(tuple(lin.weight.shape), generator).to(
            device=lin.weight.device, dtype=lin.weight.dtype
        )
        lin.weight.mul_(shrink).add_(noise, alpha=perturb)
        if lin.bias is not None:
            lin.bias.mul_(shrink)
