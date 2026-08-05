"""The MLP under study. Activation and normalisation are config fields.

Reference architecture (protocol §A.4): ``784 -> 500 -> 500 -> 500 -> 10``,
ReLU, Kaiming init. Width 500 rather than Dohare et al.'s 2000 is deliberate --
they report plasticity loss is most pronounced at smaller widths (their Fig. 2b,
middle panel) -- not a compromise.

Two things here exist purely to serve the interventions:

1. Each Linear stores the ``InitSpec`` it was initialised from, so ReDo can
   re-initialise a recycled neuron "by sampling from the *original*
   initialisation distribution for that layer" (CLAUDE.md §6) rather than from
   whatever PyTorch's default would be today.
2. ``hidden_linears`` / ``outgoing_linear`` expose the (incoming, outgoing)
   weight pair for each hidden layer, because getting the *outgoing* half wrong
   is the known reimplementation bug that silently turns ReDo into a much more
   destructive intervention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

#: Extreme values of bounded activations, used by `probes.saturated_mask`.
#: ``None`` means unbounded -- saturation is not defined and must be reported as
#: None, never 0 (CLAUDE.md §5.1).
ACTIVATION_EXTREMES: Dict[str, Optional[Tuple[float, ...]]] = {
    "relu": None,
    "leaky_relu": None,
    "identity": None,
    "tanh": (-1.0, 1.0),
    "sigmoid": (0.0, 1.0),
}


def make_activation(name: str, param: float = 0.01) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        # `param` is the negative slope, the epsilon of the demoted eps-sweep
        # (protocol §B.4).
        return nn.LeakyReLU(negative_slope=param)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"unknown activation {name!r}; known: {sorted(ACTIVATION_EXTREMES)}")


def _init_gain(activation: str, param: float) -> float:
    """Kaiming gain for an activation, via torch's own table so our numbers
    match anything else initialised with `nn.init.kaiming_*`."""
    if activation == "relu":
        return nn.init.calculate_gain("relu")
    if activation == "leaky_relu":
        return nn.init.calculate_gain("leaky_relu", param)
    if activation == "tanh":
        return nn.init.calculate_gain("tanh")
    if activation in ("sigmoid", "identity"):
        # torch's gain for sigmoid is 1.0, same as linear.
        return nn.init.calculate_gain("linear")
    raise ValueError(f"unknown activation {activation!r}")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InitSpec:
    """The exact distribution a layer's weights were drawn from.

    Stored on the model so a recycling event can resample from it years later
    (well, tasks later) without re-deriving anything. ``kind`` matches PyTorch's
    ``nn.init.kaiming_uniform_`` / ``kaiming_normal_`` in ``fan_in`` mode.
    """

    kind: str  # 'kaiming_uniform' | 'kaiming_normal'
    gain: float
    fan_in: int
    bias_value: float = 0.0

    @property
    def std(self) -> float:
        return self.gain / math.sqrt(self.fan_in)

    @property
    def bound(self) -> float:
        # kaiming_uniform_ draws from U(-bound, bound) with bound = sqrt(3)*std.
        return math.sqrt(3.0) * self.std

    def sample(
        self, shape: Sequence[int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Draw weights from the original distribution.

        Always sampled on CPU (in float32) and moved by the caller. Sampling on
        CPU with an explicit generator makes recycling bit-identical whether the
        run is on a T4 or on a laptop, which matters because a resumed run may
        not land on the same hardware.
        """
        shape = tuple(int(s) for s in shape)
        if self.kind == "kaiming_uniform":
            u = torch.rand(shape, generator=generator, dtype=torch.float32)
            return (u * 2.0 - 1.0) * self.bound
        if self.kind == "kaiming_normal":
            return torch.randn(shape, generator=generator, dtype=torch.float32) * self.std
        raise ValueError(f"unknown init kind {self.kind!r}")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "gain": self.gain,
            "fan_in": self.fan_in,
            "bias_value": self.bias_value,
            "std": self.std,
            "bound": self.bound,
        }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def make_norm(name: str, dim: int) -> nn.Module:
    if name in ("none", None):
        return nn.Identity()
    if name == "layer":
        return nn.LayerNorm(dim)
    if name == "batch":
        return nn.BatchNorm1d(dim)
    if name == "online":
        # Online Normalization (Chiley et al. 2019), the "online norm" arm of
        # Dohare et al. Fig. 4b and therefore of C3 (protocol §B.2).
        #
        # Deliberately NOT implemented from memory. Its forward pass keeps
        # per-sample running estimates of mean/variance AND its backward pass
        # applies two separate control processes; an approximation that gets the
        # backward controls wrong would still train, still produce a plausible
        # dead-unit curve, and silently invalidate C3. Implement it against the
        # paper (and ideally against the reference implementation) before
        # running §B.2.
        raise NotImplementedError(
            "norm='online' (Chiley et al. 2019 Online Normalization) is not "
            "implemented yet. Required for the C3 arm in protocol §B.2; not "
            "required for the Week-1 gate. See src/models.py:make_norm."
        )
    raise ValueError(f"unknown norm {name!r}; known: none, layer, batch, online")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Feed-forward net with explicit access to per-layer activations.

    Layer ``i`` (0-indexed over hidden layers) is computed as

        z_i = norm_i( linear_i( h_{i-1} ) )      <- "pre-activation"
        a_i = act( z_i )                          <- "post-activation"
        h_i = dropout( a_i )

    The probe measures ``z_i`` (sign entropy, preact stats) and ``a_i``
    (everything else). Dropout is applied after ``a_i`` so the recorded
    activations are the network's, not the mask's; probes run in eval mode
    anyway.

    Activations are returned explicitly rather than captured with forward hooks:
    hooks are easy to leave attached, easy to double-register, and silently
    return stale tensors -- none of which is worth the tidiness here.
    """

    def __init__(
        self,
        in_features: int = 784,
        hidden_dims: Sequence[int] = (500, 500, 500),
        out_features: int = 10,
        activation: str = "relu",
        activation_param: float = 0.01,
        norm: str = "none",
        dropout: float = 0.0,
        init: str = "kaiming_uniform",
        output_nonlinearity: Optional[str] = None,
        bias_init: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("MLP needs at least one hidden layer")
        if init not in ("kaiming_uniform", "kaiming_normal"):
            raise ValueError(f"unknown init {init!r}")

        self.in_features = int(in_features)
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.out_features = int(out_features)
        self.activation_name = activation
        self.activation_param = float(activation_param)
        self.norm_name = norm
        self.dropout_p = float(dropout)
        self.init_kind = init
        self.bias_init = float(bias_init)

        dims = [self.in_features, *self.hidden_dims, self.out_features]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.norms = nn.ModuleList(
            [make_norm(norm, d) for d in self.hidden_dims]
        )
        self.act = make_activation(activation, self.activation_param)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # Dohare et al.'s reference implementation initialises every layer,
        # including the output layer, with the hidden nonlinearity's gain. We
        # keep that as the default and expose it so the choice is visible rather
        # than buried.
        out_nl = output_nonlinearity or activation
        gains = [_init_gain(activation, self.activation_param)] * len(self.hidden_dims)
        gains.append(_init_gain(out_nl, self.activation_param))
        self.output_nonlinearity = out_nl

        self.init_specs: List[InitSpec] = [
            InitSpec(kind=init, gain=g, fan_in=dims[i], bias_value=self.bias_init)
            for i, g in enumerate(gains)
        ]
        self.reset_parameters(generator)

    # -- construction ---------------------------------------------------------

    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        """Initialise every Linear from its stored InitSpec, on CPU.

        Biases are zeroed (Dohare et al.). Normalisation affine parameters keep
        their module defaults (weight=1, bias=0).
        """
        with torch.no_grad():
            for lin, spec in zip(self.linears, self.init_specs):
                w = spec.sample(tuple(lin.weight.shape), generator)
                lin.weight.copy_(w.to(lin.weight.device))
                if lin.bias is not None:
                    lin.bias.fill_(spec.bias_value)

    # -- structure ------------------------------------------------------------

    @property
    def n_hidden(self) -> int:
        return len(self.hidden_dims)

    @property
    def hidden_linears(self) -> List[nn.Linear]:
        """The Linear whose *outputs* are hidden layer i's neurons."""
        return [self.linears[i] for i in range(self.n_hidden)]

    def incoming_linear(self, layer_idx: int) -> nn.Linear:
        self._check_hidden(layer_idx)
        return self.linears[layer_idx]

    def outgoing_linear(self, layer_idx: int) -> nn.Linear:
        """The Linear that *consumes* hidden layer i. For the last hidden layer
        this is the output layer -- which is exactly the case a buggy ReDo
        implementation forgets."""
        self._check_hidden(layer_idx)
        return self.linears[layer_idx + 1]

    def _check_hidden(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.n_hidden:
            raise IndexError(
                f"hidden layer {layer_idx} out of range (0..{self.n_hidden - 1})"
            )

    @property
    def activation_extremes(self) -> Optional[Tuple[float, ...]]:
        return ACTIVATION_EXTREMES[self.activation_name]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- forward --------------------------------------------------------------

    def forward_with_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Returns (logits, pre_activations, post_activations)."""
        h = x
        pres: List[torch.Tensor] = []
        posts: List[torch.Tensor] = []
        for i in range(self.n_hidden):
            z = self.norms[i](self.linears[i](h))
            pres.append(z)
            a = self.act(z)
            posts.append(a)
            h = self.dropout(a)
        return self.linears[-1](h), pres, posts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_activations(x)
        return logits

    # -- recycling support ----------------------------------------------------

    def sample_incoming_weights(
        self, layer_idx: int, k: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """(k, fan_in) fresh weights from hidden layer `layer_idx`'s *original*
        init distribution (CLAUDE.md §6)."""
        self._check_hidden(layer_idx)
        spec = self.init_specs[layer_idx]
        return spec.sample((k, spec.fan_in), generator)

    def init_bias_value(self, layer_idx: int) -> float:
        self._check_hidden(layer_idx)
        return self.init_specs[layer_idx].bias_value

    # -- serialisation --------------------------------------------------------

    def describe(self) -> dict:
        return {
            "in_features": self.in_features,
            "hidden_dims": list(self.hidden_dims),
            "out_features": self.out_features,
            "activation": self.activation_name,
            "activation_param": self.activation_param,
            "norm": self.norm_name,
            "dropout": self.dropout_p,
            "init": self.init_kind,
            "output_nonlinearity": self.output_nonlinearity,
            "bias_init": self.bias_init,
            "n_parameters": self.n_parameters(),
            "init_specs": [s.to_dict() for s in self.init_specs],
        }


def build_model(cfg: dict, generator: Optional[torch.Generator] = None) -> MLP:
    """Construct an MLP from the ``model`` block of a run config."""
    cfg = dict(cfg or {})
    return MLP(
        in_features=cfg.get("in_features", 784),
        hidden_dims=cfg.get("hidden_dims", (500, 500, 500)),
        out_features=cfg.get("out_features", 10),
        activation=cfg.get("activation", "relu"),
        activation_param=cfg.get("activation_param", 0.01),
        norm=cfg.get("norm", "none"),
        dropout=cfg.get("dropout", 0.0),
        init=cfg.get("init", "kaiming_uniform"),
        output_nonlinearity=cfg.get("output_nonlinearity"),
        bias_init=cfg.get("bias_init", 0.0),
        generator=generator,
    )
