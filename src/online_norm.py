"""Online Normalization (Chiley et al., NeurIPS 2019, arXiv:1905.05894).

Implemented against the paper, not from memory -- this is the `online-norm` arm
of C3 (protocol §B.2), where the claim under test is that a method *designed to
prevent* dead units *increases* them. An approximation that trained fine but got
the dynamics wrong would produce a plausible dead-unit curve and silently
invalidate that result, which is why `models.make_norm` refused to guess.

The load-bearing detail, and the reason this needs a custom autograd.Function:
**the backward pass is not the derivative of the forward pass.** Online Norm
replaces it with a control process that keeps back-propagated gradients within a
bounded distance of the true gradients of the ideal normalizer. Writing the
forward with running statistics and letting autograd differentiate it gives a
different algorithm with the same forward behaviour.

Equations, with the paper's numbering, per feature:

    forward
      y_t   = (x_t - mu_{t-1}) / sigma_{t-1}                               (8a)
      mu_t  = a_f*mu_{t-1} + (1 - a_f)*mu(x_t)                             (8b)
      var_t = a_f*var_{t-1} + (1 - a_f)*var(x_t)
              + a_f*(1 - a_f)*(mu(x_t) - mu_{t-1})^2                       (8c)

    layer scaling, over all features (needed for unbounded activations)
      z_t   = y_t / zeta_t,   zeta_t = sqrt(mean({y_t^2}))                 (9)

    backward, in reverse order
      y'_t  = (z'_t - z_t*mean({z_t z'_t})) / zeta_t                       (10)
      xt'_t = y'_t - (1 - a_b)*eps_y_{t-1}*y_t                            (11a)
      eps_y_t = eps_y_{t-1} + mean(xt'_t * y_t)                           (11b)
      x'_t  = xt'_t / sigma_{t-1} - (1 - a_b)*eps_1_{t-1}                 (12a)
      eps_1_t = eps_1_{t-1} + mean(x'_t)                                  (12b)

`mu(x_t)` and `var(x_t)` are the *feature-wide* statistics of one sample: for a
conv feature map, the spatial mean and variance; for a fully-connected layer
with one scalar per feature, `mu(x_t) = x_t` and `var(x_t) = 0`.

**Documented deviation -- batch semantics.** The paper defines the recurrence
per sample, with each sample in a batch normalised by the state left by the one
before it. Our runs use batch 128, and applying 128 sequential state updates per
layer per step would dominate the run. This module advances the state **once per
batch**, taking `mu(x_t)` and `var(x_t)` over the batch and spatial axes
together. That is exact for batch size 1 (asserted in
`tests/test_online_norm.py`) and is the same simplification the reference
implementation makes when handed a batch. It must be stated in the paper
alongside any C3 online-norm result.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

#: Reference-implementation defaults (Cerebras/online-normalization).
DEFAULT_ALPHA_FWD = 0.999
DEFAULT_ALPHA_BKW = 0.99
DEFAULT_EPS = 1e-5


def _feature_reduce_dims(x: torch.Tensor) -> Tuple[int, ...]:
    """Axes to reduce over so that only the feature axis (1) survives."""
    return (0,) + tuple(range(2, x.dim()))


def _expand(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-feature vector against x's shape."""
    return v.reshape((1, -1) + (1,) * (x.dim() - 2))


class _OnlineNormFn(torch.autograd.Function):
    """Forward (8a-8c) with a control-process backward (11a-12b).

    State tensors are mutated in place and are NOT part of the autograd graph:
    they are estimator state, exactly as the running statistics of BatchNorm are.
    """

    @staticmethod
    def forward(ctx, x, mu, var, eps_y, eps_1, alpha_f, alpha_b, eps, training):
        dims = _feature_reduce_dims(x)

        if training:
            # (8a) normalise with the state from BEFORE this step.
            scale = torch.sqrt(var + eps)
            y = (x - _expand(mu, x)) / _expand(scale, x)

            batch_mu = x.mean(dim=dims)
            # Population variance: this is a statistic of the sample, not an
            # estimate of a wider population.
            batch_var = x.var(dim=dims, unbiased=False)

            # (8b), (8c) -- both computed from mu_{t-1}, so update var FIRST.
            var.mul_(alpha_f).add_(
                (1.0 - alpha_f) * batch_var
                + alpha_f * (1.0 - alpha_f) * (batch_mu - mu).pow(2)
            )
            mu.mul_(alpha_f).add_((1.0 - alpha_f) * batch_mu)
        else:
            scale = torch.sqrt(var + eps)
            y = (x - _expand(mu, x)) / _expand(scale, x)

        ctx.save_for_backward(y, scale, eps_y, eps_1)
        ctx.alpha_b = float(alpha_b)
        ctx.training = bool(training)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        y, scale, eps_y, eps_1 = ctx.saved_tensors
        a_b = ctx.alpha_b
        dims = _feature_reduce_dims(y)

        if not ctx.training:
            return (grad_out / _expand(scale, y),) + (None,) * 8

        # (11a) control for orthogonality to y
        g = grad_out - (1.0 - a_b) * _expand(eps_y, y) * y
        # (11b) accumulate the deviation this step leaves behind
        eps_y.add_((g * y).mean(dim=dims))

        # (12a) undo the forward scaling, then control for the mean-zero condition
        g = g / _expand(scale, y)
        g = g - (1.0 - a_b) * _expand(eps_1, y)
        # (12b)
        eps_1.add_(g.mean(dim=dims))

        return (g,) + (None,) * 8


class LayerScaling(nn.Module):
    """Equation (9): z = y / sqrt(mean over features of y^2).

    "Required for unbounded activation functions, e.g. ReLU" (Fig. 6). It is the
    mechanism that stops activations growing or decaying exponentially with
    depth when the forward statistics carry error (§3.3), and every network in
    this project uses ReLU-family activations -- so for us it is not optional.

    Its gradient (10) is the exact derivative of this operation, so autograd
    handles it correctly and no custom backward is needed here.
    """

    def __init__(self, eps: float = DEFAULT_EPS):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, x.dim()))  # all features of this sample
        moment2 = x.pow(2).mean(dim=dims, keepdim=True)
        return x / torch.sqrt(moment2 + self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class OnlineNorm(nn.Module):
    """Online Normalization for (N, C) or (N, C, H, W) activations.

    Order of operations follows the reference implementation: normalise, then
    the learnable affine, then layer scaling.
    """

    def __init__(
        self,
        num_features: int,
        alpha_fwd: float = DEFAULT_ALPHA_FWD,
        alpha_bkw: float = DEFAULT_ALPHA_BKW,
        eps: float = DEFAULT_EPS,
        affine: bool = True,
        layer_scaling: bool = True,
    ):
        super().__init__()
        if not 0.0 < alpha_fwd < 1.0 or not 0.0 < alpha_bkw < 1.0:
            raise ValueError("alpha_fwd and alpha_bkw must lie in (0, 1)")
        self.num_features = int(num_features)
        self.alpha_fwd = float(alpha_fwd)
        self.alpha_bkw = float(alpha_bkw)
        self.eps = float(eps)

        # Estimator state, not parameters -- registered as buffers so they are
        # checkpointed and moved with the module, exactly like BatchNorm's
        # running statistics.
        self.register_buffer("mu", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.register_buffer("eps_y", torch.zeros(num_features))
        self.register_buffer("eps_1", torch.zeros(num_features))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.layer_scaling = LayerScaling(eps) if layer_scaling else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (2, 4):
            raise ValueError(f"expected (N, C) or (N, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} features, got {x.shape[1]}"
            )

        y = _OnlineNormFn.apply(
            x, self.mu, self.var, self.eps_y, self.eps_1,
            self.alpha_fwd, self.alpha_bkw, self.eps, self.training,
        )
        if self.weight is not None:
            y = y * _expand(self.weight, y) + _expand(self.bias, y)
        if self.layer_scaling is not None:
            y = self.layer_scaling(y)
        return y

    def reset_state(self) -> None:
        with torch.no_grad():
            self.mu.zero_()
            self.var.fill_(1.0)
            self.eps_y.zero_()
            self.eps_1.zero_()

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, alpha_fwd={self.alpha_fwd}, "
            f"alpha_bkw={self.alpha_bkw}, eps={self.eps}, "
            f"affine={self.weight is not None}, "
            f"layer_scaling={self.layer_scaling is not None}"
        )
