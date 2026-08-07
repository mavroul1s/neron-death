"""Online Normalization, checked against the paper's equations.

The point of these tests is that the BACKWARD pass is a control process, not the
derivative of the forward pass. A implementation that gets the forward right and
lets autograd handle the rest would pass a naive smoke test, train perfectly
well, and give a wrong answer for C3 -- which is the one arm where the claim is
about the method's own dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models import make_norm
from src.online_norm import (
    DEFAULT_ALPHA_BKW,
    DEFAULT_ALPHA_FWD,
    LayerScaling,
    OnlineNorm,
)


def _reference_forward(xs, alpha_f, eps, n_features):
    """Equations 8a-8c applied literally, one step at a time, in numpy."""
    mu = np.zeros(n_features)
    var = np.ones(n_features)
    outs = []
    for x in xs:  # x: (N, C) for one step
        scale = np.sqrt(var + eps)
        outs.append((x - mu) / scale)
        bm = x.mean(axis=0)
        bv = x.var(axis=0)
        var = alpha_f * var + (1 - alpha_f) * bv + alpha_f * (1 - alpha_f) * (bm - mu) ** 2
        mu = alpha_f * mu + (1 - alpha_f) * bm
    return outs


def test_forward_matches_equations_8a_8c():
    torch.manual_seed(0)
    n_features = 6
    steps = [torch.randn(4, n_features, dtype=torch.float64) for _ in range(12)]

    norm = OnlineNorm(n_features, affine=False, layer_scaling=False).double()
    got = [norm(s).detach().numpy() for s in steps]
    want = _reference_forward(
        [s.numpy() for s in steps], DEFAULT_ALPHA_FWD, norm.eps, n_features
    )
    for g, w in zip(got, want):
        assert np.allclose(g, w, rtol=1e-12, atol=1e-12)


def test_state_update_order_uses_the_previous_mean():
    """(8c) uses mu_{t-1}, so variance must be updated before the mean.

    Updating the mean first changes the variance term by a factor of alpha_f^2 --
    a difference that trains fine and is invisible without this check.
    """
    n = 3
    norm = OnlineNorm(n, affine=False, layer_scaling=False).double()
    a = norm.alpha_fwd
    x = torch.full((1, n), 4.0, dtype=torch.float64)
    mu0, var0 = norm.mu.clone(), norm.var.clone()

    norm(x)
    expected_var = a * var0 + a * (1 - a) * (x[0] - mu0) ** 2  # var(x)=0 for N=1
    expected_mu = a * mu0 + (1 - a) * x[0]
    assert torch.allclose(norm.var, expected_var, rtol=1e-12)
    assert torch.allclose(norm.mu, expected_mu, rtol=1e-12)


def test_backward_is_a_control_process_not_the_autograd_derivative():
    """THE test. If these agree, the control process was not implemented and the
    C3 online-norm arm would be measuring plain running-statistic normalisation.
    """
    torch.manual_seed(1)
    n = 5
    norm = OnlineNorm(n, affine=False, layer_scaling=False).double()
    for _ in range(20):  # warm the estimator so the state is non-trivial
        norm(torch.randn(8, n, dtype=torch.float64))
    norm.eps_y.copy_(torch.linspace(0.1, 0.5, n).double())
    norm.eps_1.copy_(torch.linspace(-0.3, 0.3, n).double())

    x = torch.randn(8, n, dtype=torch.float64, requires_grad=True)
    mu, var = norm.mu.clone(), norm.var.clone()
    norm(x).backward(torch.ones_like(x))
    control_grad = x.grad.clone()

    # What autograd would give for the same forward with frozen statistics.
    x2 = x.detach().clone().requires_grad_(True)
    ((x2 - mu) / torch.sqrt(var + norm.eps)).backward(torch.ones_like(x2))

    assert not torch.allclose(control_grad, x2.grad, rtol=1e-6), (
        "backward equals the plain derivative -- the control process (11a-12b) "
        "is missing"
    )


def test_backward_matches_equations_11a_12b():
    torch.manual_seed(2)
    n = 4
    norm = OnlineNorm(n, affine=False, layer_scaling=False).double()
    for _ in range(5):
        norm(torch.randn(6, n, dtype=torch.float64))

    x = torch.randn(6, n, dtype=torch.float64, requires_grad=True)
    scale = torch.sqrt(norm.var + norm.eps).clone()
    mu = norm.mu.clone()
    ey, e1 = norm.eps_y.clone(), norm.eps_1.clone()
    a_b = DEFAULT_ALPHA_BKW

    y = norm(x)
    grad_out = torch.randn(6, n, dtype=torch.float64)
    y.backward(grad_out)

    y_ref = (x.detach() - mu) / scale
    g = grad_out - (1 - a_b) * ey * y_ref             # 11a
    ey_new = ey + (g * y_ref).mean(dim=0)             # 11b
    g = g / scale                                     # 12a, first half
    g = g - (1 - a_b) * e1                            # 12a, second half
    e1_new = e1 + g.mean(dim=0)                       # 12b

    assert torch.allclose(x.grad, g, rtol=1e-12, atol=1e-12)
    assert torch.allclose(norm.eps_y, ey_new, rtol=1e-12)
    assert torch.allclose(norm.eps_1, e1_new, rtol=1e-12)


def test_control_errors_stay_bounded():
    """§4 Formal Properties: the accumulated errors eps_y and eps_1 remain
    bounded. If they diverge, the arm is broken and any dead-unit curve from it
    is meaningless."""
    torch.manual_seed(3)
    n = 8
    norm = OnlineNorm(n, affine=False, layer_scaling=False)
    for _ in range(400):
        x = torch.randn(16, n, requires_grad=True)
        norm(x).backward(torch.randn(16, n))
    assert torch.isfinite(norm.eps_y).all() and torch.isfinite(norm.eps_1).all()
    assert norm.eps_y.abs().max() < 50, norm.eps_y
    assert norm.eps_1.abs().max() < 50, norm.eps_1


def test_forward_normalises_after_warmup():
    torch.manual_seed(4)
    n = 16
    norm = OnlineNorm(n, affine=False, layer_scaling=False)
    for _ in range(3000):  # alpha_f=0.999 has a ~1000-step time constant
        norm(torch.randn(8, n) * 3.0 + 5.0)
    out = norm(torch.randn(2048, n) * 3.0 + 5.0)
    assert abs(float(out.mean())) < 0.15
    assert abs(float(out.std()) - 1.0) < 0.15


def test_layer_scaling_is_equation_9():
    x = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
    out = LayerScaling(eps=0.0)(x)
    assert torch.allclose(out[0], x[0] / np.sqrt(12.5))
    assert torch.allclose(out[1], x[1] / 1.0)


def test_works_on_conv_activations():
    torch.manual_seed(5)
    norm = OnlineNorm(4)
    x = torch.randn(3, 4, 5, 5, requires_grad=True)
    norm(x).sum().backward()
    assert x.grad.shape == x.shape and torch.isfinite(x.grad).all()
    assert norm.mu.shape == (4,)


def test_eval_mode_leaves_the_estimator_alone():
    """Probing must not advance the state, or measuring the network would change
    it -- every probe batch would perturb the very thing being measured."""
    torch.manual_seed(6)
    norm = OnlineNorm(4, affine=False, layer_scaling=False)
    for _ in range(10):
        norm(torch.randn(8, 4))
    norm.eval()
    before = (norm.mu.clone(), norm.var.clone())
    norm(torch.randn(2048, 4))
    assert torch.equal(norm.mu, before[0]) and torch.equal(norm.var, before[1])


def test_models_can_now_build_online_norm():
    """`make_norm('online')` used to raise NotImplementedError on purpose."""
    layer = make_norm("online", 32)
    assert isinstance(layer, OnlineNorm)
    assert layer.num_features == 32


def test_mlp_with_online_norm_trains():
    from src.models import MLP

    torch.manual_seed(7)
    model = MLP(in_features=12, hidden_dims=(16, 16), out_features=3, norm="online")
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x, y = torch.randn(32, 12), torch.randint(0, 3, (32,))
    first = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    assert torch.isfinite(loss) and float(loss) < first
