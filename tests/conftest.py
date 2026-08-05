"""Shared fixtures.

Tests never touch real MNIST: a run must be testable on a laptop with no data
cached, or the tests will not get run before GPU batches -- which is the one
thing CLAUDE.md §8 asks for.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models import MLP


@pytest.fixture(autouse=True)
def _restore_deterministic_flag():
    """`torch.use_deterministic_algorithms` is process-global; runs set it and
    tests must not leak it into each other."""
    was = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(was)


@pytest.fixture
def gen():
    g = torch.Generator()
    g.manual_seed(20260805)
    return g


@pytest.fixture
def tiny_mnist_root(tmp_path):
    """A structurally-valid but synthetic MNIST cache.

    Same shapes, dtypes and value range as the real thing, so `load_mnist` and
    every downstream shape assumption is exercised; the pixels are noise, which
    is fine because no test here asserts anything about accuracy.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    root.mkdir()
    np.savez(
        root / "mnist.npz",
        x_train=rng.integers(0, 256, size=(512, 28, 28), dtype=np.uint8),
        y_train=rng.integers(0, 10, size=512, dtype=np.uint8),
        x_test=rng.integers(0, 256, size=(128, 28, 28), dtype=np.uint8),
        y_test=rng.integers(0, 10, size=128, dtype=np.uint8),
    )
    return root


def make_model(
    gen, hidden=(16, 16), in_features=32, out_features=4, bias_init=5.0, **kw
) -> MLP:
    """A small MLP in which every unit is alive unless a test kills it.

    ``bias_init=5.0`` is not the experiment's initialisation (runs use 0.0); it
    is a test fixture choice. With zero bias and the all-positive inputs these
    tests use, a fair number of ReLU units are negative on *every* input and so
    are genuinely, correctly reported dead -- which would make "the probe found
    4 dead units" an assertion about the random seed rather than about the
    probe. A large positive bias removes that ambiguity: any dead unit in these
    tests is one the test killed.
    """
    return MLP(
        in_features=in_features,
        hidden_dims=hidden,
        out_features=out_features,
        bias_init=bias_init,
        generator=gen,
        **kw,
    )


def kill_units(model: MLP, layer_idx: int, units) -> None:
    """Force the given units of a hidden layer to emit exactly zero for any
    bounded input, by driving their pre-activation far negative."""
    with torch.no_grad():
        model.incoming_linear(layer_idx).bias[list(units)] = -1e6


def quieten_units(model: MLP, layer_idx: int, units, value: float = 0.01) -> None:
    """Make units *alive but quiet*: constant small positive output on every
    input, so `frac_inputs_active == 1` and `mean_abs_act == value`.

    This is the population C1 is about -- neurons ReDo recycles that were never
    dead. Constructed deterministically rather than hoped for, so the
    composition test cannot become flaky.
    """
    with torch.no_grad():
        units = list(units)
        model.incoming_linear(layer_idx).weight[units] = 0.0
        model.incoming_linear(layer_idx).bias[units] = value
