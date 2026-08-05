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


def make_model(gen, hidden=(16, 16), in_features=32, out_features=4, **kw) -> MLP:
    return MLP(
        in_features=in_features,
        hidden_dims=hidden,
        out_features=out_features,
        generator=gen,
        **kw,
    )


def kill_units(model: MLP, layer_idx: int, units) -> None:
    """Force the given units of a hidden layer to emit exactly zero for any
    bounded input, by driving their pre-activation far negative."""
    with torch.no_grad():
        model.incoming_linear(layer_idx).bias[list(units)] = -1e6
