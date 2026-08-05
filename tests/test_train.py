"""Training-loop tests. Covers CLAUDE.md §8 item 6 (checkpoint round-trip).

Everything runs on synthetic data on CPU in a few seconds, so there is no excuse
for not running the suite before a GPU batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from src.config import resolve_config
from src.train import Trainer, build_optimizer


def _cfg(root, run_id, **over):
    cfg = {
        "run_id": run_id,
        "seed": 3,
        "device": "cpu",
        "data": {"root": str(root), "n_tasks": 4, "batch_size": 128},
        "model": {"hidden_dims": [8, 8]},
        "optim": {"name": "sgd", "lr": 0.05, "momentum": 0.9},
        # freq=3 with 4 steps/task puts recycling events on both sides of the
        # checkpoint, so the recycler's RNG state is exercised by the resume.
        "recycling": {"kind": "redo", "tau": 0.1, "freq": 3, "score_batch_size": 32},
        "probe": {"n_probe": 64},
        "checkpoint": {"every_tasks": 2, "keep_last": 5},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


class _Interrupt(Exception):
    """Stands in for Kaggle's unannounced 12-hour kill."""


class InterruptingTrainer(Trainer):
    stop_after = None

    def _save_checkpoint(self, task_idx):
        path = super()._save_checkpoint(task_idx)
        if task_idx == self.stop_after:
            raise _Interrupt(str(path))
        return path


# ---------------------------------------------------------------------------
# CLAUDE.md §8.6 -- checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_matches_an_uninterrupted_run(tiny_mnist_root, tmp_path):
    runs = tmp_path / "runs"

    Trainer(_cfg(tiny_mnist_root, "clean"), runs_root=runs).run()

    killed = InterruptingTrainer(_cfg(tiny_mnist_root, "killed"), runs_root=runs)
    killed.stop_after = 1
    with pytest.raises(_Interrupt):
        killed.run()
    assert not (runs / "killed" / "tasks.parquet").exists(), "run was not finalized"

    # A fresh process, given only the config, picks up where it left off.
    Trainer(_cfg(tiny_mnist_root, "killed"), runs_root=runs).run(resume=True)

    a = pq.read_table(runs / "clean" / "tasks.parquet").to_pydict()
    b = pq.read_table(runs / "killed" / "tasks.parquet").to_pydict()

    assert a["task_idx"] == b["task_idx"] == [-1, 0, 1, 2, 3]
    assert np.allclose(
        a["online_accuracy"], b["online_accuracy"], equal_nan=True, rtol=0, atol=0
    ), "resumed run diverged from the uninterrupted one"
    assert np.allclose(a["mean_loss"], b["mean_loss"], equal_nan=True, rtol=0, atol=0)
    assert a["global_step"] == b["global_step"]

    # The per-neuron dataset must match too, not just the summary curve.
    na = pq.read_table(runs / "clean" / "neurons.parquet").to_pydict()
    nb = pq.read_table(runs / "killed" / "neurons.parquet").to_pydict()
    assert len(na["mean_abs_act"]) == len(nb["mean_abs_act"])
    assert np.allclose(na["mean_abs_act"], nb["mean_abs_act"], rtol=0, atol=0)
    assert na["was_recycled_this_task"] == nb["was_recycled_this_task"]

    # ...and so must the recycled-set composition table.
    ra = pq.read_table(runs / "clean" / "recycling.parquet").to_pydict()
    rb = pq.read_table(runs / "killed" / "recycling.parquet").to_pydict()
    assert ra["step"] == rb["step"]
    assert ra["k"] == rb["k"]
    assert ra["n_dead_exact"] == rb["n_dead_exact"]


def test_resume_refuses_a_different_config(tiny_mnist_root, tmp_path):
    """CLAUDE.md §7: a changed hyperparameter is a new run_id, not an edit."""
    runs = tmp_path / "runs"
    Trainer(_cfg(tiny_mnist_root, "r"), runs_root=runs).run()
    with pytest.raises(RuntimeError, match="different config"):
        Trainer(
            _cfg(tiny_mnist_root, "r", optim={"lr": 0.09}), runs_root=runs
        ).run()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def test_run_writes_every_required_output(tiny_mnist_root, tmp_path):
    runs = tmp_path / "runs"
    run_dir = Trainer(_cfg(tiny_mnist_root, "outputs"), runs_root=runs).run()

    for name in ("config.json", "environment.json", "summary.json"):
        assert (run_dir / name).exists()
    for name in ("tasks", "metrics", "neurons", "recycling"):
        assert (run_dir / f"{name}.parquet").exists(), f"{name}.parquet missing"
    assert not (run_dir / "_shards").exists(), "shards were not cleaned up"

    n_tasks, hidden = 4, [8, 8]
    neurons = pq.read_table(run_dir / "neurons.parquet")
    # One row per (task, layer, neuron), plus the pre-training init probe.
    assert neurons.num_rows == (n_tasks + 1) * sum(hidden)

    metrics = pq.read_table(run_dir / "metrics.parquet").to_pydict()
    assert len(metrics["layer_idx"]) == (n_tasks + 1) * len(hidden)
    # Both probe batches present, all four death definitions, every threshold.
    for col in (
        "dead_exact_frac", "dead_exact_frac_ref",
        "dormant_frac_tau_0", "dormant_frac_tau_0p1", "dormant_frac_tau_0p25",
        "dead_abs_frac_1em06", "dead_abs_frac_1em04", "dead_abs_frac_1em02",
        "erank", "erank_ref", "sign_entropy", "saturated_frac",
        "weight_l2", "weight_mean_abs", "grad_norm_layer",
    ):
        assert col in metrics, f"metrics.parquet is missing {col}"
    # ReLU is unbounded: saturation must be null, never 0.0.
    assert all(v is None for v in metrics["saturated_frac"])

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["n_tasks"] == n_tasks


def test_per_neuron_log_failure_is_fatal(tiny_mnist_root, tmp_path):
    """A run missing the C4 dataset must raise, not warn (CLAUDE.md §5.4)."""
    runs = tmp_path / "runs"
    t = Trainer(_cfg(tiny_mnist_root, "verify"), runs_root=runs)
    run_dir = t.run()
    with pytest.raises(RuntimeError, match="not recoverable retrospectively"):
        t._verify_per_neuron_log(4, None)
    # A truncated log (here: claiming 98 tasks when 4 were logged) is fatal too.
    with pytest.raises(RuntimeError, match="rows, expected"):
        t._verify_per_neuron_log(98, run_dir / "neurons.parquet")


def test_probe_batches_differ_between_current_and_reference(tiny_mnist_root, tmp_path):
    """If these two ever coincide, the reference batch stops doing its job of
    separating 'the neuron died' from 'the inputs moved'."""
    runs = tmp_path / "runs"
    t = Trainer(_cfg(tiny_mnist_root, "probes"), runs_root=runs)
    t.setup()
    cur, _ = t.dataset.probe_batch(0)
    ref, _ = t.dataset.reference_batch()
    assert cur.shape == ref.shape
    assert not torch.equal(cur, ref)
    # Same underlying images, only the permutation differs.
    assert torch.allclose(cur.sum(dim=1), ref.sum(dim=1), atol=1e-5)


# ---------------------------------------------------------------------------
# Config guards
# ---------------------------------------------------------------------------


def test_weight_decay_is_rejected_for_non_adamw():
    """Two routes to L2 is how a run silently gets twice the regularisation."""
    from src.models import MLP

    model = MLP(in_features=4, hidden_dims=(4,), out_features=2)
    cfg = resolve_config(
        {"run_id": "x", "optim": {"name": "sgd", "weight_decay": 1e-4}}
    )
    with pytest.raises(ValueError, match="only meaningful for AdamW"):
        build_optimizer(cfg, model)
    cfg["optim"]["name"] = "adamw"
    assert build_optimizer(cfg, model) is not None


def test_online_norm_refuses_to_run_rather_than_approximate():
    """C3 depends on this arm being faithful; a plausible approximation would be
    worse than an error."""
    from src.models import MLP

    with pytest.raises(NotImplementedError, match="Online Normalization"):
        MLP(in_features=4, hidden_dims=(4,), out_features=2, norm="online")


def test_config_hash_pins_resolved_values():
    from src.config import config_hash

    a = resolve_config({"run_id": "x", "optim": {"lr": 0.01}})
    b = resolve_config({"run_id": "x"})  # 0.01 is the default
    assert config_hash(a) == config_hash(b)
    c = resolve_config({"run_id": "x", "optim": {"lr": 0.003}})
    assert config_hash(a) != config_hash(c)
