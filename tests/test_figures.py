"""Every figure function runs end to end on a synthetic extract.

Four of the figures (C3, C5, Setting 3, Setting 2) have no data on any machine
yet -- their Kaggle jobs are still queued. Untested plotting code that first
executes on the morning the results land is how a session gets spent debugging a
`KeyError` instead of reading a result. So the schemas are faked here, exactly as
`src/probes.py` writes them, and each function is required to produce a file.

These tests check that the figures *build* and that the guards fire. They cannot
check that a figure is true -- that is what the instrument tests in
`test_probes.py` and `test_survival.py` are for.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.analysis import figures


TAUS = ["0", "0p01", "0p025", "0p05", "0p1", "0p25"]
ABS = ["1em06", "1em04", "1em02"]


def _config(run_id, seed, lr, *, arm="none", tau=0.0, activation="relu",
            dataset="permuted_mnist", n_tasks=200, optimizer="sgd"):
    return {
        "run_id": run_id,
        "seed": seed,
        "optim": {"name": optimizer, "lr": lr},
        "model": {"activation": activation, "norm": "none"},
        "l2": {"lambda": 0.0},
        "data": {"name": dataset, "n_tasks": n_tasks},
        "recycling": {"kind": arm, "tau": tau},
    }


def _metric_row(run_id, task, layer, *, dead=0.2, spatial=False):
    row = {
        "run_id": run_id, "task_idx": task, "probe_point": "task_end",
        "layer_idx": layer, "is_spatial": spatial,
    }
    for suffix in ("", "_ref"):
        row[f"dead_exact_frac{suffix}"] = dead
        row[f"mean_frac_zero_positions{suffix}"] = min(1.0, dead * 3)
        row[f"erank{suffix}"] = 100.0
        for t in TAUS:
            row[f"dormant_frac_tau_{t}{suffix}"] = min(1.0, dead * 2)
        for a in ABS:
            row[f"dead_abs_frac_{a}{suffix}"] = min(1.0, dead * 2.2)
    return row


def write_extract(root: Path, name: str, configs, *, accuracy, dead,
                  n_tasks=200, layers=(0, 1, 2), spatial=(), intra=False):
    """One extract directory in the layout `make_analysis_extract.py` writes."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    tasks, metrics, intra_rows = [], [], []
    for cfg in configs:
        rid = cfg["run_id"]
        for t in range(n_tasks):
            tasks.append({
                "run_id": rid, "task_idx": t, "probe_point": "task_end",
                "online_accuracy": accuracy(cfg, t), "mean_loss": 0.3,
                "probe_accuracy": accuracy(cfg, t), "n_recycled": 0,
            })
            for layer in layers:
                metrics.append(_metric_row(rid, t, layer, dead=dead(cfg, t, layer),
                                           spatial=layer in spatial))
        if intra:
            for t in range(0, n_tasks, 10):
                for step in (0, 25, 50, 100, 200, 450):
                    for layer in layers:
                        intra_rows.append({
                            "run_id": rid, "task_idx": t, "step": t * 469 + step,
                            "step_in_task": step, "layer_idx": layer,
                            "dead_exact_frac": dead(cfg, t, layer),
                        })
    pd.DataFrame(tasks).to_parquet(d / "tasks.parquet")
    pd.DataFrame(metrics).to_parquet(d / "metrics.parquet")
    if intra_rows:
        pd.DataFrame(intra_rows).to_parquet(d / "intra_task.parquet")
    with open(d / "runs.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"run_id": c["run_id"], "config": c, "summary": {"status": "complete"}}
             for c in configs],
            f,
        )
    return d


@pytest.fixture
def c3_extract(tmp_path):
    arms = {"backprop": (0.88, 0.20), "l2_1em4": (0.90, 0.24),
            "l2_1em3": (0.91, 0.30), "l2_1em2": (0.89, 0.35),
            "sp": (0.90, 0.18), "dropout01": (0.87, 0.15),
            "online_norm": (0.89, 0.40)}
    cfgs = [_config(f"c3_{a}_lr0p1_s{s}", s, 0.1) for a in arms for s in range(5)]
    acc = lambda c, t: arms[figures._arm_from_run_id(c["run_id"], "c3_")][0]
    dead = lambda c, t, l: arms[figures._arm_from_run_id(c["run_id"], "c3_")][1]
    write_extract(tmp_path, "c3", cfgs, accuracy=acc, dead=dead)
    return tmp_path


def test_c3_figure_builds(c3_extract, tmp_path):
    figures.use_paper_style()
    out = tmp_path / "figs"
    assert figures.fig_c3_anomaly(figures.Extracts(c3_extract), out) is not None
    assert (out / "fig7_c3_anomaly.pdf").exists()


def test_arm_names_survive_both_run_id_conventions():
    """C3 writes `_lr0p1_s2`; C5 writes only `_s0`, because its arms differ in
    optimizer hyperparameters rather than learning rate. A parser that assumes
    the learning-rate segment returns 'adam_lyle_s0' as an arm name and the
    figure dies on a KeyError -- which is exactly what happened."""
    assert figures._arm_from_run_id("c5_adam_lyle_s0", "c5_") == "adam_lyle"
    assert figures._arm_from_run_id("c5_adamw_s4", "c5_") == "adamw"
    assert figures._arm_from_run_id("c3_l2_1em3_lr0p1_s2", "c3_") == "l2_1em3"
    assert figures._arm_from_run_id("c3_backprop_lr0p1_s0", "c3_") == "backprop"


def test_c5_figure_builds(tmp_path):
    arms = {"sgd": 0.20, "adam_default": 0.59, "adam_lyle": 0.25, "adamw": 0.58}
    # Real C5 run_id shape: no `_lr` segment (see the test above).
    cfgs = [_config(f"c5_{a}_s{s}", s, 0.001, optimizer=a.split("_")[0])
            for a in arms for s in range(5)]
    root = write_extract(
        tmp_path, "c5", cfgs,
        accuracy=lambda c, t: 0.90,
        dead=lambda c, t, l: arms[figures._arm_from_run_id(c["run_id"], "c5_")],
        intra=True,
    ).parent
    figures.use_paper_style()
    out = tmp_path / "figs"
    assert figures.fig_c5_optimizers(figures.Extracts(root), out) is not None
    assert (out / "fig8_c5_optimizers.pdf").exists()


def test_c5_figure_degrades_when_the_extract_predates_the_intra_task_fix(
    tmp_path, capsys
):
    """The first C5 extract omitted intra_task.parquet. Drawing nothing loses the
    cross-task result too; drawing panel (b) silently would hide that the
    *pre-registered* panel is the missing one."""
    arms = {"sgd": 0.20, "adam_default": 0.59}
    cfgs = [_config(f"c5_{a}_s{s}", s, 0.001, optimizer=a.split("_")[0])
            for a in arms for s in range(5)]
    root = write_extract(
        tmp_path, "c5", cfgs, accuracy=lambda c, t: 0.90,
        dead=lambda c, t, l: arms[figures._arm_from_run_id(c["run_id"], "c5_")],
        intra=False,
    ).parent
    figures.use_paper_style()
    assert figures.fig_c5_optimizers(figures.Extracts(root), tmp_path / "figs")
    assert "NOT drawn" in capsys.readouterr().out


def test_setting2_figure_builds_on_a_short_run(tmp_path):
    """Setting 2 is 50 tasks, so it has no frozen late window; the figure must
    fall back to its own last fifth rather than raising."""
    cfgs = [_config(f"s2_cifar_cnn_none_lr0p1_s{s}", s, 0.1,
                    dataset="label_shuffled_cifar10", n_tasks=50) for s in range(5)]
    root = write_extract(
        tmp_path, "setting2", cfgs, accuracy=lambda c, t: 0.45,
        dead=lambda c, t, l: 0.0 if l < 3 else 0.26,
        n_tasks=50, layers=(0, 1, 2, 3), spatial=(0, 1, 2),
    ).parent
    figures.use_paper_style()
    out = tmp_path / "figs"
    assert figures.fig_setting2_channels(figures.Extracts(root), out) is not None
    assert (out / "fig10_setting2_channels.pdf").exists()


# -- the guard that matters ---------------------------------------------------


def _setting3_extract(tmp_path, tanh_accuracy, tanh_lr=None):
    acts = {"relu": 0.877, "leaky_relu": 0.900, "gelu": 0.884, "silu": 0.897}
    cfgs = [_config(f"s3_act_{a}_lr0p1_s{s}", s, 0.1, activation=a)
            for a in acts for s in range(5)]
    cfgs += [_config(f"s3_act_tanh_lr0p1_s{s}", s, 0.1, activation="tanh")
             for s in range(5)]
    if tanh_lr is not None:
        cfgs += [_config(f"s3tanh_lr0p01_s{s}", s, tanh_lr, activation="tanh")
                 for s in range(5)]

    def acc(c, t):
        a = c["model"]["activation"]
        if a != "tanh":
            return acts[a]
        return 0.86 if c["run_id"].startswith("s3tanh_") else tanh_accuracy

    write_extract(tmp_path, "setting3", cfgs, accuracy=acc,
                  dead=lambda c, t, l: 0.0 if c["model"]["activation"] != "relu" else 0.2)
    return tmp_path


def test_setting3_drops_an_arm_sitting_at_chance(tmp_path, capsys):
    """tanh at lr=0.1 reached 10.05% -- chance for ten classes. Its death
    metrics are meaningless and must not be drawn as a row."""
    root = _setting3_extract(tmp_path, tanh_accuracy=0.1005)
    figures.use_paper_style()
    path = figures.fig_setting3_activations(figures.Extracts(root), tmp_path / "figs")
    assert path is not None
    assert "dropped tanh" in capsys.readouterr().out


def test_setting3_uses_the_calibrated_tanh_runs_when_given_one(tmp_path, capsys):
    root = _setting3_extract(tmp_path, tanh_accuracy=0.1005, tanh_lr=0.01)
    figures.use_paper_style()
    path = figures.fig_setting3_activations(
        figures.Extracts(root), tmp_path / "figs", tanh_lr=0.01
    )
    assert path is not None
    assert "dropped" not in capsys.readouterr().out


def test_build_all_names_the_figures_it_could_not_draw(tmp_path, capsys):
    """A four-arm figure that quietly becomes a three-arm figure is the kind of
    thing that survives into a submission."""
    cfgs = [_config(f"c3_backprop_lr0p1_s{s}", s, 0.1) for s in range(5)]
    root = write_extract(tmp_path, "c3", cfgs, accuracy=lambda c, t: 0.88,
                         dead=lambda c, t, l: 0.2).parent
    figures.build_all(root, out=tmp_path / "figs")
    printed = capsys.readouterr().out
    assert "MISSING ARMS" in printed
    assert "fig_c5_optimizers" in printed  # skipped, but named
