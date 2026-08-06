"""The C4 extract must lose nothing that cannot be recomputed.

`scripts/make_analysis_extract.py` drops `sokar_score` from the per-neuron
extract because it was 30% of the file and is derived. That is only acceptable
if the recomputation is exact -- otherwise C4 quietly analyses a different
quantity from the one the probe measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src import probes  # noqa: E402
from src.analysis.load import add_sokar_score  # noqa: E402


def test_recomputed_sokar_score_matches_the_probe_exactly():
    """Layer means are per (run, task, layer); getting the grouping wrong would
    silently normalise across layers and change every dormancy count."""
    rng = np.random.default_rng(0)
    rows = []
    expected = []
    for run in ("r0", "r1"):
        for task in (0, 7):
            for layer, h in enumerate((5, 8)):
                acts = torch.from_numpy(
                    np.abs(rng.normal(size=(32, h))).astype(np.float32)
                )
                mean_abs = probes.mean_abs_activation(acts)
                scores = probes.sokar_scores(mean_abs).numpy()
                for i in range(h):
                    rows.append(
                        dict(
                            run_id=run,
                            task_idx=task,
                            layer_idx=layer,
                            neuron_idx=i,
                            mean_abs_act=float(mean_abs[i]),
                        )
                    )
                    expected.append(scores[i])

    got = add_sokar_score(pd.DataFrame(rows))
    assert np.allclose(got.sokar_score.to_numpy(), np.array(expected), rtol=1e-12)


def test_recomputed_score_handles_a_wholly_dead_layer():
    """0/0 := 0, matching `probes.sokar_scores`, so every unit in a dead layer
    is dormant at every tau rather than NaN."""
    df = pd.DataFrame(
        dict(
            run_id=["r"] * 4,
            task_idx=[0] * 4,
            layer_idx=[0, 0, 1, 1],
            neuron_idx=[0, 1, 0, 1],
            mean_abs_act=[0.0, 0.0, 1.0, 3.0],
        )
    )
    got = add_sokar_score(df)
    assert got.sokar_score.tolist()[:2] == [0.0, 0.0]
    assert got.sokar_score.tolist()[2:] == pytest.approx([0.5, 1.5])


def test_c4_columns_cover_what_the_claim_needs():
    """C4 is per-neuron mortality dynamics; C1 needs to follow an individual
    unit across recycling events. Both must survive the slimming."""
    from make_analysis_extract import C4_COLUMNS

    for needed in (
        "run_id", "task_idx", "layer_idx", "neuron_idx",  # identity + time
        "exact_zero_flag",                                # the death event
        "mean_abs_act",                                   # recomputes the rest
        "was_recycled_this_task",                         # the C1 trajectory
    ):
        assert needed in C4_COLUMNS, needed
    assert "sokar_score" not in C4_COLUMNS, "dropped on purpose; recomputable"
