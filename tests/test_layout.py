"""Structural rules from CLAUDE.md §4, made enforceable.

These are cheap to check and expensive to notice by eye once the repo grows.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _training_modules():
    return [p for p in SRC.glob("*.py") if p.name != "__init__.py"]


def test_training_code_never_imports_analysis():
    """`src/analysis/` is post-hoc only and must never be imported by training
    code. A one-way dependency is the only thing keeping analysis choices out of
    the runs they analyse."""
    offenders = []
    for path in _training_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                ("analysis", "src.analysis")
            ):
                offenders.append(f"{path.name}: from {node.module}")
            if isinstance(node, ast.ImportFrom) and node.level and node.module == "analysis":
                offenders.append(f"{path.name}: from .analysis")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("src.analysis", "analysis.")):
                        offenders.append(f"{path.name}: import {alias.name}")
    assert not offenders, "training code imports src/analysis: " + "; ".join(offenders)


def test_probes_is_the_only_module_that_defines_metrics():
    """Every death-metric name must be defined in probes.py and used elsewhere
    only by reference. Catches a metric being quietly reimplemented inline."""
    names = (
        "dead_exact_mask",
        "dormant_mask",
        "dead_absolute_mask",
        "saturated_mask",
        "sokar_scores",
        "effective_rank",
        "sign_entropy",
    )
    offenders = []
    for path in _training_modules():
        if path.name == "probes.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                offenders.append(f"{path.name}:{node.name}")
    assert not offenders, "metric redefined outside probes.py: " + "; ".join(offenders)


def test_notebooks_contain_no_logic():
    """CLAUDE.md §4: notebooks import from src/ and call one function."""
    import json

    nb_dir = SRC.parent / "notebooks"
    for nb in nb_dir.glob("*.ipynb"):
        source = json.loads(nb.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(c["source"]) for c in source["cells"] if c["cell_type"] == "code"
        )
        tree = ast.parse(code)
        defs = [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        ]
        assert not defs, f"{nb.name} defines {defs}; logic belongs in src/"


def test_stats_helpers_behave():
    from src.analysis.stats import classify_difference, iqm, stratified_bootstrap

    # IQM discards the outer quartiles: two wild outliers must not move it.
    clean = np.array([10.0, 11.0, 12.0, 13.0])
    assert iqm(clean) == pytest.approx(11.5)
    assert iqm(np.array([-1000.0, 10.0, 11.0, 12.0, 13.0, 1000.0])) == pytest.approx(11.5)

    scores = np.tile(np.array([[0.5], [0.5], [0.5], [0.5]]), (1, 3))
    est = stratified_bootstrap(scores, n_bootstrap=200, seed=1)
    assert est.point == pytest.approx(0.5)
    assert est.lo == pytest.approx(0.5) and est.hi == pytest.approx(0.5)
    assert est.excludes_zero

    assert classify_difference(est, approx_pp=1.0, much_greater_pp=2.0) == "much_greater"
    zero = stratified_bootstrap(np.zeros((4, 3)), n_bootstrap=200, seed=1)
    assert classify_difference(zero, approx_pp=1.0, much_greater_pp=2.0) == "approx_equal"
