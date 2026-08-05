"""Interquartile mean with stratified bootstrap CIs (Agarwal et al. 2021).

CLAUDE.md §7: report IQM with stratified bootstrap 95% CIs, never bare means
over seeds. The field has been criticised specifically for weak evaluation
practice, and this paper's whole argument is a measurement argument, so the
statistics have to be above reproach.

The score matrix everywhere in here is ``(n_runs, n_tasks)``: rows are seeds,
columns are the tasks in the analysis window. "Stratified" means each bootstrap
replicate resamples *runs*, independently within each task column -- the
resampling scheme of ``rliable``'s ``StratifiedBootstrap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle 50% of the values.

    ``scipy.stats.trim_mean(x, 0.25)``, which is what Agarwal et al.'s reference
    implementation uses.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return float("nan")
    return float(sp_stats.trim_mean(x, proportiontocut=0.25))


def mean(x: np.ndarray) -> float:
    """Provided only so a plot can show what IQM is protecting against."""
    return float(np.asarray(x, dtype=np.float64).mean())


@dataclass
class Estimate:
    point: float
    lo: float
    hi: float
    n_runs: int
    n_tasks: int
    n_bootstrap: int
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        return (self.lo > 0.0) or (self.hi < 0.0)

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "n_runs": self.n_runs,
            "n_tasks": self.n_tasks,
            "n_bootstrap": self.n_bootstrap,
            "confidence": self.confidence,
            "excludes_zero": self.excludes_zero,
        }

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def _resample_indices(
    n_runs: int, n_tasks: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_runs, n_tasks) run-indices, drawn with replacement per task column."""
    return rng.integers(0, n_runs, size=(n_runs, n_tasks))


def stratified_bootstrap(
    scores: np.ndarray,
    statistic: Callable[[np.ndarray], float] = iqm,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Point estimate and percentile CI of `statistic` over a score matrix.

    Percentile intervals (not BCa): they are what ``rliable`` reports by
    default, so our numbers stay comparable to papers that use it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    n_runs, n_tasks = scores.shape
    rng = np.random.default_rng(seed)

    point = statistic(scores)
    reps = np.empty(n_bootstrap, dtype=np.float64)
    cols = np.arange(n_tasks)
    for b in range(n_bootstrap):
        idx = _resample_indices(n_runs, n_tasks, rng)
        reps[b] = statistic(scores[idx, cols])

    alpha = 1.0 - confidence
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), n_runs, n_tasks, n_bootstrap, confidence)


def stratified_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    statistic: Callable[[np.ndarray], float] = iqm,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
    paired: bool = True,
) -> Estimate:
    """CI for ``statistic(a) - statistic(b)``.

    This is the estimator behind the pre-registered arm comparisons -- e.g.
    ReDo(tau) minus Random-matched(tau) at each tau (protocol §B.1).

    ``paired=True`` resamples the *same* run indices from both arms. Use it when
    the arms share seeds, which they do by construction here: seed s of ReDo(tau)
    and seed s of Random-matched(tau) start from the identical initialisation
    and see the identical task stream, so pairing removes the initialisation
    variance that would otherwise swamp a 1-2 pp effect.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if paired and a.shape != b.shape:
        raise ValueError(
            f"paired bootstrap needs matching shapes, got {a.shape} and {b.shape}"
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError("arms must span the same task window")

    rng = np.random.default_rng(seed)
    point = statistic(a) - statistic(b)
    reps = np.empty(n_bootstrap, dtype=np.float64)
    cols = np.arange(a.shape[1])
    for i in range(n_bootstrap):
        idx_a = _resample_indices(a.shape[0], a.shape[1], rng)
        idx_b = idx_a if paired else _resample_indices(b.shape[0], b.shape[1], rng)
        reps[i] = statistic(a[idx_a, cols]) - statistic(b[idx_b, cols])

    alpha = 1.0 - confidence
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(
        point, float(lo), float(hi), a.shape[0], a.shape[1], n_bootstrap, confidence
    )


def classify_difference(
    est: Estimate, approx_pp: float, much_greater_pp: float
) -> str:
    """Apply the frozen decision rule to a difference, in percentage points.

    From ``configs/analysis_plan.json`` (protocol §B.1):
      "~="  : the 95% CI of the difference contains zero AND |delta| < approx_pp
      ">>"  : delta > much_greater_pp AND the CI excludes zero

    Anything else is "inconclusive", whose pre-specified response is *add
    seeds* -- never adjust a threshold (CLAUDE.md §7).
    """
    delta_pp = est.point * 100.0
    if (not est.excludes_zero) and abs(delta_pp) < approx_pp:
        return "approx_equal"
    if delta_pp > much_greater_pp and est.excludes_zero:
        return "much_greater"
    if delta_pp < -much_greater_pp and est.excludes_zero:
        return "much_less"
    return "inconclusive"
