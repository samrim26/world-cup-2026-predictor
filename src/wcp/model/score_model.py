"""Dixon-Coles correct-score model.

Given (lambda_a, lambda_b), build the full correct-score probability matrix with
the Dixon-Coles low-score dependence correction (tau), then derive W/D/L,
over/under, BTTS, and top scorelines. Also provides fast Monte-Carlo sampling of
scorelines used by the tournament simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from ..utils.config import load_config


def _tau(i: np.ndarray, j: np.ndarray, lam_a: float, lam_b: float,
         rho: float) -> np.ndarray:
    """Dixon-Coles dependence factor for low scores (0-0,1-0,0-1,1-1)."""
    t = np.ones_like(i, dtype=float)
    t = np.where((i == 0) & (j == 0), 1 - lam_a * lam_b * rho, t)
    t = np.where((i == 0) & (j == 1), 1 + lam_a * rho, t)
    t = np.where((i == 1) & (j == 0), 1 + lam_b * rho, t)
    t = np.where((i == 1) & (j == 1), 1 - rho, t)
    return t


@dataclass
class ScoreResult:
    matrix: np.ndarray            # P(score_a=i, score_b=j)
    p_a: float                    # team A win
    p_draw: float
    p_b: float
    exp_a: float                  # expected goals A
    exp_b: float
    top_scorelines: list[tuple[str, float]]
    over25: float
    under25: float
    btts: float

    @property
    def most_likely_score(self) -> str:
        return self.top_scorelines[0][0]


def score_matrix(lam_a: float, lam_b: float, max_goals: int | None = None,
                 rho: float | None = None) -> np.ndarray:
    """Return the (max_goals+1)x(max_goals+1) Dixon-Coles score matrix."""
    cfg = load_config()
    mg = max_goals if max_goals is not None else cfg.get("xg.max_goals", 7)
    rho = rho if rho is not None else cfg.get("xg.dixon_coles_rho", -0.07)

    a = poisson.pmf(np.arange(mg + 1), lam_a)
    b = poisson.pmf(np.arange(mg + 1), lam_b)
    mat = np.outer(a, b)
    ii, jj = np.meshgrid(np.arange(mg + 1), np.arange(mg + 1), indexing="ij")
    mat = mat * _tau(ii, jj, lam_a, lam_b, rho)
    mat = np.clip(mat, 0, None)
    mat /= mat.sum()              # renormalise (tau + truncation)
    return mat


def analyse_match(lam_a: float, lam_b: float, team_a: str = "A",
                  team_b: str = "B", top_n: int = 5) -> ScoreResult:
    """Full probabilistic summary of a single match from its goal rates."""
    mat = score_matrix(lam_a, lam_b)
    mg = mat.shape[0] - 1
    ii, jj = np.meshgrid(np.arange(mg + 1), np.arange(mg + 1), indexing="ij")

    p_a = float(mat[ii > jj].sum())
    p_draw = float(np.trace(mat))
    p_b = float(mat[ii < jj].sum())

    flat = [((i, j), mat[i, j]) for i in range(mg + 1) for j in range(mg + 1)]
    flat.sort(key=lambda kv: kv[1], reverse=True)
    top = [(f"{i}-{j}", round(float(p), 4)) for (i, j), p in flat[:top_n]]

    totals = ii + jj
    over25 = float(mat[totals >= 3].sum())
    btts = float(mat[(ii >= 1) & (jj >= 1)].sum())

    return ScoreResult(
        matrix=mat, p_a=p_a, p_draw=p_draw, p_b=p_b,
        exp_a=float((np.arange(mg + 1)[:, None] * mat).sum()),
        exp_b=float((np.arange(mg + 1)[None, :] * mat).sum()),
        top_scorelines=top, over25=over25, under25=1 - over25, btts=btts,
    )


def sample_scores(lam_a: np.ndarray, lam_b: np.ndarray,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised scoreline sampling for the simulator.

    Uses independent Poisson draws (the DC tau correction affects only the
    lowest-score *cell probabilities*, a second-order effect on simulated
    outcomes) for speed. Accepts arrays of rates and returns integer goals.
    """
    ga = rng.poisson(np.asarray(lam_a))
    gb = rng.poisson(np.asarray(lam_b))
    return ga, gb
