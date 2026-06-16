"""Odds conversion and vig removal.

Supports decimal, American, and fractional odds. Provides both proportional
and Shin de-vigging so model probabilities can be compared against a fair
(overround-removed) market.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def to_decimal(odds: float, fmt: str = "decimal") -> float:
    """Convert American / fractional / decimal odds to decimal odds.

    ``fmt`` is one of ``decimal``, ``american``, ``fractional``. Fractional
    odds are passed as the value ``num/den`` already divided (e.g. 5/2 -> 2.5),
    so ``to_decimal(2.5, "fractional")`` returns 3.5.
    """
    fmt = fmt.lower()
    if fmt == "decimal":
        if odds <= 1.0:
            raise ValueError(f"Decimal odds must be > 1.0, got {odds}")
        return float(odds)
    if fmt == "american":
        if odds == 0:
            raise ValueError("American odds cannot be 0")
        if odds > 0:
            return 1.0 + odds / 100.0
        return 1.0 + 100.0 / abs(odds)
    if fmt == "fractional":
        if odds < 0:
            raise ValueError("Fractional odds ratio must be >= 0")
        return 1.0 + odds
    raise ValueError(f"Unknown odds format: {fmt}")


def implied_prob(odds: float, fmt: str = "decimal") -> float:
    """Raw implied probability (with vig) from odds of any supported format."""
    return 1.0 / to_decimal(odds, fmt)


def devig_proportional(probs: Iterable[float]) -> np.ndarray:
    """Remove vig by normalising raw implied probabilities to sum to 1."""
    p = np.asarray(list(probs), dtype=float)
    total = p.sum()
    if total <= 0:
        raise ValueError("Implied probabilities must be positive")
    return p / total


def devig_shin(probs: Iterable[float], max_iter: int = 100,
               tol: float = 1e-10) -> np.ndarray:
    """Remove vig with Shin's (1992) model, which assumes a proportion ``z`` of
    insider money and is generally fairer for longshots than proportional.

    Falls back to proportional de-vig when the implied book is essentially fair
    (overround <= 0) or the solver does not converge.
    """
    pi = np.asarray(list(probs), dtype=float)
    booksum = pi.sum()
    if booksum <= 1.0 + 1e-9:
        return devig_proportional(pi)

    # Solve for z in (0, 1) such that the recovered probabilities sum to 1.
    z = 0.0
    for _ in range(max_iter):
        root = np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / booksum)
        q = (root - z) / (2 * (1 - z))
        s = q.sum()
        if abs(s - 1.0) < tol:
            break
        # Newton-ish bisection update on z.
        z = np.clip(z + (s - 1.0) * 0.5, 0.0, 0.999)
    q = np.clip(q, 1e-9, None)
    return q / q.sum()


def fair_probs(odds: Iterable[float], fmt: str = "decimal",
               method: str = "proportional") -> np.ndarray:
    """Convert a set of odds for one market into fair (vig-free) probabilities."""
    raw = [implied_prob(o, fmt) for o in odds]
    if method == "shin":
        return devig_shin(raw)
    return devig_proportional(raw)


def fair_odds_from_prob(prob: float) -> float:
    """Fair decimal odds implied by a model probability (no margin)."""
    if not 0.0 < prob <= 1.0:
        raise ValueError(f"Probability out of range: {prob}")
    return 1.0 / prob


def kelly_fraction(prob: float, dec_odds: float) -> float:
    """Full-Kelly stake fraction for a binary bet. Negative -> no bet."""
    b = dec_odds - 1.0
    if b <= 0:
        return 0.0
    return (prob * b - (1.0 - prob)) / b
