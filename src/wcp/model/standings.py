"""Pure, vectorised group-standings helpers shared by the simulator and tests.

Implements the FIFA group ordering: points, then overall goal difference, then
overall goals scored, then a seeded random tiebreak standing in for the rarely-
reached head-to-head / fair-play / drawing-of-lots steps (those only trigger on
an exact points+GD+GF tie).
"""
from __future__ import annotations

import numpy as np


def group_sortkey(points: np.ndarray, gd: np.ndarray, gf: np.ndarray,
                  rand: np.ndarray | None = None) -> np.ndarray:
    """Composite lexicographic sort key (higher = better).

    Scales each criterion so it strictly dominates the next: a 1-point edge beats
    any GD edge, a 1-GD edge beats any GF edge, GF beats the random tiebreak.
    """
    points = np.asarray(points, dtype=float)
    gd = np.asarray(gd, dtype=float)
    gf = np.asarray(gf, dtype=float)
    if rand is None:
        rand = np.zeros_like(points)
    return (points * 1e9
            + (gd + 100.0) * 1e5
            + gf * 1e2
            + rand)


def rank_order(points: np.ndarray, gd: np.ndarray, gf: np.ndarray,
               rand: np.ndarray | None = None) -> np.ndarray:
    """Return row indices ordered best-to-worst along axis 0."""
    key = group_sortkey(points, gd, gf, rand)
    return np.argsort(-key, axis=0)


def best_n_indices(keys: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask of the top-``n`` rows per column for a key array [G, sims]."""
    order = np.argsort(-keys, axis=0)
    mask = np.zeros_like(keys, dtype=bool)
    cols = np.arange(keys.shape[1]) if keys.ndim == 2 else None
    top = order[:n]
    if keys.ndim == 2:
        for r in range(n):
            mask[top[r], cols] = True
    else:
        mask[top] = True
    return mask
