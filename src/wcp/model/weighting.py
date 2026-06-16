"""Match weighting for fitting on real results.

Recent and competitive matches inform a team's current strength more than old
friendlies. Weight = time-decay × competition importance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.config import load_config

# Competition-importance multipliers by keyword (case-insensitive substring).
_COMP_WEIGHTS = [
    ("friendly", 0.55),
    ("qualification", 1.00),
    ("nations league", 1.05),
    ("world cup", 1.30),
    ("copa", 1.20),
    ("euro", 1.20),
    ("african cup", 1.15),
    ("asian cup", 1.10),
    ("gold cup", 1.10),
    ("confederations", 1.20),
]


def competition_weight(name: str) -> float:
    n = str(name).lower()
    for key, w in _COMP_WEIGHTS:
        if key in n:
            return w
    return 0.9  # unknown/minor competition


def match_weights(history: pd.DataFrame,
                  ref_date: pd.Timestamp | None = None) -> np.ndarray:
    """Per-match fitting weight = recency decay × competition importance."""
    cfg = load_config()
    half_life = cfg.get("ratings.recency_half_life_days", 540)
    dates = pd.to_datetime(history["date"])
    ref = ref_date if ref_date is not None else dates.max()
    age_days = (ref - dates).dt.days.clip(lower=0).to_numpy()
    decay = 0.5 ** (age_days / float(half_life))
    comp = history["competition"].map(competition_weight).to_numpy() \
        if "competition" in history.columns else np.ones(len(history))
    return decay * comp
