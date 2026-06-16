"""Probability calibration and per-match model-confidence scoring.

Calibration: isotonic regression mapping raw model win-probabilities onto
empirically observed frequencies, validated on historical results. When there
is too little labelled data the identity map is returned (no false precision).

Confidence: a 0-1 score combining (a) how decisive the prediction is, (b) the
ensemble agreement, and (c) the data-quality of the two teams. Feeds the value
engine's uncertainty penalty and the report's confidence labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class Calibrator:
    iso: IsotonicRegression | None
    fitted: bool

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if not self.fitted or self.iso is None:
            return p
        return np.clip(self.iso.predict(np.clip(p, 0, 1)), 1e-4, 1 - 1e-4)


def fit_calibrator(pred_probs: np.ndarray, outcomes: np.ndarray,
                   min_samples: int = 200) -> Calibrator:
    """Fit isotonic calibration of predicted win-prob vs binary outcome."""
    pred_probs = np.asarray(pred_probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if len(pred_probs) < min_samples:
        return Calibrator(None, False)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pred_probs, outcomes)
    return Calibrator(iso, True)


def reliability_curve(pred: np.ndarray, outcome: np.ndarray,
                      bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean predicted, observed frequency) per probability bin."""
    pred = np.asarray(pred); outcome = np.asarray(outcome)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(pred, edges) - 1, 0, bins - 1)
    mp, of = [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            mp.append(pred[m].mean()); of.append(outcome[m].mean())
    return np.array(mp), np.array(of)


def match_confidence(p_a: float, p_draw: float, p_b: float,
                     quality_a: float, quality_b: float,
                     ensemble_agreement: float = 1.0) -> float:
    """0-1 confidence for a single match prediction.

    Decisiveness uses normalised entropy of the 1X2 distribution (a near-coinflip
    is low confidence); blended with mean data-quality and ensemble agreement.
    """
    probs = np.array([p_a, p_draw, p_b])
    probs = probs / probs.sum()
    probs = np.clip(probs, 1e-9, 1)
    entropy = -(probs * np.log(probs)).sum() / np.log(3)   # 0..1, 1 = max unc.
    decisiveness = 1.0 - entropy
    quality = 0.5 * (quality_a + quality_b)
    conf = 0.5 * decisiveness + 0.35 * quality + 0.15 * ensemble_agreement
    return float(np.clip(conf, 0.0, 1.0))
