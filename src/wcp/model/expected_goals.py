"""Layer-4 expected-goals: turn team strengths + match adjustments into the
Poisson rate parameters (lambda) for each side.

Vectorised: every input may be a scalar or a NumPy array, so the simulator can
evaluate all 72 group matches across thousands of perturbed-strength draws at
once.
"""
from __future__ import annotations

import numpy as np

from ..utils.config import load_config


def match_xg(attack_a, defense_a, attack_b, defense_b,
             adj_a=0.0, adj_b=0.0, *, knockout: bool = False):
    """Return (lambda_a, lambda_b) expected goals.

    ``attack_*`` / ``defense_*`` are standardised (z-scored) ratings. The log
    goal rate for A rises with A's attack over B's defence and with A's
    match-specific adjustment ``adj_a``.
    """
    cfg = load_config()
    base = np.log(cfg.get("xg.base_goal_rate", 1.35))
    scale = cfg.get("xg.strength_to_xg_scale", 0.55)

    log_a = base + scale * (np.asarray(attack_a) - np.asarray(defense_b)) + np.asarray(adj_a)
    log_b = base + scale * (np.asarray(attack_b) - np.asarray(defense_a)) + np.asarray(adj_b)

    if knockout:
        tempo = np.log(cfg.get("xg.knockout_tempo", 0.92))
        log_a = log_a + tempo
        log_b = log_b + tempo

    lam_a = np.exp(log_a)
    lam_b = np.exp(log_b)
    # Clip to a sane football range to avoid pathological tails.
    return np.clip(lam_a, 0.15, 5.0), np.clip(lam_b, 0.15, 5.0)


def driver_breakdown(attack_a, defense_b, adj_a, label_a="A") -> dict[str, float]:
    """Explainability: decompose A's log goal rate into named contributions."""
    cfg = load_config()
    scale = cfg.get("xg.strength_to_xg_scale", 0.55)
    return {
        "base_rate": round(float(np.log(cfg.get("xg.base_goal_rate", 1.35))), 3),
        "attack_vs_defense": round(float(scale * (attack_a - defense_b)), 3),
        "match_adjustment": round(float(adj_a), 3),
    }
