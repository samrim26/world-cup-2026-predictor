"""ROI / value-betting engine.

Compares model probabilities against vig-removed market probabilities, computes
edge, expected value and fractional-Kelly stake, applies an uncertainty penalty
so thin or low-confidence edges are not over-bet, and flags each opportunity.

Gated entirely on odds availability: with no market loaded, the board reports
"no odds" rather than inventing prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.config import load_config
from ..utils.odds_math import fair_probs, implied_prob, kelly_fraction, to_decimal


def _flag(edge: float, cfg) -> str:
    if edge >= cfg.get("value.min_edge_strong", 0.06):
        return "STRONG"
    if edge >= cfg.get("value.min_edge_moderate", 0.03):
        return "MODERATE"
    if edge >= cfg.get("value.min_edge_thin", 0.015):
        return "THIN"
    if edge <= -cfg.get("value.min_edge_thin", 0.015):
        return "MARKET_AGAINST"
    return "NO_BET"


def build_value_board(model_probs: dict[str, float], odds: pd.DataFrame,
                      confidence: dict[str, float] | None = None,
                      data_quality: dict[str, float] | None = None,
                      market: str = "tournament_winner") -> pd.DataFrame:
    """Build a value board for one market.

    ``model_probs`` maps selection -> model probability. ``odds`` is the odds
    table; rows with the requested ``market`` are used. De-vigging is applied
    across the selections present so the comparison is margin-free.
    """
    cfg = load_config()
    confidence = confidence or {}
    data_quality = data_quality or {}

    mk = odds[odds["market"] == market].copy()
    if mk.empty:
        return pd.DataFrame(columns=["selection", "note"]).assign(
            note="no odds loaded for market")

    fmt = mk["format"].iloc[0] if "format" in mk.columns else "decimal"
    fair = fair_probs(mk["odds"].tolist(), fmt,
                      method=cfg.get("value.devig_method", "proportional"))
    mk = mk.assign(market_prob=fair,
                   raw_implied=[implied_prob(o, fmt) for o in mk["odds"]])

    rows = []
    pen = cfg.get("value.uncertainty_penalty", 0.5)
    kfrac = cfg.get("value.kelly_fraction", 0.25)
    kcap = cfg.get("value.kelly_cap", 0.05)
    for _, r in mk.iterrows():
        sel = r["selection"]
        mp = model_probs.get(sel)
        if mp is None:
            continue
        dec = to_decimal(r["odds"], fmt)
        edge = mp - r["market_prob"]
        conf = confidence.get(sel, 0.6)
        dq = data_quality.get(sel, 0.6)
        # Shrink edge by an uncertainty penalty driven by confidence*quality.
        shrunk = edge - pen * edge * (1 - conf * dq) if edge > 0 else edge
        ev = mp * (dec - 1) - (1 - mp)        # per 1 unit staked
        kelly = max(0.0, kelly_fraction(mp, dec)) * kfrac
        kelly = min(kelly, kcap)
        # Do not stake on a negative shrunk edge.
        if shrunk <= 0:
            kelly = 0.0
        rows.append({
            "market": market, "selection": sel,
            "odds": r["odds"], "market_prob": round(float(r["market_prob"]), 4),
            "model_prob": round(float(mp), 4),
            "edge": round(float(edge), 4),
            "edge_adj": round(float(shrunk), 4),
            "expected_value": round(float(ev), 4),
            "kelly_stake": round(float(kelly), 4),
            "confidence": round(float(conf), 3),
            "data_quality": round(float(dq), 3),
            "flag": _flag(shrunk, cfg),
        })
    board = pd.DataFrame(rows)
    if board.empty:
        return board
    return board.sort_values("edge_adj", ascending=False).reset_index(drop=True)
