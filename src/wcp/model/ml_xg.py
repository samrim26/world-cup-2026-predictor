"""Machine-learning expected-goals estimator.

Trains a gradient-boosted regressor on historical matches to predict goals
scored as a function of rating features, then derives a per-team attack and
defence rating by asking the model how many goals each team scores/concedes
against a league-average opponent.

Backend: scikit-learn's HistGradientBoostingRegressor by default (always
available); upgrades to XGBoost when installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .weighting import match_weights

log = get_logger("ml_xg")


@dataclass
class MLEstimate:
    attack_z: pd.Series
    defense_z: pd.Series
    backend: str


def _make_regressor():
    try:
        from xgboost import XGBRegressor  # type: ignore
        return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                            subsample=0.9, random_state=42), "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                            max_iter=300, random_state=42), "sklearn-hgb"


def _features(elo_for: np.ndarray, elo_against: np.ndarray,
              rank_for: np.ndarray, rank_against: np.ndarray,
              is_home: np.ndarray) -> np.ndarray:
    return np.column_stack([
        elo_for, elo_against, elo_for - elo_against,
        rank_for, rank_against, is_home,
    ])


def estimate(history: pd.DataFrame, teams_df: pd.DataFrame) -> MLEstimate:
    teams = list(teams_df["team"])
    elo = teams_df.set_index("team")["elo"].to_dict()
    rank = teams_df.set_index("team")["fifa_rank"].to_dict()
    known = set(teams)

    h = history[history["home_team"].isin(known)
                & history["away_team"].isin(known)].copy()
    w_match = match_weights(h)
    # Two rows per match: each team's goals-for as the target (shared weight).
    elo_for, elo_ag, rk_for, rk_ag, home, target, weight = ([] for _ in range(7))
    for mw, (_, m) in zip(w_match, h.iterrows()):
        for side, opp, gf, is_h in (("home_team", "away_team", "home_score", 1),
                                    ("away_team", "home_team", "away_score", 0)):
            t, o = m[side], m[opp]
            elo_for.append(elo[t]); elo_ag.append(elo[o])
            rk_for.append(rank[t]); rk_ag.append(rank[o])
            home.append(is_h); target.append(m[gf]); weight.append(mw)

    X = _features(np.array(elo_for), np.array(elo_ag), np.array(rk_for),
                  np.array(rk_ag), np.array(home))
    y = np.array(target, dtype=float)
    sw = np.array(weight, dtype=float)

    model, backend = _make_regressor()
    try:
        model.fit(X, y, sample_weight=sw)
    except TypeError:
        model.fit(X, y)
    log.info("ML xG backend=%s trained on %d weighted rows", backend, len(y))

    avg_elo = np.mean(list(elo.values()))
    avg_rank = np.mean(list(rank.values()))
    atk, dfn = {}, {}
    for t in teams:
        # Goals this team scores vs an average opponent (neutral) -> attack.
        xf = _features(np.array([elo[t]]), np.array([avg_elo]),
                       np.array([rank[t]]), np.array([avg_rank]), np.array([0]))
        atk[t] = float(model.predict(xf)[0])
        # Goals an average opponent scores vs this team -> defence (invert).
        xd = _features(np.array([avg_elo]), np.array([elo[t]]),
                       np.array([avg_rank]), np.array([rank[t]]), np.array([0]))
        dfn[t] = -float(model.predict(xd)[0])

    atk_s = pd.Series(atk)
    dfn_s = pd.Series(dfn)
    atk_z = (atk_s - atk_s.mean()) / (atk_s.std(ddof=0) or 1.0)
    dfn_z = (dfn_s - dfn_s.mean()) / (dfn_s.std(ddof=0) or 1.0)
    return MLEstimate(atk_z.reindex(teams), dfn_z.reindex(teams), backend)
