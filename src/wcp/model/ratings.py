"""Layer-1 components: turn raw inputs into normalised rating components.

Each component is returned on a common standardised scale (mean 0, sd 1 across
the 48 teams) so the configured weights are directly comparable. The composite
is assembled in :mod:`wcp.model.team_strength`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest.match_history import MatchHistoryLoader
from ..utils.config import load_config


def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def elo_component(teams: pd.DataFrame) -> pd.Series:
    return _zscore(teams["elo"])


def fifa_component(teams: pd.DataFrame) -> pd.Series:
    # Higher rank number = worse, so invert before standardising.
    return _zscore(-teams["fifa_rank"].astype(float))


def squad_value_component(teams: pd.DataFrame) -> pd.Series:
    # Market value is heavy-tailed; log first.
    return _zscore(np.log1p(teams["squad_value_m"]))


def qualification_component(teams: pd.DataFrame) -> pd.Series:
    return _zscore(teams["qualification_pts"])


def manager_component(teams: pd.DataFrame) -> pd.Series:
    # Stability proxy: tenure with diminishing returns past ~3 years.
    tenure = teams["manager_tenure_months"].clip(upper=60)
    return _zscore(np.sqrt(tenure))


def confederation_component(teams: pd.DataFrame) -> pd.Series:
    cfg = load_config()
    adj = cfg.get("confederation_adjustment", {})
    raw = teams["confederation"].map(adj).fillna(0.85)
    return _zscore(raw)


def experience_component(teams: pd.DataFrame) -> pd.Series:
    return _zscore(teams["tournament_experience"])


def gk_penalty_component(teams: pd.DataFrame) -> pd.Series:
    return _zscore(0.6 * teams["gk_rating"] + 0.4 * teams["penalty_rating"])


def attack_defense_balance_component(teams: pd.DataFrame) -> pd.Series:
    """A team that is strong in both possession and transition (two-way) gets a
    small balance bonus; lopsided sides are penalised."""
    bal = teams[["possession_rating", "transition_rating",
                 "pressing_rating"]].mean(axis=1)
    spread = teams[["possession_rating", "transition_rating",
                    "pressing_rating"]].std(axis=1)
    return _zscore(bal - 0.3 * spread)


def recent_form_component(teams: pd.DataFrame,
                          history: pd.DataFrame | None = None) -> pd.Series:
    """Opposition-adjusted recent form from match history.

    Form = average (goal_diff + opponent_strength_credit) over a team's most
    recent matches, exponentially weighted toward the latest games. Falls back
    to zeros when no history is available.
    """
    if history is None:
        history = MatchHistoryLoader().load()
    if history is None or history.empty:
        return pd.Series(np.zeros(len(teams)), index=teams.index)

    elo = dict(zip(teams["team"], teams["elo"]))
    elo_mean = np.mean(list(elo.values()))
    elo_sd = np.std(list(elo.values())) or 1.0
    scores: dict[str, float] = {}
    hist = history.sort_values("date")
    for team in teams["team"]:
        as_home = hist[hist["home_team"] == team].assign(
            gf=lambda d: d["home_score"], ga=lambda d: d["away_score"],
            opp=lambda d: d["away_team"])
        as_away = hist[hist["away_team"] == team].assign(
            gf=lambda d: d["away_score"], ga=lambda d: d["home_score"],
            opp=lambda d: d["home_team"])
        games = pd.concat([as_home, as_away]).sort_values("date").tail(10)
        if games.empty:
            scores[team] = 0.0
            continue
        n = len(games)
        w = np.exp(np.linspace(-1.0, 0.0, n))  # recent games weigh more
        gd = (games["gf"] - games["ga"]).to_numpy(dtype=float)
        opp_strength = games["opp"].map(
            lambda o: (elo.get(o, elo_mean) - elo_mean) / elo_sd).to_numpy()
        # Reward results against stronger opponents.
        val = gd + 0.4 * opp_strength * np.sign(gd + 1e-9)
        scores[team] = float(np.average(val, weights=w))
    return _zscore(teams["team"].map(scores))
