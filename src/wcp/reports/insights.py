"""Value-add analytical modules ("the edge").

Each function takes the assembled model objects and returns a tidy DataFrame so
the report and dashboard can render them and the CLI can save them as CSVs.
All are explainable: every flag traces back to a named, comparable quantity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.odds_math import fair_probs


# --------------------------------------------------------------------------- #
def overrated_underrated(team_probs: pd.DataFrame, teams: pd.DataFrame,
                         odds: pd.DataFrame | None) -> pd.DataFrame:
    """Fraud Alert / Overrated + its mirror (underrated).

    Compares the model's champion probability to a reputation baseline. If
    tournament-winner odds exist, reputation = market; otherwise reputation is
    derived from FIFA ranking. A team the market/reputation rates well above the
    model is 'overrated' (negative gap); the reverse is 'underrated'.
    """
    tp = team_probs.set_index("team")
    rep = _reputation_prob(team_probs, teams, odds)
    rows = []
    for t in tp.index:
        model_p = float(tp.loc[t, "p_champion"])
        rep_p = float(rep.get(t, np.nan))
        if np.isnan(rep_p):
            continue
        gap = model_p - rep_p
        rows.append({"team": t, "group": tp.loc[t, "group"],
                     "model_champion": round(model_p, 4),
                     "reputation_champion": round(rep_p, 4),
                     "gap": round(gap, 4),
                     "verdict": "UNDERRATED" if gap > 0 else "OVERRATED"})
    df = pd.DataFrame(rows).sort_values("gap")
    return df.reset_index(drop=True)


def _reputation_prob(team_probs, teams, odds) -> dict[str, float]:
    if odds is not None and not odds.empty:
        mk = odds[odds["market"] == "tournament_winner"]
        if not mk.empty:
            fmt = mk["format"].iloc[0]
            fair = fair_probs(mk["odds"].tolist(), fmt)
            return dict(zip(mk["selection"], fair))
    # Fallback: reputation from FIFA rank via a softmax over -rank.
    t = teams.set_index("team")["fifa_rank"]
    z = (-t.astype(float))
    w = np.exp((z - z.mean()) / (z.std() or 1.0) * 1.3)
    p = w / w.sum()
    return p.to_dict()


# --------------------------------------------------------------------------- #
def upset_radar(predictions: pd.DataFrame, teams: pd.DataFrame,
                threshold: float = 0.33) -> pd.DataFrame:
    """Group matches where the reputational underdog (worse FIFA rank) has a
    materially higher win probability than reputation implies."""
    rank = teams.set_index("team")["fifa_rank"].to_dict()
    g = predictions[predictions["stage"] == "group"]
    rows = []
    for _, m in g.iterrows():
        a, b = m["team_a"], m["team_b"]
        if a not in rank or b not in rank:
            continue
        # Underdog = worse (higher) FIFA rank.
        if rank[a] > rank[b]:
            dog, dog_p, fav = a, m["team_a_win_prob"], b
        else:
            dog, dog_p, fav = b, m["team_b_win_prob"], a
        if dog_p >= threshold:
            rows.append({"match_id": m["match_id"], "favourite": fav,
                         "underdog": dog, "underdog_win_prob": round(dog_p, 3),
                         "predicted_score": m["predicted_score"],
                         "rank_gap": abs(rank[a] - rank[b])})
    return pd.DataFrame(rows).sort_values("underdog_win_prob",
                                          ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
def dark_horse_index(team_probs: pd.DataFrame, teams: pd.DataFrame,
                     odds: pd.DataFrame | None) -> pd.DataFrame:
    """Dark-Horse Index: teams outside the elite reputation tier whose model
    deep-run probability and odds value are high.

    Combines (a) model SF reach probability, (b) value vs reputation, and
    (c) a 'not already a favourite' factor, into a 0-100 index.
    """
    tp = team_probs.set_index("team")
    rep = _reputation_prob(team_probs, teams, odds)
    rows = []
    for t in tp.index:
        model_sf = float(tp.loc[t, "p_sf"])
        model_champ = float(tp.loc[t, "p_champion"])
        rep_p = float(rep.get(t, model_champ))
        value = model_champ - rep_p
        outsider = 1.0 - min(rep_p / 0.12, 1.0)     # 0 for top favourites
        index = 100 * (0.5 * model_sf + 0.3 * max(value, 0) * 5 + 0.2 * outsider * model_sf * 3)
        rows.append({"team": t, "group": tp.loc[t, "group"],
                     "p_semifinal": round(model_sf, 4),
                     "value_vs_reputation": round(value, 4),
                     "dark_horse_index": round(float(index), 1)})
    df = pd.DataFrame(rows).sort_values("dark_horse_index", ascending=False)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
def path_difficulty(team_probs: pd.DataFrame, strength: pd.DataFrame
                    ) -> pd.DataFrame:
    """Path-Difficulty Score: conversion efficiency from reaching the knockouts
    to going deep, controlled for team strength.

    A strong team with a low (deep-run / qualify) conversion has a hard path.
    Score is normalised 0-100 (higher = harder path than strength predicts).
    """
    tp = team_probs.set_index("team")
    s = strength.set_index("team")["composite"]
    rows = []
    for t in tp.index:
        qualify = max(float(tp.loc[t, "p_r32"]), 1e-6)
        deep = float(tp.loc[t, "p_qf"])               # reach QF
        conversion = deep / qualify
        strength_z = float(s.get(t, 0.0))
        # Expected conversion rises with strength; residual = difficulty.
        expected = 1 / (1 + np.exp(-1.1 * strength_z)) * 0.6
        difficulty = expected - conversion
        rows.append({"team": t, "group": tp.loc[t, "group"],
                     "qualify_to_QF_conversion": round(conversion, 3),
                     "path_difficulty": round(float(difficulty), 4)})
    df = pd.DataFrame(rows)
    lo, hi = df["path_difficulty"].min(), df["path_difficulty"].max()
    df["path_difficulty_score"] = ((df["path_difficulty"] - lo) /
                                   ((hi - lo) or 1) * 100).round(1)
    return df.sort_values("path_difficulty_score",
                          ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
def bracket_leverage(team_probs: pd.DataFrame) -> pd.DataFrame:
    """Bracket-Leverage Score: how much a team's deep-run probability is tied to
    winning its group (proxy: champion prob per unit of group-win prob vs per
    unit of runner-up prob). High leverage = winning the group matters a lot."""
    tp = team_probs
    df = tp[["team", "group", "p_win_group", "p_runner_up", "p_champion",
             "p_sf"]].copy()
    df["bracket_leverage"] = (df["p_win_group"] - df["p_runner_up"]).abs().round(3)
    # Scale by how much title equity the team has at all.
    df["leverage_weighted"] = (df["bracket_leverage"] * df["p_sf"] * 100).round(2)
    return df.sort_values("leverage_weighted", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
def consensus_vs_model(team_probs: pd.DataFrame, teams: pd.DataFrame,
                       odds: pd.DataFrame | None) -> pd.DataFrame:
    """Side-by-side of model champion prob vs market/reputation, ranked by the
    absolute disagreement."""
    tp = team_probs.set_index("team")
    rep = _reputation_prob(team_probs, teams, odds)
    rows = []
    for t in tp.index:
        m = float(tp.loc[t, "p_champion"])
        c = float(rep.get(t, np.nan))
        if np.isnan(c):
            continue
        rows.append({"team": t, "model_champion": round(m, 4),
                     "consensus_champion": round(c, 4),
                     "disagreement": round(m - c, 4)})
    return pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)["disagreement"].abs().sort_values(
            ascending=False).index).reset_index(drop=True)
