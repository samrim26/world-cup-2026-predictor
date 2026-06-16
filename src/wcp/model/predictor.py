"""Per-match prediction table generation (group stage + knockout).

Group-stage matches are predicted deterministically from the blended ratings and
the precomputed match context (so predictions are stable and explainable).
Knockout matches are predicted for the *most-likely* bracket occupants from the
simulation, with the regulation draw folded into an advancement probability
(extra time + skill-weighted penalties).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.matchup import matchup_adjustment
from ..utils.config import load_config
from . import bracket
from .calibration import match_confidence
from .expected_goals import driver_breakdown, match_xg
from .score_model import analyse_match
from .team_strength import key_drivers


def _drivers_string(strength: pd.DataFrame, team_a: str, team_b: str,
                    att_a, def_b, adj_a, style_notes) -> str:
    sa = strength[strength["team"] == team_a].iloc[0]
    top = key_drivers(sa, top_n=3)
    parts = [f"{k}:{v:+.2f}" for k, v in top]
    xg = driver_breakdown(att_a, def_b, adj_a)
    parts.append(f"atk-def:{xg['attack_vs_defense']:+.2f}")
    if abs(xg["match_adjustment"]) > 0.02:
        parts.append(f"ctx:{xg['match_adjustment']:+.2f}")
    if style_notes:
        parts.append(style_notes[0])
    return "; ".join(parts)


def predict_group_matches(dataset, strength: pd.DataFrame, ratings: pd.DataFrame,
                          context: pd.DataFrame,
                          quality: dict[str, float]) -> pd.DataFrame:
    r = ratings.set_index("team")
    ctx = context.set_index("match_id")
    fx = dataset.fixtures
    grp = fx[fx["stage"] == "group"]
    rows = []
    for _, m in grp.iterrows():
        a, b = m["team_a"], m["team_b"]
        c = ctx.loc[m["match_id"]] if m["match_id"] in ctx.index else None
        adj_a = float(c["adj_a"]) if c is not None else 0.0
        adj_b = float(c["adj_b"]) if c is not None else 0.0
        notes = list(c["style_notes"]) if c is not None else []
        la, lb = match_xg(r.loc[a, "attack_z"], r.loc[a, "defense_z"],
                          r.loc[b, "attack_z"], r.loc[b, "defense_z"],
                          adj_a, adj_b)
        res = analyse_match(float(la), float(lb), a, b)
        qa, qb = quality.get(a, 0.5), quality.get(b, 0.5)
        conf = match_confidence(res.p_a, res.p_draw, res.p_b, qa, qb)
        rows.append({
            "match_id": m["match_id"], "stage": "group", "group": m["group"],
            "date": m["date"].date().isoformat(), "venue": m["venue"],
            "city": m["city"], "team_a": a, "team_b": b,
            "predicted_score": res.most_likely_score,
            "team_a_xg": round(res.exp_a, 2), "team_b_xg": round(res.exp_b, 2),
            "team_a_win_prob": round(res.p_a, 3),
            "draw_prob": round(res.p_draw, 3),
            "team_b_win_prob": round(res.p_b, 3),
            "over25_prob": round(res.over25, 3), "btts_prob": round(res.btts, 3),
            "top_scorelines": "; ".join(f"{s} ({p:.0%})"
                                        for s, p in res.top_scorelines),
            "confidence": round(conf, 3),
            "data_quality": round(0.5 * (qa + qb), 3),
            "key_model_drivers": _drivers_string(
                strength, a, b, r.loc[a, "attack_z"], r.loc[b, "defense_z"],
                adj_a, notes),
        })
    return pd.DataFrame(rows)


def predict_knockout_matches(dataset, strength: pd.DataFrame,
                             ratings: pd.DataFrame, sim_result,
                             quality: dict[str, float]) -> pd.DataFrame:
    """Predict R32 matches using the modal group qualifiers.

    Provides regulation xG/scoreline plus an advancement probability that folds
    the draw into extra time + penalties.
    """
    cfg = load_config()
    r = ratings.set_index("team")
    s = strength.set_index("team")
    modal = sim_result.bracket_modal
    fx = dataset.fixtures
    ko = fx[fx["stage"] == "R32"].reset_index(drop=True)

    # Resolve modal R32 occupants from modal winners/runners (thirds approximate
    # to the most common third per host slot is complex; for the readable bracket
    # we fill winner/runner slots and label third slots generically).
    def occupant(spec):
        kind, key = spec
        if kind == "W":
            return modal.get(f"{key}1")
        if kind == "R":
            return modal.get(f"{key}2")
        return None  # third-place slot: occupant varies, left for report note

    rows = []
    for i, (a_spec, b_spec) in enumerate(bracket.R32_MATCHES):
        a = occupant(a_spec)
        b = occupant(b_spec)
        m = ko.iloc[i] if i < len(ko) else None
        rec = {
            "match_id": m["match_id"] if m is not None else f"R32-{i+1}",
            "stage": "R32", "group": "",
            "date": m["date"].date().isoformat() if m is not None else "",
            "venue": m["venue"] if m is not None else "",
            "city": m["city"] if m is not None else "",
            "team_a": a or f"Best-3rd (slot {a_spec[1]})",
            "team_b": b or f"Best-3rd (slot {b_spec[1]})",
        }
        if a and b:
            la, lb = match_xg(r.loc[a, "attack_z"], r.loc[a, "defense_z"],
                              r.loc[b, "attack_z"], r.loc[b, "defense_z"],
                              knockout=True)
            res = analyse_match(float(la), float(lb), a, b)
            # Advancement: regulation win + half of draw mass nudged by penalty skill.
            k = cfg.get("simulation.penalty_skill_scale", 0.10)
            edge = k * ((s.loc[a, "penalty_rating"] - s.loc[b, "penalty_rating"])
                        + 0.5 * (s.loc[a, "gk_rating"] - s.loc[b, "gk_rating"]))
            so_a = float(np.clip(0.5 + edge, 0.05, 0.95))
            adv_a = res.p_a + res.p_draw * so_a
            qa, qb = quality.get(a, 0.5), quality.get(b, 0.5)
            rec.update({
                "predicted_score": res.most_likely_score,
                "team_a_xg": round(res.exp_a, 2), "team_b_xg": round(res.exp_b, 2),
                "team_a_win_prob": round(res.p_a, 3),
                "draw_prob": round(res.p_draw, 3),
                "team_b_win_prob": round(res.p_b, 3),
                "team_a_advance_prob": round(adv_a, 3),
                "over25_prob": round(res.over25, 3),
                "btts_prob": round(res.btts, 3),
                "top_scorelines": "; ".join(f"{sc} ({p:.0%})"
                                            for sc, p in res.top_scorelines),
                "confidence": round(match_confidence(
                    res.p_a, res.p_draw, res.p_b, qa, qb), 3),
                "data_quality": round(0.5 * (qa + qb), 3),
                "key_model_drivers": "knockout (reduced tempo); "
                                     "advance folds ET+penalties",
            })
        rows.append(rec)
    return pd.DataFrame(rows)


def predict_all(dataset, strength, ratings, context, sim_result,
                quality: dict[str, float]) -> pd.DataFrame:
    g = predict_group_matches(dataset, strength, ratings, context, quality)
    k = predict_knockout_matches(dataset, strength, ratings, sim_result, quality)
    return pd.concat([g, k], ignore_index=True)
