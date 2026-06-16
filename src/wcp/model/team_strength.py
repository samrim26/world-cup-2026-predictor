"""Layers 1 & 2: composite team strength and attack/defense decomposition.

``compute_team_strength`` produces, for each of the 48 teams, a single
``composite`` rating (a weighted, config-driven blend of the Layer-1 components)
plus the Layer-2 decomposition (attack, defense, midfield, gk, set-piece, ...).
Every component contribution is retained for explainability ("key drivers").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.config import load_config
from . import ratings


# Map config weight keys -> Layer-1 component functions.
_COMPONENTS = {
    "elo": ratings.elo_component,
    "recent_form": ratings.recent_form_component,
    "squad_value": ratings.squad_value_component,
    "qualification": ratings.qualification_component,
    "attack_defense_balance": ratings.attack_defense_balance_component,
    "manager_stability": ratings.manager_component,
    "confederation": ratings.confederation_component,
    "tournament_experience": ratings.experience_component,
    "goalkeeper_penalty": ratings.gk_penalty_component,
}


def compute_team_strength(teams: pd.DataFrame,
                          history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a per-team strength table with composite + decomposition + the
    signed contribution of each component (columns ``contrib_<name>``)."""
    cfg = load_config()
    weights = cfg.get("strength_weights", {})

    out = teams[["team", "group", "seed_pos", "confederation", "region",
                 "host_flag", "fifa_rank", "elo"]].copy()

    composite = pd.Series(np.zeros(len(teams)), index=teams.index)
    for name, fn in _COMPONENTS.items():
        comp = fn(teams, history) if name == "recent_form" else fn(teams)
        w = float(weights.get(name, 0.0))
        contrib = w * comp
        out[f"contrib_{name}"] = contrib.round(4)
        composite = composite + contrib
    out["composite"] = composite

    _decompose(out, teams)
    # Convenience: composite expressed on a 0-100 readable scale.
    out["composite_100"] = (50 + 12 * out["composite"]).round(1)
    return out.sort_values("composite", ascending=False).reset_index(drop=True)


def _decompose(out: pd.DataFrame, teams: pd.DataFrame) -> None:
    """Layer 2: derive facet ratings from composite + style sub-ratings.

    Attack/defense are tilted by the team's offensive vs defensive style so two
    teams with equal composite can still differ in how they generate/concede
    goals — which the xG model needs.
    """
    t = teams.set_index("team")
    comp = out.set_index("team")["composite"]

    def z(col: str) -> pd.Series:
        s = t[col]
        return (s - s.mean()) / (s.std(ddof=0) or 1.0)

    attack_tilt = 0.5 * z("possession_rating") + 0.5 * z("transition_rating")
    defense_tilt = 0.5 * z("pressing_rating") + 0.5 * z("aerial_rating")

    out["attack_rating"] = (comp + 0.35 * attack_tilt).values
    out["defense_rating"] = (comp + 0.35 * defense_tilt).values
    out["midfield_rating"] = (comp + 0.25 * z("possession_rating")).values
    out["gk_rating"] = z("gk_rating").values
    out["set_piece_rating"] = z("set_piece_rating").values
    out["transition_rating"] = z("transition_rating").values
    out["possession_rating"] = z("possession_rating").values
    out["pressing_rating"] = z("pressing_rating").values
    out["aerial_rating"] = z("aerial_rating").values
    out["depth_rating"] = z("depth_rating").values
    out["penalty_rating"] = z("penalty_rating").values
    out["discipline_rating"] = z("discipline_rating").values
    out["manager_rating"] = z("manager_tenure_months").values
    # Readable squad strength 0-100.
    out["squad_rating"] = (50 + 12 * comp).round(1).values


def key_drivers(strength_row: pd.Series, top_n: int = 4) -> list[tuple[str, float]]:
    """Return the top signed component contributions for a team (for reports)."""
    contribs = {k.replace("contrib_", ""): v
                for k, v in strength_row.items() if k.startswith("contrib_")}
    ordered = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ordered[:top_n]
