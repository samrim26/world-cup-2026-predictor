"""Precompute deterministic per-fixture, per-side log-lambda adjustments.

These do NOT depend on simulated outcomes, so they are computed once and reused
across all Monte-Carlo iterations. The simulator then only adds per-sim team
strength perturbations on top.
"""
from __future__ import annotations

import pandas as pd

from . import matchup, travel, venues


def build_match_context(fixtures: pd.DataFrame, strength: pd.DataFrame,
                        venues_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per group fixture with adjustment deltas + notes.

    Columns: match_id, team_a, team_b, adj_a, adj_b, style_notes (list).
    Knockout fixtures are handled at simulation time (teams unknown here).
    """
    sched = travel.team_schedule(fixtures)
    by_team = {row["team"]: row for _, row in strength.iterrows()}
    vmap = venues_df.set_index("venue")

    rows = []
    grp = fixtures[fixtures["stage"] == "group"]
    for _, m in grp.iterrows():
        a, b = m["team_a"], m["team_b"]
        ra, rb = by_team[a], by_team[b]
        venue_row = vmap.loc[m["venue"]] if m["venue"] in vmap.index else m

        # Home / regional advantage.
        adj_a = venues.home_advantage(ra, m["country"])
        adj_b = venues.home_advantage(rb, m["country"])
        # Altitude + climate.
        adj_a += venues.altitude_climate_adjustment(ra, venue_row)
        adj_b += venues.altitude_climate_adjustment(rb, venue_row)
        # Rest + travel.
        r_a, t_a = travel.rest_travel_adjustment(a, m["match_id"], sched)
        r_b, t_b = travel.rest_travel_adjustment(b, m["match_id"], sched)
        adj_a += r_a + t_a
        adj_b += r_b + t_b
        # Tactical matchup (symmetric: each side as attacker).
        ma, notes_a = matchup.matchup_adjustment(ra, rb)
        mb, notes_b = matchup.matchup_adjustment(rb, ra)
        adj_a += ma
        adj_b += mb

        rows.append({
            "match_id": m["match_id"], "team_a": a, "team_b": b,
            "adj_a": round(adj_a, 4), "adj_b": round(adj_b, 4),
            "style_notes": notes_a + notes_b,
        })
    return pd.DataFrame(rows)
