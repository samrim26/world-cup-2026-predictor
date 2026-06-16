"""Live ranking logic: group qualification + the 8 best third-place wildcards.

Reuses the FIFA tiebreaker ordering (points -> goal difference -> goals scored)
from :mod:`wcp.model.standings`, applied to the real live standings.
"""
from __future__ import annotations

import numpy as np

from ..model.standings import group_sortkey

N_WILDCARDS = 8        # 8 best third-placed teams advance to the Round of 32


def qualification_flags(groups: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Tag each team row with a live status: top-2 = advancing, 3rd = wildcard
    contender, 4th = eliminated-track. (Provisional until the group finishes.)"""
    thirds = third_place_race(groups)
    wildcard_in = {(t["group"], t["team"]) for t in thirds
                   if t["wildcard_status"] == "IN"}
    out = {}
    for g, rows in groups.items():
        tagged = []
        for i, t in enumerate(rows):
            row = dict(t)
            if i < 2:
                row["live_status"] = "advancing (top 2)"
            elif i == 2:
                row["live_status"] = ("wildcard IN" if (g, t["team"]) in wildcard_in
                                      else "wildcard out")
            else:
                row["live_status"] = "eliminated-track"
            tagged.append(row)
        out[g] = tagged
    return out


def third_place_race(groups: dict[str, list[dict]]) -> list[dict]:
    """Rank the twelve third-placed teams; top 8 hold a wildcard spot.

    Ranked by points -> GD -> GF (FIFA criteria; fair-play/lots not modelled).
    """
    thirds = []
    for g, rows in groups.items():
        if len(rows) >= 3:
            t = dict(rows[2])      # rows are sorted by in-group rank
            t["group"] = g
            thirds.append(t)
    if not thirds:
        return []
    key = group_sortkey(
        np.array([t["Pts"] for t in thirds], dtype=float),
        np.array([t["GD"] for t in thirds], dtype=float),
        np.array([t["GF"] for t in thirds], dtype=float),
    )
    order = np.argsort(-key)
    ranked = []
    for pos, idx in enumerate(order):
        t = thirds[int(idx)]
        t["wildcard_rank"] = pos + 1
        t["wildcard_status"] = "IN" if pos < N_WILDCARDS else "OUT"
        ranked.append(t)
    return ranked


def tournament_complete_groups(groups: dict[str, list[dict]]) -> int:
    """How many groups have every team on 3 games played (group stage done)."""
    done = 0
    for rows in groups.values():
        if rows and all(t["P"] >= 3 for t in rows):
            done += 1
    return done
