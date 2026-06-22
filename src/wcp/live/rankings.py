"""Full 48-team ranking, ordered by live odds to reach the Round of 32."""
from __future__ import annotations

from .advance_odds import ABBR_ELO
from .clinch import group_clinch


def build_rankings(groups: dict[str, list[dict]], odds: dict[str, float],
                   remaining_by_group: dict | None = None) -> list[dict]:
    remaining_by_group = remaining_by_group or {}
    rows = []
    for g, teams in groups.items():
        cl = group_clinch(teams, remaining_by_group.get(g, []))
        for t in teams:
            ab = t["abbr"]
            c = cl.get(ab, {})
            rows.append({
                "abbr": ab, "name": t["team"], "group": g,
                "P": t["P"], "W": t["W"], "D": t["D"], "L": t["L"],
                "GF": t["GF"], "GA": t["GA"], "GD": t["GD"], "Pts": t["Pts"],
                "elo": ABBR_ELO.get(ab, 1700),
                "r32_odds": odds.get(ab, 0.0),
                "clinched_first": c.get("clinched_first", False),
                "clinched_advance": c.get("clinched_advance", False),
                "eliminated": c.get("out_entirely", False),
            })
    # Rank by R32 odds, then points, GD, Elo as stable tiebreaks.
    rows.sort(key=lambda r: (r["r32_odds"], r["Pts"], r["GD"], r["elo"]),
              reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows
