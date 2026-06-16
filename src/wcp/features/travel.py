"""Rest-days and travel-fatigue adjustments (group stage, where the schedule is
known per team)."""
from __future__ import annotations

import pandas as pd

from ..utils.config import load_config
from ..utils.geo import approx_timezone_offset_hours, haversine_km


def team_schedule(fixtures: pd.DataFrame) -> dict[str, list[dict]]:
    """Return, per team, its ordered list of group matches with venue geo."""
    grp = fixtures[fixtures["stage"] == "group"].sort_values("date")
    sched: dict[str, list[dict]] = {}
    for _, m in grp.iterrows():
        for side in ("team_a", "team_b"):
            sched.setdefault(m[side], []).append({
                "date": m["date"], "lat": m["lat"], "lon": m["lon"],
                "match_id": m["match_id"],
            })
    return sched


def rest_travel_adjustment(team: str, match_id: str,
                           sched: dict[str, list[dict]]) -> tuple[float, float]:
    """Return (rest_adj, travel_adj) log-lambda deltas for ``team`` entering
    ``match_id``, relative to its previous match. First match -> (0, 0)."""
    cfg = load_config()
    games = sched.get(team, [])
    idx = next((i for i, g in enumerate(games) if g["match_id"] == match_id), None)
    if idx is None or idx == 0:
        return 0.0, 0.0
    prev, cur = games[idx - 1], games[idx]
    rest_days = (cur["date"] - prev["date"]).days
    # Reference rest is ~4 days; deviation scaled by config.
    rest_adj = cfg.get("adjustments.rest_day_value", 0.015) * (rest_days - 4)

    dist = haversine_km(prev["lat"], prev["lon"], cur["lat"], cur["lon"])
    tz = abs(approx_timezone_offset_hours(cur["lon"])
             - approx_timezone_offset_hours(prev["lon"]))
    travel_adj = (cfg.get("adjustments.travel_per_1000km", -0.02) * dist / 1000.0
                  + cfg.get("adjustments.timezone_per_hour", -0.01) * tz)
    return rest_adj, travel_adj
