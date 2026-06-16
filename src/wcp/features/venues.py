"""Venue-driven adjustments: home/regional advantage, altitude, climate."""
from __future__ import annotations

import pandas as pd

from ..utils.config import load_config

# Which host nation each region is "at home" / "near home" in.
HOST_COUNTRIES = {"United States", "Mexico", "Canada"}


def home_advantage(team_row: pd.Series, match_country: str) -> float:
    """Additive log-lambda home/regional advantage for ``team`` in a match.

    - True host playing in its own country -> full ``home_advantage``.
    - Other CONCACAF sides -> ``regional_advantage`` (continental familiarity).
    - Everyone else -> 0.
    """
    cfg = load_config()
    full = cfg.get("xg.home_advantage", 0.25)
    regional = cfg.get("xg.regional_advantage", 0.10)
    team_country = team_row["team"]
    host_flag = int(team_row.get("host_flag", 0))
    if host_flag and team_country == match_country:
        return full
    if host_flag:  # host nation but playing in a co-host country
        return regional
    if team_row.get("confederation") == "CONCACAF":
        return regional
    if team_row.get("confederation") == "CONMEBOL":
        return 0.4 * regional  # climate/time-zone familiarity, modest
    return 0.0


def altitude_climate_adjustment(team_row: pd.Series, venue_row: pd.Series) -> float:
    """Penalty (log-lambda) for unfamiliar altitude / extreme heat.

    Altitude-resident teams (Mexico, Andean sides) are exempt from the altitude
    penalty; heat penalty applies to both sides but is returned per-team so it
    can be scaled by squad depth later.
    """
    cfg = load_config()
    adj = 0.0
    altitude = float(venue_row.get("altitude_m", 0))
    ref = cfg.get("adjustments.altitude_ref_m", 1000)
    if altitude > ref:
        altitude_natives = {"Mexico", "Ecuador", "Colombia", "Bolivia", "Peru"}
        if team_row["team"] not in altitude_natives:
            adj += cfg.get("adjustments.altitude_per_1000m", -0.05) * \
                (altitude - ref) / 1000.0
    if venue_row.get("climate") in ("hot_humid", "hot_dry"):
        adj += cfg.get("adjustments.heat_penalty", -0.03)
    return adj
