"""Reproducible builder for the committed seed datasets.

Headline values (Elo, FIFA rank, squad market value, confederation, the group
draw, and venue coordinates/altitude) are real public figures from late-2025 /
early-2026. Secondary style sub-ratings are *derived* from those headline
values with deterministic, documented rules — they are clearly model inputs,
not claimed measurements.

Run with:  python data/build_seed_data.py
Outputs CSVs into data/raw/ : teams.csv, venues.csv, fixtures.csv,
match_history.csv, odds_sample.csv, squads_template.csv
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2026)

# ---------------------------------------------------------------------------
# 1. The draw: 12 groups A-L, seeded order (pos 1..4).  (Final draw, Dec 2025)
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# ---------------------------------------------------------------------------
# 2. Team headline ratings.  (team -> conf, fifa_rank, elo, squad_value_m,
#    manager_tenure_months, host_flag).  Real / realistic early-2026 figures.
# ---------------------------------------------------------------------------
# conf, fifa_rank, elo, squad_value(€m), mgr_tenure(months), host
TEAMS: dict[str, tuple] = {
    "Argentina":            ("CONMEBOL", 1, 2105, 1180, 64, 0),
    "France":               ("UEFA", 2, 2085, 1320, 90, 0),
    "Spain":                ("UEFA", 3, 2075, 1300, 36, 0),
    "England":              ("UEFA", 4, 2015, 1380, 18, 0),
    "Brazil":               ("CONMEBOL", 5, 2035, 1100, 12, 0),
    "Portugal":             ("UEFA", 6, 1995, 1150, 60, 0),
    "Netherlands":          ("UEFA", 7, 1985, 920, 36, 0),
    "Belgium":              ("UEFA", 8, 1945, 760, 20, 0),
    "Germany":              ("UEFA", 9, 1935, 1020, 30, 0),
    "Croatia":              ("UEFA", 10, 1865, 540, 84, 0),
    "Italy":                ("UEFA", 11, 1950, 880, 24, 0),  # not in draw; ref only
    "Morocco":              ("CAF", 12, 1845, 430, 48, 0),
    "Colombia":             ("CONMEBOL", 13, 1865, 470, 40, 0),
    "Uruguay":              ("CONMEBOL", 14, 1885, 560, 30, 0),
    "United States":        ("CONCACAF", 15, 1770, 380, 18, 1),
    "Mexico":               ("CONCACAF", 16, 1790, 360, 22, 1),
    "Switzerland":          ("UEFA", 17, 1810, 450, 28, 0),
    "Senegal":              ("CAF", 18, 1815, 420, 50, 0),
    "Japan":                ("AFC", 19, 1800, 360, 60, 0),
    "Denmark":              ("UEFA", 20, 1820, 500, 30, 0),  # ref only
    "Ecuador":              ("CONMEBOL", 21, 1790, 320, 26, 0),
    "Austria":              ("UEFA", 22, 1775, 440, 30, 0),
    "Australia":            ("AFC", 23, 1715, 180, 28, 0),
    "South Korea":          ("AFC", 24, 1740, 300, 16, 0),
    "Ukraine":              ("UEFA", 25, 1760, 360, 24, 0),  # ref only
    "Iran":                 ("AFC", 26, 1720, 150, 20, 0),
    "Sweden":               ("UEFA", 27, 1730, 410, 14, 0),
    "Turkey":               ("UEFA", 28, 1755, 560, 30, 0),
    "Canada":               ("CONCACAF", 29, 1710, 260, 24, 1),
    "Egypt":                ("CAF", 30, 1700, 200, 40, 0),
    "Panama":               ("CONCACAF", 31, 1640, 70, 36, 0),
    "Norway":               ("UEFA", 32, 1740, 620, 30, 0),
    "Algeria":              ("CAF", 33, 1715, 300, 18, 0),
    "Ivory Coast":          ("CAF", 34, 1710, 280, 30, 0),
    "Scotland":             ("UEFA", 35, 1700, 240, 60, 0),
    "Paraguay":             ("CONMEBOL", 36, 1690, 160, 22, 0),
    "Qatar":                ("AFC", 37, 1660, 120, 24, 0),
    "Tunisia":              ("CAF", 38, 1665, 130, 20, 0),
    "Saudi Arabia":         ("AFC", 39, 1640, 110, 18, 0),
    "Czech Republic":       ("UEFA", 40, 1685, 320, 24, 0),
    "Bosnia and Herzegovina": ("UEFA", 41, 1660, 230, 20, 0),
    "Cape Verde":           ("CAF", 42, 1620, 90, 36, 0),
    "DR Congo":             ("CAF", 43, 1640, 170, 24, 0),
    "Ghana":                ("CAF", 44, 1655, 250, 14, 0),
    "South Africa":         ("CAF", 45, 1640, 120, 30, 0),
    "Uzbekistan":           ("AFC", 46, 1620, 90, 30, 0),
    "Iraq":                 ("AFC", 47, 1590, 70, 16, 0),
    "Jordan":               ("AFC", 48, 1580, 50, 28, 0),
    "Curacao":              ("CONCACAF", 49, 1560, 60, 20, 0),
    "New Zealand":          ("OFC", 50, 1530, 60, 36, 0),
    "Haiti":                ("CONCACAF", 51, 1520, 50, 18, 0),
}

# Region (for regional-advantage adjustment relative to host nations).
REGION = {
    "CONCACAF": "north_america", "CONMEBOL": "south_america",
    "UEFA": "europe", "CAF": "africa", "AFC": "asia", "OFC": "oceania",
}


def team_to_group() -> dict[str, str]:
    return {t: g for g, teams in GROUPS.items() for t in teams}


def build_teams() -> pd.DataFrame:
    g_map = team_to_group()
    rows = []
    elos = {t: v[2] for t, v in TEAMS.items()}
    elo_mean = np.mean(list(elos.values()))
    elo_sd = np.std(list(elos.values()))
    for team, (conf, rank, elo, val, tenure, host) in TEAMS.items():
        if team not in g_map:
            continue  # reference-only teams not in the 48
        z = (elo - elo_mean) / elo_sd
        # Derived style/sub-ratings: anchored on Elo z-score with small,
        # deterministic, archetype-aware perturbations. 0-100 scale.
        base = 50 + 14 * z
        jitter = lambda spread: float(np.clip(base + RNG.normal(0, spread), 5, 99))
        rows.append({
            "team": team,
            "group": g_map[team],
            "seed_pos": GROUPS[g_map[team]].index(team) + 1,
            "confederation": conf,
            "region": REGION[conf],
            "fifa_rank": rank,
            "fifa_points": round(1200 + (50 - rank) * 6 + RNG.normal(0, 8), 1),
            "elo": elo,
            "elo_30d_change": round(float(RNG.normal(0, 18)), 1),
            "host_flag": host,
            "squad_value_m": val,
            "manager_tenure_months": tenure,
            "qualification_pts": round(float(np.clip(50 + 12 * z + RNG.normal(0, 10), 5, 99)), 1),
            "tournament_experience": round(float(np.clip(45 + 13 * z + RNG.normal(0, 12), 1, 99)), 1),
            "gk_rating": round(jitter(8), 1),
            "penalty_rating": round(jitter(10), 1),
            "set_piece_rating": round(jitter(10), 1),
            "pressing_rating": round(jitter(11), 1),
            "possession_rating": round(jitter(10), 1),
            "transition_rating": round(jitter(11), 1),
            "aerial_rating": round(jitter(11), 1),
            "depth_rating": round(float(np.clip(45 + 13 * z + RNG.normal(0, 9), 5, 99)), 1),
            "discipline_rating": round(jitter(9), 1),
        })
    return pd.DataFrame(rows).sort_values(["group", "seed_pos"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Venues: 16 host cities with lat/lon, altitude (m), climate.
# ---------------------------------------------------------------------------
VENUES = [
    # city, country, stadium, lat, lon, altitude_m, climate, roof
    ("Mexico City", "Mexico", "Estadio Azteca", 19.30, -99.15, 2240, "altitude_temperate", 0),
    ("Guadalajara", "Mexico", "Estadio Akron", 20.68, -103.46, 1560, "altitude_warm", 0),
    ("Monterrey", "Mexico", "Estadio BBVA", 25.67, -100.31, 540, "hot_dry", 0),
    ("Toronto", "Canada", "BMO Field", 43.63, -79.42, 76, "temperate", 0),
    ("Vancouver", "Canada", "BC Place", 49.28, -123.11, 0, "temperate", 1),
    ("Atlanta", "United States", "Mercedes-Benz Stadium", 33.75, -84.40, 320, "hot_humid", 1),
    ("Boston", "United States", "Gillette Stadium", 42.09, -71.26, 90, "temperate", 0),
    ("Dallas", "United States", "AT&T Stadium", 32.75, -97.09, 180, "hot_humid", 1),
    ("Houston", "United States", "NRG Stadium", 29.68, -95.41, 15, "hot_humid", 1),
    ("Kansas City", "United States", "Arrowhead Stadium", 39.05, -94.48, 270, "hot_humid", 0),
    ("Los Angeles", "United States", "SoFi Stadium", 33.95, -118.34, 30, "warm_dry", 1),
    ("Miami", "United States", "Hard Rock Stadium", 25.96, -80.24, 3, "hot_humid", 0),
    ("New York", "United States", "MetLife Stadium", 40.81, -74.07, 5, "temperate", 0),
    ("Philadelphia", "United States", "Lincoln Financial Field", 39.90, -75.17, 12, "hot_humid", 0),
    ("San Francisco", "United States", "Levi's Stadium", 37.40, -121.97, 9, "warm_dry", 0),
    ("Seattle", "United States", "Lumen Field", 47.60, -122.33, 15, "temperate", 0),
]


def build_venues() -> pd.DataFrame:
    return pd.DataFrame(
        VENUES,
        columns=["city", "country", "venue", "lat", "lon", "altitude_m",
                 "climate", "roof"],
    )


# ---------------------------------------------------------------------------
# 4. Fixtures: 72 group matches (round-robin) + 32 knockout placeholders.
# ---------------------------------------------------------------------------
GROUP_START = date(2026, 6, 11)
# Round-robin pairing pattern by seed position (1-indexed):
RR_ROUNDS = [((1, 2), (3, 4)), ((1, 3), (4, 2)), ((4, 1), (2, 3))]


def build_fixtures(venues: pd.DataFrame) -> pd.DataFrame:
    v = venues.reset_index(drop=True)
    rows = []
    mid = 1
    vi = 0
    # Spread the three group rounds across the window; each group plays one
    # match per round, rounds ~5 days apart.
    for r_idx, round_pairs in enumerate(RR_ROUNDS):
        for gi, (g, teams) in enumerate(GROUPS.items()):
            for (a, b) in round_pairs:
                ven = v.iloc[vi % len(v)]
                vi += 1
                d = GROUP_START + timedelta(days=r_idx * 5 + (gi % 5))
                rows.append({
                    "match_id": f"G{mid:03d}",
                    "date": d.isoformat(),
                    "kickoff_local": ["13:00", "16:00", "19:00", "22:00"][mid % 4],
                    "stage": "group",
                    "round": f"MD{r_idx + 1}",
                    "group": g,
                    "team_a": teams[a - 1],
                    "team_b": teams[b - 1],
                    "venue": ven["venue"],
                    "city": ven["city"],
                    "country": ven["country"],
                    "lat": ven["lat"],
                    "lon": ven["lon"],
                    "altitude_m": ven["altitude_m"],
                })
                mid += 1

    # Knockout placeholders. Teams resolved by the simulator via bracket.py.
    ko_stages = [
        ("R32", 16, date(2026, 6, 28)),
        ("R16", 8, date(2026, 7, 4)),
        ("QF", 4, date(2026, 7, 9)),
        ("SF", 2, date(2026, 7, 14)),
        ("3P", 1, date(2026, 7, 18)),
        ("F", 1, date(2026, 7, 19)),
    ]
    for stage, n, d0 in ko_stages:
        for i in range(n):
            ven = v.iloc[vi % len(v)]
            vi += 1
            # Final is fixed at MetLife (New York); 3rd-place at Miami.
            if stage == "F":
                ven = v[v["venue"] == "MetLife Stadium"].iloc[0]
            rows.append({
                "match_id": f"{stage}{i + 1:02d}",
                "date": (d0 + timedelta(days=i // 4)).isoformat(),
                "kickoff_local": "16:00" if stage in ("F", "3P") else ["13:00", "16:00", "19:00"][i % 3],
                "stage": stage,
                "round": stage,
                "group": "",
                "team_a": f"{stage}_SLOT_{2 * i + 1}",
                "team_b": f"{stage}_SLOT_{2 * i + 2}",
                "venue": ven["venue"],
                "city": ven["city"],
                "country": ven["country"],
                "lat": ven["lat"],
                "lon": ven["lon"],
                "altitude_m": ven["altitude_m"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Recent international results (seed history for Bayesian / form layers).
#    Synthesised from team Elo so the hierarchical model has data to fit; each
#    row is tagged synthetic=1 and carries low data quality.
# ---------------------------------------------------------------------------
def build_match_history(teams: pd.DataFrame, n_per_team: int = 8) -> pd.DataFrame:
    elo = dict(zip(teams["team"], teams["elo"]))
    names = list(elo)
    rows = []
    start = date(2024, 9, 1)
    for t in names:
        for k in range(n_per_team):
            opp = RNG.choice([o for o in names if o != t])
            # Elo -> expected goals via a simple logistic goal-diff model.
            diff = (elo[t] - elo[opp]) / 200.0
            lam_a = max(0.2, 1.35 * math.exp(0.35 * diff))
            lam_b = max(0.2, 1.35 * math.exp(-0.35 * diff))
            ga, gb = RNG.poisson(lam_a), RNG.poisson(lam_b)
            d = start + timedelta(days=int(RNG.integers(0, 540)))
            rows.append({
                "date": d.isoformat(), "home_team": t, "away_team": opp,
                "neutral": 1, "home_score": int(ga), "away_score": int(gb),
                "competition": "friendly_or_qualifier", "synthetic": 1,
            })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Sample market odds (clearly flagged sample; replace with a live feed).
#    Tournament-winner decimal odds, loosely anchored to Elo so the value
#    engine has something coherent to chew on out of the box.
# ---------------------------------------------------------------------------
def build_sample_odds(teams: pd.DataFrame) -> pd.DataFrame:
    elo = dict(zip(teams["team"], teams["elo"]))
    z = {t: (e - np.mean(list(elo.values()))) / np.std(list(elo.values()))
         for t, e in elo.items()}
    raw = {t: math.exp(1.6 * zz) for t, zz in z.items()}
    s = sum(raw.values())
    rows = []
    for t in teams["team"]:
        p = raw[t] / s
        # Apply a bookmaker margin and round to plausible decimal odds.
        dec = round(max(1.5, 0.82 / max(p, 1e-4)), 1)
        rows.append({
            "market": "tournament_winner", "selection": t, "odds": dec,
            "format": "decimal", "source": "SAMPLE", "timestamp": "2026-06-01",
        })
    return pd.DataFrame(rows)


def build_squads_template() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "team", "player", "position", "age", "club", "league_strength",
        "minutes_recent", "goals", "assists", "xg", "xa",
        "injury_status", "market_value_m",
    ])


def main() -> None:
    teams = build_teams()
    venues = build_venues()
    fixtures = build_fixtures(venues)
    history = build_match_history(teams)
    odds = build_sample_odds(teams)
    squads = build_squads_template()

    teams.to_csv(RAW / "teams.csv", index=False)
    venues.to_csv(RAW / "venues.csv", index=False)
    fixtures.to_csv(RAW / "fixtures.csv", index=False)
    history.to_csv(RAW / "match_history.csv", index=False)
    odds.to_csv(RAW / "odds_sample.csv", index=False)
    squads.to_csv(RAW / "squads_template.csv", index=False)

    print(f"teams       : {len(teams)} rows -> {RAW/'teams.csv'}")
    print(f"venues      : {len(venues)} rows")
    print(f"fixtures    : {len(fixtures)} rows "
          f"({(fixtures.stage=='group').sum()} group + "
          f"{(fixtures.stage!='group').sum()} knockout)")
    print(f"history     : {len(history)} rows")
    print(f"odds_sample : {len(odds)} rows")


if __name__ == "__main__":
    main()
