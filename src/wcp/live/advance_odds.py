"""Live 'odds to reach the Round of 32' for all 48 teams.

Pure-Python Monte-Carlo: take the current real standings, simulate the remaining
group fixtures (Poisson scorelines from an Elo-based expected-goal model), apply
the FIFA tiebreakers, take each group's top 2 + the 8 best third-placed teams,
and tally how often each team makes the Round of 32. Memoised on the played-
results state so it only recomputes when a result actually changes.
"""
from __future__ import annotations

import math
import random

# Elo ratings (baked from data/raw/teams.csv) keyed by ESPN abbreviation.
ABBR_ELO = {
    "ARG": 2105, "FRA": 2085, "ESP": 2075, "BRA": 2035, "ENG": 2015, "POR": 1995,
    "NED": 1985, "BEL": 1945, "GER": 1935, "URU": 1885, "COL": 1865, "CRO": 1865,
    "MAR": 1845, "SEN": 1815, "SUI": 1810, "JPN": 1800, "MEX": 1790, "ECU": 1790,
    "AUT": 1775, "USA": 1770, "TUR": 1755, "KOR": 1740, "NOR": 1740, "SWE": 1730,
    "IRN": 1720, "AUS": 1715, "ALG": 1715, "CAN": 1710, "CIV": 1710, "SCO": 1700,
    "EGY": 1700, "PAR": 1690, "CZE": 1685, "TUN": 1665, "BIH": 1660, "QAT": 1660,
    "GHA": 1655, "RSA": 1640, "KSA": 1640, "COD": 1640, "PAN": 1640, "CPV": 1620,
    "UZB": 1620, "IRQ": 1590, "JOR": 1580, "CUW": 1560, "NZL": 1530, "HAI": 1520,
}
BASE_RATE = 1.35          # league-average goals per team per match
SCALE = 0.0022            # Elo-diff -> log expected-goal scaling
N_SIMS = 4000
_CACHE: dict[str, dict] = {}


def _lambdas(a: str, b: str) -> tuple[float, float]:
    d = (ABBR_ELO.get(a, 1700) - ABBR_ELO.get(b, 1700)) * SCALE
    return (max(0.2, BASE_RATE * math.exp(d)),
            max(0.2, BASE_RATE * math.exp(-d)))


def _poisson(lam: float, rng: random.Random) -> int:
    # Knuth's algorithm — fine for the small lambdas here.
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _state_key(groups: dict, remaining: list) -> str:
    played = sorted((g, t["abbr"], t["P"], t["Pts"], t["GD"], t["GF"])
                    for g, rows in groups.items() for t in rows)
    return repr(played) + "|" + str(len(remaining))


def r32_odds(groups: dict[str, list[dict]], remaining: list[dict],
             n: int = N_SIMS) -> dict[str, float]:
    """Probability each team reaches the Round of 32, given current standings and
    the remaining fixtures (list of dicts with 'group','home','away')."""
    key = _state_key(groups, remaining)
    if key in _CACHE:
        return _CACHE[key]

    teams = {t["abbr"]: g for g, rows in groups.items() for t in rows}
    base = {t["abbr"]: (t["Pts"], t["GF"] - t["GA"], t["GF"])
            for rows in groups.values() for t in rows}
    rem = [(m["group"], m["home"], m["away"]) for m in remaining
           if m["home"] in teams and m["away"] in teams]

    counts = {ab: 0 for ab in teams}
    rng = random.Random(20260618)
    for _ in range(n):
        pts = {ab: base[ab][0] for ab in base}
        gd = {ab: base[ab][1] for ab in base}
        gf = {ab: base[ab][2] for ab in base}
        for g, h, a in rem:
            lh, la = _lambdas(h, a)
            gh, ga = _poisson(lh, rng), _poisson(la, rng)
            if gh > ga:
                pts[h] += 3
            elif ga > gh:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
            gd[h] += gh - ga; gd[a] += ga - gh
            gf[h] += gh; gf[a] += ga

        # Rank each group (points -> GD -> GF); collect top-2 and the thirds.
        thirds = []
        for g, rows in groups.items():
            order = sorted((t["abbr"] for t in rows),
                           key=lambda ab: (pts[ab], gd[ab], gf[ab]), reverse=True)
            counts[order[0]] += 1
            counts[order[1]] += 1
            thirds.append(order[2])
        # 8 best third-placed teams.
        for ab in sorted(thirds, key=lambda x: (pts[x], gd[x], gf[x]),
                         reverse=True)[:8]:
            counts[ab] += 1

    odds = {ab: round(counts[ab] / n, 4) for ab in counts}
    _CACHE[key] = odds
    return odds
