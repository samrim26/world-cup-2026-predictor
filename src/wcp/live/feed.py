"""Live data feed for the real 2026 World Cup.

Default source is ESPN's public (no-key) soccer API, which returns the real
group standings and live scores. The feed is swappable: subclass ``BaseFeed``
and point it at a keyed provider (API-Football, SportMonks, ...) without
touching the tracker or dashboard.
"""
from __future__ import annotations

import json
import ssl
import subprocess
import urllib.request
from abc import ABC, abstractmethod

from ..utils.logging import get_logger

log = get_logger("live.feed")

ESPN_BASE = "https://site.api.espn.com/apis"
STANDINGS_URL = f"{ESPN_BASE}/v2/sports/soccer/fifa.world/standings"
SCOREBOARD_URL = f"{ESPN_BASE}/site/v2/sports/soccer/fifa.world/scoreboard"


def _get_json(url: str, timeout: int = 20) -> dict:
    """Fetch JSON, tolerating macOS SSL-cert quirks; fall back to curl."""
    for attempt in ("default", "unverified"):
        try:
            ctx = ssl._create_unverified_context() if attempt == "unverified" else None
            with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            continue
    out = subprocess.run(["curl", "-s", "--max-time", str(timeout), url],
                         capture_output=True, timeout=timeout + 10, check=True)
    return json.loads(out.stdout.decode("utf-8"))


class BaseFeed(ABC):
    source = "unknown"

    @abstractmethod
    def standings(self) -> dict[str, list[dict]]:
        """Return {group_letter: [team_row, ...]} sorted by in-group rank.
        Each team_row: team, abbr, P, W, D, L, GF, GA, GD, Pts, rank."""

    @abstractmethod
    def today(self) -> list[dict]:
        """Return today's matches: home/away (+abbr), scores, status, state, clock."""


class ESPNFeed(BaseFeed):
    source = "ESPN public API (fifa.world)"

    def standings(self) -> dict[str, list[dict]]:
        d = _get_json(STANDINGS_URL)
        groups: dict[str, list[dict]] = {}
        for child in d.get("children", []):
            name = child.get("name", "")          # e.g. "Group A"
            letter = name.split()[-1] if name else "?"
            rows = []
            for e in child.get("standings", {}).get("entries", []):
                s = {x["name"]: x.get("value") for x in e.get("stats", [])}
                team = e.get("team", {})
                rows.append({
                    "team": team.get("displayName", "?"),
                    "abbr": team.get("abbreviation", "?"),
                    "P": int(s.get("gamesPlayed", 0) or 0),
                    "W": int(s.get("wins", 0) or 0),
                    "D": int(s.get("ties", 0) or 0),
                    "L": int(s.get("losses", 0) or 0),
                    "GF": int(s.get("pointsFor", 0) or 0),
                    "GA": int(s.get("pointsAgainst", 0) or 0),
                    "GD": int(s.get("pointDifferential", 0) or 0),
                    "Pts": int(s.get("points", 0) or 0),
                    "rank": int(s.get("rank", 99) or 99),
                })
            rows.sort(key=lambda t: (t["rank"] if t["rank"] else 99,
                                     -t["Pts"], -t["GD"], -t["GF"]))
            groups[letter] = rows
        return dict(sorted(groups.items()))

    @staticmethod
    def _goals(comp: dict) -> list[dict]:
        """Extract goal-scoring plays from a competition's inline details."""
        id2abbr = {c.get("team", {}).get("id"): c.get("team", {}).get("abbreviation")
                   for c in comp.get("competitors", [])}
        goals = []
        for d in comp.get("details", []) or []:
            if not d.get("scoringPlay"):
                continue
            ath = d.get("athletesInvolved") or [{}]
            goals.append({
                "clock": d.get("clock", {}).get("displayValue", ""),
                "scorer": ath[0].get("displayName", "?") if ath else "?",
                "team": id2abbr.get(d.get("team", {}).get("id"), "?"),
                "type": d.get("type", {}).get("text", "Goal"),
            })
        return goals

    @staticmethod
    def _parse(d: dict) -> list[dict]:
        out = []
        for e in d.get("events", []):
            comp = (e.get("competitions") or [{}])[0]
            cs = comp.get("competitors", [])
            home = next((c for c in cs if c.get("homeAway") == "home"), {})
            away = next((c for c in cs if c.get("homeAway") == "away"), {})
            st = e.get("status", {}).get("type", {})
            out.append({
                "id": e.get("id"),
                "goals": ESPNFeed._goals(comp),
                "kickoff": e.get("date", ""),           # full ISO UTC, e.g. 2026-06-24T19:00Z
                "date": (e.get("date", "") or "")[:10],
                "home": home.get("team", {}).get("abbreviation", "?"),
                "home_name": home.get("team", {}).get("displayName", "?"),
                "home_score": home.get("score", "0"),
                "away": away.get("team", {}).get("abbreviation", "?"),
                "away_name": away.get("team", {}).get("displayName", "?"),
                "away_score": away.get("score", "0"),
                "status": st.get("description", ""),
                "state": st.get("state", ""),          # pre / in / post
                "clock": e.get("status", {}).get("displayClock", ""),
                "detail": st.get("shortDetail", ""),
            })
        return out

    def today(self) -> list[dict]:
        return self._parse(_get_json(SCOREBOARD_URL))

    def day(self, yyyymmdd: str) -> list[dict]:
        """All matches on a given date (YYYYMMDD), incl. finished results."""
        return self._parse(_get_json(f"{SCOREBOARD_URL}?dates={yyyymmdd}"))
