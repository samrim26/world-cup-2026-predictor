"""Vercel serverless function: returns the full live snapshot as JSON.

Pure-stdlib (the live package was refactored to need no numpy/scipy), so the
function bundle is tiny and cold-starts fast. Served at /api/live.
"""
import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler

# Make the `wcp` package importable (bundled via vercel.json includeFiles).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wcp.live.feed import ESPNFeed                                  # noqa: E402
from wcp.live.tracker import (qualification_flags, third_place_race,  # noqa: E402
                              tournament_complete_groups, wildcard_swings)
from wcp.live import user_picks                                     # noqa: E402
from wcp.live.bracket_tracker import goal_feed, bracket_status      # noqa: E402
from wcp.live import compare                                        # noqa: E402
from wcp.live.live_bracket import build as build_bracket, by_round  # noqa: E402
from wcp.live.user_picks import attach_predictions                  # noqa: E402
from wcp.live.schedule import (build_schedule, score_from_schedule,  # noqa: E402
                               team_schedule)
from wcp.live.advance_odds import r32_odds                          # noqa: E402
from wcp.live.rankings import build_rankings                        # noqa: E402
from wcp.live import clinch                                         # noqa: E402


def snapshot() -> dict:
    feed = ESPNFeed()
    groups = feed.standings()                       # fetched once, reused below
    today = attach_predictions(feed.today())
    schedule = build_schedule(feed, groups)
    # "pre" and "in" (live): both are still undecided, so keep their scenarios
    # up until the match actually ends (state == "post").
    remaining = [m for m in schedule if m["state"] in ("pre", "in")]
    rem_by_group = {}
    for m in remaining:
        rem_by_group.setdefault(m["group"], []).append((m["home"], m["away"]))
    odds = r32_odds(groups, remaining)
    locked = {g: clinch.locked_positions(rows, rem_by_group.get(g, []))
              for g, rows in groups.items()}
    return {
        "updated": datetime.now(timezone.utc).strftime("%H:%M UTC, %b %d"),
        "source": feed.source,
        "groups_done": tournament_complete_groups(groups),
        "today": today,
        "goals": goal_feed(today),
        "thirds": third_place_race(groups),
        "groups": qualification_flags(groups),
        "score": score_from_schedule(schedule),     # derived; no extra fetches
        "qual": compare.qualifier_accuracy(groups),
        "bracket_survival": bracket_status(feed, groups),
        "live_bracket": by_round(build_bracket(feed, groups, locked)),
        "schedule": schedule,
        "advance_odds": odds,
        "rankings": build_rankings(groups, odds, rem_by_group),
        "scenarios": clinch.game_implications(groups, remaining),
        "wildcard_swings": wildcard_swings(groups, remaining),
        "picks_bracket": user_picks.predicted_bracket(),
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            payload, code = snapshot(), 200
        except Exception as exc:                     # surface errors to the client
            payload, code = {"error": str(exc)}, 500
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        # CDN-cache briefly so rapid refreshes don't hammer the upstream API.
        self.send_header("Cache-Control", "s-maxage=20, stale-while-revalidate=40")
        self.end_headers()
        self.wfile.write(body)
