"""Knockout bracket survival tracking + goal feed.

Lights up the user's bracket picks as the tournament progresses: each predicted
team is ALIVE or ELIMINATED, with per-round survival counts. Becomes fully
meaningful as groups finish and the knockout rounds begin.
"""
from __future__ import annotations

import time
from datetime import date, timedelta

from . import user_picks as up
from .tracker import third_place_race, tournament_complete_groups

KO_START = date(2026, 6, 28)
FINAL = date(2026, 7, 19)

# Finalised past knockout days never change -> cache them forever; the whole
# result is memoised for a few seconds so the several callers in one snapshot
# share a single set of fetches while still refreshing live between polls.
_KO_PAST_DAYS: dict[str, list[dict]] = {}
_KO_MEMO: dict = {"ts": 0.0, "val": None}


def _ko_result(m: dict):
    """(winner, loser) for a finished knockout match, incl. penalty shootouts."""
    try:
        hs, as_ = int(m["home_score"]), int(m["away_score"])
    except (ValueError, TypeError):
        return None
    if hs > as_:
        return m["home"], m["away"]
    if as_ > hs:
        return m["away"], m["home"]
    if m.get("home_winner"):            # level after ET -> decided on penalties
        return m["home"], m["away"]
    if m.get("away_winner"):
        return m["away"], m["home"]
    return None                          # shootout winner not yet known


def actual_qualifiers(groups: dict[str, list[dict]]) -> set[str]:
    """The 32 advancing teams once the group stage is complete (top 2 + best 8 thirds)."""
    quals = set()
    for rows in groups.values():
        for t in rows[:2]:
            quals.add(t["abbr"])
    for t in third_place_race(groups):
        if t["wildcard_status"] == "IN":
            quals.add(t["abbr"])
    return quals


def finished_knockouts(feed) -> list[dict]:
    """Finished knockout matches (winner/loser by abbr), shootouts included."""
    now = time.time()
    if _KO_MEMO["val"] is not None and now - _KO_MEMO["ts"] < 3:
        return _KO_MEMO["val"]
    out = []
    today = date.today()
    d = KO_START
    while d <= min(FINAL, today):
        key = d.strftime("%Y%m%d")
        if d < today and key in _KO_PAST_DAYS:       # finalised -> cached forever
            out.extend(_KO_PAST_DAYS[key])
            d += timedelta(days=1)
            continue
        matches, ok = [], False
        try:
            matches, ok = feed.day(key), True
        except Exception:
            ok = False
        day_res = []
        for m in matches:
            if m["state"] != "post":
                continue
            res = _ko_result(m)
            if res:
                day_res.append({"winner": res[0], "loser": res[1]})
        # Only cache a past day once it's fully final (avoids caching transients).
        if d < today and ok and matches and all(x["state"] == "post" for x in matches):
            _KO_PAST_DAYS[key] = day_res
        out.extend(day_res)
        d += timedelta(days=1)
    _KO_MEMO["ts"], _KO_MEMO["val"] = now, out
    return out


def eliminated_teams(feed, groups: dict[str, list[dict]]) -> tuple[set[str], bool]:
    """Set of definitively-eliminated team abbrs, plus group-stage-complete flag."""
    elim: set[str] = set()
    groups_done = tournament_complete_groups(groups) == 12
    if groups_done:
        quals = actual_qualifiers(groups)
        for rows in groups.values():
            for t in rows:
                if t["abbr"] not in quals:
                    elim.add(t["abbr"])
    else:
        # In completed groups the 4th-placed team is definitely out.
        for rows in groups.values():
            if rows and all(t["P"] >= 3 for t in rows):
                elim.add(rows[3]["abbr"])
    for ko in finished_knockouts(feed):
        elim.add(ko["loser"])
    return elim, groups_done


def _location(abbr: str, groups: dict[str, list[dict]]) -> str:
    for g, rows in groups.items():
        for i, t in enumerate(rows):
            if t["abbr"] == abbr:
                return f"Grp {g} ({['1st','2nd','3rd','4th'][i]})"
    return "—"


def bracket_status(feed, groups: dict[str, list[dict]] | None = None) -> dict:
    groups = groups or feed.standings()
    elim, groups_done = eliminated_teams(feed, groups)
    rounds = up.predicted_rounds()
    tiers = []
    for label, teams in rounds.items():
        rows = []
        alive = 0
        for ab in teams:
            is_alive = ab not in elim
            alive += int(is_alive)
            rows.append({"team": ab, "alive": is_alive,
                         "loc": _location(ab, groups) if is_alive else "OUT"})
        tiers.append({"label": label, "alive": alive, "total": len(teams), "teams": rows})
    return {"tiers": tiers, "groups_done": groups_done,
            "champion": up.NAME2ABBR.get(up.CHAMPION, up.CHAMPION),
            "champion_alive": up.NAME2ABBR.get(up.CHAMPION, up.CHAMPION) not in elim}


def goal_feed(today: list[dict], limit: int = 12) -> list[dict]:
    """Flatten today's matches into a goal-by-goal feed (most recent first)."""
    feed = []
    for m in today:
        for g in m.get("goals", []):
            feed.append({
                "match": f"{m['home']}-{m['away']}",
                "clock": g["clock"], "scorer": g["scorer"],
                "team": g["team"], "type": g["type"],
                "live": m["state"] == "in",
            })
    # Sort by minute (parse leading number from clock like "90'+6'").
    def minute(c):
        import re
        nums = re.findall(r"\d+", c.get("clock", "") or "")
        return sum(int(n) for n in nums) if nums else 0
    feed.sort(key=minute, reverse=True)
    return feed[:limit]
