"""Knockout bracket survival tracking + goal feed.

Lights up the user's bracket picks as the tournament progresses: each predicted
team is ALIVE or ELIMINATED, with per-round survival counts. Becomes fully
meaningful as groups finish and the knockout rounds begin.
"""
from __future__ import annotations

from datetime import date, timedelta

from . import user_picks as up
from .tracker import third_place_race, tournament_complete_groups

KO_START = date(2026, 6, 28)
FINAL = date(2026, 7, 19)


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
    """Finished knockout matches (winner/loser by abbr)."""
    out = []
    d = KO_START
    while d <= min(FINAL, date.today()):
        try:
            for m in feed.day(d.strftime("%Y%m%d")):
                if m["state"] != "post":
                    continue
                try:
                    hs, as_ = int(m["home_score"]), int(m["away_score"])
                except (ValueError, TypeError):
                    continue
                if hs == as_:                       # shootout: winner unknown here
                    continue
                win, lose = ((m["home"], m["away"]) if hs > as_ else (m["away"], m["home"]))
                out.append({"winner": win, "loser": lose})
        except Exception:
            pass
        d += timedelta(days=1)
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
