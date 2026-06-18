"""Full group-stage schedule (all 72 matches) with real ET kickoff times, live
results, the user's prediction, and points scored — built from the live feed.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from . import user_picks as up

GROUP_START = date(2026, 6, 11)
GROUP_END = date(2026, 6, 27)
ET = timezone(timedelta(hours=-4))               # EDT (June/July)
_DAY_CACHE: dict[str, list[dict]] = {}           # finalised days cached


def _kickoff_et(iso: str) -> tuple[str, str]:
    """Return (YYYY-MM-DD ET, 'h:MM AM/PM ET') from an ISO-UTC kickoff string."""
    if not iso:
        return "", ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(ET)
    except ValueError:
        return iso[:10], ""
    hour = dt.strftime("%I:%M %p").lstrip("0")
    return dt.strftime("%Y-%m-%d"), f"{hour} ET"


def _outcome(hs: int, as_: int) -> str:
    return "H" if hs > as_ else ("A" if as_ > hs else "D")


def _points(pred: str | None, hs: int, as_: int) -> int | None:
    """Pool points for a finished match given the predicted 'h-a' scoreline."""
    if not pred:
        return None
    try:
        ph, pa = (int(x) for x in pred.split("-"))
    except ValueError:
        return None
    pts = 3 if _outcome(ph, pa) == _outcome(hs, as_) else 0
    pts += 1 if ph == hs else 0
    pts += 1 if pa == as_ else 0
    return pts


def build_schedule(feed, groups: dict | None = None) -> list[dict]:
    """All 72 group matches in chronological order, enriched for the UI."""
    groups = groups if groups is not None else feed.standings()
    team_group = {t["abbr"]: g for g, rows in groups.items() for t in rows}

    rows: list[dict] = []
    d = GROUP_START
    while d <= GROUP_END:
        key = d.strftime("%Y%m%d")
        is_today = (d == date.today())
        if key in _DAY_CACHE and not is_today:
            matches = _DAY_CACHE[key]
        else:
            try:
                matches = feed.day(key)
            except Exception:
                matches = []
            if not is_today and matches and all(m["state"] == "post" for m in matches):
                _DAY_CACHE[key] = matches      # cache only fully-finalised days
        for m in matches:
            ed, et = _kickoff_et(m.get("kickoff", ""))
            pred = up.predicted_scoreline(m["home"], m["away"])
            done = m["state"] == "post"
            pts = None
            if done:
                try:
                    pts = _points(pred, int(m["home_score"]), int(m["away_score"]))
                except (ValueError, TypeError):
                    pts = None
            rows.append({
                "et_date": ed, "et_time": et, "kickoff": m.get("kickoff", ""),
                "group": team_group.get(m["home"]) or team_group.get(m["away"]) or "?",
                "home": m["home"], "home_name": m["home_name"], "home_score": m["home_score"],
                "away": m["away"], "away_name": m["away_name"], "away_score": m["away_score"],
                "state": m["state"], "status": m["status"], "clock": m.get("clock", ""),
                "detail": m.get("detail", ""), "predicted": pred, "points": pts,
            })
        d += timedelta(days=1)

    rows.sort(key=lambda r: (r["kickoff"] or "9", r["group"]))
    return rows


def team_schedule(schedule: list[dict], abbr: str) -> list[dict]:
    """That team's matches (chronological) for the click-through team view."""
    return [m for m in schedule if m["home"] == abbr or m["away"] == abbr]


def score_from_schedule(schedule: list[dict]) -> dict:
    """Pool score derived from the schedule (no extra fetches): +3 outcome,
    +1 each correct score; 5 = exact."""
    rows, total = [], 0
    exact = correct = 0
    for m in schedule:
        if m["state"] != "post" or m["points"] is None:
            continue
        pts = m["points"]
        total += pts
        if pts == 5:
            exact += 1
        if pts >= 3:
            correct += 1
        rows.append({
            "date": m["et_date"], "match": f"{m['home']} {m['home_score']}-{m['away_score']} {m['away']}",
            "predicted": m["predicted"], "actual": f"{m['home_score']}-{m['away_score']}",
            "points": pts,
        })
    rows.sort(key=lambda r: r["date"], reverse=True)
    return {"rows": rows, "total": total, "n": len(rows), "exact": exact,
            "correct_outcome": correct, "max_possible": len(rows) * 5}
