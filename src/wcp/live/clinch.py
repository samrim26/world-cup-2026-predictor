"""Mathematical clinch detection for the group stage.

Conservative, never-wrong logic using points bounds: a team is only flagged as
having clinched something when it is *guaranteed* regardless of any remaining
result (it may be flagged slightly late on goal-difference-only clinches, but
it is never flagged wrongly). Powers the Scenarios tab, the Rankings highlights,
and the early bracket locking.
"""
from __future__ import annotations


def _maxpts(t: dict) -> int:
    return t["Pts"] + 3 * (3 - t["P"])          # current pts + 3 per game left


def group_clinch(rows: list[dict]) -> dict[str, dict]:
    """Per-team clinch flags for one group's standings rows."""
    out = {}
    for a in rows:
        ab = a["abbr"]
        rivals = [r for r in rows if r["abbr"] != ab]
        # Worst case for A: A gains nothing (floor = A.Pts), rivals max out.
        clinched_first = all(a["Pts"] > _maxpts(r) for r in rivals)
        could_reach_a = sum(1 for r in rivals if _maxpts(r) >= a["Pts"])
        clinched_advance = could_reach_a <= 1           # at most one team above A
        above_for_sure = sum(1 for r in rivals if r["Pts"] > _maxpts(a))
        out[ab] = {
            "clinched_first": clinched_first,
            "clinched_advance": clinched_advance,
            "eliminated": above_for_sure >= 2,          # cannot reach top 2
            "out_entirely": above_for_sure >= 3,        # cannot even reach 3rd
        }
    return out


def locked_positions(rows: list[dict]) -> tuple[str | None, str | None]:
    """(team locked into 1st, team locked into 2nd) for the bracket, or None."""
    if not rows:
        return None, None
    complete = all(t["P"] >= 3 for t in rows)
    if complete:
        return rows[0]["abbr"], (rows[1]["abbr"] if len(rows) > 1 else None)
    cl = group_clinch(rows)
    first = next((t["abbr"] for t in rows if cl[t["abbr"]]["clinched_first"]), None)
    second = None
    if first:
        for b in rows:
            if b["abbr"] == first:
                continue
            others = [r for r in rows if r["abbr"] not in (first, b["abbr"])]
            if cl[b["abbr"]]["clinched_advance"] and all(b["Pts"] > _maxpts(r) for r in others):
                second = b["abbr"]
                break
    return first, second


def apply_result(rows: list[dict], home: str, away: str, outcome: str) -> list[dict]:
    """Copy of rows with one game (home vs away) played as 'H'|'D'|'A'."""
    new = [dict(t) for t in rows]
    by = {t["abbr"]: t for t in new}
    h, a = by.get(home), by.get(away)
    if not h or not a:
        return new
    h["P"] += 1; a["P"] += 1
    if outcome == "H":
        h["Pts"] += 3
    elif outcome == "A":
        a["Pts"] += 3
    else:
        h["Pts"] += 1; a["Pts"] += 1
    return new


_NAME = {"H": "home win", "D": "draw", "A": "away win"}


def game_implications(groups: dict[str, list[dict]],
                      remaining: list[dict]) -> list[dict]:
    """For each remaining game, what each of its 3 outcomes guarantees."""
    by_group = {g: rows for g, rows in groups.items()}
    out = []
    for m in remaining:
        g = m["group"]
        rows = by_group.get(g)
        if not rows:
            continue
        before = group_clinch(rows)
        labels = {t["abbr"]: t.get("team", t["abbr"]) for t in rows}
        outcomes = []
        for oc in ("H", "D", "A"):
            after = group_clinch(apply_result(rows, m["home"], m["away"], oc))
            notes = []
            for ab in (m["home"], m["away"], *(r["abbr"] for r in rows)):
                if ab in [n["abbr"] for n in notes]:
                    continue
                b, a = before.get(ab, {}), after.get(ab, {})
                if a.get("clinched_first") and not b.get("clinched_first"):
                    notes.append({"abbr": ab, "text": f"{ab} clinches 1st", "kind": "first"})
                elif a.get("clinched_advance") and not b.get("clinched_advance"):
                    notes.append({"abbr": ab, "text": f"{ab} clinches a knockout spot", "kind": "advance"})
                if a.get("eliminated") and not b.get("eliminated"):
                    notes.append({"abbr": ab, "text": f"{ab} eliminated from top 2", "kind": "elim"})
            who = m["home"] if oc == "H" else (m["away"] if oc == "A" else "")
            outcomes.append({
                "outcome": oc,
                "label": f"{who} win" if who else "draw",
                "notes": notes or [{"abbr": "", "text": "nothing decided — group stays open", "kind": "none"}],
            })
        out.append({
            "group": g, "home": m["home"], "away": m["away"],
            "home_name": m.get("home_name", m["home"]), "away_name": m.get("away_name", m["away"]),
            "et_date": m.get("et_date", ""), "et_time": m.get("et_time", ""),
            "kickoff": m.get("kickoff", ""), "outcomes": outcomes,
        })
    out.sort(key=lambda x: (x["kickoff"] or "9", x["group"]))
    return out
