"""Mathematical clinch detection for the group stage.

Conservative, never-wrong logic using points bounds: a team is only flagged as
having clinched something when it is *guaranteed* regardless of any remaining
result (it may be flagged slightly late on goal-difference-only clinches, but
it is never flagged wrongly). Powers the Scenarios tab, the Rankings highlights,
and the early bracket locking.
"""
from __future__ import annotations

from itertools import product


def _final_points(rows: list[dict], remaining: list[tuple[str, str]]):
    """Yield {abbr: final_points} for every win/draw/loss combination of the
    remaining games — so head-to-head structure (e.g. two rivals playing each
    other) is handled exactly, not double-counted."""
    base = {t["abbr"]: t["Pts"] for t in rows}
    for combo in product("HDA", repeat=len(remaining)):
        pts = dict(base)
        for (h, a), oc in zip(remaining, combo):
            if oc == "H":
                pts[h] += 3
            elif oc == "A":
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
        yield pts


def group_clinch(rows: list[dict],
                 remaining: list[tuple[str, str]] | None = None) -> dict[str, dict]:
    """Per-team clinch flags via exact enumeration of the group's remaining games.

    A flag is True only when it holds in *every* possible remaining-results
    scenario (still conservative on goal-difference-only ties — points only —
    but exact on the head-to-head structure).
    """
    remaining = remaining or []
    abbrs = [t["abbr"] for t in rows]
    worst_at_or_above = {ab: 0 for ab in abbrs}     # max over scenarios
    best_strictly_above = {ab: 99 for ab in abbrs}  # min over scenarios
    for pts in _final_points(rows, remaining):
        for ab in abbrs:
            above = sum(1 for o in abbrs if o != ab and pts[o] > pts[ab])
            tie = sum(1 for o in abbrs if o != ab and pts[o] == pts[ab])
            worst_at_or_above[ab] = max(worst_at_or_above[ab], above + tie)
            best_strictly_above[ab] = min(best_strictly_above[ab], above)
    return {ab: {
        "clinched_first": worst_at_or_above[ab] == 0,     # always strictly top
        "clinched_advance": worst_at_or_above[ab] <= 1,   # always top 2
        "eliminated": best_strictly_above[ab] >= 2,       # never reaches top 2
        "out_entirely": best_strictly_above[ab] >= 3,     # never reaches top 3
    } for ab in abbrs}


def locked_positions(rows: list[dict],
                     remaining: list[tuple[str, str]] | None = None
                     ) -> tuple[str | None, str | None]:
    """(team locked into 1st, team locked into 2nd) for the bracket, or None."""
    if not rows:
        return None, None
    if all(t["P"] >= 3 for t in rows):              # group complete
        return rows[0]["abbr"], (rows[1]["abbr"] if len(rows) > 1 else None)
    cl = group_clinch(rows, remaining)
    first = next((ab for ab in cl if cl[ab]["clinched_first"]), None)
    # If 1st is locked, the (unique) other team that has clinched top-2 is 2nd.
    second = None
    if first:
        second = next((ab for ab in cl
                       if ab != first and cl[ab]["clinched_advance"]), None)
    return first, second


def apply_result(rows: list[dict], home: str, away: str, outcome: str) -> list[dict]:
    """Copy of rows with one game (home vs away) played as 'H'|'D'|'A'.

    Uses a nominal 1-0 margin (0-0 for a draw) so tied teams order sensibly;
    the clinch flags themselves only use points, so the margin is display-only.
    """
    new = [dict(t) for t in rows]
    by = {t["abbr"]: t for t in new}
    h, a = by.get(home), by.get(away)
    if not h or not a:
        return new
    h["P"] += 1; a["P"] += 1
    if outcome == "H":
        h["Pts"] += 3; h["GF"] += 1; h["GD"] += 1; a["GA"] += 1; a["GD"] -= 1
    elif outcome == "A":
        a["Pts"] += 3; a["GF"] += 1; a["GD"] += 1; h["GA"] += 1; h["GD"] -= 1
    else:
        h["Pts"] += 1; a["Pts"] += 1
    return new


_NAME = {"H": "home win", "D": "draw", "A": "away win"}


def _status(c: dict) -> str:
    if c.get("clinched_first"):
        return "first"
    if c.get("clinched_advance"):
        return "advance"
    if c.get("out_entirely"):
        return "elim"
    return "none"


def game_implications(groups: dict[str, list[dict]],
                      remaining: list[dict]) -> list[dict]:
    """For each remaining game, the group's projected ranking after each of its
    three outcomes, ordered by points->GD->GF, with each team's clinch status."""
    rem_by_group: dict[str, list[tuple[str, str]]] = {}
    for m in remaining:
        rem_by_group.setdefault(m["group"], []).append((m["home"], m["away"]))
    out = []
    for m in remaining:
        g = m["group"]
        rows = groups.get(g)
        if not rows:
            continue
        # The other games still open in this group after this one is played.
        others = [pr for pr in rem_by_group[g] if pr != (m["home"], m["away"])]
        outcomes = []
        for oc in ("H", "D", "A"):
            after_rows = apply_result(rows, m["home"], m["away"], oc)
            cl = group_clinch(after_rows, others)
            ordered = sorted(after_rows, key=lambda t: (t["Pts"], t["GD"], t["GF"]),
                             reverse=True)
            ranking = [{
                "pos": i + 1, "abbr": t["abbr"], "name": t.get("team", t["abbr"]),
                "pts": t["Pts"], "gd": t["GD"],
                "status": _status(cl.get(t["abbr"], {})),
            } for i, t in enumerate(ordered)]
            who = m["home"] if oc == "H" else (m["away"] if oc == "A" else "")
            outcomes.append({"outcome": oc,
                             "label": f"{who} win" if who else "draw",
                             "ranking": ranking})
        out.append({
            "group": g, "home": m["home"], "away": m["away"],
            "home_name": m.get("home_name", m["home"]), "away_name": m.get("away_name", m["away"]),
            "et_date": m.get("et_date", ""), "et_time": m.get("et_time", ""),
            "kickoff": m.get("kickoff", ""), "outcomes": outcomes,
        })
    out.sort(key=lambda x: (x["kickoff"] or "9", x["group"]))
    return out
