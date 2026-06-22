"""Live ranking logic: group qualification + the 8 best third-place wildcards.

Reuses the FIFA tiebreaker ordering (points -> goal difference -> goals scored)
from :mod:`wcp.model.standings`, applied to the real live standings.
"""
from __future__ import annotations

N_WILDCARDS = 8        # 8 best third-placed teams advance to the Round of 32


def qualification_flags(groups: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Tag each team row with a live status: top-2 = advancing, 3rd = wildcard
    contender, 4th = eliminated-track. (Provisional until the group finishes.)"""
    thirds = third_place_race(groups)
    wildcard_in = {(t["group"], t["team"]) for t in thirds
                   if t["wildcard_status"] == "IN"}
    out = {}
    for g, rows in groups.items():
        tagged = []
        for i, t in enumerate(rows):
            row = dict(t)
            if i < 2:
                row["live_status"] = "advancing (top 2)"
            elif i == 2:
                row["live_status"] = ("wildcard IN" if (g, t["team"]) in wildcard_in
                                      else "wildcard out")
            else:
                row["live_status"] = "eliminated-track"
            tagged.append(row)
        out[g] = tagged
    return out


def third_place_race(groups: dict[str, list[dict]]) -> list[dict]:
    """Rank the twelve third-placed teams; top 8 hold a wildcard spot.

    Ranked by points -> GD -> GF (FIFA criteria; fair-play/lots not modelled).
    """
    thirds = []
    for g, rows in groups.items():
        if len(rows) >= 3:
            t = dict(rows[2])      # rows are sorted by in-group rank
            t["group"] = g
            thirds.append(t)
    # FIFA criteria: points -> goal difference -> goals scored.
    thirds.sort(key=lambda t: (t["Pts"], t["GD"], t["GF"]), reverse=True)
    for pos, t in enumerate(thirds):
        t["wildcard_rank"] = pos + 1
        t["wildcard_status"] = "IN" if pos < N_WILDCARDS else "OUT"
    return thirds


def _wildcard_in_set(groups: dict[str, list[dict]]) -> set[str]:
    """Abbrs of the third-placed teams currently holding a top-8 wildcard."""
    return {t["abbr"] for t in third_place_race(groups)
            if t["wildcard_status"] == "IN"}


def wildcard_swings(groups: dict[str, list[dict]],
                    remaining: list[dict]) -> list[dict]:
    """Which upcoming games can move a team across the top-8 wildcard line.

    For every remaining game and each of its three outcomes, project that one
    result onto its group (re-sorted by the FIFA tiebreak), recompute the
    cross-group third-place race, and report any change in the IN (top-8) set.
    Only games with at least one membership-changing outcome are returned.
    """
    from .clinch import apply_result            # local import: avoid cycle

    base_in = _wildcard_in_set(groups)
    out = []
    for m in remaining:
        g = m.get("group")
        rows = groups.get(g)
        if not rows:
            continue
        swung = []
        for oc in ("H", "D", "A"):
            new_rows = sorted(apply_result(rows, m["home"], m["away"], oc),
                              key=lambda t: (t["Pts"], t["GD"], t["GF"]),
                              reverse=True)
            projected = dict(groups)
            projected[g] = new_rows
            new_in = _wildcard_in_set(projected)
            entered = sorted(new_in - base_in)
            dropped = sorted(base_in - new_in)
            if entered or dropped:
                who = m["home"] if oc == "H" else (m["away"] if oc == "A" else "")
                swung.append({"outcome": oc,
                              "label": f"{who} win" if who else "draw",
                              "entered": entered, "dropped": dropped})
        if swung:
            out.append({
                "group": g, "home": m["home"], "away": m["away"],
                "home_name": m.get("home_name", m["home"]),
                "away_name": m.get("away_name", m["away"]),
                "et_date": m.get("et_date", ""), "et_time": m.get("et_time", ""),
                "kickoff": m.get("kickoff", ""), "outcomes": swung,
            })
    out.sort(key=lambda x: (x["kickoff"] or "9", x["group"]))
    return out


def tournament_complete_groups(groups: dict[str, list[dict]]) -> int:
    """How many groups have every team on 3 games played (group stage done)."""
    done = 0
    for rows in groups.values():
        if rows and all(t["P"] >= 3 for t in rows):
            done += 1
    return done
