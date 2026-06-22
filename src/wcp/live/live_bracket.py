"""Live Round-of-32 bracket, filled from current standings and advanced by live
knockout results.

Uses the OFFICIAL 2026 bracket template (matches 73-104). Group winner/runner-up
slots fill as each group finishes; the eight third-place slots fill once the
group stage is complete, assigned to their eligible slots by bipartite matching
(the same rule FIFA's combination table encodes). Knockout rounds then advance
automatically as real results come in; unresolved slots show placeholders.
"""
from __future__ import annotations

from .bracket_tracker import finished_knockouts
from .tracker import third_place_race, tournament_complete_groups

# Round of 32: match_no -> (slotA, slotB).  Slots: ("W",g) ("R",g) ("3",match_no)
R32 = {
    73: (("R", "A"), ("R", "B")), 74: (("W", "E"), ("3", 74)),
    75: (("W", "F"), ("R", "C")), 76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("3", 77)), 78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("3", 79)), 80: (("W", "L"), ("3", 80)),
    81: (("W", "D"), ("3", 81)), 82: (("W", "G"), ("3", 82)),
    83: (("R", "K"), ("R", "L")), 84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("3", 85)), 86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("3", 87)), 88: (("R", "D"), ("R", "G")),
}
# Eligible group set for each third-place slot.
THIRD_SLOTS = {
    74: set("ABCDF"), 77: set("CDFGH"), 79: set("CEFHI"), 80: set("EHIJK"),
    81: set("BEFIJ"), 82: set("AEHIJ"), 85: set("EFGIJ"), 87: set("DEIJL"),
}
# Later rounds: match_no -> (feeder_match_a, feeder_match_b).
TREE = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100), 104: (101, 102),
}
ROUND_OF = {**{m: "Round of 32" for m in R32},
            **{m: "Round of 16" for m in range(89, 97)},
            **{m: "Quarterfinal" for m in range(97, 101)},
            101: "Semifinal", 102: "Semifinal", 104: "Final"}


def _assign_thirds(qualifying_groups: set[str]) -> dict[int, str]:
    """Match the 8 qualifying third-place groups to their eligible R32 slots.

    Pure-Python backtracking with a most-constrained-slot-first order (a perfect
    matching is guaranteed to exist by FIFA's eligible-set design).
    """
    groups = sorted(qualifying_groups)
    if len(groups) != 8:
        return {}
    # Process the slots with the fewest eligible qualifying groups first.
    slot_order = sorted(THIRD_SLOTS, key=lambda m:
                        sum(1 for g in groups if g in THIRD_SLOTS[m]))
    result: dict[int, str] = {}
    used: set[str] = set()

    def solve(i: int) -> bool:
        if i == len(slot_order):
            return True
        m = slot_order[i]
        for g in groups:
            if g not in used and g in THIRD_SLOTS[m]:
                used.add(g); result[m] = g
                if solve(i + 1):
                    return True
                used.discard(g); result.pop(m, None)
        return False

    solve(0)
    return dict(result)


def build(feed, groups: dict | None = None, locked: dict | None = None) -> dict:
    """``locked`` maps group letter -> (first_abbr|None, second_abbr|None): teams
    mathematically clinched into 1st/2nd, used to fill R32 slots before the group
    is even complete."""
    groups = groups or feed.standings()
    locked = locked or {}
    all_done = tournament_complete_groups(groups) == 12

    # Per-group winner/runner/third (current order) + whether the group is final.
    win, run, decided = {}, {}, {}
    for g, rows in groups.items():
        decided[g] = bool(rows) and all(t["P"] >= 3 for t in rows)
        win[g] = rows[0]["abbr"] if rows else None
        run[g] = rows[1]["abbr"] if len(rows) > 1 else None

    third_team = {}
    if all_done:
        qual = {t["group"] for t in third_place_race(groups)
                if t["wildcard_status"] == "IN"}
        mapping = _assign_thirds(qual)               # slot match_no -> group letter
        third_by_group = {t["group"]: t["abbr"] for t in
                          third_place_race(groups) if t["wildcard_status"] == "IN"}
        third_team = {m: third_by_group[gl] for m, gl in mapping.items()}

    # Knockout results so far -> winner by team pair.
    ko = {frozenset((k["winner"], k["loser"])): k["winner"]
          for k in finished_knockouts(feed)}

    def slot(spec):
        """Return (team_abbr_or_None, label, is_final)."""
        kind, key = spec
        lk = locked.get(key) or (None, None)
        if kind == "W":
            if lk[0]:                                    # 1st mathematically clinched
                return lk[0], lk[0], True
            return win[key], (win[key] if decided[key]
                              else f"{win[key]}?" if win[key] else f"1{key}"), decided[key]
        if kind == "R":
            if lk[1]:                                    # 2nd mathematically clinched
                return lk[1], lk[1], True
            return run[key], (run[key] if decided[key]
                              else f"{run[key]}?" if run[key] else f"2{key}"), decided[key]
        # third-place slot
        if key in third_team:
            return third_team[key], third_team[key], True
        return None, "3rd[" + "".join(sorted(THIRD_SLOTS[key])) + "]", False

    resolved: dict[int, dict] = {}

    def winner_of(m: int):
        a = resolved[m]["a_team"]; b = resolved[m]["b_team"]
        if a and b:
            return ko.get(frozenset((a, b)))
        return None

    # Round of 32.
    for m, (sa, sb) in R32.items():
        a_team, a_lbl, a_fin = slot(sa)
        b_team, b_lbl, b_fin = slot(sb)
        resolved[m] = {"round": "Round of 32", "match": m,
                       "a_team": a_team, "b_team": b_team,
                       "a": a_lbl, "b": b_lbl,
                       "ready": a_fin and b_fin}
        resolved[m]["winner"] = winner_of(m)

    # Later rounds (depend on feeders' winners).
    for m, (f1, f2) in TREE.items():
        a_team = resolved[f1].get("winner")
        b_team = resolved[f2].get("winner")
        a_lbl = a_team or f"W{f1}"
        b_lbl = b_team or f"W{f2}"
        resolved[m] = {"round": ROUND_OF[m], "match": m,
                       "a_team": a_team, "b_team": b_team,
                       "a": a_lbl, "b": b_lbl,
                       "ready": bool(a_team and b_team)}
        resolved[m]["winner"] = winner_of(m)

    return {"matches": resolved, "all_groups_done": all_done}


def by_round(bracket: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for m in sorted(bracket["matches"]):
        info = bracket["matches"][m]
        out.setdefault(info["round"], []).append(info)
    return out


# --- text render (CLI) ----------------------------------------------------- #
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
GREEN = "\033[32m"; YELLOW = "\033[33m"; CYAN = "\033[36m"


def render_text(bracket: dict) -> str:
    rounds = by_round(bracket)
    out = [f"{BOLD}{CYAN}LIVE BRACKET{RESET}  "
           f"{DIM}(official 2026 template; '?' = provisional, "
           f"3rd[..] = wildcard TBD){RESET}"]
    if not bracket["all_groups_done"]:
        out.append(f"  {DIM}group stage in progress - winners/runners-up lock as "
                   f"groups finish; thirds lock when all 12 are done{RESET}")
    for rnd in ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"]:
        if rnd not in rounds:
            continue
        out.append(f"\n{BOLD}{rnd}{RESET}")
        for info in rounds[rnd]:
            w = info["winner"]
            def fmt(team, lbl):
                if w and team == w:
                    return f"{GREEN}{BOLD}{lbl}{RESET}"
                if info["ready"]:
                    return f"{lbl}"
                return f"{DIM}{lbl}{RESET}"
            arrow = f"  ->  {GREEN}{w}{RESET}" if w else ""
            out.append(f"  M{info['match']:>3}  {fmt(info['a_team'], info['a']):>16} "
                       f"vs {fmt(info['b_team'], info['b']):<16}{arrow}")
    return "\n".join(out)
