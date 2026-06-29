"""Live pool scoring + the head-to-head race against the rival ARG backer.

Pool scoring (per correctly-placed team, per round the team *reaches*):
    Round of 32 +10 · Round of 16 +20 · Quarterfinal +40 · Semifinal +80
    Third place +40 · Finalists (reach final) +160 · Winner +320

Both the user and the rival picked the same Final (Argentina over Spain), so the
+320 winner and +160 finalist swings cancel between them — the race is decided by
where their deeper-round picks diverge (the user's Brazil/USA/Netherlands vs the
rival's Belgium/Germany/England/France).
"""
from __future__ import annotations

from . import user_picks as up
from .rival_picks import RIVAL

SCORING = {"R32": 10, "R16": 20, "QF": 40, "SF": 80, "FINAL": 160,
           "WINNER": 320, "THIRD": 40}
ROUND_ORDER = ["R32", "R16", "QF", "SF", "FINAL", "WINNER"]
ROUND_LABEL = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarterfinal",
               "SF": "Semifinal", "FINAL": "Final", "WINNER": "Winner", "THIRD": "Third place"}

# Current total pool scores supplied by the user (anchor, Jun 29, mid-R32), and
# each side's knockout-bracket points at that same moment. The fixed group-stage
# component = anchor total - anchor bracket points; live totals then grow as more
# knockout results come in. (ANCHOR_BRACKET["you"] is computed by tests to match.)
ANCHOR_TOTAL = {"you": 453, "rival": 437}
ANCHOR_BRACKET = {"you": 270, "rival": 300}
GROUP_BASE = {p: ANCHOR_TOTAL[p] - ANCHOR_BRACKET[p] for p in ANCHOR_TOTAL}

# Map the live-bracket round names to canonical keys.
_LB_NAME = {"Round of 32": "R32", "Round of 16": "R16", "Quarterfinal": "QF",
            "Semifinal": "SF", "Final": "FINAL"}
# Map user_picks.predicted_rounds() keys to canonical keys.
_UP_NAME = {"Round of 32 (32)": "R32", "Round of 16 (16)": "R16",
            "Quarterfinals (8)": "QF", "Semifinals (4)": "SF",
            "Final (2)": "FINAL", "Champion (1)": "WINNER"}


def reached_rounds(live_bracket: dict) -> dict[str, set[str]]:
    """Teams that have *reached* each round, read from the live-bracket tree."""
    reached = {k: set() for k in ROUND_ORDER}
    final_winner = None
    for rnd, matches in live_bracket.items():
        key = _LB_NAME.get(rnd)
        if not key:
            continue
        for m in matches:
            for t in (m.get("a_team"), m.get("b_team")):
                if t:
                    reached[key].add(t)
            if rnd == "Final" and m.get("winner"):
                final_winner = m["winner"]
    if final_winner:
        reached["WINNER"].add(final_winner)
    return reached


def user_placements() -> dict[str, list[str]]:
    rounds = up.predicted_rounds()
    return {_UP_NAME[k]: v for k, v in rounds.items() if k in _UP_NAME}


def user_third() -> str:
    return up.NAME2ABBR.get(up.THIRD, up.THIRD)


def _round_breakdown(placed: set[str], reached: set[str], eliminated: set[str],
                     pts: int) -> dict:
    hits = sorted(placed & reached)
    alive = sorted(t for t in placed if t not in reached and t not in eliminated)
    dead = sorted(t for t in placed if t not in reached and t in eliminated)
    return {"hits": hits, "alive": alive, "dead": dead,
            "earned": len(hits) * pts, "potential": (len(hits) + len(alive)) * pts}


def score_breakdown(placements: dict[str, list[str]], third_pick, third_candidates,
                    reached: dict[str, set[str]], third_winner, eliminated: set[str]) -> dict:
    """Per-round hit/alive/dead breakdown + bracket totals for one entrant."""
    rounds = {}
    earned = potential = 0
    for r in ROUND_ORDER:
        bd = _round_breakdown(set(placements.get(r, [])), reached.get(r, set()),
                              eliminated, SCORING[r])
        rounds[r] = bd
        earned += bd["earned"]
        potential += bd["potential"]
    # Third place: a single pick, or undetermined among candidates.
    cands = [third_pick] if third_pick else list(third_candidates or [])
    if third_winner is not None:
        third_hit = third_winner in cands
        third_alive = False
    else:
        third_hit = False
        third_alive = any(c not in eliminated for c in cands)
    third_earned = SCORING["THIRD"] if third_hit else 0
    rounds["THIRD"] = {
        "hits": [third_winner] if third_hit else [],
        "alive": cands if third_alive else [],
        "dead": [] if (third_hit or third_alive) else cands,
        "earned": third_earned,
        "potential": third_earned + (SCORING["THIRD"] if third_alive else 0),
        "undetermined": third_pick is None and bool(cands),
    }
    earned += rounds["THIRD"]["earned"]
    potential += rounds["THIRD"]["potential"]
    return {"rounds": rounds, "bracket_earned": earned, "bracket_potential": potential}


def _state(team: str, reached_set: set[str], eliminated: set[str]) -> str:
    if team in reached_set:
        return "hit"
    return "out" if team in eliminated else "alive"


def _add_lever(by_team, t, rnd, st):
    e = by_team.setdefault(t, {"team": t, "rounds": [], "upside": 0, "banked": 0})
    e["rounds"].append({"round": rnd, "label": ROUND_LABEL[rnd],
                        "pts": SCORING[rnd], "state": st})
    if st == "alive":
        e["upside"] += SCORING[rnd]
    elif st == "hit":
        e["banked"] += SCORING[rnd]


def _levers(mine: dict[str, list[str]], theirs: dict[str, list[str]],
            my_third: str, my_third_cands, reached: dict[str, set[str]],
            third_winner, eliminated: set[str]) -> list[dict]:
    """Teams unique to my bracket (vs the rival) that still have *forward* points
    in play — the live race differentiators. ``upside`` = points still winnable;
    settled-and-banked R32/R16 edges (no upside left) are dropped."""
    by_team: dict[str, dict] = {}
    for r in ROUND_ORDER:
        for t in set(mine.get(r, [])) - set(theirs.get(r, [])):
            _add_lever(by_team, t, r, _state(t, reached.get(r, set()), eliminated))
    cands = [my_third] if my_third else list(my_third_cands or [])
    for t in cands:
        if third_winner is not None:
            st = "hit" if t == third_winner else "out"
        else:
            st = "out" if t in eliminated else "alive"
        _add_lever(by_team, t, "THIRD", st)
    levers = [e for e in by_team.values() if e["upside"] > 0]
    return sorted(levers, key=lambda e: (-e["upside"], e["team"]))


def head_to_head(reached: dict[str, set[str]], third_winner, eliminated: set[str]) -> dict:
    you_pl = user_placements()
    you_third = user_third()
    rival_pl = RIVAL["rounds"]
    rival_third = RIVAL["third"]
    rival_cands = RIVAL["third_candidates"]

    you_bd = score_breakdown(you_pl, you_third, None, reached, third_winner, eliminated)
    rival_bd = score_breakdown(rival_pl, rival_third, rival_cands, reached,
                               third_winner, eliminated)

    you_total = GROUP_BASE["you"] + you_bd["bracket_earned"]
    rival_total = GROUP_BASE["rival"] + rival_bd["bracket_earned"]
    you_max = you_total + (you_bd["bracket_potential"] - you_bd["bracket_earned"])
    rival_max = rival_total + (rival_bd["bracket_potential"] - rival_bd["bracket_earned"])

    def pack(name, base, bd, total, mx):
        return {"name": name, "total": total, "max_possible": mx,
                "group_base": base, "bracket_earned": bd["bracket_earned"],
                "rounds": bd["rounds"]}

    return {
        "prize": {"pool": 380, "buy_in": 20,
                  "note": "You're the only two with Argentina as champion."},
        "you": pack("You", GROUP_BASE["you"], you_bd, you_total, you_max),
        "rival": pack(RIVAL["name"], GROUP_BASE["rival"], rival_bd, rival_total, rival_max),
        "lead": you_total - rival_total,
        "shared": {"champion": "ARG", "runner_up": "ESP",
                   "note": "Identical Final (ARG def. ESP) — winner + finalist points cancel."},
        "you_levers": _levers(you_pl, rival_pl, you_third, None, reached, third_winner, eliminated),
        "rival_levers": _levers(rival_pl, you_pl, rival_third, rival_cands, reached,
                                third_winner, eliminated),
        "third_undetermined": rival_third is None,
        # every team each side still has advancing somewhere (for live-game badges)
        "you_all": sorted({t for r in ROUND_ORDER for t in you_pl.get(r, [])} | {you_third}),
        "rival_all": sorted({t for r in ROUND_ORDER for t in rival_pl.get(r, [])}
                            | set(rival_cands)),
        # per-round placements (for the bracket pick-overlay)
        "you_placed": {r: list(you_pl.get(r, [])) for r in ROUND_ORDER},
        "rival_placed": {r: list(rival_pl.get(r, [])) for r in ROUND_ORDER},
    }
