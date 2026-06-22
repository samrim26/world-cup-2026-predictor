"""Property-based stress tests for the live clinch engine.

Generates thousands of realistic random group states (a partially-played
round-robin) and checks the clinch flags against an *independent* brute-force
reference, plus the structural invariants and the bracket-locking logic.
"""
import random
from itertools import combinations, product

import pytest

from wcp.live import clinch

PAIRS = list(combinations(range(4), 2))     # the 6 games of a 4-team group


def random_group_state(rng):
    """Return (rows, remaining_pairs) for a partially-played 4-team group."""
    pts = [0, 0, 0, 0]
    gf = [0, 0, 0, 0]
    ga = [0, 0, 0, 0]
    played = [0, 0, 0, 0]
    remaining = []
    for (i, j) in PAIRS:
        if rng.random() < 0.5:                       # game still to play
            remaining.append((f"T{i}", f"T{j}"))
            continue
        hs, as_ = rng.randint(0, 4), rng.randint(0, 4)
        gf[i] += hs; ga[i] += as_; gf[j] += as_; ga[j] += hs
        played[i] += 1; played[j] += 1
        if hs > as_:
            pts[i] += 3
        elif as_ > hs:
            pts[j] += 3
        else:
            pts[i] += 1; pts[j] += 1
    rows = [{"abbr": f"T{i}", "team": f"T{i}", "Pts": pts[i], "P": played[i],
             "GF": gf[i], "GA": ga[i], "GD": gf[i] - ga[i]} for i in range(4)]
    return rows, remaining


def reference_clinch(rows, remaining):
    """Independent points-only brute force (different code path than clinch.py)."""
    abbrs = [t["abbr"] for t in rows]
    base = {t["abbr"]: t["Pts"] for t in rows}
    scenarios = []
    for combo in product("HDA", repeat=len(remaining)):
        p = dict(base)
        for (h, a), oc in zip(remaining, combo):
            if oc == "H":
                p[h] += 3
            elif oc == "A":
                p[a] += 3
            else:
                p[h] += 1; p[a] += 1
        scenarios.append(p)
    res = {}
    for ab in abbrs:
        strictly = [sum(1 for o in abbrs if o != ab and p[o] > p[ab]) for p in scenarios]
        at_or = [sum(1 for o in abbrs if o != ab and p[o] >= p[ab]) for p in scenarios]
        res[ab] = {
            "clinched_first": all(x == 0 for x in at_or),
            "clinched_advance": all(x <= 1 for x in at_or),
            "eliminated": all(x >= 2 for x in strictly),
            "out_entirely": all(x >= 3 for x in strictly),
        }
    return res


def test_clinch_matches_bruteforce():
    rng = random.Random(1)
    for _ in range(4000):
        rows, rem = random_group_state(rng)
        got = clinch.group_clinch(rows, rem)
        exp = reference_clinch(rows, rem)
        assert got == exp, (rows, rem, got, exp)


def test_clinch_invariants():
    rng = random.Random(2)
    for _ in range(4000):
        rows, rem = random_group_state(rng)
        cl = clinch.group_clinch(rows, rem)
        for ab, f in cl.items():
            if f["clinched_first"]:
                assert f["clinched_advance"]                 # 1st => through
            if f["out_entirely"]:
                assert f["eliminated"]                       # out => not top-2
            if f["clinched_advance"]:
                assert not f["eliminated"]                   # can't be both
            # at most one team per group can clinch 1st
        firsts = [a for a, f in cl.items() if f["clinched_first"]]
        assert len(firsts) <= 1
        advs = [a for a, f in cl.items() if f["clinched_advance"]]
        assert len(advs) <= 2                                # at most 2 go through here


def test_locked_positions_never_wrong():
    """A locked 1st can never be overtaken on points (no team finishes strictly
    above it in any scenario); a locked 2nd can have at most one team above it.

    Rows are passed pre-sorted by standing — the contract `feed.standings()`
    guarantees in production.
    """
    rng = random.Random(3)
    for _ in range(4000):
        rows, rem = random_group_state(rng)
        rows.sort(key=lambda t: (t["Pts"], t["GD"], t["GF"]), reverse=True)
        first, second = clinch.locked_positions(rows, rem)
        abbrs = [t["abbr"] for t in rows]
        base = {t["abbr"]: t["Pts"] for t in rows}
        scen = []
        for combo in product("HDA", repeat=len(rem)):
            p = dict(base)
            for (h, a), oc in zip(rem, combo):
                if oc == "H":
                    p[h] += 3
                elif oc == "A":
                    p[a] += 3
                else:
                    p[h] += 1; p[a] += 1
            scen.append(p)
        if first:
            assert all(sum(1 for o in abbrs if o != first and p[o] > p[first]) == 0
                       for p in scen)
        if second:
            assert first is not None
            assert all(sum(1 for o in abbrs if o != second and p[o] > p[second]) <= 1
                       for p in scen)


def test_complete_group_is_final():
    rows = [{"abbr": "A", "Pts": 9, "P": 3, "GF": 6, "GA": 1, "GD": 5, "team": "A"},
            {"abbr": "B", "Pts": 6, "P": 3, "GF": 4, "GA": 3, "GD": 1, "team": "B"},
            {"abbr": "C", "Pts": 3, "P": 3, "GF": 2, "GA": 4, "GD": -2, "team": "C"},
            {"abbr": "D", "Pts": 0, "P": 3, "GF": 1, "GA": 5, "GD": -4, "team": "D"}]
    cl = clinch.group_clinch(rows, [])
    assert cl["A"]["clinched_first"] and cl["A"]["clinched_advance"]
    assert cl["B"]["clinched_advance"] and not cl["B"]["clinched_first"]
    assert cl["D"]["out_entirely"] and cl["D"]["eliminated"]
    assert clinch.locked_positions(rows, []) == ("A", "B")


def test_head_to_head_clinch():
    """The USA case: leader on 6, two chasers on 3 who play each other -> leader
    has clinched a knockout spot even though each chaser *could* reach 6."""
    rows = [{"abbr": "USA", "Pts": 6, "P": 2, "GF": 5, "GA": 0, "GD": 5, "team": "USA"},
            {"abbr": "AUS", "Pts": 3, "P": 2, "GF": 2, "GA": 2, "GD": 0, "team": "AUS"},
            {"abbr": "PAR", "Pts": 3, "P": 2, "GF": 2, "GA": 4, "GD": -2, "team": "PAR"},
            {"abbr": "TUR", "Pts": 0, "P": 2, "GF": 1, "GA": 4, "GD": -3, "team": "TUR"}]
    remaining = [("PAR", "AUS"), ("TUR", "USA")]
    cl = clinch.group_clinch(rows, remaining)
    assert cl["USA"]["clinched_advance"] is True
    assert cl["USA"]["clinched_first"] is False     # can be tied on points
    assert cl["TUR"]["out_entirely"] is False        # TUR can still reach 3rd (3 pts)


def test_implications_structure():
    rows = [{"abbr": "A", "Pts": 4, "P": 2, "GF": 3, "GA": 1, "GD": 2, "team": "A"},
            {"abbr": "B", "Pts": 3, "P": 2, "GF": 2, "GA": 2, "GD": 0, "team": "B"},
            {"abbr": "C", "Pts": 2, "P": 2, "GF": 2, "GA": 3, "GD": -1, "team": "C"},
            {"abbr": "D", "Pts": 1, "P": 2, "GF": 1, "GA": 2, "GD": -1, "team": "D"}]
    groups = {"X": rows}
    remaining = [{"group": "X", "home": "A", "away": "D"},
                 {"group": "X", "home": "B", "away": "C"}]
    imps = clinch.game_implications(groups, remaining)
    assert len(imps) == 2
    for im in imps:
        for o in im["outcomes"]:
            r = o["ranking"]
            assert len(r) == 4
            assert [x["pos"] for x in r] == [1, 2, 3, 4]
            # ordered by points descending
            assert all(r[i]["pts"] >= r[i + 1]["pts"] for i in range(3))
            assert all(x["status"] in ("first", "advance", "elim", "none") for x in r)
