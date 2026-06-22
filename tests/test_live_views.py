"""Tests for the Phase-6 live views: predicted bracket, wildcard swings, rankings sort."""
from wcp.live import user_picks
from wcp.live.tracker import wildcard_swings, third_place_race
from wcp.live.rankings import build_rankings

_KO_ORDER = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]


def test_predicted_bracket_chains_and_valid():
    pb = user_picks.predicted_bracket()
    counts = {r: len(pb["rounds"][r]) for r in _KO_ORDER}
    assert counts == {"Round of 32": 16, "Round of 16": 8, "Quarterfinals": 4,
                      "Semifinals": 2, "Final": 1}
    valid = set(user_picks.NAME2ABBR.values())
    for r in _KO_ORDER:
        for m in pb["rounds"][r]:
            assert {m["a"], m["b"], m["winner"]} <= valid          # real abbrs
            assert m["winner"] in (m["a"], m["b"])                 # winner is a participant
    # each round's winners are exactly the next round's entrants (in any order)
    for a, b in zip(_KO_ORDER, _KO_ORDER[1:]):
        winners = sorted(m["winner"] for m in pb["rounds"][a])
        entrants = sorted(t for m in pb["rounds"][b] for t in (m["a"], m["b"]))
        assert winners == entrants
    assert pb["champion"] == pb["rounds"]["Final"][0]["winner"] == "ARG"


def _group(abbr_pts):
    rows = []
    for ab, p, gd, gf in abbr_pts:
        rows.append({"abbr": ab, "team": ab, "Pts": p, "P": 2,
                     "GF": gf, "GA": gf - gd, "GD": gd, "W": 0, "D": 0, "L": 0})
    rows.sort(key=lambda t: (t["Pts"], t["GD"], t["GF"]), reverse=True)
    return rows


def test_wildcard_swings_structure():
    # 12 groups, each with a clear 3rd-placed team and one remaining game.
    groups, remaining = {}, []
    for i, g in enumerate("ABCDEFGHIJKL"):
        a, b, c, d = f"{g}1", f"{g}2", f"{g}3", f"{g}4"
        groups[g] = _group([(a, 6, 3, 4), (b, 4, 1, 3), (c, 3 if i < 6 else 1, 0, 2), (d, 0, -4, 0)])
        remaining.append({"group": g, "home": c, "away": d, "kickoff": f"2026-06-2{i%10}",
                          "et_date": "2026-06-24", "et_time": "12:00 PM"})
    sw = wildcard_swings(groups, remaining)
    in_set = {t["abbr"] for t in third_place_race(groups) if t["wildcard_status"] == "IN"}
    for s in sw:
        assert s["outcomes"], "swing game must have >=1 changing outcome"
        for o in s["outcomes"]:
            assert o["label"]
            assert set(o["entered"]).isdisjoint(o["dropped"])      # disjoint
            # a dropped team was IN; an entered team was OUT
            for ab in o["dropped"]:
                assert ab in in_set
            for ab in o["entered"]:
                assert ab not in in_set


def test_rankings_sorted_by_points_no_odds_dependency():
    groups = {
        "A": _group([("MEX", 6, 3, 5), ("KOR", 4, 1, 3), ("CZE", 1, -1, 2), ("RSA", 1, -3, 1)]),
        "B": _group([("SUI", 6, 4, 6), ("CAN", 3, 0, 3), ("BIH", 3, 0, 2), ("QAT", 0, -4, 0)]),
    }
    # Pass deliberately "wrong" odds to prove the ranking ignores them now.
    odds = {t["abbr"]: 0.5 for g in groups.values() for t in g}
    rk = build_rankings(groups, odds, {})
    keys = [(r["Pts"], r["GD"], r["GF"]) for r in rk]
    assert keys == sorted(keys, reverse=True)
    assert rk[0]["abbr"] in ("MEX", "SUI")            # top points
    assert all("clinched_advance" in r for r in rk)   # flags preserved
