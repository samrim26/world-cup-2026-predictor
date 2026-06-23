"""Tests for the Phase-6 live views: predicted bracket, wildcard swings, rankings sort."""
from wcp.live import user_picks
from wcp.live.tracker import wildcard_swings, advancement_set
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
    adv = advancement_set(groups)
    for s in sw:
        assert s["outcomes"], "swing game must have >=1 changing outcome"
        for o in s["outcomes"]:
            assert o["label"]
            assert set(o["entered"]).isdisjoint(o["dropped"])      # disjoint
            # a dropped team was advancing; an entered team was not
            for abv in o["dropped"]:
                assert abv in adv
            for abv in o["entered"]:
                assert abv not in adv


def test_dropped_team_is_never_top2():
    """Invariant for the 'POR OUT' regression: a team that finishes in the top
    two of its group after an outcome can never be reported as dropping OUT of
    the advancement set (top-2 always advances)."""
    from wcp.live.clinch import apply_result

    groups, remaining = {}, []
    for i, g in enumerate("ABCDEFGHIJKL"):
        a, b, c, d = f"{g}1", f"{g}2", f"{g}3", f"{g}4"
        groups[g] = _group([(a, 6, 3, 4), (b, 3, 0, 3), (c, 3, 0, 2), (d, 0, -3, 0)])
        remaining.append({"group": g, "home": b, "away": c, "kickoff": f"2026-06-2{i%10}",
                          "et_date": "2026-06-24", "et_time": "12:00 PM"})
    sw = wildcard_swings(groups, remaining)
    for s in sw:
        rows = groups[s["group"]]
        for o in s["outcomes"]:
            new = sorted(apply_result(rows, s["home"], s["away"], o["outcome"]),
                         key=lambda t: (t["Pts"], t["GD"], t["GF"]), reverse=True)
            top2 = {new[0]["abbr"], new[1]["abbr"]}
            assert top2.isdisjoint(o["dropped"]), \
                f"a top-2 team was wrongly marked OUT: {top2 & set(o['dropped'])}"


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
