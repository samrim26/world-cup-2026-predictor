"""Tests for the knockout pool scoring + head-to-head race engine."""
from wcp.live import pool


def test_scoring_buckets():
    assert pool.SCORING == {"R32": 10, "R16": 20, "QF": 40, "SF": 80,
                            "FINAL": 160, "WINNER": 320, "THIRD": 40}


def test_reached_rounds_from_bracket():
    lb = {
        "Round of 32": [{"a_team": "A", "b_team": "B", "winner": "A"},
                        {"a_team": "C", "b_team": "D", "winner": None}],
        "Round of 16": [{"a_team": "A", "b_team": None, "winner": None}],
        "Quarterfinal": [], "Semifinal": [],
        "Final": [{"a_team": "X", "b_team": "Y", "winner": "X"}],
    }
    r = pool.reached_rounds(lb)
    assert r["R32"] == {"A", "B", "C", "D"}
    assert r["R16"] == {"A"}
    assert r["QF"] == set() and r["SF"] == set()
    assert r["FINAL"] == {"X", "Y"}
    assert r["WINNER"] == {"X"}


def test_score_breakdown_hit_alive_dead():
    placements = {"QF": ["A", "B", "C"]}
    reached = {"QF": {"A"}}
    bd = pool.score_breakdown(placements, third_pick="Z", third_candidates=None,
                              reached=reached, third_winner=None, eliminated={"B"})
    qf = bd["rounds"]["QF"]
    assert qf["hits"] == ["A"] and qf["earned"] == 40
    assert qf["alive"] == ["C"]          # not reached, not eliminated
    assert qf["dead"] == ["B"]           # eliminated before reaching QF
    assert qf["potential"] == 80         # A (hit) + C (alive)


def test_anchor_reproduces_reported_totals():
    # The fixed group component + each side's anchor bracket points = the
    # user-reported current scores (453 / 437).
    assert pool.GROUP_BASE["you"] + pool.ANCHOR_BRACKET["you"] == 453
    assert pool.GROUP_BASE["rival"] + pool.ANCHOR_BRACKET["rival"] == 437


def test_head_to_head_cancel_and_levers():
    # Mid-R32-ish state: Brazil & Canada into the R16, nothing eliminated of note.
    reached = {"R32": set(), "R16": {"BRA", "CAN"}, "QF": set(),
               "SF": set(), "FINAL": set(), "WINNER": set()}
    h = pool.head_to_head(reached, third_winner=None, eliminated=set())
    you = {l["team"] for l in h["you_levers"]}
    rival = {l["team"] for l in h["rival_levers"]}
    # the real differentiators
    assert {"BRA", "USA", "NED"} <= you
    assert {"BEL", "GER"} <= rival
    # the shared Final cancels — Argentina & Spain are nobody's "edge"
    assert "ARG" not in you and "ARG" not in rival
    assert "ESP" not in you and "ESP" not in rival
    # Brazil is the biggest lever (QF + SF + 3rd = 160)
    bra = next(l for l in h["you_levers"] if l["team"] == "BRA")
    assert bra["upside"] == 160
    # totals obey the anchor model and the lead is consistent
    assert h["you"]["total"] == pool.GROUP_BASE["you"] + h["you"]["bracket_earned"]
    assert h["lead"] == h["you"]["total"] - h["rival"]["total"]
    assert h["third_undetermined"] is True       # rival's 3rd pick unknown


def test_eliminated_lever_drops_out():
    reached = {k: set() for k in pool.ROUND_ORDER}
    # If Brazil is knocked out, it's no longer a live lever for you.
    h = pool.head_to_head(reached, third_winner=None, eliminated={"BRA"})
    assert "BRA" not in {l["team"] for l in h["you_levers"]}
