"""Tests for the Kalshi edge tracker (no network — results injected)."""
from wcp.markets import edge_tracker as et
from wcp.markets.r32_markets import MARKETS


def test_model_p_symmetry_and_direction():
    # complementary, and the stronger team is favoured
    assert abs(et.model_p("FRA", "SWE") + et.model_p("SWE", "FRA") - 1) < 1e-9
    assert et.model_p("FRA", "SWE") > 0.5
    # host bump helps the host
    assert et.model_p("USA", "BIH") > et.model_p("BIH", "USA")


def test_analyze_flags_value_side():
    a = et.analyze({"t1": "BEL", "m1": 1.62, "t2": "SEN", "m2": 2.34, "vol": 1})
    assert abs(a["p1"] + a["p2"] - 1) < 1e-9
    # model has Belgium ~68% vs implied ~62% -> Belgium is the +EV value side
    assert a["value_side"] == "BEL" and a["value_ev"] > 0


def test_grade_pnl_and_brier():
    markets = [{"t1": "BEL", "m1": 1.62, "t2": "SEN", "m2": 2.34, "vol": 1}]
    # Belgium (the value side) advances -> winning bet pays mult-1
    g = et.grade(markets=markets, results={frozenset(("BEL", "SEN")): "BEL"})
    assert g["wins"] == 1 and g["losses"] == 0
    assert abs(g["pnl"] - (1.62 - 1)) < 1e-9
    assert g["staked"] == 1 and g["graded"] == 1
    assert g["model_brier"] is not None and g["market_brier"] is not None
    # ... and a losing grade
    g2 = et.grade(markets=markets, results={frozenset(("BEL", "SEN")): "SEN"})
    assert g2["losses"] == 1 and g2["pnl"] == -1


def test_pending_games_are_not_graded():
    g = et.grade(markets=list(MARKETS), results={})    # nothing settled
    assert g["graded"] == 0 and g["staked"] == 0 and g["model_brier"] is None
