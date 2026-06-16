import pandas as pd

from wcp.model.value_betting import build_value_board


def _odds(market="tournament_winner"):
    return pd.DataFrame([
        {"market": market, "selection": "A", "odds": 5.0, "format": "decimal",
         "source": "t", "timestamp": "2026"},
        {"market": market, "selection": "B", "odds": 3.0, "format": "decimal",
         "source": "t", "timestamp": "2026"},
        {"market": market, "selection": "C", "odds": 4.0, "format": "decimal",
         "source": "t", "timestamp": "2026"},
    ])


def test_value_board_flags_strong_edge():
    # Model loves A far more than the (de-vigged) market implies.
    model = {"A": 0.45, "B": 0.30, "C": 0.20}
    board = build_value_board(model, _odds(),
                              confidence={"A": 0.9, "B": 0.9, "C": 0.9},
                              data_quality={"A": 0.9, "B": 0.9, "C": 0.9})
    row_a = board[board["selection"] == "A"].iloc[0]
    assert row_a["edge"] > 0
    assert row_a["flag"] in ("STRONG", "MODERATE")
    assert row_a["kelly_stake"] > 0


def test_no_bet_when_model_matches_market():
    model = {"A": 0.20, "B": 0.333, "C": 0.25}  # ~ market
    board = build_value_board(model, _odds())
    assert (board["flag"].isin(["NO_BET", "THIN", "MARKET_AGAINST"])).any()


def test_kelly_zero_on_negative_edge():
    model = {"A": 0.05, "B": 0.05, "C": 0.05}
    board = build_value_board(model, _odds())
    assert (board["kelly_stake"] == 0).all()


def test_empty_when_market_absent():
    board = build_value_board({"A": 0.5}, _odds(), market="nonexistent_market")
    assert board.empty or "note" in board.columns


def test_market_probs_sum_to_one_after_devig():
    model = {"A": 0.33, "B": 0.34, "C": 0.33}
    board = build_value_board(model, _odds())
    # market_prob is stored rounded to 4 dp; tolerance reflects that rounding.
    assert abs(board["market_prob"].sum() - 1.0) < 1e-3
