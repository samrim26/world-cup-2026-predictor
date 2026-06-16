import pandas as pd

from wcp.model.power_index import build_power_index


def _inputs():
    teams = ["A", "B", "C", "D", "E", "F"]
    strength = pd.DataFrame({"team": teams,
                             "composite": [3.0, 2.0, 1.0, 0.0, -1.0, -2.0]})
    teams_df = pd.DataFrame({"team": teams,
                             "elo": [2000, 1900, 1800, 1700, 1600, 1500],
                             "fifa_rank": [1, 2, 3, 4, 5, 6]})
    sim = pd.DataFrame({"team": teams,
                        "p_champion": [0.4, 0.25, 0.15, 0.1, 0.07, 0.03]})
    odds = pd.DataFrame([
        {"market": "tournament_winner", "selection": t, "odds": o,
         "format": "decimal", "source": "x", "timestamp": "2026"}
        for t, o in zip(teams, [2.5, 4.0, 6.0, 9.0, 15.0, 30.0])
    ])
    empty_history = pd.DataFrame(columns=["date", "home_team", "away_team",
                                          "home_score", "away_score"])
    return strength, teams_df, sim, empty_history, odds


def test_blended_rank_is_permutation():
    strength, teams_df, sim, hist, odds = _inputs()
    pi = build_power_index(strength, teams_df, sim, hist, odds)
    assert sorted(pi["blended_rank"]) == list(range(1, len(teams_df) + 1))


def test_all_lens_ranks_present_and_valid():
    strength, teams_df, sim, hist, odds = _inputs()
    pi = build_power_index(strength, teams_df, sim, hist, odds)
    for lens in ["rank_model", "rank_sim", "rank_market", "rank_elo", "rank_form"]:
        assert lens in pi.columns
        assert sorted(pi[lens]) == list(range(1, len(teams_df) + 1))


def test_disagreement_non_negative():
    strength, teams_df, sim, hist, odds = _inputs()
    pi = build_power_index(strength, teams_df, sim, hist, odds)
    assert (pi["disagreement"] >= 0).all()


def test_aligned_methods_put_best_team_on_top():
    # All lenses agree A is best -> A should be blended #1.
    strength, teams_df, sim, hist, odds = _inputs()
    pi = build_power_index(strength, teams_df, sim, hist, odds)
    assert pi.iloc[0]["team"] == "A"


def test_works_without_odds():
    strength, teams_df, sim, hist, _ = _inputs()
    pi = build_power_index(strength, teams_df, sim, hist, odds=None)
    # Market lens absent but the index still ranks everyone.
    assert sorted(pi["blended_rank"]) == list(range(1, len(teams_df) + 1))
