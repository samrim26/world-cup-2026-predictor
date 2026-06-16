import numpy as np
import pytest

from wcp.features.context import build_match_context
from wcp.ingest.dataset import load_dataset
from wcp.model.ensemble import build_team_ratings
from wcp.model.simulator import TournamentSimulator
from wcp.model.team_strength import compute_team_strength


@pytest.fixture(scope="module")
def sim_result():
    ds = load_dataset()
    s = compute_team_strength(ds.teams, ds.history)
    tr = build_team_ratings(s, ds.history, ds.teams, use_pymc=False, use_ml=True)
    ctx = build_match_context(ds.fixtures, s, ds.venues)
    sim = TournamentSimulator(tr.table, s, ds.teams, ctx, seed=7)
    return sim.run(4000)


def test_champion_probs_sum_to_one(sim_result):
    assert sim_result.team_probs["p_champion"].sum() == pytest.approx(1.0, abs=1e-6)


def test_exactly_32_qualify(sim_result):
    assert sim_result.team_probs["p_r32"].sum() == pytest.approx(32.0, abs=0.05)


def test_twelve_group_winners(sim_result):
    assert sim_result.team_probs["p_win_group"].sum() == pytest.approx(12.0, abs=0.05)


def test_round_probabilities_monotone(sim_result):
    tp = sim_result.team_probs
    assert (tp["p_r32"] >= tp["p_qf"] - 1e-9).all()
    assert (tp["p_qf"] >= tp["p_sf"] - 1e-9).all()
    assert (tp["p_sf"] >= tp["p_final"] - 1e-9).all()
    assert (tp["p_final"] >= tp["p_champion"] - 1e-9).all()


def test_all_probabilities_in_range(sim_result):
    tp = sim_result.team_probs
    for col in ["p_r32", "p_qf", "p_sf", "p_final", "p_champion",
                "p_win_group", "p_runner_up", "p_third"]:
        assert tp[col].between(0.0, 1.0).all()


def test_finalists_and_semifinalists_counts(sim_result):
    tp = sim_result.team_probs
    assert tp["p_final"].sum() == pytest.approx(2.0, abs=0.05)
    assert tp["p_sf"].sum() == pytest.approx(4.0, abs=0.05)


def test_reproducible_with_seed():
    ds = load_dataset()
    s = compute_team_strength(ds.teams, ds.history)
    tr = build_team_ratings(s, ds.history, ds.teams, use_pymc=False, use_ml=True)
    ctx = build_match_context(ds.fixtures, s, ds.venues)
    a = TournamentSimulator(tr.table, s, ds.teams, ctx, seed=99).run(2000)
    b = TournamentSimulator(tr.table, s, ds.teams, ctx, seed=99).run(2000)
    np.testing.assert_allclose(a.team_probs.set_index("team")["p_champion"],
                               b.team_probs.set_index("team")["p_champion"])


def test_knockout_never_draws():
    """A knockout round must always return one of the two teams as winner."""
    ds = load_dataset()
    s = compute_team_strength(ds.teams, ds.history)
    tr = build_team_ratings(s, ds.history, ds.teams, use_pymc=False, use_ml=True)
    ctx = build_match_context(ds.fixtures, s, ds.venues)
    sim = TournamentSimulator(tr.table, s, ds.teams, ctx, seed=3)
    att, deff = sim._eff_ratings(5000)
    # Force evenly-matched teams (same index) to stress draws -> shootouts.
    a = np.zeros(5000, dtype=int)
    b = np.ones(5000, dtype=int)
    winners = sim._knockout_round(a, b, att, deff)
    assert set(np.unique(winners)).issubset({0, 1})
    # Both outcomes occur (no silent draw collapse).
    assert winners.min() == 0 and winners.max() == 1
