import numpy as np
import pytest

from wcp.model.expected_goals import match_xg
from wcp.model.score_model import analyse_match, sample_scores, score_matrix


def test_score_matrix_sums_to_one():
    for la, lb in [(1.2, 1.0), (2.5, 0.4), (0.3, 0.3), (3.1, 2.7)]:
        mat = score_matrix(la, lb)
        assert mat.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(mat >= 0)


def test_wdl_probabilities_sum_to_one():
    res = analyse_match(1.6, 1.1, "A", "B")
    assert res.p_a + res.p_draw + res.p_b == pytest.approx(1.0, abs=1e-6)


def test_stronger_team_favoured():
    res = analyse_match(2.4, 0.6, "Strong", "Weak")
    assert res.p_a > res.p_b
    assert res.exp_a > res.exp_b


def test_expected_goals_match_lambda():
    # Expected goals from the matrix should track the input rates closely.
    res = analyse_match(1.8, 1.2)
    assert res.exp_a == pytest.approx(1.8, abs=0.15)
    assert res.exp_b == pytest.approx(1.2, abs=0.15)


def test_top_scorelines_sorted_and_normalised():
    res = analyse_match(1.5, 1.0)
    probs = [p for _, p in res.top_scorelines]
    assert probs == sorted(probs, reverse=True)
    assert all(0 <= p <= 1 for p in probs)


def test_match_xg_monotonic_in_strength():
    la_weak, _ = match_xg(0.0, 0.0, 0.0, 0.0)
    la_strong, _ = match_xg(2.0, 0.0, 0.0, 0.0)
    assert float(la_strong) > float(la_weak)


def test_sample_scores_shapes_and_nonneg():
    rng = np.random.default_rng(0)
    la = np.full(1000, 1.5)
    lb = np.full(1000, 1.0)
    ga, gb = sample_scores(la, lb, rng)
    assert ga.shape == (1000,) and gb.shape == (1000,)
    assert ga.min() >= 0 and gb.min() >= 0
    # Sample mean near lambda.
    assert ga.mean() == pytest.approx(1.5, abs=0.15)
