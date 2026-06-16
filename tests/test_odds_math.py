import numpy as np
import pytest

from wcp.utils import odds_math as om


def test_decimal_passthrough():
    assert om.to_decimal(2.5, "decimal") == 2.5


def test_american_conversion():
    assert om.to_decimal(100, "american") == pytest.approx(2.0)
    assert om.to_decimal(-200, "american") == pytest.approx(1.5)
    assert om.to_decimal(150, "american") == pytest.approx(2.5)


def test_fractional_conversion():
    # 6/4 = 1.5 ratio -> decimal 2.5
    assert om.to_decimal(1.5, "fractional") == pytest.approx(2.5)


def test_implied_prob_inverse():
    assert om.implied_prob(2.0) == pytest.approx(0.5)
    assert om.implied_prob(-110, "american") == pytest.approx(110 / 210, abs=1e-6)


def test_devig_proportional_sums_to_one():
    probs = [0.55, 0.30, 0.25]  # overround book
    fair = om.devig_proportional(probs)
    assert fair.sum() == pytest.approx(1.0)
    assert np.all(fair > 0)


def test_devig_shin_sums_to_one():
    probs = [0.55, 0.30, 0.25]
    fair = om.devig_shin(probs)
    assert fair.sum() == pytest.approx(1.0, abs=1e-6)


def test_fair_probs_from_odds():
    fair = om.fair_probs([2.1, 3.5, 4.0], "decimal")
    assert fair.sum() == pytest.approx(1.0)


def test_kelly_positive_when_edge():
    # prob 0.6 at decimal 2.0 (b=1): f = (0.6*1 - 0.4)/1 = 0.2
    assert om.kelly_fraction(0.6, 2.0) == pytest.approx(0.2)


def test_kelly_zero_when_no_edge():
    assert om.kelly_fraction(0.4, 2.0) <= 0


def test_invalid_decimal_raises():
    with pytest.raises(ValueError):
        om.to_decimal(0.9, "decimal")
