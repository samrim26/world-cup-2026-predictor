import itertools

import pytest

from wcp.model import bracket


def test_bracket_has_16_matches():
    assert len(bracket.R32_MATCHES) == 16


def test_each_qualifier_used_once():
    """All 12 winners, 12 runners-up and 8 third-slots appear exactly once."""
    winners, runners, thirds = [], [], []
    for a, b in bracket.R32_MATCHES:
        for kind, key in (a, b):
            {"W": winners, "R": runners, "3": thirds}[kind].append(key)
    assert sorted(winners) == list("ABCDEFGHIJKL")
    assert sorted(runners) == list("ABCDEFGHIJKL")
    assert sorted(thirds) == list(range(bracket.N_THIRD_SLOTS))


def test_no_same_group_winner_runner_meetings():
    for a, b in bracket.R32_MATCHES:
        if a[0] in ("W", "R") and b[0] in ("W", "R"):
            assert a[1] != b[1], f"same-group R32 meeting: {a} vs {b}"


def test_assign_thirds_is_valid_for_all_combinations():
    """For every C(12,8) selection of qualifying thirds the mapping must be a
    bijection that never sends a third into a forbidden (same-group) slot."""
    groups = list("ABCDEFGHIJKL")
    for combo in itertools.combinations(groups, 8):
        mapping = bracket.assign_thirds(combo)
        assert len(mapping) == bracket.N_THIRD_SLOTS
        # Bijection: each qualifying group used exactly once.
        assert sorted(mapping.values()) == sorted(combo)
        # No forbidden assignment.
        for slot, g in mapping.items():
            assert g != bracket.THIRD_SLOT_FORBIDDEN[slot]


def test_assign_thirds_is_deterministic():
    combo = tuple("ABCDEFGH")
    assert bracket.assign_thirds(combo) == bracket.assign_thirds(combo)


def test_assign_thirds_rejects_wrong_count():
    with pytest.raises(ValueError):
        bracket.assign_thirds(tuple("ABC"))
