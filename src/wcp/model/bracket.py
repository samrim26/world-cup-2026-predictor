"""Knockout bracket structure + the best-third-place combination logic.

The Round of 32 takes 12 group winners, 12 runners-up and the 8 best third-
placed teams. The bracket below is a *balanced* valid 2026-format bracket:
each half of the draw receives 6 winners, 6 runners-up and 4 third-placed teams,
no team meets a side from its own group in the Round of 32, and a group's winner
and runner-up are placed on opposite halves. (It is not claimed to reproduce
FIFA's exact published slot letters; swap ``R32_MATCHES`` to match an official
sheet if desired — everything downstream is agnostic to the specific letters.)

The eight third-placed teams are slotted via ``assign_thirds``, a pure, memoised
function of *which* groups' thirds qualify — the same property FIFA's official
combination table has — guaranteeing a valid, reproducible draw every sim.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

GROUPS = list("ABCDEFGHIJKL")

# Eight third-place slots (index 0..7). Each cannot be filled by a third-placed
# team from the group that already occupies the *other* side of that R32 match.
THIRD_SLOT_FORBIDDEN = ["C", "G", "K", "D", "D", "H", "L", "C"]

# --- Round of 32: 16 balanced matches. Slot specs:
#   ("W", g) winner of group g | ("R", g) runner-up | ("3", k) third-slot k.
R32_MATCHES: list[tuple[tuple, tuple]] = [
    (("W", "A"), ("R", "B")),     # half 1
    (("W", "C"), ("3", 0)),
    (("W", "E"), ("R", "F")),
    (("W", "G"), ("3", 1)),
    (("W", "I"), ("R", "J")),
    (("W", "K"), ("3", 2)),
    (("R", "D"), ("3", 3)),
    (("R", "H"), ("R", "L")),
    (("W", "B"), ("R", "A")),     # half 2
    (("W", "D"), ("3", 4)),
    (("W", "F"), ("R", "E")),
    (("W", "H"), ("3", 5)),
    (("W", "J"), ("R", "I")),
    (("W", "L"), ("3", 6)),
    (("R", "C"), ("3", 7)),
    (("R", "G"), ("R", "K")),
]
assert len(R32_MATCHES) == 16
N_THIRD_SLOTS = len(THIRD_SLOT_FORBIDDEN)


def _round_pairs(n_matches: int) -> list[tuple[int, int]]:
    """Adjacent pairings feeding the next round: (0,1),(2,3),..."""
    return [(i, i + 1) for i in range(0, n_matches, 2)]


R16_PAIRS = _round_pairs(16)
QF_PAIRS = _round_pairs(8)
SF_PAIRS = _round_pairs(4)
FINAL_PAIR = _round_pairs(2)


@lru_cache(maxsize=512)
def assign_thirds(qualifying_groups: tuple[str, ...]) -> dict[int, str]:
    """Map each third-slot index -> the qualifying group filling it.

    Pure function of the (sorted) set of 8 qualifying third-place groups. No
    third-placed team is sent to a slot whose match already contains its own
    group; deterministic, so identical inputs always yield identical brackets.
    """
    groups = sorted(qualifying_groups)
    if len(groups) != N_THIRD_SLOTS:
        raise ValueError(f"Expected {N_THIRD_SLOTS} qualifying thirds, "
                         f"got {len(groups)}")

    cost = np.zeros((N_THIRD_SLOTS, N_THIRD_SLOTS))
    for si, forbid in enumerate(THIRD_SLOT_FORBIDDEN):
        for gi, g in enumerate(groups):
            cost[si, gi] = 1e6 if g == forbid else abs(si - (ord(g) - ord("A")))
    rows, cols = linear_sum_assignment(cost)
    mapping = {int(si): groups[gi] for si, gi in zip(rows, cols)}
    for si, g in mapping.items():
        if g == THIRD_SLOT_FORBIDDEN[si]:
            raise RuntimeError("Infeasible third-place assignment")
    return mapping


def describe_bracket() -> list[str]:
    """Human-readable R32 pairing description (slot level)."""
    def fmt(s: tuple) -> str:
        kind, key = s
        return {"W": f"Winner {key}", "R": f"Runner-up {key}",
                "3": f"3rd-place slot {key}"}[kind]
    return [f"R32-{i+1:02d}: {fmt(a)} vs {fmt(b)}"
            for i, (a, b) in enumerate(R32_MATCHES)]
