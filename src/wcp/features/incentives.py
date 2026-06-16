"""Matchday-3 group incentive adjustments.

A coarse, transparent model of end-of-group motivation: a side that is already
safely through may ease off; a side that must win pushes harder. Applied as a
small symmetric tempo nudge during simulation, given the live (simulated) table
state before the final round.
"""
from __future__ import annotations


def md3_tempo_delta(pts_for: int, pts_against: int, max_other: int,
                    already_top2: bool, eliminated: bool) -> float:
    """Return a log-lambda tempo delta for a team in its MD3 match.

    Positive -> more aggressive (needs goals/result); negative -> rotation/ease.
    Inputs are this team's current points, the opponent's, and the best points
    among the *other two* group teams, plus simple qualification flags.
    """
    if eliminated:
        return -0.06          # nothing to play for, often rotates
    if already_top2 and pts_for - max_other >= 3:
        return -0.04          # comfortably through, manage minutes
    if pts_for < max_other:
        return 0.05           # must chase the result
    return 0.0
