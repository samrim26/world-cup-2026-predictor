"""Style-vs-style tactical matchup adjustments and human-readable notes.

The adjustments are small log-lambda nudges capturing well-known tactical
interactions (e.g. a high-press side disrupting a possession side; an
aerially strong side exploiting a weak set-piece defence).
"""
from __future__ import annotations

import pandas as pd

# Each rule: (attacker_facet, defender_facet, sign, scale, note_template)
# A positive sign means attacker's facet advantage over defender's facet *adds*
# expected goals for the attacker.
_RULES = [
    ("transition_rating", "pressing_rating", 1.0, 0.06,
     "{a} transition threat vs {b}'s high line"),
    ("aerial_rating", "set_piece_rating", 1.0, 0.05,
     "{a} aerial/set-piece edge over {b}"),
    ("pressing_rating", "possession_rating", 1.0, 0.05,
     "{a} press disrupts {b}'s build-up"),
    ("possession_rating", "pressing_rating", 1.0, 0.04,
     "{a} controls tempo against {b}'s mid-block"),
]


def matchup_adjustment(att: pd.Series, deff: pd.Series) -> tuple[float, list[str]]:
    """Return (log-lambda delta for the attacking team, list of notes).

    ``att`` and ``deff`` are decomposition rows (z-scored facet ratings) for the
    attacking and defending teams respectively.
    """
    delta = 0.0
    notes: list[str] = []
    for af, df, sign, scale, tmpl in _RULES:
        edge = float(att.get(af, 0.0)) - float(deff.get(df, 0.0))
        delta += sign * scale * edge
        if edge > 0.8:  # only surface clear, material mismatches
            notes.append(tmpl.format(a=att["team"], b=deff["team"]))
    return delta, notes
