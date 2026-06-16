"""Full match-by-match prediction renderer.

Enumerates every group-stage match (grouped by group → matchday) and the Round
of 32, each with its predicted scoreline, expected goals, and 1X2 probabilities.
Pulls straight from the ``predict_all`` output — no new modelling.
"""
from __future__ import annotations

import pandas as pd


def _fmt_match(m: pd.Series) -> str:
    a, b = m["team_a"], m["team_b"]
    score = m.get("predicted_score")
    date = m.get("date", "")
    # A knockout slot whose opponent is a best-third-place team is undecided
    # until the group stage finishes — render it as a pairing, not a fake score.
    if pd.isna(score) or pd.isna(m.get("team_a_xg")):
        return f"- **{a}** vs **{b}** — _opponent set after group stage_  _{date}_"
    xa, xb = m["team_a_xg"], m["team_b_xg"]
    pa, pd_, pb = m["team_a_win_prob"], m["draw_prob"], m["team_b_win_prob"]
    wdl = f" · {pa:.0%}/{pd_:.0%}/{pb:.0%}" if pd.notna(pa) else ""
    return (f"- `{score}`  **{a}** {score} **{b}**  "
            f"(xG {xa:.1f}–{xb:.1f}{wdl})  _{date}_")


def render_match_predictions(predictions: pd.DataFrame) -> str:
    lines: list[str] = ["# World Cup 2026 — Full Match Predictions", ""]
    g = predictions[predictions["stage"] == "group"].copy()

    lines.append("## Group stage — all 72 matches\n")
    for grp in sorted(g["group"].dropna().unique()):
        sub = g[g["group"] == grp].sort_values(["date", "match_id"])
        lines.append(f"### Group {grp}")
        for _, m in sub.iterrows():
            lines.append(_fmt_match(m))
        lines.append("")

    ko = predictions[predictions["stage"] != "group"]
    if not ko.empty:
        lines.append("## Round of 32 (modal qualifiers)\n")
        for _, m in ko.iterrows():
            adv = m.get("team_a_advance_prob")
            extra = f" · adv {adv:.0%}" if pd.notna(adv) else ""
            lines.append(_fmt_match(m) + extra)
        lines.append("")

    lines.append("---")
    lines.append("_Scores are the single most-likely correct score; xG are "
                 "expected goals; percentages are win/draw/loss. Knockout rows "
                 "use the most-likely group qualifiers and reduced knockout "
                 "tempo. See `predictions.csv` for full scoreline distributions "
                 "and key drivers._")
    return "\n".join(lines)
