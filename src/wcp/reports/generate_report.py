"""Investor-style Markdown report generator.

Renders the model output as a sharp, betting-aware brief — power rankings, group
projections, the projected bracket, the value board, and the analytical edge
modules — with an explicit data-quality appendix so the reader knows exactly how
well-supported each number is.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ..pipeline import Results


def _md_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    d = df[cols].head(n) if n else df[cols]
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def generate_report(res: Results) -> str:
    tp = res.sim.team_probs
    cs = res.config_summary
    lines: list[str] = []
    w = lines.append

    fav = tp.iloc[0]
    w(f"# World Cup 2026 — Quantamental Model Brief")
    w(f"*Generated {date.today().isoformat()} · "
      f"{cs['n_sims']:,} Monte-Carlo tournaments · "
      f"ensemble = {', '.join(f'{k} {v:.0%}' for k, v in cs['ensemble_weights'].items())}*")
    w("")
    w("## Executive thesis")
    w(f"- **Model favourite:** {fav['team']} at "
      f"**{fav['p_champion']:.1%}** (fair odds {fav['fair_champion_odds']:.1f}). "
      f"The title race is led by {', '.join(tp['team'].head(3))}.")
    top_dh = res.insights['dark_horse'].iloc[0]
    w(f"- **Sharpest dark horse:** {top_dh['team']} "
      f"(index {top_dh['dark_horse_index']}, semis prob {top_dh['p_semifinal']:.1%}).")
    over = res.insights['overrated']
    if not over.empty:
        worst = over.iloc[0]
        w(f"- **Most overrated vs reputation:** {worst['team']} "
          f"(model {worst['model_champion']:.1%} vs reputation "
          f"{worst['reputation_champion']:.1%}).")
    if not res.value_board.empty and "flag" in res.value_board.columns:
        strong = res.value_board[res.value_board["flag"].isin(["STRONG", "MODERATE"])]
        if not strong.empty:
            b = strong.iloc[0]
            w(f"- **Top market edge:** {b['selection']} to win @ {b['odds']} "
              f"(model {b['model_prob']:.1%} vs market {b['market_prob']:.1%}, "
              f"edge {b['edge']:+.1%}, {b['flag']}).")
        else:
            w("- **Market:** no material edges on the loaded (sample) odds — "
              "board is informational until a live feed is wired.")
    w("")

    w("## Power rankings (composite strength)")
    pr = res.strength.copy()
    pr["rank"] = range(1, len(pr) + 1)
    w(_md_table(pr, ["rank", "team", "group", "composite_100", "squad_rating"], 16))
    w("")

    w("## Power Index — multi-lens ranking")
    w("*Each team's rank under independent methods, blended into a consensus. "
      "High `disagreement` = the methods (and pundits) part ways.*")
    pi = res.power_index
    pi_cols = [c for c in ["team", "blended_rank", "rank_model", "rank_sim",
                           "rank_market", "rank_elo", "rank_form", "rank_external",
                           "disagreement"] if c in pi.columns]
    w(_md_table(pi, pi_cols, 16))
    if "rank_market" in pi.columns:
        # Focus on title-relevant teams (consensus top 16), not longshot noise.
        contenders = pi[pi["blended_rank"] <= 16].copy()
        contenders["gap"] = (contenders["rank_model"] - contenders["rank_market"]).abs()
        div = contenders.sort_values("gap", ascending=False).head(3)
        notes = ", ".join(f"{r['team']} (model #{int(r['rank_model'])} vs "
                          f"market #{int(r['rank_market'])})" for _, r in div.iterrows())
        w(f"\n**Biggest model-vs-market disagreements among contenders:** {notes}.")
    w("")

    w("## Champion & advancement probabilities (top 16)")
    w(_md_table(tp, ["team", "group", "p_r32", "p_qf", "p_sf", "p_final",
                     "p_champion", "fair_champion_odds"], 16))
    w("")

    w("## Group-stage projections")
    gt = res.sim.group_tables
    for g in sorted(gt["group"].unique()):
        sub = gt[gt["group"] == g]
        w(f"**Group {g}**")
        w(_md_table(sub, ["team", "proj_points", "proj_gd", "p_win_group",
                          "p_r32"]))
        w("")

    w("## All 72 group-stage match predictions")
    w("*Predicted most-likely score · expected goals · win/draw/loss. Full "
      "scoreline distributions & drivers in `predictions.csv`; a readable "
      "match-by-match file is written to `match_predictions.md`.*")
    gp = res.predictions[res.predictions["stage"] == "group"].copy()
    gp["score"] = gp["predicted_score"]
    gp["W/D/L"] = gp.apply(
        lambda m: f"{m['team_a_win_prob']:.0%}/{m['draw_prob']:.0%}/"
                  f"{m['team_b_win_prob']:.0%}", axis=1)
    gp["xG"] = gp.apply(lambda m: f"{m['team_a_xg']:.1f}-{m['team_b_xg']:.1f}", axis=1)
    w(_md_table(gp.sort_values(["group", "match_id"]),
                ["match_id", "group", "team_a", "team_b", "score", "xG", "W/D/L"]))
    w("")

    w("## Projected bracket (modal group qualifiers)")
    modal = res.sim.bracket_modal
    for g in sorted({k[0] for k in modal}):
        w(f"- Group {g}: 🥇 {modal[g+'1']} · 🥈 {modal[g+'2']}")
    w("")

    w("## Value board — tournament winner")
    if res.value_board.empty or "flag" not in res.value_board.columns:
        w("*No odds market loaded.*")
    else:
        w(_md_table(res.value_board, ["selection", "odds", "model_prob",
                    "market_prob", "edge", "edge_adj", "kelly_stake", "flag"], 12))
    w("")

    w("## Upset radar (group stage)")
    ur = res.insights["upset_radar"]
    w("*No high-variance underdog spots flagged.*" if ur.empty
      else _md_table(ur, ["favourite", "underdog", "underdog_win_prob",
                          "predicted_score", "rank_gap"], 10))
    w("")

    w("## Dark-horse index")
    w(_md_table(res.insights["dark_horse"], ["team", "group", "p_semifinal",
                "value_vs_reputation", "dark_horse_index"], 8))
    w("")

    w("## Overrated / underrated vs reputation")
    w(_md_table(res.insights["overrated"], ["team", "model_champion",
                "reputation_champion", "gap", "verdict"], 8))
    w("")

    w("## Path difficulty (hardest projected routes)")
    w(_md_table(res.insights["path_difficulty"], ["team", "group",
                "qualify_to_QF_conversion", "path_difficulty_score"], 8))
    w("")

    w("## Data-quality & model notes")
    dq = res.data_quality["data_quality"].value_counts().to_dict()
    w(f"- Per-team data quality: {dq}")
    w(f"- Data sources: {cs['sources']}")
    w(f"- Ensemble components: {cs['ensemble_components']}")
    if cs["validation_warnings"]:
        w(f"- Validation warnings: {cs['validation_warnings']}")
    w("- Odds shipped as **sample** data; wire a live feed for actionable value flags.")
    w("- Probabilities are simulation estimates with Monte-Carlo noise; "
      "no confidence is hardcoded. Re-run with more sims to tighten bands.")
    w("")
    return "\n".join(lines)
