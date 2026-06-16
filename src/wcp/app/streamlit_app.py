"""Streamlit dashboard for the World Cup 2026 quant model.

Run:  streamlit run src/wcp/app/streamlit_app.py

Exposes power rankings, champion/advancement probabilities, group tables, the
value board, the analytical edge modules, and an interactive Scenario Mode
(injure a star / boost a host / custom strength deltas) that re-simulates live.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `wcp` importable when run via `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wcp.features.context import build_match_context          # noqa: E402
from wcp.ingest.dataset import load_dataset                   # noqa: E402
from wcp.model.ensemble import build_team_ratings             # noqa: E402
from wcp.model.predictor import predict_all                   # noqa: E402
from wcp.model.simulator import TournamentSimulator           # noqa: E402
from wcp.model.team_strength import compute_team_strength     # noqa: E402
from wcp.model.power_index import build_power_index           # noqa: E402
from wcp.model import value_betting                           # noqa: E402
from wcp.reports import insights                              # noqa: E402

st.set_page_config(page_title="WC2026 Quant Model", layout="wide")


@st.cache_data(show_spinner=False)
def _base():
    ds = load_dataset()
    strength = compute_team_strength(ds.teams, ds.history)
    ratings = build_team_ratings(strength, ds.history, ds.teams, use_ml=True)
    context = build_match_context(ds.fixtures, strength, ds.venues)
    return ds, strength, ratings, context


def _simulate(ratings_table, strength, teams, context, n_sims, seed):
    sim = TournamentSimulator(ratings_table, strength, teams, context, seed=seed)
    return sim.run(n_sims)


def main():
    st.title("⚽ World Cup 2026 — Quantamental Model")
    st.caption("Dixon-Coles + Bayesian + ML ensemble · Monte-Carlo tournament "
               "simulation · value/ROI engine")

    ds, strength, ratings, context = _base()
    teams_list = sorted(ds.teams["team"])

    with st.sidebar:
        st.header("Controls")
        n_sims = st.select_slider("Simulations",
                                  options=[2000, 10000, 30000, 100000],
                                  value=10000)
        seed = st.number_input("Seed", value=42, step=1)
        st.subheader("Scenario mode")
        injure = st.multiselect("Key player(s) out", teams_list)
        boost = st.multiselect("Crowd/home boost", teams_list)

    rt = ratings.table.copy()
    notes = []
    for t in injure:
        m = rt["team"] == t
        rt.loc[m, "attack_z"] -= 0.4
        rt.loc[m, "defense_z"] -= 0.24
        notes.append(f"−{t}")
    for t in boost:
        m = rt["team"] == t
        rt.loc[m, "attack_z"] += 0.25
        rt.loc[m, "defense_z"] += 0.15
        notes.append(f"+{t}")
    if notes:
        st.info("Scenario applied: " + ", ".join(notes))

    sim = _simulate(rt, strength, ds.teams, context, int(n_sims), int(seed))
    tp = sim.team_probs

    tab_titles = ["🏆 Title race", "🧭 Power Index", "📊 Power rankings",
                  "🅰️ Groups", "🎯 Predictions", "💰 Value board", "🔮 Edge modules"]
    t1, tpi, t2, t3, t4, t5, t6 = st.tabs(tab_titles)

    with t1:
        st.subheader("Champion & advancement probabilities")
        show = tp[["team", "group", "p_r32", "p_qf", "p_sf", "p_final",
                   "p_champion", "fair_champion_odds"]].head(24)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.bar_chart(tp.head(12).set_index("team")["p_champion"])

    with tpi:
        st.subheader("Multi-lens Power Index")
        st.caption("Each team's rank under independent methods — model, sim, "
                   "market, Elo, recent form — blended into a consensus. High "
                   "`disagreement` = the methods (and pundits) part ways.")
        pi = build_power_index(strength, ds.teams, tp, ds.history, ds.odds)
        st.dataframe(pi.head(24), use_container_width=True, hide_index=True)
        if "rank_market" in pi.columns:
            st.caption("Tip: big model-vs-market gaps are where the edge (or the "
                       "trap) lives — e.g. teams your model rates far above the book.")

    with t2:
        pr = strength.copy()
        pr.insert(0, "rank", range(1, len(pr) + 1))
        st.dataframe(pr[["rank", "team", "group", "composite_100", "squad_rating",
                         "attack_rating", "defense_rating", "gk_rating"]],
                     use_container_width=True, hide_index=True)

    with t3:
        for g in sorted(sim.group_tables["group"].unique()):
            st.markdown(f"**Group {g}**")
            sub = sim.group_tables[sim.group_tables["group"] == g]
            st.dataframe(sub[["team", "proj_points", "proj_gd", "p_win_group",
                              "p_r32"]], use_container_width=True, hide_index=True)

    with t4:
        dq = dict(zip(ds.teams["team"], [0.6] * len(ds.teams)))
        preds = predict_all(ds, strength, rt, context, sim, dq)
        stage = st.selectbox("Stage", ["group", "R32"])
        cols = ["match_id", "team_a", "team_b", "predicted_score",
                "team_a_win_prob", "draw_prob", "team_b_win_prob",
                "confidence", "key_model_drivers"]
        cols = [c for c in cols if c in preds.columns]
        st.dataframe(preds[preds["stage"] == stage][cols],
                     use_container_width=True, hide_index=True)

    with t5:
        model_champ = dict(zip(tp["team"], tp["p_champion"]))
        board = value_betting.build_value_board(model_champ, ds.odds)
        if board.empty or "flag" not in board.columns:
            st.warning("No odds market loaded.")
        else:
            st.dataframe(board, use_container_width=True, hide_index=True)
            st.caption("Odds are SAMPLE data — wire a live feed for actionable flags.")

    with t6:
        st.markdown("**Dark-horse index**")
        st.dataframe(insights.dark_horse_index(tp, ds.teams, ds.odds).head(10),
                     use_container_width=True, hide_index=True)
        st.markdown("**Overrated / underrated vs reputation**")
        st.dataframe(insights.overrated_underrated(tp, ds.teams, ds.odds).head(10),
                     use_container_width=True, hide_index=True)
        st.markdown("**Path difficulty**")
        st.dataframe(insights.path_difficulty(tp, strength).head(10),
                     use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
