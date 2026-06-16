"""Streamlit live web dashboard for the 2026 World Cup.

Run:  streamlit run src/wcp/live/live_app.py
Auto-refreshes the live data every 30s; shows standings, the third-place
wildcard race, and predicted-vs-actual scoring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wcp.live.feed import ESPNFeed                       # noqa: E402
from wcp.live import compare                             # noqa: E402
from wcp.live.bracket_tracker import bracket_status, goal_feed  # noqa: E402
from wcp.live.live_bracket import build as build_bracket, by_round  # noqa: E402
from wcp.live.tracker import (qualification_flags,       # noqa: E402
                              third_place_race, tournament_complete_groups)

st.set_page_config(page_title="WC2026 Live", layout="wide")
FEED = ESPNFeed()


def _style_wildcard(row):
    color = "#1b5e20" if row["Status"] == "IN" else "#7f1d1d"
    return [f"background-color: {color}; color: white"] * len(row)


@st.fragment(run_every="30s")
def live_view():
    from wcp.live.user_picks import attach_predictions
    groups = FEED.standings()
    today = attach_predictions(FEED.today())
    flagged = qualification_flags(groups)
    thirds = third_place_race(groups)
    score = compare.score_predictions(FEED)
    qual = compare.qualifier_accuracy(groups)

    from datetime import datetime
    st.caption(f"Updated {datetime.now():%H:%M:%S} · source: {FEED.source} · "
               f"groups completed {tournament_complete_groups(groups)}/12 · "
               f"auto-refresh 30s")

    # Today
    st.subheader("Today")
    if today:
        cols = st.columns(min(len(today), 4))
        for i, m in enumerate(today):
            tag = ("🔴 LIVE " + (m.get("clock") or "") if m["state"] == "in"
                   else "FT" if m["state"] == "post" else m.get("detail", "—"))
            label = f"{m['home']} v {m['away']}"
            if m.get("predicted"):
                label += f"  ·  pred {m['predicted']}"
            cols[i % len(cols)].metric(
                label, f"{m['home_score']}-{m['away_score']}", tag)
    else:
        st.write("No matches scheduled today.")

    # Goal feed
    goals = goal_feed(today)
    if goals:
        st.subheader("⚽ Goals (latest first)")
        st.dataframe(pd.DataFrame([{
            "Min": g["clock"], "Team": g["team"], "Scorer": g["scorer"],
            "Type": g["type"], "Match": g["match"],
            "Live": "🔴" if g["live"] else ""} for g in goals]),
            hide_index=True, use_container_width=True)

    # Bracket survival
    bracket = bracket_status(FEED, groups)
    st.subheader("🗺️ Your bracket — survival")
    champ = bracket["champion"]
    st.caption(f"Champion pick: **{champ}** — "
               f"{'🟢 ALIVE' if bracket['champion_alive'] else '🔴 OUT'}")
    bcols = st.columns(len(bracket["tiers"]))
    for col, tier in zip(bcols, bracket["tiers"]):
        col.metric(tier["label"], f"{tier['alive']}/{tier['total']}", "alive")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("🟡 Third-place wildcard race")
        st.caption("Top 8 advance to the Round of 32")
        tdf = pd.DataFrame([{"#": t["wildcard_rank"], "Team": t["team"],
                             "Grp": t["group"], "P": t["P"], "Pts": t["Pts"],
                             "GD": t["GD"], "GF": t["GF"],
                             "Status": t["wildcard_status"]} for t in thirds])
        st.dataframe(tdf.style.apply(_style_wildcard, axis=1),
                     hide_index=True, use_container_width=True)
    with c2:
        st.subheader("🎯 Predicted vs actual")
        m1, m2, m3 = st.columns(3)
        m1.metric("Pool points", score["total"], f"of {score['max_possible']}")
        m2.metric("Exact scores", score["exact"])
        m3.metric("Correct results", score["correct_outcome"])
        q_both = sum(1 for q in qual if q["match"] == "2/2")
        st.caption(f"Group qualifiers: **{q_both}/12** groups with both top-2 correct")
        if score["rows"]:
            st.dataframe(pd.DataFrame(score["rows"])[
                ["date", "match", "predicted", "actual", "points"]],
                hide_index=True, use_container_width=True, height=260)

    # Live bracket (official 2026 template, filled from current standings)
    st.subheader("🏆 Live bracket")
    st.caption("Official 2026 template · `?` = provisional (group not final) · "
               "`3rd[..]` = wildcard slot (locks when all groups finish) · "
               "advances automatically as knockout results come in")
    brk = by_round(build_bracket(FEED, groups))
    order = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"]
    bcols = st.columns(len(order))
    for col, rnd in zip(bcols, order):
        with col:
            st.markdown(f"**{rnd}**")
            for info in brk.get(rnd, []):
                w = info["winner"]
                def tag(team, lbl):
                    if w and team == w:
                        return f"**:green[{lbl}]**"
                    return lbl if info["ready"] else f":gray[{lbl}]"
                st.markdown(f"<small>{tag(info['a_team'], info['a'])} v "
                            f"{tag(info['b_team'], info['b'])}</small>",
                            unsafe_allow_html=True)

    st.subheader("Group standings")
    cols = st.columns(4)
    for i, (g, rows) in enumerate(flagged.items()):
        with cols[i % 4]:
            st.markdown(f"**Group {g}**")
            df = pd.DataFrame([{"Team": t["abbr"], "P": t["P"],
                                "W-D-L": f"{t['W']}-{t['D']}-{t['L']}",
                                "GD": t["GD"], "Pts": t["Pts"]} for t in rows])
            st.dataframe(df, hide_index=True, use_container_width=True)


st.title("⚽ World Cup 2026 — Live Tracker")
live_view()
