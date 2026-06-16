"""Command-line interface.

    python -m wcp ingest                 # (re)build + validate seed data
    python -m wcp rate                   # power rankings
    python -m wcp simulate --sims 100000 # full sim, write CSV outputs
    python -m wcp value                  # value board
    python -m wcp report                 # everything + investor report.md
    python -m wcp all --sims 200000      # alias for report

Scenario flags (apply before simulating):
    --injure TEAM        drop a team's strength (key player unavailable)
    --boost TEAM         add a crowd/home boost to a team
    --scenario "T:da:dd" custom attack/defense z deltas (repeatable)
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from ..pipeline import Results, run_pipeline
from ..reports.generate_report import generate_report
from ..utils.config import load_config
from ..utils.io import write_table
from ..utils.logging import get_logger

log = get_logger("cli")


# --------------------------------------------------------------------------- #
def _apply_scenarios(res_ratings, strength, injure, boost, scenario):
    """Mutate the ratings table in place for scenario analysis."""
    cfg = load_config()
    tbl = res_ratings.table
    notes = []
    star = abs(cfg.get("adjustments.injury_star_out", 0.08)) * 5  # ~ z penalty
    for t in injure or []:
        m = tbl["team"] == t
        if m.any():
            tbl.loc[m, "attack_z"] -= star
            tbl.loc[m, "defense_z"] -= star * 0.6
            notes.append(f"injury: {t} -{star:.2f} attack z")
    for t in boost or []:
        m = tbl["team"] == t
        if m.any():
            tbl.loc[m, "attack_z"] += 0.25
            tbl.loc[m, "defense_z"] += 0.15
            notes.append(f"boost: {t} +0.25 attack z")
    for spec in scenario or []:
        try:
            team, da, dd = spec.split(":")
            m = tbl["team"] == team
            tbl.loc[m, "attack_z"] += float(da)
            tbl.loc[m, "defense_z"] += float(dd)
            notes.append(f"scenario: {spec}")
        except ValueError:
            log.warning("bad --scenario spec '%s' (want TEAM:da:dd)", spec)
    return notes


def _run(args) -> Results:
    # Scenario application requires re-deriving ratings, so we hook in by running
    # the pipeline then re-simulating if any scenario flags are present.
    res = run_pipeline(n_sims=args.sims, use_pymc=args.pymc,
                       use_ml=not args.no_ml, seed=args.seed)
    if args.injure or args.boost or args.scenario:
        notes = _apply_scenarios(res.ratings, res.strength, args.injure,
                                 args.boost, args.scenario)
        log.info("re-simulating under scenario: %s", notes)
        from ..model.simulator import TournamentSimulator
        sim = TournamentSimulator(res.ratings.table, res.strength,
                                  res.dataset.teams, res.context, seed=args.seed)
        res.sim = sim.run(args.sims or load_config().get("simulation.n_sims"))
        # Refresh derived artefacts that depend on the simulation.
        from ..model.power_index import build_power_index
        from ..model.predictor import predict_all
        dq_map = dict(zip(res.data_quality["team"], res.data_quality["dq_score"]))
        res.predictions = predict_all(res.dataset, res.strength,
                                      res.ratings.table, res.context, res.sim, dq_map)
        res.power_index = build_power_index(res.strength, res.dataset.teams,
                                            res.sim.team_probs, res.dataset.history,
                                            res.dataset.odds)
        res.config_summary["scenario"] = notes
    return res


# --------------------------------------------------------------------------- #
def _write_outputs(res: Results, with_report: bool) -> None:
    write_table(res.predictions, "predictions.csv")
    write_table(res.sim.team_probs, "team_probabilities.csv")
    write_table(res.sim.group_tables, "group_tables.csv")
    write_table(res.strength.assign(rank=range(1, len(res.strength) + 1)),
                "power_rankings.csv")
    write_table(res.data_quality, "data_quality.csv")
    write_table(res.power_index, "power_index.csv")
    if not res.value_board.empty:
        write_table(res.value_board, "value_bets.csv")
    for name, df in res.insights.items():
        write_table(df, f"insight_{name}.csv")
    # Full readable match-by-match file.
    from ..reports.match_table import render_match_predictions
    mp_path = load_config().path("outputs") / "match_predictions.md"
    mp_path.write_text(render_match_predictions(res.predictions), encoding="utf-8")
    log.info("wrote match predictions -> %s", mp_path)
    # Modal bracket.
    modal = res.sim.bracket_modal
    bracket_rows = [{"group": g, "winner": modal[g + "1"],
                     "runner_up": modal[g + "2"]}
                    for g in sorted({k[0] for k in modal})]
    write_table(pd.DataFrame(bracket_rows), "bracket_projection.csv")
    if with_report:
        report = generate_report(res)
        path = load_config().path("outputs") / "report.md"
        path.write_text(report, encoding="utf-8")
        log.info("wrote report -> %s", path)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wcp",
                                description="World Cup 2026 quant predictor")
    p.add_argument("command",
                   choices=["ingest", "rate", "simulate", "value", "predict",
                            "report", "all", "live", "bracket"])
    p.add_argument("--sims", type=int, default=None, help="number of simulations")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pymc", action="store_true", help="use PyMC Bayesian layer")
    p.add_argument("--no-ml", action="store_true", help="disable ML xG layer")
    p.add_argument("--injure", action="append", help="team with key player out")
    p.add_argument("--boost", action="append", help="team with crowd/home boost")
    p.add_argument("--scenario", action="append", help="TEAM:attack_dz:defense_dz")
    p.add_argument("--stage", choices=["group", "R32", "all"], default="all",
                   help="filter `predict` output by stage")
    p.add_argument("--refresh", type=int, default=30,
                   help="`live` dashboard refresh interval (seconds)")
    p.add_argument("--once", action="store_true",
                   help="`live`: render a single snapshot and exit")
    args = p.parse_args(argv)

    if args.command == "live":
        from ..live.dashboard import run as run_live
        run_live(refresh=args.refresh, once=args.once)
        return 0

    if args.command == "bracket":
        import time
        from ..live.feed import ESPNFeed
        from ..live.live_bracket import build, render_text
        feed = ESPNFeed()
        while True:
            print(("\033[2J\033[H" if not args.once else "")
                  + render_text(build(feed)))
            if args.once:
                return 0
            print(f"\n\033[2mrefreshing every {args.refresh}s - Ctrl-C to stop\033[0m")
            try:
                time.sleep(args.refresh)
            except KeyboardInterrupt:
                return 0

    if args.command == "ingest":
        from ..ingest.dataset import load_dataset
        ds = load_dataset()
        print(f"Loaded {len(ds.teams)} teams, {len(ds.fixtures)} fixtures, "
              f"{len(ds.history)} historical results.")
        print("Sources:", {k: v["data_quality"] for k, v in ds.sources.items()})
        print("Validation warnings:", ds.warnings or "none")
        return 0

    if args.command == "rate":
        from ..ingest.dataset import load_dataset
        from ..model.team_strength import compute_team_strength
        ds = load_dataset()
        s = compute_team_strength(ds.teams, ds.history)
        s["rank"] = range(1, len(s) + 1)
        print(s[["rank", "team", "group", "composite_100", "squad_rating",
                 "attack_rating", "defense_rating"]].head(24).to_string(index=False))
        write_table(s, "power_rankings.csv")
        return 0

    res = _run(args)

    if args.command == "value":
        if res.value_board.empty:
            print("No odds loaded — value board unavailable.")
        else:
            print(res.value_board.to_string(index=False))
        _write_outputs(res, with_report=False)
        return 0

    if args.command == "predict":
        preds = res.predictions
        if args.stage and args.stage != "all":
            preds = preds[preds["stage"] == args.stage]
        cols = ["match_id", "stage", "group", "team_a", "team_b",
                "predicted_score", "team_a_xg", "team_b_xg", "team_a_win_prob",
                "draw_prob", "team_b_win_prob", "confidence"]
        cols = [c for c in cols if c in preds.columns]
        with pd.option_context("display.max_rows", None):
            print(preds[cols].to_string(index=False))
        _write_outputs(res, with_report=False)
        return 0

    if args.command == "simulate":
        tp = res.sim.team_probs
        print(tp[["team", "group", "p_r32", "p_qf", "p_sf", "p_champion",
                  "fair_champion_odds"]].head(16).to_string(index=False))
        _write_outputs(res, with_report=False)
        return 0

    # report / all
    _write_outputs(res, with_report=True)
    print(generate_report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
