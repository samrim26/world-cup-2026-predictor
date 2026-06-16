"""End-to-end orchestration: data -> ratings -> simulation -> predictions ->
value -> insights. Returns a single results container consumed by the CLI,
report generator, and Streamlit app.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features.context import build_match_context
from .ingest.dataset import Dataset, load_dataset
from .model import value_betting
from .model.ensemble import TeamRatings, build_team_ratings
from .model.power_index import build_power_index
from .model.predictor import predict_all
from .model.simulator import SimResult, TournamentSimulator
from .model.team_strength import compute_team_strength
from .reports import insights
from .utils.config import load_config
from .utils.logging import get_logger
from .utils.validation import data_quality_score

log = get_logger("pipeline")


@dataclass
class Results:
    dataset: Dataset
    strength: pd.DataFrame
    ratings: TeamRatings
    context: pd.DataFrame
    sim: SimResult
    predictions: pd.DataFrame
    value_board: pd.DataFrame
    power_index: pd.DataFrame
    data_quality: pd.DataFrame
    insights: dict[str, pd.DataFrame]
    config_summary: dict


def _team_data_quality(ds: Dataset) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-team data-quality label + score from source availability."""
    history_teams = set(ds.history["home_team"]) | set(ds.history["away_team"])
    squad_teams = set(ds.squads["team"]) if not ds.squads.empty else set()
    has_odds = ds.has_odds
    rows = {}
    for t in ds.teams["team"]:
        flags = {
            "recent_results": t in history_teams,
            "ratings": True,
            "venue": True,
            "squad": t in squad_teams,
            "odds": has_odds,
            "injuries": False,
        }
        label, score = data_quality_score(flags)
        rows[t] = (label, score)
    df = pd.DataFrame([{"team": t, "data_quality": lbl, "dq_score": sc}
                       for t, (lbl, sc) in rows.items()])
    return df, {t: sc for t, (lbl, sc) in rows.items()}


def run_pipeline(n_sims: int | None = None, use_pymc: bool = False,
                 use_ml: bool = True, seed: int | None = None) -> Results:
    cfg = load_config()
    n_sims = n_sims or cfg.get("simulation.n_sims", 100000)

    log.info("loading data ...")
    ds = load_dataset()
    dq_df, dq_map = _team_data_quality(ds)

    log.info("computing team strength + ensemble ratings ...")
    strength = compute_team_strength(ds.teams, ds.history)
    ratings = build_team_ratings(strength, ds.history, ds.teams,
                                 use_pymc=use_pymc, use_ml=use_ml)

    log.info("building match context ...")
    context = build_match_context(ds.fixtures, strength, ds.venues)

    log.info("simulating tournament (%d sims) ...", n_sims)
    sim = TournamentSimulator(ratings.table, strength, ds.teams, context,
                              seed=seed if seed is not None else cfg.get("seed", 42))
    sim_res = sim.run(n_sims)

    log.info("generating per-match predictions ...")
    preds = predict_all(ds, strength, ratings.table, context, sim_res, dq_map)

    log.info("building value board + insights ...")
    model_champ = dict(zip(sim_res.team_probs["team"],
                           sim_res.team_probs["p_champion"]))
    conf_map = {t: 0.6 for t in ds.teams["team"]}
    value_board = value_betting.build_value_board(
        model_champ, ds.odds, conf_map, dq_map, market="tournament_winner")

    log.info("building multi-lens power index ...")
    power_index = build_power_index(strength, ds.teams, sim_res.team_probs,
                                    ds.history, ds.odds)

    ins = {
        "overrated": insights.overrated_underrated(sim_res.team_probs, ds.teams, ds.odds),
        "upset_radar": insights.upset_radar(preds, ds.teams),
        "dark_horse": insights.dark_horse_index(sim_res.team_probs, ds.teams, ds.odds),
        "path_difficulty": insights.path_difficulty(sim_res.team_probs, strength),
        "bracket_leverage": insights.bracket_leverage(sim_res.team_probs),
        "consensus_vs_model": insights.consensus_vs_model(sim_res.team_probs, ds.teams, ds.odds),
    }

    config_summary = {
        "n_sims": n_sims,
        "ensemble_weights": ratings.weights,
        "ensemble_components": ratings.components,
        "sources": {k: v["data_quality"] for k, v in ds.sources.items()},
        "has_odds": ds.has_odds,
        "validation_warnings": ds.warnings,
    }
    return Results(dataset=ds, strength=strength, ratings=ratings, context=context,
                   sim=sim_res, predictions=preds, value_board=value_board,
                   power_index=power_index, data_quality=dq_df, insights=ins,
                   config_summary=config_summary)
