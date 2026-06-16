"""Multi-lens Power Index.

Ranks every team under several *independent* methods, then blends them into a
consensus rank and quantifies disagreement. The point is transparency: you can
see exactly where the methods (and pundits) part ways — e.g. a team the model
loves but the market doesn't, or a popular dark-horse pick that only one lens
supports.

Lenses
- ``model``   : our composite team strength
- ``elo``     : raw World-Football Elo
- ``market``  : vig-removed tournament-winner probability (real odds)
- ``sim``     : Monte-Carlo champion probability
- ``form``    : opposition-adjusted recent form
- ``external``: optional user-supplied ranking (data/external/external_ranking.csv)

Each lens yields a 1..N rank (1 = best). ``blended_rank`` is a config-weighted
average of the available lens ranks, re-ranked. ``disagreement`` is the spread
(sd) of a team's ranks across lenses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import load_config
from ..utils.logging import get_logger
from ..utils.odds_math import fair_probs
from . import ratings as ratings_mod

log = get_logger("power_index")


def _rank_from_score(teams: list[str], score: dict[str, float]) -> pd.Series:
    """Dense 1..N rank (1 = highest score). Missing teams sink to the bottom."""
    s = pd.Series({t: score.get(t, -np.inf) for t in teams})
    # rank: highest score -> rank 1. Stable, integer ranks.
    return s.rank(ascending=False, method="first").astype(int)


def _market_scores(odds: pd.DataFrame) -> dict[str, float]:
    if odds is None or odds.empty:
        return {}
    mk = odds[odds["market"] == "tournament_winner"]
    if mk.empty:
        return {}
    fmt = mk["format"].iloc[0]
    fair = fair_probs(mk["odds"].tolist(), fmt)
    return dict(zip(mk["selection"], fair))


def _external_scores(path: Path) -> tuple[dict[str, float], str]:
    """Read an optional external ranking. Accepts a `team` column plus either
    `score` (higher better) or `rank` (lower better)."""
    if not path.exists():
        return {}, ""
    df = pd.read_csv(path)
    if "team" not in df.columns:
        return {}, ""
    if "score" in df.columns:
        return dict(zip(df["team"], df["score"].astype(float))), "external"
    if "rank" in df.columns:
        # Convert rank -> score (higher better) so it ranks consistently.
        return dict(zip(df["team"], -df["rank"].astype(float))), "external"
    return {}, ""


def build_power_index(strength: pd.DataFrame, teams_df: pd.DataFrame,
                      sim_team_probs: pd.DataFrame, history: pd.DataFrame,
                      odds: pd.DataFrame | None,
                      external_path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    weights = dict(cfg.get("power_index_weights", {}))
    teams = list(teams_df["team"])

    # --- lens scores (higher = better) ---
    model_score = dict(zip(strength["team"], strength["composite"]))
    elo_score = dict(zip(teams_df["team"], teams_df["elo"].astype(float)))
    market_score = _market_scores(odds)
    sim_score = dict(zip(sim_team_probs["team"], sim_team_probs["p_champion"]))
    form = ratings_mod.recent_form_component(teams_df, history)
    form_score = dict(zip(teams_df["team"], form))

    lenses = {
        "model": model_score, "elo": elo_score, "market": market_score,
        "sim": sim_score, "form": form_score,
    }
    # Optional external lens.
    ext_path = Path(external_path) if external_path else \
        (cfg.root / "data" / "external" / "external_ranking.csv")
    ext_score, ext_label = _external_scores(ext_path)
    if ext_score:
        lenses["external"] = ext_score
        weights.setdefault("external", 0.15)
        log.info("Power Index: external ranking loaded (%d teams)", len(ext_score))

    rank_cols = {}
    for name, score in lenses.items():
        rank_cols[f"rank_{name}"] = _rank_from_score(teams, score)

    ranks = pd.DataFrame(rank_cols, index=teams)

    # Blend available lenses by weight; re-rank to a consensus.
    avail = [c for c in ranks.columns if weights.get(c.replace("rank_", ""), 0) > 0]
    if not avail:
        avail = list(ranks.columns)
    w = np.array([weights.get(c.replace("rank_", ""), 1.0) for c in avail])
    w = w / w.sum()
    blended_score = (ranks[avail].to_numpy() * w).sum(axis=1)
    ranks["blended_value"] = blended_score
    ranks["blended_rank"] = pd.Series(blended_score, index=teams).rank(
        ascending=True, method="first").astype(int)
    ranks["disagreement"] = ranks[[c for c in ranks.columns
                                   if c.startswith("rank_")]].std(axis=1).round(2)

    out = ranks.reset_index().rename(columns={"index": "team"})
    out = out.sort_values("blended_rank").reset_index(drop=True)
    # Reorder for readability.
    front = ["team", "blended_rank"] + \
        [c for c in out.columns if c.startswith("rank_")] + ["disagreement"]
    return out[front]
