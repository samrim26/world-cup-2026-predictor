"""Blend the Dixon-Coles (composite), Bayesian, and ML attack/defence ratings
into the final per-team ratings consumed by the simulator and predictors.

Each estimator contributes standardised (z-scored) attack and defence ratings.
Weights come from ``config.ensemble_weights`` and are renormalised over whichever
components are actually available, so the engine degrades gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..utils.config import load_config
from ..utils.logging import get_logger
from . import bayesian, ml_xg

log = get_logger("ensemble")


@dataclass
class TeamRatings:
    """Final blended ratings table, indexed by team."""
    table: pd.DataFrame           # columns: attack_z, defense_z + provenance
    components: dict[str, str]    # estimator -> backend/method label
    weights: dict[str, float]     # effective (renormalised) weights used


def build_team_ratings(strength: pd.DataFrame, history: pd.DataFrame,
                       teams_df: pd.DataFrame,
                       use_pymc: bool = False, use_ml: bool = True) -> TeamRatings:
    cfg = load_config()
    w = dict(cfg.get("ensemble_weights", {}))

    teams = list(strength["team"])
    # 1) Dixon-Coles / composite estimator (always present).
    dc = strength.set_index("team")
    dc_att = dc["attack_rating"]
    dc_def = dc["defense_rating"]
    # Standardise so all estimators share a scale.
    dc_att = (dc_att - dc_att.mean()) / (dc_att.std(ddof=0) or 1.0)
    dc_def = (dc_def - dc_def.mean()) / (dc_def.std(ddof=0) or 1.0)

    estimators: dict[str, tuple[pd.Series, pd.Series]] = {
        "dixon_coles": (dc_att.reindex(teams), dc_def.reindex(teams)),
    }
    components = {"dixon_coles": "composite"}

    # 2) Bayesian estimator.
    try:
        be = bayesian.estimate(history, teams_df, use_pymc=use_pymc)
        estimators["bayesian"] = (be.attack_z.reindex(teams),
                                  be.defense_z.reindex(teams))
        components["bayesian"] = be.method
    except Exception as exc:  # pragma: no cover - safety net
        log.warning("Bayesian estimator failed, dropping: %s", exc)
        w.pop("bayesian", None)

    # 3) ML estimator.
    if use_ml:
        try:
            me = ml_xg.estimate(history, teams_df)
            estimators["ml"] = (me.attack_z.reindex(teams),
                                me.defense_z.reindex(teams))
            components["ml"] = me.backend
        except Exception as exc:  # pragma: no cover
            log.warning("ML estimator failed, dropping: %s", exc)
            w.pop("ml", None)
    else:
        w.pop("ml", None)

    # Renormalise weights over available estimators.
    avail = {k: w.get(k, 0.0) for k in estimators}
    total = sum(avail.values()) or 1.0
    eff = {k: v / total for k, v in avail.items()}

    att = sum(eff[k] * estimators[k][0] for k in estimators)
    deff = sum(eff[k] * estimators[k][1] for k in estimators)

    table = pd.DataFrame({"attack_z": att, "defense_z": deff})
    table.index.name = "team"
    log.info("ensemble weights=%s components=%s",
             {k: round(v, 3) for k, v in eff.items()}, components)
    return TeamRatings(table=table.reset_index(), components=components, weights=eff)
