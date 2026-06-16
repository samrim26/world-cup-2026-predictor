"""Bayesian / statistical attack-defense estimator from match history.

Always-available path: a Dixon-Coles-style Poisson maximum-likelihood fit of
per-team attack and defence effects on the historical results, with empirical-
Bayes shrinkage toward the Elo-based prior (so teams with little/synthetic
history are pulled to their rating, not over-fit).

Optional upgrade: if PyMC is installed, a hierarchical Poisson model is fit by
MCMC to obtain genuine posterior means *and* uncertainty. Controlled by the
``use_pymc`` flag; falls back silently if unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..utils.logging import get_logger
from .weighting import match_weights

log = get_logger("bayesian")


def _standardise_subset(values: pd.Series, subset: list[str]) -> pd.Series:
    """Restrict a rating series to ``subset`` teams and z-score over them."""
    s = values.reindex(subset).astype(float)
    s = s.fillna(s.mean() if s.notna().any() else 0.0)
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


@dataclass
class StrengthEstimate:
    attack_z: pd.Series      # standardised attacking rating per team
    defense_z: pd.Series     # standardised defensive rating (higher = better D)
    attack_sd: pd.Series     # posterior sd (0 for the MLE path)
    method: str


def _prepare(history: pd.DataFrame, idx: dict[str, int]):
    h = history[history["home_team"].isin(idx) & history["away_team"].isin(idx)]
    hi = h["home_team"].map(idx).to_numpy()
    ai = h["away_team"].map(idx).to_numpy()
    hg = h["home_score"].to_numpy(dtype=float)
    ag = h["away_score"].to_numpy(dtype=float)
    w = match_weights(h)
    return hi, ai, hg, ag, w


def fit_poisson_mle(history: pd.DataFrame, teams_df: pd.DataFrame,
                    shrinkage: float = 1.0) -> StrengthEstimate:
    """Fit per-team attack/defence by recency-weighted penalised Poisson MLE.

    Fits over the *union* of all teams appearing in the (real) history — so each
    of our 48 is informed by every real opponent it has faced — then extracts and
    standardises the 48 participants. Recent, competitive matches weigh more.

    Model: log E[home goals] = mu + home + att[h] - def[a]
           log E[away goals] = mu + att[a] - def[h]
    Penalty pulls att/def toward an Elo prior for the 48 (empirical Bayes).
    """
    participants = list(teams_df["team"])
    # Universe = all teams in history (cap to those with enough games for stability).
    universe = sorted(set(history["home_team"]) | set(history["away_team"]))
    if not universe:
        universe = participants
    idx = {t: i for i, t in enumerate(universe)}
    n = len(universe)
    hi, ai, hg, ag, w = _prepare(history, idx)

    # Elo prior (z-scored) for the 48; unknown opponents get a 0 (league-avg) prior.
    elo = teams_df.set_index("team")["elo"]
    elo_z = (elo - elo.mean()) / (elo.std(ddof=0) or 1.0)
    prior = np.array([elo_z.get(t, 0.0) for t in universe])

    def unpack(x):
        return x[0], x[1], x[2:2 + n], x[2 + n:2 + 2 * n]

    def negloglik(x):
        mu, home, att, deff = unpack(x)
        log_h = mu + home + att[hi] - deff[ai]
        log_a = mu + att[ai] - deff[hi]
        lam_h, lam_a = np.exp(log_h), np.exp(log_a)
        ll = np.sum(w * (hg * log_h - lam_h)) + np.sum(w * (ag * log_a - lam_a))
        pen = shrinkage * (np.sum((att - 0.3 * prior) ** 2)
                           + np.sum((deff - 0.3 * prior) ** 2))
        pen += 50.0 * (att.mean() ** 2 + deff.mean() ** 2)
        return -ll + pen

    x0 = np.concatenate([[np.log(1.3), 0.2], 0.3 * prior, 0.3 * prior])
    res = minimize(negloglik, x0, method="L-BFGS-B", options={"maxiter": 800})
    _, _, att, deff = unpack(res.x)

    att_full = pd.Series(att, index=universe)
    def_full = pd.Series(deff, index=universe)
    att_z = _standardise_subset(att_full, participants)
    def_z = _standardise_subset(def_full, participants)
    log.info("Poisson MLE: %d teams (%d participants), %d weighted matches, conv=%s",
             n, len(participants), len(hi), res.success)
    return StrengthEstimate(att_z, def_z,
                            pd.Series(0.0, index=participants), "poisson_mle")


def fit_pymc(history: pd.DataFrame, teams_df: pd.DataFrame,
             draws: int = 800, tune: int = 800) -> StrengthEstimate | None:
    """Hierarchical Poisson via PyMC, returning posterior means + sds.

    Returns ``None`` if PyMC is not installed so the caller can fall back.
    """
    try:
        import pymc as pm  # type: ignore
    except Exception:
        log.info("PyMC not available; using Poisson-MLE Bayesian path")
        return None

    teams = list(teams_df["team"])
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    hi, ai, hg, ag, _w = _prepare(history, idx)
    with pm.Model():
        mu = pm.Normal("mu", 0.3, 1.0)
        home = pm.Normal("home", 0.2, 0.5)
        sd_att = pm.HalfNormal("sd_att", 0.5)
        sd_def = pm.HalfNormal("sd_def", 0.5)
        att = pm.Normal("att", 0.0, sd_att, shape=n)
        deff = pm.Normal("deff", 0.0, sd_def, shape=n)
        lam_h = pm.math.exp(mu + home + att[hi] - deff[ai])
        lam_a = pm.math.exp(mu + att[ai] - deff[hi])
        pm.Poisson("hg", lam_h, observed=hg)
        pm.Poisson("ag", lam_a, observed=ag)
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=1,
                          progressbar=False, random_seed=42)

    post = idata.posterior
    att_m = post["att"].mean(("chain", "draw")).to_numpy()
    def_m = post["deff"].mean(("chain", "draw")).to_numpy()
    att_s = post["att"].std(("chain", "draw")).to_numpy()
    att_z = pd.Series((att_m - att_m.mean()) / (att_m.std() or 1.0), index=teams)
    def_z = pd.Series((def_m - def_m.mean()) / (def_m.std() or 1.0), index=teams)
    return StrengthEstimate(att_z, def_z, pd.Series(att_s, index=teams), "pymc")


def estimate(history: pd.DataFrame, teams_df: pd.DataFrame,
             use_pymc: bool = False) -> StrengthEstimate:
    """Public entry: PyMC if requested+available, else penalised Poisson MLE."""
    if use_pymc:
        est = fit_pymc(history, teams_df)
        if est is not None:
            return est
    return fit_poisson_mle(history, teams_df)
