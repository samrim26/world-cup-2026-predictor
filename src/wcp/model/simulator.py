"""Monte-Carlo tournament simulator.

Fully vectorised across the simulation dimension: every match is evaluated for
all N sims at once. Per sim, each team gets a strength shock (form uncertainty);
the group stage is played with FIFA tiebreakers and a matchday-3 incentive
nudge; the best eight third-placed teams are slotted into the Round of 32 via
``bracket.assign_thirds``; knockouts run with reduced extra-time tempo and a
skill-weighted penalty shootout. A draw can never survive a knockout tie.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.config import load_config
from ..utils.logging import get_logger
from . import bracket, standings
from .expected_goals import match_xg

log = get_logger("simulator")

GROUPS = list("ABCDEFGHIJKL")
ROUND_NAMES = ["reach_R32", "reach_R16", "reach_QF", "reach_SF",
               "reach_final", "champion"]


@dataclass
class SimResult:
    team_probs: pd.DataFrame      # advancement + champion probabilities per team
    group_tables: pd.DataFrame    # projected group standings
    n_sims: int
    bracket_modal: dict           # most-likely occupant of each bracket slot


class TournamentSimulator:
    def __init__(self, ratings: pd.DataFrame, strength: pd.DataFrame,
                 teams_df: pd.DataFrame, context: pd.DataFrame,
                 seed: int | None = None):
        cfg = load_config()
        self.cfg = cfg
        self.rng = np.random.default_rng(seed if seed is not None
                                         else cfg.get("seed", 42))
        self.strength_sd = cfg.get("simulation.strength_sd", 0.18)

        # Team index space.
        self.teams = list(teams_df["team"])
        self.idx = {t: i for i, t in enumerate(self.teams)}
        r = ratings.set_index("team").reindex(self.teams)
        s = strength.set_index("team").reindex(self.teams)
        self.attack = r["attack_z"].to_numpy(dtype=float)
        self.defense = r["defense_z"].to_numpy(dtype=float)
        self.pen = s["penalty_rating"].to_numpy(dtype=float)
        self.gk = s["gk_rating"].to_numpy(dtype=float)
        self.group_of = teams_df.set_index("team")["group"].to_dict()

        # Group membership (fixed team indices per group).
        self.group_team_idx = {
            g: [self.idx[t] for t in teams_df[teams_df["group"] == g]["team"]]
            for g in GROUPS
        }

        # Group fixtures with per-side adjustments and matchday.
        self._build_group_fixtures(context, teams_df)

    # ------------------------------------------------------------------ setup
    def _build_group_fixtures(self, context: pd.DataFrame,
                              teams_df: pd.DataFrame) -> None:
        ctx = context.set_index("match_id")
        from ..ingest.fixtures import FixturesLoader
        fx = FixturesLoader().load()
        grp = fx[fx["stage"] == "group"]
        fixtures = []
        for _, m in grp.iterrows():
            c = ctx.loc[m["match_id"]] if m["match_id"] in ctx.index else None
            fixtures.append({
                "group": m["group"],
                "a": self.idx[m["team_a"]], "b": self.idx[m["team_b"]],
                "adj_a": float(c["adj_a"]) if c is not None else 0.0,
                "adj_b": float(c["adj_b"]) if c is not None else 0.0,
                "md": m["round"],
            })
        self.group_fixtures = fixtures

    # -------------------------------------------------------------- mechanics
    def _eff_ratings(self, n: int):
        """Per-sim attack/defense with a shared per-team form shock."""
        shock = self.rng.normal(0.0, self.strength_sd, size=(len(self.teams), n))
        return self.attack[:, None] + shock, self.defense[:, None] + shock

    @staticmethod
    def _gather(rating: np.ndarray, idx):
        """Effective rating per sim for team ``idx``.

        ``rating`` is shaped (n_teams, n_sims). ``idx`` may be a scalar team
        index (same team across all sims -> group stage) or a per-sim array of
        team indices (knockouts), in which case we gather the matching sim.
        """
        if np.isscalar(idx):
            return rating[idx]
        cols = np.arange(rating.shape[1])
        return rating[idx, cols]

    def _play(self, a, b, att, deff, adj_a, adj_b, knockout=False):
        """Sample goals for a batch of matches. a,b are team indices (scalar or
        per-sim arrays)."""
        lam_a, lam_b = match_xg(self._gather(att, a), self._gather(deff, a),
                                self._gather(att, b), self._gather(deff, b),
                                adj_a, adj_b, knockout=knockout)
        ga = self.rng.poisson(lam_a)
        gb = self.rng.poisson(lam_b)
        return ga, gb

    def _shootout(self, a, b):
        """Return boolean array: True where team A wins the shootout."""
        k = self.cfg.get("simulation.penalty_skill_scale", 0.10)
        pen_a, pen_b = self.pen[a], self.pen[b]
        gk_a, gk_b = self.gk[a], self.gk[b]
        edge = k * ((pen_a - pen_b) + 0.5 * (gk_a - gk_b))
        p_a = np.clip(0.5 + edge, 0.05, 0.95)
        return self.rng.random(len(a)) < p_a

    def _knockout_round(self, a, b, att, deff):
        """Resolve a batch of knockout ties; return winner team-index array.

        a, b are per-sim arrays of team indices.
        """
        ga, gb = self._play(a, b, att, deff, 0.0, 0.0, knockout=True)
        win_a = ga > gb
        win_b = gb > ga
        tied = ~(win_a | win_b)
        if tied.any():
            # Extra time (reduced tempo).
            et_scale = self.cfg.get("xg.extra_time_tempo", 0.34)
            lam_a, lam_b = match_xg(
                self._gather(att, a), self._gather(deff, a),
                self._gather(att, b), self._gather(deff, b),
                np.log(et_scale), np.log(et_scale))
            ega = self.rng.poisson(lam_a)
            egb = self.rng.poisson(lam_b)
            win_a = win_a | (tied & (ega > egb))
            win_b = win_b | (tied & (egb > ega))
            still = tied & (ega == egb)
            if still.any():
                a_wins_so = self._shootout(a, b)
                win_a = win_a | (still & a_wins_so)
                win_b = win_b | (still & ~a_wins_so)
        return np.where(win_a, a, b)

    # ------------------------------------------------------------------- main
    def run(self, n_sims: int) -> SimResult:
        n = n_sims
        att, deff = self._eff_ratings(n)
        log.info("simulating %d tournaments ...", n)

        # --- Group stage -------------------------------------------------
        pts = {g: np.zeros((4, n), dtype=np.int32) for g in GROUPS}
        gd = {g: np.zeros((4, n), dtype=np.int32) for g in GROUPS}
        gf = {g: np.zeros((4, n), dtype=np.int32) for g in GROUPS}
        local = {g: {ti: k for k, ti in enumerate(self.group_team_idx[g])}
                 for g in GROUPS}

        def accumulate(g, a, b, ga, gb):
            la, lb = local[g][a], local[g][b]
            pts[g][la] += np.where(ga > gb, 3, np.where(ga == gb, 1, 0))
            pts[g][lb] += np.where(gb > ga, 3, np.where(ga == gb, 1, 0))
            gd[g][la] += ga - gb
            gd[g][lb] += gb - ga
            gf[g][la] += ga
            gf[g][lb] += gb

        # Matchdays 1 & 2 first (needed for MD3 incentives).
        for m in self.group_fixtures:
            if m["md"] == "MD3":
                continue
            ga, gb = self._play(m["a"], m["b"], att, deff, m["adj_a"], m["adj_b"])
            accumulate(m["group"], m["a"], m["b"], ga, gb)

        # Matchday 3 with incentive nudge from the interim table.
        for m in self.group_fixtures:
            if m["md"] != "MD3":
                continue
            g = m["group"]
            da = self._incentive(g, m["a"], pts)
            db = self._incentive(g, m["b"], pts)
            ga, gb = self._play(m["a"], m["b"], att, deff,
                                m["adj_a"] + da, m["adj_b"] + db)
            accumulate(g, m["a"], m["b"], ga, gb)

        ranked, third_keys = self._rank_groups(pts, gd, gf, n)

        # --- Qualification: winners, runners, best-8 thirds --------------
        winners = {g: ranked[g][0] for g in GROUPS}
        runners = {g: ranked[g][1] for g in GROUPS}
        third_groups = self._best_eight_thirds(third_keys, n)

        # --- Round of 32 occupancy --------------------------------------
        r32 = self._build_r32(winners, runners, ranked, third_groups, n)

        # --- Knockouts ---------------------------------------------------
        reach = {name: np.zeros(len(self.teams)) for name in ROUND_NAMES}
        # Everyone in R32 reaches R32.
        for slot in r32:
            np.add.at(reach["reach_R32"], slot, 1)

        matches = r32
        for round_name, pairs in [("reach_R16", bracket.R16_PAIRS),
                                  ("reach_QF", bracket.QF_PAIRS),
                                  ("reach_SF", bracket.SF_PAIRS),
                                  ("reach_final", bracket.FINAL_PAIR)]:
            # Pair current slots and resolve.
            winners_round = []
            # `matches` is a list of team-index arrays (one per slot).
            for i in range(0, len(matches), 2):
                w = self._knockout_round(matches[i], matches[i + 1], att, deff)
                winners_round.append(w)
            for w in winners_round:
                np.add.at(reach[round_name], w, 1)
            matches = winners_round

        # Final: one match remains -> champion.
        champ = self._knockout_round(matches[0], matches[1], att, deff) \
            if len(matches) == 2 else matches[0]
        np.add.at(reach["champion"], champ, 1)

        return self._assemble(reach, pts, gd, ranked, n)

    # ------------------------------------------------------------- helpers
    def _incentive(self, g, team_idx, pts):
        """Vectorised MD3 tempo delta for one team given the interim table."""
        la = local_pos = list(self.group_team_idx[g]).index(team_idx)
        my = pts[g][la]
        others = np.delete(pts[g], la, axis=0)          # [3,n]
        best_other = others.max(axis=0)
        delta = np.zeros_like(my, dtype=float)
        delta = np.where(my < best_other, 0.05, delta)          # must chase
        delta = np.where(my >= best_other + 3, -0.04, delta)    # comfortable
        return delta

    def _sortkey(self, pts_g, gd_g, gf_g, n):
        rand = self.rng.random((pts_g.shape[0], n)) * 0.9
        return standings.group_sortkey(pts_g, gd_g, gf_g, rand)

    def _rank_groups(self, pts, gd, gf, n):
        """Rank each group once; return (ranked indices [4,n], third-key [12,n]).

        Using a single sortkey per group keeps the third-placed team identified
        for the bracket consistent with the team scored for best-8 selection.
        """
        ranked = {}
        third_keys = np.zeros((12, n))
        for gi, g in enumerate(GROUPS):
            key = self._sortkey(pts[g], gd[g], gf[g], n)         # [4,n]
            order = np.argsort(-key, axis=0)                      # best-first rows
            gidx = np.array(self.group_team_idx[g])
            ranked[g] = gidx[order]                              # [4,n] global idx
            third_keys[gi] = np.take_along_axis(key, order[2:3], axis=0)[0]
        return ranked, third_keys

    def _best_eight_thirds(self, third_keys, n):
        """Boolean mask [12,n]: which groups' third-placed team is in the best 8."""
        order_thirds = np.argsort(-third_keys, axis=0)            # [12,n]
        qualifies = np.zeros((12, n), dtype=bool)
        rows = order_thirds[:8]                                   # [8,n]
        cols = np.arange(n)
        for s in range(8):
            qualifies[rows[s], cols] = True
        return qualifies

    def _build_r32(self, winners, runners, ranked, qualifies, n):
        """Resolve all 32 R32 slots to team-index arrays [n], honouring the
        memoised third-place combination mapping per qualifying-group set."""
        # third team index per group [12,n]
        thirds_team = np.stack([ranked[g][2] for g in GROUPS])    # [12,n]
        # bitmask of qualifying groups per sim
        powers = (1 << np.arange(12))[:, None]
        masks = (qualifies * powers).sum(axis=0)                  # [n]

        # third-slot index -> team index array [n]
        third_slot_team = {k: np.full(n, -1, dtype=np.int64)
                           for k in range(bracket.N_THIRD_SLOTS)}
        for mask in np.unique(masks):
            sims = np.where(masks == mask)[0]
            groups = tuple(GROUPS[b] for b in range(12) if (mask >> b) & 1)
            mapping = bracket.assign_thirds(groups)               # slot->group letter
            for slot, glet in mapping.items():
                gi = GROUPS.index(glet)
                third_slot_team[slot][sims] = thirds_team[gi, sims]

        # Resolve each R32 match's two slots.
        win_arr = {g: winners[g] for g in GROUPS}
        run_arr = {g: runners[g] for g in GROUPS}
        slots = []
        for a_spec, b_spec in bracket.R32_MATCHES:
            slots.append(self._resolve(a_spec, win_arr, run_arr, third_slot_team))
            slots.append(self._resolve(b_spec, win_arr, run_arr, third_slot_team))
        return slots

    @staticmethod
    def _resolve(spec, win_arr, run_arr, third_slot_team):
        kind, key = spec
        if kind == "W":
            return win_arr[key]
        if kind == "R":
            return run_arr[key]
        return third_slot_team[key]

    def _assemble(self, reach, pts, gd, ranked, n):
        df = pd.DataFrame({"team": self.teams})
        df["group"] = df["team"].map(self.group_of)
        for name in ROUND_NAMES:
            df[name.replace("reach_", "p_") if name != "champion" else "p_champion"] = \
                reach[name] / n
        # Rename to clean probability columns.
        df = df.rename(columns={"p_R32": "p_r32", "p_R16": "p_r16",
                                "p_QF": "p_qf", "p_SF": "p_sf",
                                "p_final": "p_final"})

        # Group-finish probabilities + projected points.
        win_p, ru_p, th_p, proj_pts, proj_gd = [], [], [], [], []
        for t in self.teams:
            g = self.group_of[t]
            ti = self.idx[t]
            la = self.group_team_idx[g].index(ti)
            r0 = ranked[g][0] == ti
            r1 = ranked[g][1] == ti
            r2 = ranked[g][2] == ti
            win_p.append(r0.mean()); ru_p.append(r1.mean()); th_p.append(r2.mean())
            proj_pts.append(pts[g][la].mean()); proj_gd.append(gd[g][la].mean())
        df["p_win_group"] = win_p
        df["p_runner_up"] = ru_p
        df["p_third"] = th_p
        df["proj_points"] = np.round(proj_pts, 2)
        df["proj_gd"] = np.round(proj_gd, 2)
        df["fair_champion_odds"] = np.where(df["p_champion"] > 0,
                                            (1 / df["p_champion"]).round(1), np.inf)

        df = df.sort_values("p_champion", ascending=False).reset_index(drop=True)

        group_tables = df[["team", "group", "proj_points", "proj_gd",
                           "p_win_group", "p_runner_up", "p_third",
                           "p_r32"]].sort_values(
            ["group", "proj_points"], ascending=[True, False]).reset_index(drop=True)

        modal = self._modal_bracket(ranked, n)
        return SimResult(team_probs=df, group_tables=group_tables,
                         n_sims=n, bracket_modal=modal)

    def _modal_bracket(self, ranked, n):
        """Most-likely group winner & runner-up per group (modal qualifiers)."""
        modal = {}
        for g in GROUPS:
            w = ranked[g][0]
            r = ranked[g][1]
            modal[f"{g}1"] = self.teams[int(np.bincount(w, minlength=len(self.teams)).argmax())]
            modal[f"{g}2"] = self.teams[int(np.bincount(r, minlength=len(self.teams)).argmax())]
        return modal
