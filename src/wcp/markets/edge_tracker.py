"""Model-vs-market edge tracker for the 2026 knockout markets.

For each recorded Kalshi market it: (1) computes the model's advance probability
(Elo head-to-head with a host bump), (2) compares it to the market's implied
probability to flag a value side and its EV, and (3) once the game settles,
grades the bet's P/L and scores model vs. market calibration (Brier).

This is the disciplined version of "is this worth betting": it turns one-off
guesses into a logged, gradable track record — the thing that actually tells you
whether your model beats the market.

Run:  python -m wcp.markets.edge_tracker
"""
from __future__ import annotations

import json
import urllib.request

from ..live.advance_odds import ABBR_ELO
from .r32_markets import MARKETS

HOSTS = {"USA", "MEX", "CAN"}
HOST_BUMP = 70            # Elo-points home edge for host nations (a modest prior)
EV_THRESHOLD = 0.02       # only call it a "value bet" above +2% EV
PROD_URL = "https://world-cup-predictor-nine-wine.vercel.app/api/live"


def model_p(a: str, b: str) -> float:
    """Model P(team a advances past team b) — Elo expected score, draws split."""
    d = ABBR_ELO.get(a, 1700) - ABBR_ELO.get(b, 1700)
    if a in HOSTS:
        d += HOST_BUMP
    if b in HOSTS:
        d -= HOST_BUMP
    return 1.0 / (1.0 + 10 ** (-d / 400))


def analyze(market: dict) -> dict:
    """Add model probs, implied probs, the value side and its EV to a market."""
    t1, t2 = market["t1"], market["t2"]
    p1 = model_p(t1, t2)
    imp1, imp2 = 1 / market["m1"], 1 / market["m2"]
    # EV per $1 staked on each side at the offered multiplier.
    ev1 = p1 * market["m1"] - 1
    ev2 = (1 - p1) * market["m2"] - 1
    if max(ev1, ev2) < EV_THRESHOLD:
        side, ev, mult, p = None, max(ev1, ev2), None, None
    elif ev1 >= ev2:
        side, ev, mult, p = t1, ev1, market["m1"], p1
    else:
        side, ev, mult, p = t2, ev2, market["m2"], 1 - p1
    return {**market, "p1": p1, "p2": 1 - p1, "imp1": imp1, "imp2": imp2,
            "fav": t1 if market["m1"] <= market["m2"] else t2,
            "value_side": side, "value_ev": ev, "value_mult": mult, "value_p": p}


def fetch_results(url: str = PROD_URL) -> dict:
    """{frozenset(abbr,abbr): winner_or_None} from the live R32 bracket."""
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.loads(r.read().decode())
    except Exception:
        return {}
    out = {}
    for m in d.get("live_bracket", {}).get("Round of 32", []):
        a, b = m.get("a_team"), m.get("b_team")
        if a and b:
            out[frozenset((a, b))] = m.get("winner")
    return out


def grade(markets: list[dict] | None = None, results: dict | None = None) -> dict:
    markets = [analyze(m) for m in (markets or MARKETS)]
    results = fetch_results() if results is None else results
    rows, staked, pnl, wins, losses = [], 0.0, 0.0, 0, 0
    model_brier = market_brier = 0.0
    graded = 0
    for m in markets:
        winner = results.get(frozenset((m["t1"], m["t2"])))
        row = {**m, "winner": winner, "bet_pnl": None, "bet_result": None}
        if winner:                                       # settled -> grade
            graded += 1
            a_adv = 1.0 if winner == m["t1"] else 0.0
            model_brier += (m["p1"] - a_adv) ** 2
            # de-vig the market to a fair prob for a fair Brier comparison
            fair1 = m["imp1"] / (m["imp1"] + m["imp2"])
            market_brier += (fair1 - a_adv) ** 2
            if m["value_side"]:                          # we had a value bet
                staked += 1
                won = (m["value_side"] == winner)
                bp = (m["value_mult"] - 1) if won else -1
                pnl += bp
                wins += int(won); losses += int(not won)
                row["bet_pnl"] = bp
                row["bet_result"] = "WON" if won else "LOST"
        rows.append(row)
    return {"rows": rows, "graded": graded, "staked": staked, "pnl": pnl,
            "wins": wins, "losses": losses,
            "model_brier": model_brier / graded if graded else None,
            "market_brier": market_brier / graded if graded else None}


def report(url: str = PROD_URL) -> str:
    g = grade(results=fetch_results(url))
    L = ["WC2026 knockout — model vs Kalshi edge tracker",
         "=" * 78,
         f"{'Match':16}{'Mkt fav':>12}{'Model':>10}{'Value bet':>16}{'EV':>7}{'Result':>9}{'P/L':>7}"]
    for r in g["rows"]:
        favpc = f"{r['fav']} {round((1/r['m1'] if r['fav']==r['t1'] else 1/r['m2'])*100)}%"
        modpc = f"{r['fav']} {round((r['p1'] if r['fav']==r['t1'] else r['p2'])*100)}%"
        if r["value_side"]:
            vb = f"{r['value_side']}@{r['value_mult']:.2f}"
            ev = f"{r['value_ev']*100:+.0f}%"
        else:
            vb, ev = "— none", "—"
        res = r["bet_result"] or (("adv:" + r["winner"]) if r["winner"] else "pending")
        pl = f"{r['bet_pnl']:+.2f}" if r["bet_pnl"] is not None else "—"
        L.append(f"{r['t1']+' v '+r['t2']:16}{favpc:>12}{modpc:>10}{vb:>16}{ev:>7}{res:>9}{pl:>7}")
    L.append("-" * 78)
    roi = (g["pnl"] / g["staked"] * 100) if g["staked"] else 0.0
    L.append(f"Value bets graded: {g['wins']+g['losses']}  (record {g['wins']}-{g['losses']})  "
             f"staked ${g['staked']:.0f}  P/L ${g['pnl']:+.2f}  ROI {roi:+.1f}%")
    if g["graded"]:
        verdict = "MODEL beats market" if g["model_brier"] < g["market_brier"] else "market beats model"
        L.append(f"Calibration over {g['graded']} settled games — "
                 f"model Brier {g['model_brier']:.3f} vs market {g['market_brier']:.3f}  → {verdict}")
        L.append("(lower Brier = sharper probabilities; this is the number that actually matters)")
    else:
        L.append("No games settled yet — rerun after kickoffs to auto-grade P/L and calibration.")
    return "\n".join(L)


if __name__ == "__main__":
    print(report())
