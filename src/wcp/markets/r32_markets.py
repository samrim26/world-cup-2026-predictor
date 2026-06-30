"""Recorded Kalshi "to advance" prices for the 2026 Round of 32.

Snapshot taken Jun 30 2026 (pre-kickoff). ``mult`` is the Kalshi payout
multiplier = decimal odds (a $1 winning bet returns $mult); implied probability
= 1/mult. ``vol`` is the market's reported dollar volume (a proxy for how sharp
the price is — high volume = harder to beat). Order is as displayed.
"""

# Each market: two sides with their payout multiplier, plus volume.
MARKETS = [
    {"date": "2026-06-30", "t1": "FRA", "m1": 1.12, "t2": "SWE", "m2": 7.85, "vol": 11_017_881},
    {"date": "2026-06-30", "t1": "MEX", "m1": 1.52, "t2": "ECU", "m2": 2.52, "vol": 11_761_575},
    {"date": "2026-07-01", "t1": "ENG", "m1": 1.13, "t2": "COD", "m2": 7.25, "vol": 1_008_748},
    {"date": "2026-07-01", "t1": "BEL", "m1": 1.62, "t2": "SEN", "m2": 2.34, "vol": 365_948},
    {"date": "2026-07-01", "t1": "USA", "m1": 1.18, "t2": "BIH", "m2": 5.25, "vol": 1_136_164},
    {"date": "2026-07-02", "t1": "ESP", "m1": 1.14, "t2": "AUT", "m2": 6.74, "vol": 267_382},
    {"date": "2026-07-02", "t1": "POR", "m1": 1.38, "t2": "CRO", "m2": 3.08, "vol": 466_461},
    {"date": "2026-07-02", "t1": "SUI", "m1": 1.48, "t2": "ALG", "m2": 2.73, "vol": 125_013},
    {"date": "2026-07-03", "t1": "AUS", "m1": 2.19, "t2": "EGY", "m2": 1.70, "vol": 189_370},
    {"date": "2026-07-03", "t1": "ARG", "m1": 1.09, "t2": "CPV", "m2": 9.41, "vol": 1_920_752},
    {"date": "2026-07-03", "t1": "COL", "m1": 1.22, "t2": "GHA", "m2": 4.73, "vol": 281_994},
]
