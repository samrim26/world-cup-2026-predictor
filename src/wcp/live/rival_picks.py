"""The one other pool entrant who also picked Argentina as champion.

Their per-round bracket placements (which teams they predicted to *reach* each
knockout round), read from their league scorecard. Used by :mod:`wcp.live.pool`
for the live head-to-head race. Because both brackets share the same Final
(Argentina over Spain), the title race between these two comes down purely to
where their deeper-round picks diverge.
"""
from __future__ import annotations

RIVAL = {
    "name": "Rival (ARG backer)",
    "champion": "ARG",
    "runner_up": "ESP",
    "third": None,                       # undetermined: England or France
    "third_candidates": ["ENG", "FRA"],  # their two semifinal losers
    "rounds": {
        # 32 teams they predicted to qualify from the groups (26 hit, 6 missed).
        "R32": ["ARG", "AUS", "AUT", "BEL", "BIH", "BRA", "CAN", "COL", "CRO",
                "ECU", "EGY", "ENG", "FRA", "GER", "JPN", "MEX", "MAR", "NED",
                "NOR", "POR", "SEN", "RSA", "ESP", "SWE", "SUI", "USA",
                "JOR", "NZL", "KOR", "TUR", "URU", "UZB"],
        "R16": ["BRA", "CAN", "ARG", "AUS", "BEL", "COL", "ECU", "ENG", "FRA",
                "GER", "MEX", "NED", "POR", "KOR", "ESP", "USA"],
        "QF": ["ARG", "BEL", "ENG", "FRA", "GER", "NED", "POR", "ESP"],
        "SF": ["ARG", "ENG", "FRA", "ESP"],
        "FINAL": ["ARG", "ESP"],
        "WINNER": ["ARG"],
    },
}
