"""The user's final predictions, for live 'predicted vs actual' comparison.

Group predictions are (home, away, home_goals, away_goals) in the real fixture
orientation. Teams are matched to live ESPN results by abbreviation.
"""
from __future__ import annotations

# Map our team names -> ESPN abbreviations (used to match live results).
NAME2ABBR = {
    "Mexico": "MEX", "South Africa": "RSA", "South Korea": "KOR",
    "Czech Republic": "CZE", "Canada": "CAN", "Bosnia and Herzegovina": "BIH",
    "Qatar": "QAT", "Switzerland": "SUI", "Brazil": "BRA", "Morocco": "MAR",
    "Haiti": "HAI", "Scotland": "SCO", "United States": "USA", "Paraguay": "PAR",
    "Australia": "AUS", "Turkey": "TUR", "Germany": "GER", "Curacao": "CUW",
    "Ivory Coast": "CIV", "Ecuador": "ECU", "Netherlands": "NED", "Japan": "JPN",
    "Sweden": "SWE", "Tunisia": "TUN", "Belgium": "BEL", "Egypt": "EGY",
    "Iran": "IRN", "New Zealand": "NZL", "Spain": "ESP", "Cape Verde": "CPV",
    "Saudi Arabia": "KSA", "Uruguay": "URU", "France": "FRA", "Senegal": "SEN",
    "Iraq": "IRQ", "Norway": "NOR", "Argentina": "ARG", "Algeria": "ALG",
    "Austria": "AUT", "Jordan": "JOR", "Portugal": "POR", "DR Congo": "COD",
    "Uzbekistan": "UZB", "Colombia": "COL", "England": "ENG", "Croatia": "CRO",
    "Ghana": "GHA", "Panama": "PAN",
}
ABBR2NAME = {v: k for k, v in NAME2ABBR.items()}

# The user's 72 group-stage score predictions.
GROUP_PREDICTIONS = {
 "A": [("Mexico","South Africa",2,0),("South Korea","Czech Republic",1,0),
       ("Czech Republic","South Africa",1,0),("Mexico","South Korea",2,1),
       ("South Africa","South Korea",0,1),("Czech Republic","Mexico",0,2)],
 "B": [("Bosnia and Herzegovina","Qatar",1,0),("Qatar","Switzerland",0,4),
       ("Switzerland","Bosnia and Herzegovina",3,1),("Switzerland","Canada",2,0),
       ("Canada","Qatar",2,0),("Canada","Bosnia and Herzegovina",1,0)],
 "C": [("Scotland","Brazil",0,4),("Brazil","Morocco",1,0),("Haiti","Scotland",0,3),
       ("Morocco","Haiti",4,0),("Brazil","Haiti",4,0),("Scotland","Morocco",0,2)],
 "D": [("Paraguay","Australia",0,1),("United States","Australia",2,1),
       ("Australia","Turkey",1,2),("United States","Paraguay",2,1),
       ("Turkey","Paraguay",1,0),("Turkey","United States",1,2)],
 "E": [("Germany","Curacao",4,0),("Curacao","Ivory Coast",0,2),("Ecuador","Curacao",3,0),
       ("Germany","Ivory Coast",2,0),("Ivory Coast","Ecuador",0,1),("Ecuador","Germany",0,1)],
 "F": [("Tunisia","Japan",0,1),("Netherlands","Sweden",4,0),("Tunisia","Netherlands",0,2),
       ("Sweden","Tunisia",0,1),("Netherlands","Japan",2,1),("Japan","Sweden",3,0)],
 "G": [("New Zealand","Egypt",0,2),("New Zealand","Belgium",0,4),("Belgium","Egypt",2,0),
       ("Egypt","Iran",0,1),("Belgium","Iran",2,0),("Iran","New Zealand",2,0)],
 "H": [("Spain","Cape Verde",6,0),("Cape Verde","Saudi Arabia",0,0),("Spain","Saudi Arabia",4,0),
       ("Saudi Arabia","Uruguay",0,3),("Uruguay","Cape Verde",2,0),("Uruguay","Spain",0,2)],
 "I": [("France","Senegal",3,0),("Senegal","Iraq",2,0),("Norway","France",0,4),
       ("Norway","Senegal",1,2),("Iraq","Norway",0,2),("France","Iraq",4,0)],
 "J": [("Austria","Jordan",3,0),("Jordan","Algeria",0,2),("Jordan","Argentina",0,4),
       ("Algeria","Austria",0,1),("Argentina","Austria",3,0),("Argentina","Algeria",4,0)],
 "K": [("Portugal","Uzbekistan",4,0),("Colombia","DR Congo",2,0),("Portugal","DR Congo",4,0),
       ("Colombia","Portugal",0,2),("Uzbekistan","Colombia",0,3),("DR Congo","Uzbekistan",0,1)],
 "L": [("Ghana","Panama",1,2),("England","Croatia",2,0),("England","Ghana",5,0),
       ("Croatia","Ghana",3,0),("Panama","England",0,3),("Panama","Croatia",0,3)],
}

# Predicted knockout winners (champion etc.) for reference.
CHAMPION = "Argentina"
RUNNER_UP = "Spain"
THIRD = "Brazil"

# The user's predicted knockout bracket: each round's matchups (home, away, winner).
KO_MATCHES = {
 "Round of 32": [("Germany","Australia","Germany"),("France","Tunisia","France"),
   ("South Korea","Canada","South Korea"),("Netherlands","Morocco","Netherlands"),
   ("Colombia","Croatia","Colombia"),("Spain","Austria","Spain"),
   ("United States","Bosnia and Herzegovina","United States"),("Belgium","Czech Republic","Belgium"),
   ("Brazil","Japan","Brazil"),("Ecuador","Senegal","Senegal"),
   ("Mexico","Scotland","Mexico"),("England","Norway","England"),
   ("Argentina","Uruguay","Argentina"),("Turkey","Iran","Turkey"),
   ("Switzerland","Egypt","Switzerland"),("Portugal","Ivory Coast","Portugal")],
 "Round of 16": [("Germany","France","France"),("South Korea","Netherlands","Netherlands"),
   ("Colombia","Spain","Spain"),("United States","Belgium","United States"),
   ("Brazil","Senegal","Brazil"),("Mexico","England","England"),
   ("Argentina","Turkey","Argentina"),("Switzerland","Portugal","Portugal")],
 "Quarterfinals": [("France","Netherlands","Netherlands"),("Spain","United States","Spain"),
   ("Brazil","England","Brazil"),("Argentina","Portugal","Argentina")],
 "Semifinals": [("Netherlands","Spain","Spain"),("Brazil","Argentina","Argentina")],
 "Final": [("Spain","Argentina","Argentina")],
}


def predicted_rounds() -> dict[str, list[str]]:
    """Teams (abbr) the user predicted to *reach* each knockout round."""
    def abbrs(names):
        return [NAME2ABBR.get(n, n) for n in names]
    r32 = [t for m in KO_MATCHES["Round of 32"] for t in m[:2]]
    return {
        "Round of 32 (32)": abbrs(r32),
        "Round of 16 (16)": abbrs([m[2] for m in KO_MATCHES["Round of 32"]]),
        "Quarterfinals (8)": abbrs([m[2] for m in KO_MATCHES["Round of 16"]]),
        "Semifinals (4)": abbrs([m[2] for m in KO_MATCHES["Quarterfinals"]]),
        "Final (2)": abbrs([m[2] for m in KO_MATCHES["Semifinals"]]),
        "Champion (1)": abbrs([KO_MATCHES["Final"][0][2]]),
    }


def predicted_score_by_abbrs(a: str, b: str):
    """Return (home_abbr, home_goals, away_goals) for the user's prediction of the
    match between abbreviations a and b, or None if not found."""
    for group in GROUP_PREDICTIONS.values():
        for h, aw, hs, as_ in group:
            ph, pa = NAME2ABBR.get(h), NAME2ABBR.get(aw)
            if {ph, pa} == {a, b}:
                return ph, hs, as_
    return None


def predicted_scoreline(home_abbr: str, away_abbr: str):
    """Predicted score as 'home-away' aligned to this match's orientation, or None."""
    pred = predicted_score_by_abbrs(home_abbr, away_abbr)
    if not pred:
        return None
    ph, hs, as_ = pred
    return f"{hs}-{as_}" if ph == home_abbr else f"{as_}-{hs}"


def attach_predictions(matches: list[dict]) -> list[dict]:
    """Add a 'predicted' scoreline field to each match dict (in place)."""
    for m in matches:
        m["predicted"] = predicted_scoreline(m.get("home"), m.get("away"))
    return matches
