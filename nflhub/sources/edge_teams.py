"""Team-code normalization for the edge feature, targeting nfl-hub's own convention
(from nflhub/sources/nfl_schedule.py's ESPN scoreboard data) rather than pickem-edge's
original table, which used a few different canonical spellings (WAS vs WSH here).
"""

from __future__ import annotations

TEAMS: frozenset[str] = frozenset(
    {
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
        "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WSH",
    }
)

# Alternate spellings/city codes seen in scrapes or hand-typed/pasted input, mapped to
# nfl-hub's canonical abbreviation above.
_ALIASES: dict[str, str] = {
    "WAS": "WSH", "JAC": "JAX", "LA": "LAR", "STL": "LAR", "SD": "LAC",
    "OAK": "LV", "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB",
}


class UnknownTeamError(ValueError):
    pass


def normalize_team(raw: str) -> str:
    code = (raw or "").strip().upper()
    if code in TEAMS:
        return code
    if code in _ALIASES:
        return _ALIASES[code]
    raise UnknownTeamError(f"unknown team code: {raw!r}")
