"""Per-game betting lines and de-vigged win probabilities.

Default provider "espn" reuses the odds already attached to the scoreboard payload, so no
key is needed. "sportsgameodds" and "theoddsapi" fetch fuller lines with ODDS_API_KEY.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)
TIMEOUT = 15

# NFL result spread std deviation; used to turn a point spread into a win probability
# when no moneyline is available. ~13.45 is the commonly cited historical value.
_SPREAD_SIGMA = 13.45


def _implied_from_moneyline(ml: float) -> float:
    return (-ml) / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def _winprob_from_spread(home_spread: float) -> float:
    """P(home wins) from the home point spread via a normal model."""
    return 0.5 * (1.0 + math.erf((-home_spread) / (_SPREAD_SIGMA * math.sqrt(2.0))))


def compute_probs(spread: Optional[float], ml_home: Optional[float], ml_away: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Return (implied_home, implied_away), de-vigged when both moneylines are present."""
    if ml_home is not None and ml_away is not None:
        ph = _implied_from_moneyline(float(ml_home))
        pa = _implied_from_moneyline(float(ml_away))
        s = ph + pa
        if s > 0:
            return round(ph / s, 4), round(pa / s, 4)
    if spread is not None:
        ph = _winprob_from_spread(float(spread))
        return round(ph, 4), round(1.0 - ph, 4)
    return None, None


def _from_scoreboard(games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for g in games:
        raw = g.get("_odds") or {}
        spread = raw.get("spread")
        total = raw.get("total")
        ml_home = raw.get("ml_home")
        ml_away = raw.get("ml_away")
        ih, ia = compute_probs(spread, ml_home, ml_away)
        out[g["game_id"]] = {
            "game_id": g["game_id"],
            "week": g["week"],
            "spread": spread,
            "total": float(total) if total is not None else None,
            "ml_home": int(ml_home) if ml_home is not None else None,
            "ml_away": int(ml_away) if ml_away is not None else None,
            "implied_home": ih,
            "implied_away": ia,
            "book": raw.get("book", "ESPN"),
        }
    return out


def _match_game(games: list[dict[str, Any]], home_name: str, away_name: str) -> Optional[dict[str, Any]]:
    hn, an = home_name.lower(), away_name.lower()
    for g in games:
        if g["home_full"].lower() in hn or hn in g["home_full"].lower():
            if g["away_full"].lower() in an or an in g["away_full"].lower():
                return g
    return None


def _from_theoddsapi(games: list[dict[str, Any]], api_key: str) -> dict[str, dict[str, Any]]:
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals,h2h",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    events = resp.json()
    out: dict[str, dict[str, Any]] = {}
    for ev in events:
        g = _match_game(games, ev.get("home_team", ""), ev.get("away_team", ""))
        if not g:
            continue
        book = (ev.get("bookmakers") or [{}])[0]
        spread = total = ml_home = ml_away = None
        for mk in book.get("markets", []):
            for oc in mk.get("outcomes", []):
                if mk["key"] == "spreads" and oc["name"] == ev["home_team"]:
                    spread = oc.get("point")
                elif mk["key"] == "totals" and oc["name"].lower() == "over":
                    total = oc.get("point")
                elif mk["key"] == "h2h" and oc["name"] == ev["home_team"]:
                    ml_home = oc.get("price")
                elif mk["key"] == "h2h" and oc["name"] == ev["away_team"]:
                    ml_away = oc.get("price")
        ih, ia = compute_probs(spread, ml_home, ml_away)
        out[g["game_id"]] = {
            "game_id": g["game_id"], "week": g["week"], "spread": spread,
            "total": total, "ml_home": ml_home, "ml_away": ml_away,
            "implied_home": ih, "implied_away": ia,
            "book": book.get("title", "the-odds-api"),
        }
    return out


def _from_sportsgameodds(games: list[dict[str, Any]], api_key: str) -> dict[str, dict[str, Any]]:
    url = "https://api.sportsgameodds.com/v2/events"
    params = {"apiKey": api_key, "leagueID": "NFL", "type": "match", "oddsAvailable": "true"}
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    out: dict[str, dict[str, Any]] = {}
    for ev in payload.get("data", []):
        teams = ev.get("teams", {})
        home_name = teams.get("home", {}).get("names", {}).get("long", "")
        away_name = teams.get("away", {}).get("names", {}).get("long", "")
        g = _match_game(games, home_name, away_name)
        if not g:
            continue
        odds = ev.get("odds", {})
        spread = odds.get("points_spread_home") or odds.get("spread_home")
        total = odds.get("points_total_over") or odds.get("total")
        ml_home = odds.get("moneyline_home")
        ml_away = odds.get("moneyline_away")
        ih, ia = compute_probs(spread, ml_home, ml_away)
        out[g["game_id"]] = {
            "game_id": g["game_id"], "week": g["week"], "spread": spread,
            "total": total, "ml_home": ml_home, "ml_away": ml_away,
            "implied_home": ih, "implied_away": ia, "book": "sportsgameodds",
        }
    return out


def get_week_odds(games: list[dict[str, Any]], provider: str, api_key: str) -> dict[str, dict[str, Any]]:
    """game_id -> odds dict. Falls back to scoreboard odds if a keyed provider fails."""
    provider = (provider or "espn").lower()
    if provider == "espn" or not api_key:
        return _from_scoreboard(games)
    try:
        if provider == "theoddsapi":
            merged = _from_scoreboard(games)
            merged.update(_from_theoddsapi(games, api_key))
            return merged
        if provider == "sportsgameodds":
            merged = _from_scoreboard(games)
            merged.update(_from_sportsgameodds(games, api_key))
            return merged
    except requests.RequestException as exc:
        log.warning("odds provider %s failed (%s); using scoreboard odds", provider, exc)
    return _from_scoreboard(games)
