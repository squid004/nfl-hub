"""NFL schedule, scores, injuries and news from ESPN's public site API (no key required)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
TIMEOUT = 15


def _get(path: str, params: Optional[dict] = None) -> dict:
    resp = requests.get(f"{BASE}/{path}", params=params or {}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _iso_utc(s: str) -> str:
    """ESPN timestamps come as '2025-09-05T00:20Z'; return a normalized UTC ISO string."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).isoformat()


def current_week() -> tuple[int, int]:
    """Return (season_year, week_number) for the current NFL week."""
    data = _get("scoreboard")
    season = int(data.get("season", {}).get("year") or datetime.now().year)
    week = int(data.get("week", {}).get("number") or 1)
    return season, week


def _parse_spread(odds_obj: dict, home_abbr: str, away_abbr: str) -> Optional[float]:
    """Normalize ESPN odds to a home-team point spread (negative = home favored)."""
    details = (odds_obj or {}).get("details")
    if isinstance(details, str) and details.strip():
        d = details.strip()
        if d.upper() in {"EVEN", "PK", "PICK"}:
            return 0.0
        # format: "KC -3.5"
        parts = d.rsplit(" ", 1)
        if len(parts) == 2:
            team, num = parts
            try:
                mag = float(num)
            except ValueError:
                mag = None
            if mag is not None:
                if team.strip() == home_abbr:
                    return mag if mag < 0 else -mag
                if team.strip() == away_abbr:
                    return abs(mag)
    # fallbacks
    raw = (odds_obj or {}).get("spread")
    if isinstance(raw, (int, float)):
        home_fav = (odds_obj.get("homeTeamOdds", {}) or {}).get("favorite")
        val = -abs(float(raw)) if home_fav else abs(float(raw))
        # if ESPN already signed it as a home spread, trust the sign when favorite flag is absent
        if home_fav is None:
            val = float(raw)
        return val
    return None


def fetch_scoreboard(season: int, week: int) -> list[dict[str, Any]]:
    """List of normalized game dicts for one week, including raw odds when present."""
    data = _get("scoreboard", {"seasontype": 2, "week": week, "dates": season})
    games: list[dict[str, Any]] = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        home_team = home.get("team", {})
        away_team = away.get("team", {})
        state = (comp.get("status", {}).get("type", {}) or {}).get("state", "pre")
        odds_list = comp.get("odds") or []
        odds_obj = odds_list[0] if odds_list else {}
        home_abbr = home_team.get("abbreviation", "")
        away_abbr = away_team.get("abbreviation", "")

        games.append(
            {
                "game_id": str(ev.get("id")),
                "season": season,
                "week": week,
                "kickoff": _iso_utc(ev.get("date")),
                "home": home_abbr,
                "away": away_abbr,
                "home_full": home_team.get("displayName", home_abbr),
                "away_full": away_team.get("displayName", away_abbr),
                "state": state,
                "home_score": int(home.get("score") or 0),
                "away_score": int(away.get("score") or 0),
                "_odds": {
                    "spread": _parse_spread(odds_obj, home_abbr, away_abbr),
                    "total": odds_obj.get("overUnder"),
                    "ml_home": (odds_obj.get("homeTeamOdds", {}) or {}).get("moneyLine"),
                    "ml_away": (odds_obj.get("awayTeamOdds", {}) or {}).get("moneyLine"),
                    "book": (odds_obj.get("provider", {}) or {}).get("name", "ESPN"),
                },
            }
        )
    return games


def fetch_news() -> list[dict[str, Any]]:
    """Recent league news; each item carries the athlete names ESPN tagged on it."""
    try:
        data = _get("news")
    except requests.RequestException as exc:  # news is non-critical
        log.warning("news fetch failed: %s", exc)
        return []
    out = []
    for art in data.get("articles", []):
        names = [
            c.get("description")
            for c in art.get("categories", [])
            if c.get("type") == "athlete" and c.get("description")
        ]
        out.append(
            {
                "id": str(art.get("id") or art.get("headline", ""))[:120],
                "published": art.get("published", ""),
                "headline": art.get("headline", ""),
                "description": art.get("description", ""),
                "athletes": names,
                "link": (art.get("links", {}).get("web", {}) or {}).get("href", ""),
            }
        )
    return out


def fetch_injuries() -> dict[str, list[dict[str, Any]]]:
    """Map of team abbreviation -> list of {player, status, detail}."""
    try:
        data = _get("injuries")
    except requests.RequestException as exc:
        log.warning("injuries fetch failed: %s", exc)
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for team_block in data.get("injuries", []):
        abbr = (team_block.get("team", {}) or {}).get("abbreviation") or team_block.get("abbreviation", "")
        entries = []
        for inj in team_block.get("injuries", []):
            entries.append(
                {
                    "player": (inj.get("athlete", {}) or {}).get("displayName", ""),
                    "status": inj.get("status", ""),
                    "detail": inj.get("shortComment") or inj.get("longComment") or "",
                }
            )
        if abbr:
            out[abbr] = entries
    return out
