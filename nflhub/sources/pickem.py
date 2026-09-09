"""Weekly pick'em slate built from the NFL schedule + odds, with your recorded picks.

CBS has no usable pick'em API, so the slate is derived locally and you enter picks in the
dashboard. The reminder engine still nags you before the weekly lock.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import store


def weekly_slate(week: int) -> dict[str, Any]:
    games = store.games_for_week(week)
    odds = store.odds_for_week(week)
    picks = store.get_pickem_picks(week)

    rows = []
    for g in games:
        o = odds.get(g["game_id"], {})
        spread = o.get("spread")
        favorite = None
        if spread is not None:
            favorite = g["home"] if spread < 0 else (g["away"] if spread > 0 else "EVEN")
        rows.append(
            {
                "game_id": g["game_id"],
                "kickoff": g["kickoff"],
                "state": g["state"],
                "away": g["away"],
                "home": g["home"],
                "away_full": g["away_full"],
                "home_full": g["home_full"],
                "spread": spread,
                "total": o.get("total"),
                "favorite": favorite,
                "implied_home": o.get("implied_home"),
                "implied_away": o.get("implied_away"),
                "my_pick": picks.get(g["game_id"], {}).get("pick"),
                "confidence": picks.get(g["game_id"], {}).get("confidence"),
                "result": _result(g),
            }
        )
    deadline = min((g["kickoff"] for g in games), default=None)
    return {
        "week": week,
        "deadline": deadline,
        "rows": rows,
        "made": len(picks),
        "total_games": len(games),
    }


def _result(g: dict[str, Any]) -> Optional[str]:
    if g["state"] != "post":
        return None
    if g["home_score"] > g["away_score"]:
        return g["home"]
    if g["away_score"] > g["home_score"]:
        return g["away"]
    return "TIE"


def record_pick(week: int, game_id: str, team: str, confidence: Optional[int] = None) -> None:
    store.record_pickem_pick(week, game_id, team, confidence)
