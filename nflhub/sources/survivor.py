"""Survivor pool state: which teams you have burned, and this week's best remaining picks.

Yahoo exposes no survivor API, so your weekly pick is recorded in the dashboard and the
"already used" set is derived from that history.
"""

from __future__ import annotations

from typing import Any

from .. import store


def status(week: int) -> dict[str, Any]:
    history = store.get_survivor_picks()          # {week: team}
    used_prior = {t for w, t in history.items() if w < week}
    this_week_pick = history.get(week)

    games = store.games_for_week(week)
    odds = store.odds_for_week(week)

    options = []
    for g in games:
        o = odds.get(g["game_id"], {})
        for side, team, opp, prob in (
            ("home", g["home"], g["away"], o.get("implied_home")),
            ("away", g["away"], g["home"], o.get("implied_away")),
        ):
            options.append(
                {
                    "team": team,
                    "team_full": g["home_full"] if side == "home" else g["away_full"],
                    "opponent": opp,
                    "home_away": side,
                    "kickoff": g["kickoff"],
                    "win_prob": prob,
                    "used": team in used_prior,
                    "is_pick": team == this_week_pick,
                }
            )
    options.sort(key=lambda r: (r["win_prob"] is None, -(r["win_prob"] or 0)))

    return {
        "week": week,
        "deadline": min((g["kickoff"] for g in games), default=None),
        "used_teams": sorted(used_prior),
        "history": dict(sorted(history.items())),
        "this_week_pick": this_week_pick,
        "options": options,
    }


def record_pick(week: int, team: str) -> None:
    store.record_survivor_pick(week, team)
