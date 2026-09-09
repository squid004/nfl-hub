"""ESPN fantasy league snapshot via the community `espn-api` package (read only)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .. import store
from ..config import Config

log = logging.getLogger(__name__)

BENCH_SLOTS = {"BE", "Bench", "IR", "IR/DL"}

# ESPN fantasy pro-team abbreviations that differ from the site scoreboard.
_ABBR_ALIAS = {"WAS": "WSH", "JAC": "JAX", "OAK": "LV", "LA": "LAR"}


def _norm_abbr(abbr: str) -> str:
    return _ABBR_ALIAS.get(abbr, abbr)


def _kickoff_index(week_games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for g in week_games:
        idx[g["home"]] = {"kickoff": g["kickoff"], "state": g["state"], "opp": g["away"]}
        idx[g["away"]] = {"kickoff": g["kickoff"], "state": g["state"], "opp": g["home"]}
    return idx


def _player_dict(bp: Any, kindex: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pro = _norm_abbr(getattr(bp, "proTeam", "") or "")
    sched = kindex.get(pro, {})
    game_date = getattr(bp, "game_date", None)
    kickoff = game_date.isoformat() if hasattr(game_date, "isoformat") else sched.get("kickoff")
    return {
        "name": getattr(bp, "name", "?"),
        "position": getattr(bp, "position", "") or "",
        "elig": [getattr(bp, "position", "") or ""],
        "slot": getattr(bp, "slot_position", "") or "",
        "pro_team": pro,
        "opponent": getattr(bp, "pro_opponent", "") or sched.get("opp", ""),
        "kickoff": kickoff,
        "injury_status": (getattr(bp, "injuryStatus", "") or "").upper(),
        "points": float(getattr(bp, "points", 0) or 0),
        "projected": float(getattr(bp, "projected_points", 0) or 0),
        "game_state": sched.get("state"),
        "on_bye": pro not in kindex,
    }


def _alerts(starters: list[dict[str, Any]]) -> list[str]:
    out = []
    for p in starters:
        if p["on_bye"]:
            out.append(f"{p['name']} ({p['slot']}) is on BYE")
        elif p["injury_status"] in {"OUT", "DOUBTFUL", "SUSPENSION", "INJURY_RESERVE"}:
            out.append(f"{p['name']} ({p['slot']}) is {p['injury_status'].title()}")
    return out


def build_snapshot(cfg: Config, season: int, week: int, week_games: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not cfg.espn.enabled:
        return None
    try:
        from espn_api.football import League
    except ImportError:
        log.error("espn-api not installed; run: pip install -r requirements.txt")
        return None

    league = League(
        league_id=cfg.espn.league_id,
        year=season,
        espn_s2=cfg.espn.s2 or None,
        swid=cfg.espn.swid or None,
    )
    kindex = _kickoff_index(week_games)

    box = None
    for b in league.box_scores(week):
        if getattr(b.home_team, "team_id", None) == cfg.espn.team_id:
            box = (b, "home")
            break
        if getattr(b.away_team, "team_id", None) == cfg.espn.team_id:
            box = (b, "away")
            break
    if box is None:
        log.warning("ESPN team_id %s not found in week %s box scores", cfg.espn.team_id, week)
        return None

    b, side = box
    if side == "home":
        me_lineup, opp_lineup = b.home_lineup, b.away_lineup
        me_team, opp_team = b.home_team, b.away_team
        me_score, opp_score = b.home_score, b.away_score
    else:
        me_lineup, opp_lineup = b.away_lineup, b.home_lineup
        me_team, opp_team = b.away_team, b.home_team
        me_score, opp_score = b.away_score, b.home_score

    players = [_player_dict(p, kindex) for p in me_lineup]
    starters = [p for p in players if p["slot"] not in BENCH_SLOTS]
    bench = [p for p in players if p["slot"] in BENCH_SLOTS]
    opp_players = [_player_dict(p, kindex) for p in opp_lineup]

    snapshot = {
        "league": "espn",
        "week": week,
        "team_name": getattr(me_team, "team_name", "My Team"),
        "opponent_name": getattr(opp_team, "team_name", "Opponent"),
        "score": round(float(me_score or 0), 2),
        "opponent_score": round(float(opp_score or 0), 2),
        "projected": round(sum(p["projected"] for p in starters), 2),
        "opponent_projected": round(
            sum(p["projected"] for p in opp_players if p["slot"] not in BENCH_SLOTS), 2
        ),
        "starters": starters,
        "bench": bench,
        "alerts": _alerts(starters),
    }
    store.save_roster_snapshot("espn", week, snapshot)
    return snapshot
