"""Yahoo fantasy league snapshot via the official Fantasy Sports API (read only for v1).

Auth is a one-time OAuth2 consent handled by `python -m nflhub yahoo-auth`, which writes
oauth2.json. Setting the lineup (PUT team/{team_key}/roster) is intentionally not wired up
yet; see the plan's "out of scope for v1" section.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .. import store
from ..config import OAUTH_PATH, Config

log = logging.getLogger(__name__)

BENCH_POS = {"BN", "IR"}
_ABBR_ALIAS = {"WAS": "WSH", "WSH": "WSH", "JAC": "JAX"}
_STATUS_LABEL = {
    "Q": "QUESTIONABLE", "D": "DOUBTFUL", "O": "OUT", "IR": "INJURY_RESERVE",
    "SUSP": "SUSPENSION", "PUP": "PUP", "NA": "NOT_ACTIVE", "COVID-19": "OUT",
}


def oauth_ready() -> bool:
    if not OAUTH_PATH.exists():
        return False
    try:
        data = json.loads(OAUTH_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(data.get("refresh_token"))


def _session():
    from yahoo_oauth import OAuth2

    sc = OAuth2(None, None, from_file=str(OAUTH_PATH))
    if not sc.token_is_valid():
        sc.refresh_access_token()
    return sc


def _norm_abbr(abbr: str) -> str:
    a = (abbr or "").upper()
    return _ABBR_ALIAS.get(a, a)


def _resolve_team_key(cfg: Config, lg) -> Optional[str]:
    if cfg.yahoo.team_key:
        return cfg.yahoo.team_key
    cached = store.kv_get("yahoo_team_key")
    if cached:
        return cached
    try:
        tk = lg.team_key()
        store.kv_set("yahoo_team_key", tk)
        return tk
    except Exception as exc:  # noqa: BLE001 - surface as a warning, keep refresh alive
        log.warning("could not resolve Yahoo team_key: %s", exc)
        return None


def _parse_matchup(raw: dict, my_team_key: str) -> dict[str, Any]:
    """Pull my/opponent points and projected points out of lg.matchups() JSON."""
    out = {"score": 0.0, "opponent_score": 0.0, "projected": 0.0,
           "opponent_projected": 0.0, "opponent_name": "Opponent", "team_name": "My Team"}
    try:
        league = raw["fantasy_content"]["league"]
        scoreboard = next(x for x in league if isinstance(x, dict) and "scoreboard" in x)
        matchups = scoreboard["scoreboard"]["0"]["matchups"]
    except (KeyError, StopIteration, TypeError):
        return out

    for key, mval in matchups.items():
        if key == "count":
            continue
        teams = mval["matchup"]["0"]["teams"]
        parsed = []
        for tkey, tval in teams.items():
            if tkey == "count":
                continue
            meta = tval["team"][0]
            team_key = next(d["team_key"] for d in meta if isinstance(d, dict) and "team_key" in d)
            name = next((d["name"] for d in meta if isinstance(d, dict) and "name" in d), "Team")
            stats = tval["team"][1]
            pts = float(stats.get("team_points", {}).get("total", 0) or 0)
            proj = float(stats.get("team_projected_points", {}).get("total", 0) or 0)
            parsed.append({"team_key": team_key, "name": name, "pts": pts, "proj": proj})
        if any(p["team_key"] == my_team_key for p in parsed) and len(parsed) == 2:
            me = next(p for p in parsed if p["team_key"] == my_team_key)
            opp = next(p for p in parsed if p["team_key"] != my_team_key)
            return {
                "score": round(me["pts"], 2),
                "opponent_score": round(opp["pts"], 2),
                "projected": round(me["proj"], 2),
                "opponent_projected": round(opp["proj"], 2),
                "opponent_name": opp["name"],
                "team_name": me["name"],
            }
    return out


def _kickoff_index(week_games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for g in week_games:
        idx[g["home"]] = {"kickoff": g["kickoff"], "state": g["state"], "opp": g["away"]}
        idx[g["away"]] = {"kickoff": g["kickoff"], "state": g["state"], "opp": g["home"]}
    return idx


def build_snapshot(cfg: Config, season: int, week: int, week_games: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not cfg.yahoo.enabled:
        return None
    if not oauth_ready():
        log.info("Yahoo OAuth not set up; run: python -m nflhub yahoo-auth")
        return None
    try:
        import yahoo_fantasy_api as yfa
    except ImportError:
        log.error("yahoo_fantasy_api not installed; run: pip install -r requirements.txt")
        return None

    sc = _session()
    gm = yfa.Game(sc, "nfl")
    lg = gm.to_league(cfg.yahoo.league_id)
    team_key = _resolve_team_key(cfg, lg)
    if not team_key:
        return None

    tm = lg.to_team(team_key)
    roster = tm.roster(week)
    pids = [p["player_id"] for p in roster]
    details = {d["player_id"]: d for d in lg.player_details(pids)} if pids else {}
    kindex = _kickoff_index(week_games)

    def to_player(rp: dict) -> dict[str, Any]:
        det = details.get(rp["player_id"], {})
        pro = _norm_abbr(det.get("editorial_team_abbr", ""))
        sched = kindex.get(pro, {})
        bye = det.get("bye_weeks", {}).get("week")
        status_raw = rp.get("status") or det.get("status") or ""
        return {
            "name": rp.get("name", "?"),
            "yahoo_id": rp.get("player_id"),
            "position": "/".join(rp.get("eligible_positions", [])) or rp.get("position_type", ""),
            "elig": [x for x in rp.get("eligible_positions", []) if x in {"QB", "RB", "WR", "TE", "K", "DEF"}],
            "slot": rp.get("selected_position", ""),
            "pro_team": pro,
            "opponent": sched.get("opp", ""),
            "kickoff": sched.get("kickoff"),
            "injury_status": _STATUS_LABEL.get(status_raw, status_raw.upper()),
            "points": 0.0,
            "projected": 0.0,
            "game_state": sched.get("state"),
            "on_bye": str(bye) == str(week) if bye else (pro not in kindex),
        }

    players = [to_player(p) for p in roster]
    starters = [p for p in players if p["slot"] not in BENCH_POS]
    bench = [p for p in players if p["slot"] in BENCH_POS]

    mu = _parse_matchup(lg.matchups(week), team_key)
    alerts = []
    for p in starters:
        if p["on_bye"]:
            alerts.append(f"{p['name']} ({p['slot']}) is on BYE")
        elif p["injury_status"] in {"OUT", "DOUBTFUL", "SUSPENSION", "INJURY_RESERVE"}:
            alerts.append(f"{p['name']} ({p['slot']}) is {p['injury_status'].title()}")

    snapshot = {
        "league": "yahoo",
        "week": week,
        "team_name": mu["team_name"],
        "opponent_name": mu["opponent_name"],
        "score": mu["score"],
        "opponent_score": mu["opponent_score"],
        "projected": mu["projected"],
        "opponent_projected": mu["opponent_projected"],
        "starters": starters,
        "bench": bench,
        "alerts": alerts,
    }
    store.save_roster_snapshot("yahoo", week, snapshot)
    return snapshot
