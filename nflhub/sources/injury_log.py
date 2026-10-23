"""Side-by-side log of the two QB injury sources: ESPN's live feed (what the live prediction gate reads) and nflverse's weekly
injury report (what every backtest and the QB-gap weights were fit on).

Why: the live gate uses ESPN because nflverse's report_status often stays blank until the Friday report, but the weights were
validated against nflverse. After a season of both, the two can be compared: how often they agree on "QB out/doubtful", how much
earlier ESPN knows, and whether the ones they disagree on matter. ESPN has no historical archive, so this log is the only way to
get that comparison.

Stored in kv `injury_source_log` as {"<season>-<week>": {team: [entry, ...]}}; an entry is
{"t": iso time, "espn": [{"name", "status"}], "nflverse": [{"name", "status"}]} for that team's QBs, appended only when either side
changed since the team's previous entry. nflverse is fetched at most once an hour. Best-effort: never blocks the refresh.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone

import requests

from .edge_teams import UnknownTeamError, normalize_team
from .team_ratings import INJURIES_URL, TIMEOUT

log = logging.getLogger(__name__)

LOG_KEY = "injury_source_log"
CHECK_KEY = "injury_source_log_nflverse_checked_at"
NFLVERSE_EVERY_SECONDS = 3600
INTEREST = ("Out", "Doubtful", "Questionable", "Injured Reserve", "IR")   # statuses worth recording; healthy QBs are skipped


def _espn_qbs(injuries: dict[str, list[dict]]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for team, entries in (injuries or {}).items():
        qbs = [{"name": e.get("player", ""), "status": e.get("status", "")} for e in entries if e.get("position") == "QB"]
        if qbs:
            out[team] = sorted(qbs, key=lambda q: q["name"])
    return out


def _nflverse_qbs(season: int, week: int) -> dict[str, list[dict]]:
    resp = requests.get(INJURIES_URL.format(season=season), timeout=TIMEOUT)
    resp.raise_for_status()
    out: dict[str, list[dict]] = {}
    for row in csv.DictReader(io.StringIO(resp.text)):
        if row.get("game_type") != "REG" or row.get("position") != "QB" or str(row.get("week")) != str(week):
            continue
        status = row.get("report_status") or ""
        if status not in INTEREST:
            continue
        try:
            team = normalize_team(row["team"])
        except (UnknownTeamError, KeyError):
            continue
        out.setdefault(team, []).append({"name": row.get("full_name", ""), "status": status})
    return {t: sorted(v, key=lambda q: q["name"]) for t, v in out.items()}


def record(store, season: int, week: int, injuries: dict[str, list[dict]]) -> str:
    """Append this run's snapshot for the current week. Returns a short summary for the refresh log."""
    now = datetime.now(timezone.utc)
    espn = _espn_qbs(injuries)
    espn = {t: [q for q in v if q["status"] in INTEREST] for t, v in espn.items()}
    espn = {t: v for t, v in espn.items() if v}

    last = store.kv_get(CHECK_KEY)
    nfl_fresh = True
    if last:
        try:
            nfl_fresh = (now - datetime.fromisoformat(last)).total_seconds() >= NFLVERSE_EVERY_SECONDS
        except ValueError:
            pass
    raw = store.kv_get(LOG_KEY)
    try:
        blob = json.loads(raw) if raw else {}
    except ValueError:
        blob = {}
    key = f"{season}-{week}"
    wk = blob.setdefault(key, {})

    if nfl_fresh:
        nfl = _nflverse_qbs(season, week)
        store.kv_set(CHECK_KEY, now.isoformat())
    else:
        # reuse the nflverse side of each team's latest entry so a throttled run can still log an ESPN-only change
        nfl = {t: (entries[-1].get("nflverse") or []) for t, entries in wk.items() if entries and entries[-1].get("nflverse")}

    added = 0
    for team in sorted(set(espn) | set(nfl) | set(wk)):
        cur = {"espn": espn.get(team, []), "nflverse": nfl.get(team, [])}
        entries = wk.setdefault(team, [])
        if entries and entries[-1].get("espn") == cur["espn"] and entries[-1].get("nflverse") == cur["nflverse"]:
            continue
        if not entries and not cur["espn"] and not cur["nflverse"]:
            wk.pop(team, None)
            continue
        entries.append({"t": now.isoformat(timespec="seconds"), **cur})
        added += 1
    if added:
        store.kv_set(LOG_KEY, json.dumps(blob))
    return f"{added} change(s) logged for wk{week}"
