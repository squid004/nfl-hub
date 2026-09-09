"""FantasyPros weekly projections + Expert Consensus Rankings.

Verified endpoints (shared with the ff-draft-edge project):
    base:  https://api.fantasypros.com/public/v2/json
    auth:  x-api-key header
    GET /nfl/{season}/projections        ?position=ALL&week=N&scoring=PPR|HALF|STD
    GET /nfl/{season}/consensus-rankings ?position=ALL&week=N&scoring=...&type=weekly

The API has no league sync and no lineup read/write; it is a data feed only. The
"what FantasyPros would start" lineup is computed locally in nflhub/optimizer.py from
these numbers applied to the roster we already pull from Yahoo/ESPN.

A non-production key truncates every board to 10 rows (`count` > len(players)); that case
raises FreeTierLimited so we never optimize off a partial board.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import requests

from ..config import Config

log = logging.getLogger(__name__)
TIMEOUT = 30

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.IGNORECASE)
_NONWORD = re.compile(r"[^a-z0-9 ]")


class FreeTierLimited(RuntimeError):
    pass


def _norm_name(name: str) -> str:
    n = (name or "").lower().replace(".", "").replace("'", "").replace("-", " ")
    n = _SUFFIX.sub("", n)
    n = _NONWORD.sub("", n)
    return re.sub(r"\s+", " ", n).strip()


def _session(cfg: Config) -> requests.Session:
    s = requests.Session()
    s.headers.update({"x-api-key": cfg.fantasypros.api_key, "Accept": "application/json"})
    return s


def _get(cfg: Config, sess: requests.Session, path: str, params: dict[str, Any]) -> dict:
    url = f"{cfg.fantasypros.base_url}/{path.lstrip('/')}"
    resp = sess.get(url, params=params, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {resp.url} -> {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    rows = data.get("players")
    count = data.get("count")
    if isinstance(rows, list) and count is not None:
        try:
            if len(rows) < int(count):
                raise FreeTierLimited(
                    f"{resp.url} returned {len(rows)}/{count} rows (tier={data.get('tier')!r}); "
                    "a production FantasyPros key is required."
                )
        except (TypeError, ValueError):
            pass
    return data


def _points(stats: dict, scoring: str) -> Optional[float]:
    if not isinstance(stats, dict):
        return None
    order = {
        "PPR": ("points_ppr", "points", "points_half"),
        "HALF": ("points_half", "points_ppr", "points"),
        "STD": ("points", "points_half", "points_ppr"),
    }.get(scoring, ("points_ppr", "points", "points_half"))
    for k in order:
        v = stats.get(k)
        if v not in (None, "", "0", 0):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _index(rows: list[dict], value_fn) -> dict[str, dict]:
    """Multi-key index so a roster player can be matched by id, name+team, name, or team-DST."""
    idx: dict[str, dict] = {}
    for r in rows:
        rec = value_fn(r)
        if rec is None:
            continue
        name = r.get("player_name") or r.get("name") or ""
        team = (r.get("player_team_id") or r.get("team_id") or r.get("team") or "").upper()
        pos = (r.get("player_position_id") or r.get("position_id") or r.get("position") or "").upper()
        yid = r.get("player_yahoo_id") or r.get("yahoo_id")
        nn = _norm_name(name)
        keys = [f"n:{nn}"]
        if team:
            keys.append(f"nt:{nn}:{team}")
        if yid:
            keys.append(f"y:{yid}")
        if pos in {"DST", "DEF"} and team:
            keys.append(f"def:{team}")
        for k in keys:
            idx.setdefault(k, rec)
    return idx


def weekly_projections(cfg: Config, season: int, week: int, scoring: str) -> dict[str, dict]:
    sess = _session(cfg)
    data = _get(cfg, sess, f"nfl/{season}/projections",
                {"position": "ALL", "week": week, "scoring": scoring})
    return _index(
        data.get("players", []),
        lambda r: (
            {"points": _points(r.get("stats", {}), scoring), "src": "fp"}
            if _points(r.get("stats", {}), scoring) is not None
            else None
        ),
    )


def weekly_ecr(cfg: Config, season: int, week: int, scoring: str) -> dict[str, dict]:
    sess = _session(cfg)
    data = _get(cfg, sess, f"nfl/{season}/consensus-rankings",
                {"position": "ALL", "week": week, "scoring": scoring, "type": "weekly"})

    def rec(r: dict) -> Optional[dict]:
        try:
            ecr = int(float(r.get("rank_ecr"))) if r.get("rank_ecr") not in (None, "") else None
        except (TypeError, ValueError):
            ecr = None
        return {"ecr": ecr, "tier": r.get("tier"), "pos_rank": r.get("pos_rank")}

    return _index(data.get("players", []), rec)


def _lookup(idx: dict[str, dict], player: dict) -> Optional[dict]:
    nn = _norm_name(player.get("name", ""))
    team = (player.get("pro_team") or "").upper()
    pos = (player.get("position") or "").upper()
    for key in (
        f"y:{player.get('yahoo_id')}" if player.get("yahoo_id") else None,
        f"nt:{nn}:{team}" if team else None,
        f"def:{team}" if ("DEF" in pos or "D/ST" in pos or "DST" in pos) and team else None,
        f"n:{nn}",
    ):
        if key and key in idx:
            return idx[key]
    return None


def enrich(payload: dict, proj: dict[str, dict], ecr: dict[str, dict]) -> int:
    """Attach fp_points / fp_ecr / fp_tier to every starter and bench player. Returns match count."""
    matched = 0
    for group in ("starters", "bench"):
        for p in payload.get(group, []):
            pr = _lookup(proj, p)
            er = _lookup(ecr, p)
            if pr and pr.get("points") is not None:
                p["fp_points"] = round(float(pr["points"]), 2)
                matched += 1
            if er:
                p["fp_ecr"] = er.get("ecr")
                p["fp_tier"] = er.get("tier")
    return matched
