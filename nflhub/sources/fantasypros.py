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


# --- QB-health fantasy-gap support (team_ratings.qb_fantasy_gap_data) --------------------
#
# Everything below is a SECOND, independent consumer of the same FantasyPros API, for a
# completely different purpose: team_ratings.py's QB-health correction (validated additive
# in research/edge_signal_test_v49/v50_fantasypros_qb_gap*.py), not the fantasy-pool lineup
# optimizer above. Deliberately uses its OWN auth (a direct os.environ read) instead of the
# Config-based _session()/_get() above -- team_ratings.refresh(store, injuries=...) has no
# Config object in its call chain (team_ratings.py is architecturally independent of
# nflhub.config throughout, reading every other external source the same direct way), and
# retrofitting that would be a much larger, riskier change than two small auth paths
# coexisting in one file for two different callers.
import os

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "team_ratings")
_API_KEY_ENV = os.environ.get("FANTASYPROS_API_KEY", "")
_ID_CROSSWALK_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"
FIRST_SEASON = 2017  # FantasyPros weekly projection history this correction trusts -- see
# research/edge_signal_test_v48's own data-integrity check (Puka Nacua's week-1 2023
# projection, 4.67 points/84th among WRs before his real breakout -- genuinely pre-game, not
# hindsight-informed) for why this is trusted at all; not tested/used before 2017.


def id_crosswalk() -> dict[str, int]:
    """gsis_id -> FantasyPros fpid. Cached permanently once fetched (a player's id mapping
    doesn't change); community-maintained (dynastyprocess), the same source nfl_data_py.
    import_ids() wraps -- fetched directly here instead of adding nfl_data_py as a
    dependency nfl-hub otherwise has no use for."""
    path = os.path.join(_DATA_DIR, "fantasypros_id_crosswalk.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(_ID_CROSSWALK_URL, usecols=["gsis_id", "fantasypros_id"])
        os.makedirs(_DATA_DIR, exist_ok=True)
        df.to_parquet(path)
    df = df.dropna(subset=["gsis_id", "fantasypros_id"])
    return dict(zip(df["gsis_id"], df["fantasypros_id"].astype(int)))


def _fetch_qb_week(season: int, week: int) -> list[dict]:
    if not _API_KEY_ENV:
        return []
    try:
        resp = requests.get(
            "https://api.fantasypros.com/public/v2/json" + f"/nfl/{season}/projections",
            params={"position": "QB", "week": week, "scoring": "HALF"},
            headers={"x-api-key": _API_KEY_ENV, "Accept": "application/json"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:  # noqa: BLE001 -- best-effort, same soft-fail convention as team_ratings.py
        log.warning("FantasyPros QB projections fetch failed for %s wk%s", season, week, exc_info=True)
        return []
    rows = []
    for p in data.get("players") or []:
        stats = p.get("stats") or {}
        pts = stats.get("points_half")
        if p.get("fpid") is not None and pts is not None:
            rows.append({"season": season, "week": week, "fpid": int(p["fpid"]), "points": float(pts)})
    return rows


def qb_projections_by_season(season: int, current_season: int, current_week: int) -> dict[tuple[int, int], float]:
    """{(week, fpid): projected half-PPR points} for every QB, every week of `season`. Same
    per-SEASON caching convention as team_ratings._passer_game_stats (one parquet per
    completed season, covering every week at once, committed to the repo since this CI
    runner's disk doesn't persist between runs) -- not per-week, to avoid 150+ tiny files.
    The CURRENT season always re-fetches whatever weeks aren't cached yet and appends them
    (the current week's projection can still change as injury news breaks through the week,
    same timeliness reasoning as the live QB Out/Doubtful gate elsewhere in that module).
    Soft-fails to {} per missing week (no adjustment, same as "no change this week") rather
    than raising -- this correction is additive on top of an already-complete model, not
    something refresh() should ever hard-fail over."""
    path = os.path.join(_DATA_DIR, f"fantasypros_qb_proj_{season}.parquet")
    cached = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=["season", "week", "fpid", "points"])
    is_complete_season = season < current_season
    if is_complete_season and not cached.empty:
        return {(int(r.week), int(r.fpid)): r.points for r in cached.itertuples(index=False)}

    weeks_needed = range(1, 19) if is_complete_season else range(1, (current_week if season == current_season else 18) + 1)
    have_weeks = set(cached["week"].unique()) if not cached.empty else set()
    new_rows = []
    for week in weeks_needed:
        if week in have_weeks:
            continue
        new_rows.extend(_fetch_qb_week(season, week))
    if new_rows:
        cached = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
    if is_complete_season and not cached.empty:
        os.makedirs(_DATA_DIR, exist_ok=True)
        cached.to_parquet(path)
    return {(int(r.week), int(r.fpid)): r.points for r in cached.itertuples(index=False)} if not cached.empty else {}
