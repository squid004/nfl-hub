"""Live pool picks: every entrant's picks that are visible mid-week, merged from independent sources.

Why: picks become visible only after each game day's first kickoff, and the pool optimizer needs them right then
(not on Tuesday when the final sheet tab is written). Two sources feed the same store, for redundancy:
  * "sheet" -- the optional "Week N live" tab in the pool sheet, written by a scraper/Cowork run
    (edge_sheet.fetch_live_week);
  * "cbs"   -- a direct, read-only CBS fetch from the refresh job (cbs_pool.py), needs the CBS_COOKIE secret; rate-limited to one
    fetch per CBS_MIN_MINUTES. Add another source by writing one function that returns a Snapshot and registering it in SOURCES.

Each run collects every source that works and merges them (collect()). Deduplication: a pick that several sources
agree on is one pick; if two snapshots disagree about the same entrant's pick in the same game, the later capture
wins and the disagreement is recorded in `conflicts` (locked picks cannot legitimately change, so a conflict on a
locked game means a bad scrape). A source that fails (or is skipped by its rate limit) is recorded with its error, keeps its previous snapshot for this week (so a
transient failure never erases data already captured), and never blocks the others.
Stored in kv `pool_live_picks`; read back with load().
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .. import store
from . import cbs_pool, edge_sheet
from .edge_teams import normalize_team

log = logging.getLogger(__name__)

KEY = "pool_live_picks"


@dataclass
class Snapshot:
    source: str
    captured_at: datetime | None
    picks: dict[str, list[str]]       # entrant -> teams visible at capture time
    tiebreaks: dict[str, float] = field(default_factory=dict)   # entrant -> Monday-night total guess, where visible
    hidden: dict[str, int] = field(default_factory=dict)        # entrant -> picks still hidden (placeholder "locked") at capture


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _sheet_source(cfg, week: int) -> Snapshot:
    data = edge_sheet.fetch_live_week(cfg.edge.sheet_id, week, normalize_team)
    return Snapshot("sheet", _parse_ts(data.captured_at), data.picks, hidden=data.hidden)


CBS_MIN_MINUTES = 20
CBS_LAST_FETCH_KEY = "cbs_pool_last_fetch"


def _cbs_source(cfg, week: int) -> Snapshot | None:
    """None = skipped by the rate limit (the previous snapshot is kept)."""
    if not getattr(cfg.edge, "cbs_cookie", ""):
        raise cbs_pool.CbsPoolUnavailable("CBS_COOKIE not set")
    last = store.kv_get(CBS_LAST_FETCH_KEY)
    now = datetime.now(timezone.utc)
    if last:
        t = _parse_ts(last)
        if t and (now - t).total_seconds() < CBS_MIN_MINUTES * 60:
            return None
    store.kv_set(CBS_LAST_FETCH_KEY, now.isoformat())   # set before the call so a failing fetch is rate-limited too
    return cbs_pool.fetch(cfg, week)


SOURCES: dict[str, Callable[[Any, int], Snapshot | None]] = {"sheet": _sheet_source, "cbs": _cbs_source}


def merge(snapshots: list[Snapshot], game_of_team: dict[str, str]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Team-level merge keyed by game: later captures override earlier ones; disagreements are reported."""
    ordered = sorted(snapshots, key=lambda s: s.captured_at or datetime.min.replace(tzinfo=timezone.utc))
    by_entrant: dict[str, dict[str, tuple[str, str]]] = {}    # entrant -> game key -> (team, source)
    conflicts: list[dict[str, Any]] = []
    for snap in ordered:
        for name, teams in snap.picks.items():
            slot = by_entrant.setdefault(name, {})
            for t in teams:
                gk = game_of_team.get(t, t)
                prev = slot.get(gk)
                if prev and prev[0] != t:
                    conflicts.append({"entrant": name, "game": gk, "earlier": {"source": prev[1], "team": prev[0]},
                                      "later": {"source": snap.source, "team": t}})
                slot[gk] = (t, snap.source)
    return {n: [t for t, _src in slot.values()] for n, slot in by_entrant.items()}, conflicts


def collect(cfg, season: int, week: int, game_of_team: dict[str, str]) -> dict[str, Any] | None:
    """Run every source, merge, store (only when something changed). Returns the stored blob or None if no source had data."""
    snaps: list[Snapshot] = []
    status: dict[str, dict[str, Any]] = {}
    prev_live = load(season, week)
    prev_by_source = {s.source: s for s in (prev_live.snapshots if prev_live else [])}
    for name, fn in SOURCES.items():
        err = None
        try:
            s = fn(cfg, week)
        except Exception as exc:  # noqa: BLE001 - one source failing must never block the others
            s, err = None, str(exc)[:200]
        if s is None and name in prev_by_source:
            s = prev_by_source[name]                      # keep the last good capture (stale, but its timestamp says so)
            status[name] = {"ok": err is None, "stale": True, "captured_at": s.captured_at.isoformat() if s.captured_at else None,
                            "entrants": len(s.picks), "error": err}
            snaps.append(s)
        elif s is None:
            status[name] = {"ok": False, "stale": False, "captured_at": None, "entrants": 0, "error": err or "skipped"}
        else:
            snaps.append(s)
            status[name] = {"ok": True, "stale": False, "captured_at": s.captured_at.isoformat() if s.captured_at else None,
                            "entrants": len(s.picks), "error": None}
    if not snaps:
        return None
    merged, conflicts = merge(snaps, game_of_team)
    tiebreaks: dict[str, float] = {}
    for s in sorted(snaps, key=lambda x: x.captured_at or datetime.min.replace(tzinfo=timezone.utc)):
        tiebreaks.update(s.tiebreaks)
    stamps = [s.captured_at for s in snaps if s.captured_at]
    blob = {
        "season": season, "week": week,
        "captured_at": max(stamps).isoformat() if stamps else None,
        "sources": status, "picks": merged, "tiebreaks": tiebreaks, "conflicts": conflicts,
        "snapshots": [{"source": s.source, "captured_at": s.captured_at.isoformat() if s.captured_at else None,
                       "picks": s.picks, "tiebreaks": s.tiebreaks, "hidden": s.hidden} for s in snaps],
    }
    prev = _raw()
    if not prev or {k: v for k, v in prev.items() if k != "updated"} != blob:
        store.kv_set(KEY, json.dumps({**blob, "updated": datetime.now(timezone.utc).isoformat()}))
        blob["updated"] = datetime.now(timezone.utc).isoformat()
    else:
        blob["updated"] = prev.get("updated")
    return blob


def _raw() -> dict[str, Any] | None:
    raw = store.kv_get(KEY)
    try:
        return json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None


@dataclass
class Live:
    picks: dict[str, list[str]]
    snapshots: list[Snapshot] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    updated: str | None = None
    tiebreaks: dict[str, float] = field(default_factory=dict)

    @property
    def captured_at(self) -> datetime | None:
        stamps = [s.captured_at for s in self.snapshots if s.captured_at]
        return max(stamps) if stamps else None

    def fresh(self, day_first: datetime | None) -> bool:
        """True if some source captured AFTER this game day's first kickoff, i.e. the day's picks were visible."""
        return bool(day_first and self.captured_at and self.captured_at >= day_first)

    def consistent_with(self, games_not_started: int) -> bool:
        """For tabs with no capture timestamp: CBS hides a pick until its game kicks off and the scraper writes the placeholder
        "locked" there. If (nearly) every entrant's placeholder count equals the number of games that have NOT started yet,
        the tab reflects the current lock state, so a started game with no pick shown really has no pick."""
        for s in self.snapshots:
            if s.hidden:
                # +1: the Monday-night tiebreaker guess is hidden too, until Monday's kickoff
                ok = sum(1 for n in s.hidden.values() if n in (games_not_started, games_not_started + 1))
                if ok >= 0.9 * len(s.hidden):
                    return True
        return False


def load(season: int, week: int) -> Live | None:
    b = _raw()
    if not b or b.get("season") != season or b.get("week") != week:
        return None
    snaps = [Snapshot(s["source"], _parse_ts(s.get("captured_at")), s.get("picks", {}), s.get("tiebreaks", {}), s.get("hidden", {}))
             for s in b.get("snapshots", [])]
    return Live(picks=b.get("picks", {}), snapshots=snaps, conflicts=b.get("conflicts", []),
                sources=b.get("sources", {}), updated=b.get("updated"), tiebreaks=b.get("tiebreaks", {}))
