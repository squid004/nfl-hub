"""Compute lineup / pick deadlines for the current week and decide which reminders are due."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from . import store
from .config import Config
from .sources import pickem, survivor
from .util import fmt_local


@dataclass
class Deadline:
    kind: str          # yahoo_lineup | espn_lineup | pickem | survivor
    label: str
    when: datetime     # timezone-aware UTC
    detail: str


def _iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def current_week() -> tuple[int, int]:
    season = store.kv_get("season")
    week = store.kv_get("week")
    if season and week:
        return int(season), int(week)
    from .sources import nfl_schedule

    return nfl_schedule.current_week()


def compute(cfg: Config, season: int, week: int) -> list[Deadline]:
    games = store.games_for_week(week)
    if not games:
        return []
    kicks = {g["game_id"]: _iso(g["kickoff"]) for g in games}
    first = min(kicks.values())
    first_local = fmt_local(first, cfg.timezone)

    out: list[Deadline] = [
        Deadline("pickem", "Pick'em", first, f"Week {week} pick'em locks at first kickoff ({first_local})."),
        Deadline("survivor", "Survivor", first, f"Week {week} survivor pick locks at first kickoff ({first_local})."),
    ]

    for league in ("yahoo", "espn"):
        snap = store.latest_roster_snapshot(league)
        if not snap or snap.get("week") != week:
            continue
        payload = snap["payload"]
        teams = {p.get("pro_team") for p in payload.get("starters", []) if p.get("pro_team")}
        relevant = [
            kicks[g["game_id"]] for g in games if g["home"] in teams or g["away"] in teams
        ]
        lock = min(relevant) if relevant else first
        detail = f"{payload.get('team_name', 'My team')} vs {payload.get('opponent_name', 'opponent')}."
        alerts = payload.get("alerts", [])
        if alerts:
            detail += " Problems: " + "; ".join(alerts[:3])
        fp = payload.get("fp")
        if fp and fp.get("delta", 0) >= 0.5 and fp.get("swaps"):
            tips = [
                f"start {s['start']} over {s['sit']} (+{s['gain']})"
                for s in fp["swaps"][:2]
                if s.get("sit")
            ]
            if tips:
                detail += f" FantasyPros +{fp['delta']} pts: " + "; ".join(tips)
        out.append(Deadline(f"{league}_lineup", f"{league.title()} lineup", lock, detail))

    return out


def _in_quiet_hours(now: datetime, tz: ZoneInfo, cfg: Config) -> bool:
    h = now.astimezone(tz).hour
    qs, qe = cfg.reminders.quiet_start, cfg.reminders.quiet_end
    return (qs <= h or h < qe) if qs > qe else (qs <= h < qe)


def _suppressed(kind: str, week: int) -> bool:
    """True if the pick/lineup work is already done, so no reminder is needed."""
    if kind == "pickem":
        s = pickem.weekly_slate(week)
        return bool(s["total_games"]) and s["made"] >= s["total_games"]
    if kind == "survivor":
        return survivor.status(week)["this_week_pick"] is not None
    return False


def due_reminders(cfg: Config, now: Optional[datetime] = None) -> list[dict[str, Any]]:
    """One entry per deadline that has a reminder to send right now."""
    now = now or datetime.now(timezone.utc)
    tz = ZoneInfo(cfg.timezone)
    season, week = current_week()
    offsets = sorted(cfg.reminders.offsets_hours)  # most urgent first

    fires: list[dict[str, Any]] = []
    for d in compute(cfg, season, week):
        if now >= d.when or _suppressed(d.kind, week):
            continue
        chosen = None
        for off in offsets:
            if d.when - timedelta(hours=off) <= now:
                chosen = off
                break
        if chosen is None:
            continue
        key = f"{week}:{chosen}"
        if store.reminder_already_sent(d.kind, key):
            continue
        hours_left = (d.when - now).total_seconds() / 3600.0
        if _in_quiet_hours(now, tz, cfg) and hours_left > 6:
            continue
        # firing this urgency also retires any less-urgent (larger) offsets we skipped past
        also_mark = [f"{week}:{o}" for o in offsets if o >= chosen]
        fires.append(
            {
                "kind": d.kind,
                "key": key,
                "also_mark": also_mark,
                "hours_left": round(hours_left, 2),
                "detail": d.detail,
                "when": d.when.isoformat(),
            }
        )
    return fires
