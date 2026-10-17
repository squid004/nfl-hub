"""ntfy reminders for the pool optimizer: one when a lock window opens, one last call before the lock.

A window = the next game day that still has open (unlocked) games. It opens when every earlier game day of the week
is final (so the Sunday card is advised after Thursday's result, the Monday pick and tiebreaker after Sunday's) and the
lock is within WINDOW_HOURS. Notifications only advise -- the project never submits picks. Deduped through the same
reminder_log table the deadline reminders use (kind "pool_card").

Message: the recommended changes against the user's CURRENT picks (the CBS pick when a live scrape has it, else the
in-app pick), the tiebreaker guess, the week's win chance, and any app-vs-CBS pick mismatches.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .. import notify, store
from ..deadlines import _in_quiet_hours

log = logging.getLogger(__name__)

KIND = "pool_card"
WINDOW_HOURS = 40.0
LAST_CALL_HOURS = 3.0


def _kick(g) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(g.get("kickoff")))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def maybe_notify(cfg, rec: dict | None, games: list[dict], now: datetime | None = None) -> str:
    if not rec or rec.get("week") is None:
        return "no recommendation"
    now = now or datetime.now(timezone.utc)
    tz = ZoneInfo(cfg.timezone)
    by_id = {g["game_id"]: g for g in games}
    day_of: dict[str, str] = {}
    first_kick: dict[str, datetime] = {}
    for gid in rec["games"]:
        g = by_id.get(gid)
        k = _kick(g) if g else None
        if not k:
            continue
        d = k.astimezone(tz).date().isoformat()
        day_of[gid] = d
        if d not in first_kick or k < first_kick[d]:
            first_kick[d] = k
    open_days = sorted({day_of[gid] for gid, v in rec["games"].items() if not v.get("locked") and gid in day_of})
    if not open_days:
        return "nothing open"
    day = open_days[0]
    earlier_unfinished = [gid for gid, v in rec["games"].items()
                          if gid in day_of and day_of[gid] < day and not v.get("final")]
    if earlier_unfinished:
        return "waiting for earlier game day to finish"
    hours_left = (first_kick[day] - now).total_seconds() / 3600.0
    if hours_left <= 0:
        return "locked"
    if hours_left > WINDOW_HOURS:
        return f"window not open ({hours_left:.0f}h to lock)"

    open_games = {gid: v for gid, v in rec["games"].items() if not v.get("locked") and day_of.get(gid) == day}
    changes = []
    for gid, v in open_games.items():
        cur = v.get("cbs_pick") or v.get("app_pick")
        if cur != v["pick"]:
            changes.append((v["gain_dollars"], gid, v, cur))
    changes.sort(key=lambda t: -t[0])
    flags = [(gid, v) for gid, v in rec["games"].items() if v.get("flag")]

    for stage, wanted in (("last", hours_left <= LAST_CALL_HOURS and (changes or flags)), ("start", True)):
        key = f"{rec['season']}:{rec['week']}:{day}:{stage}"
        if not wanted or store.reminder_already_sent(KIND, key):
            continue
        if _in_quiet_hours(now, tz, cfg) and hours_left > 6:
            return "quiet hours"
        lines = []
        if changes:
            for gain, gid, v, cur in changes[:8]:
                g = by_id[gid]
                lines.append(f"{g['away']}@{g['home']}: take {v['pick']}" + (f" (you: {cur})" if cur else " (no pick yet)")
                             + f", +${gain:.2f}")
            if len(changes) > 8:
                lines.append(f"...and {len(changes) - 8} smaller")
        else:
            lines.append("No pick changes recommended.")
        mine_tb = rec.get("my_tiebreak")
        lines.append(f"Tiebreaker guess: {rec['tiebreak_guess']}" + (f" (yours: {mine_tb:g})" if mine_tb is not None else ""))
        lines.append(f"Week win chance {rec['p_win'] * 100:.1f}% (all favorites {rec['all_favorites']['p_win'] * 100:.1f}%)")
        for gid, v in flags:
            g = by_id.get(gid, {})
            lines.append(f"CHECK {g.get('away', '?')}@{g.get('home', '?')}: app {v.get('app_pick') or 'none'} vs CBS {v.get('cbs_pick') or 'none'}")
        label = datetime.fromisoformat(day).strftime("%a")
        title = (f"Pool card, {label} picks lock in {hours_left:.0f}h" if stage == "start"
                 else f"Last call: {label} picks lock in {max(1, round(hours_left * 60))} min" if hours_left < 1
                 else f"Last call: {label} picks lock in {hours_left:.1f}h")
        if notify.push(cfg, title, "\n".join(lines), tags="moneybag", priority="high" if stage == "last" else None, click=cfg.dashboard_url):
            store.mark_reminder_sent(KIND, key)
            if stage == "last":
                store.mark_reminder_sent(KIND, f"{rec['season']}:{rec['week']}:{day}:start")
            return f"sent {stage}"
        return "push failed"
    return "already sent"
