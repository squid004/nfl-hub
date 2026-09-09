"""Small shared helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def to_local(iso_or_dt, tz_name: str) -> datetime | None:
    if iso_or_dt is None:
        return None
    dt = iso_or_dt if isinstance(iso_or_dt, datetime) else datetime.fromisoformat(iso_or_dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo(tz_name))


def fmt_local(iso_or_dt, tz_name: str, with_day: bool = True) -> str:
    """Windows-safe short local time, e.g. 'Sun 1:20 PM'."""
    dt = to_local(iso_or_dt, tz_name)
    if dt is None:
        return "TBD"
    hour = dt.hour % 12 or 12
    stamp = f"{hour}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"
    return f"{dt.strftime('%a')} {stamp}" if with_day else stamp
