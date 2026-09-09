"""Push notifications via ntfy (https://ntfy.sh or a self-hosted server)."""

from __future__ import annotations

import logging

import requests

from . import deadlines, store
from .config import Config

log = logging.getLogger(__name__)


def push(cfg: Config, title: str, message: str, *, tags: str = "", priority: str | None = None,
         click: str | None = None) -> bool:
    if not cfg.ntfy.topic or cfg.ntfy.topic.startswith("CHANGE-ME"):
        log.error("ntfy topic not configured; set [ntfy].topic in config.toml")
        return False
    url = f"{cfg.ntfy.server}/{cfg.ntfy.topic}"
    headers = {
        "Title": title,
        "Priority": priority or cfg.ntfy.priority,
    }
    if tags:
        headers["Tags"] = tags
    if click:
        headers["Click"] = click
    if cfg.ntfy.token:
        headers["Authorization"] = f"Bearer {cfg.ntfy.token}"
    try:
        resp = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        log.error("ntfy push failed: %s", exc)
        return False


_KIND_META = {
    "yahoo_lineup": ("Yahoo lineup locks", "football", "high"),
    "espn_lineup": ("ESPN lineup locks", "football", "high"),
    "pickem": ("Pick'em picks due", "pencil2", "high"),
    "survivor": ("Survivor pick due", "skull", "high"),
}


def format_reminder(kind: str, hours_left: float, detail: str) -> tuple[str, str, str, str]:
    """Return (title, message, tags, priority) for a deadline reminder."""
    label, tag, prio = _KIND_META.get(kind, (kind, "alarm_clock", "high"))
    if hours_left <= 1:
        when = f"{int(round(hours_left * 60))} min"
        prio = "urgent"
    elif hours_left < 6:
        when = f"{hours_left:.1f} hr"
    else:
        when = f"{int(round(hours_left))} hr"
    return (f"{label} in {when}", detail, tag, prio)


def run_due_reminders(cfg: Config) -> list[dict]:
    """Send every reminder that is due now; dedupe via reminder_log. Returns what fired."""
    fires = deadlines.due_reminders(cfg)
    for f in fires:
        title, message, tags, prio = format_reminder(f["kind"], f["hours_left"], f["detail"])
        if push(cfg, title, message, tags=tags, priority=prio, click=cfg.dashboard_url):
            for key in f["also_mark"]:
                store.mark_reminder_sent(f["kind"], key)
            log.info("sent reminder: %s (%s)", f["kind"], title)
        else:
            log.warning("push failed, will retry next run: %s", f["kind"])
    return fires
