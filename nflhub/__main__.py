"""CLI entry point: python -m nflhub <command>.

In GitHub Actions the workflow runs `refresh` then `tick`. `yahoo-auth` is a one-time local
step whose oauth2.json is then stored as the YAHOO_OAUTH_JSON secret.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import store
from .config import OAUTH_PATH, get_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_yahoo_auth(_args) -> int:
    import os

    cid, csec = os.environ.get("YAHOO_CLIENT_ID", ""), os.environ.get("YAHOO_CLIENT_SECRET", "")
    if not cid or not csec:
        print("Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET in .env first.", file=sys.stderr)
        return 1
    if not OAUTH_PATH.exists():
        OAUTH_PATH.write_text(json.dumps({"consumer_key": cid, "consumer_secret": csec}, indent=2))
        print(f"Wrote {OAUTH_PATH}")
    try:
        from yahoo_oauth import OAuth2
    except ImportError:
        print("pip install -r requirements.txt first.", file=sys.stderr)
        return 1
    oauth = OAuth2(None, None, from_file=str(OAUTH_PATH))
    if not oauth.token_is_valid():
        oauth.refresh_access_token()
    ok = oauth.token_is_valid()
    print("Yahoo OAuth ready." if ok else "Still not valid; re-run.")
    if ok:
        print(f"\nStore the contents of {OAUTH_PATH} as the GitHub secret YAHOO_OAUTH_JSON.")
    return 0 if ok else 1


def cmd_refresh(_args) -> int:
    from .refresh import refresh_all

    summary = refresh_all()
    print(json.dumps(summary, indent=2, default=str))
    return 1 if summary.get("errors") else 0


def cmd_tick(_args) -> int:
    from . import notify

    store.init_db()
    fires = notify.run_due_reminders(get_config())
    print(f"{len(fires)} reminder(s) processed")
    for f in fires:
        print(f"  {f['kind']}: {f['hours_left']}h -> {f['detail']}")
    return 0


def cmd_notify(args) -> int:
    from . import deadlines, notify

    cfg = get_config()
    if args.test:
        ok = notify.push(cfg, "NFL Hub test", "ntfy is wired up.",
                         tags="white_check_mark", click=cfg.dashboard_url)
        print("sent" if ok else "failed")
        return 0 if ok else 1
    fires = deadlines.due_reminders(cfg)
    if not fires:
        print("No reminders due right now.")
        return 0
    for f in fires:
        title, message, tags, prio = notify.format_reminder(f["kind"], f["hours_left"], f["detail"])
        print(f"[{prio}] {title}\n  {message}\n  keys: {', '.join(f['also_mark'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nflhub", description="NFL fantasy/pickem/survivor/odds hub job")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("yahoo-auth", help="one-time Yahoo OAuth consent")
    sub.add_parser("refresh", help="pull every source into Supabase")
    sub.add_parser("tick", help="send any reminders that are due now")
    p_notify = sub.add_parser("notify", help="test push or dry-run pending reminders")
    p_notify.add_argument("--test", action="store_true")
    p_notify.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return {
        "yahoo-auth": cmd_yahoo_auth,
        "refresh": cmd_refresh,
        "tick": cmd_tick,
        "notify": cmd_notify,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
