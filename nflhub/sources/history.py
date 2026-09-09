"""Historical favorite performance vs. the closing spread, from the nflverse games dataset.

Produces a small blob (stored in kv as `hist_distribution`) that the dashboard joins to
each week's spreads to answer "how many games should I expect to be upsets?":

- per spread-size bucket x week-bucket (Week 1 vs Weeks 2+) x home/away favorite:
  P(favorite wins straight up) and P(favorite covers), shrunk toward the all-weeks rate
  for that bucket so thin cells don't read as 0% / 100%.
- pooled Week-1 vs Weeks-2+ summary (large n, matches published analyses).
- a per-week series so the week-over-week trend is visible.

nflverse `spread_line` convention: positive = home favored. (The dashboard's own odds use
the opposite sign; the frontend passes a `home_fav` boolean, not a signed number.)
"""

from __future__ import annotations

import csv
import io
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

GAMES_CSV = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TIMEOUT = 30
MIN_SEASON = 2007
SHRINK_K = 40  # pseudo-count pulling a bucket cell toward its all-weeks prior

# (lo, hi, label) on the absolute spread; 0.5-pt increments, so these tile the line cleanly.
BUCKETS = [(0.0, 2.5, "≤2.5"), (3.0, 3.0, "3"), (3.5, 6.0, "3.5–6"),
           (6.5, 9.5, "6.5–9.5"), (10.0, 99.0, "10+")]
BUCKET_LABELS = [b[2] for b in BUCKETS]


def _bucket(abs_spread: float) -> str | None:
    for lo, hi, lab in BUCKETS:
        if lo <= abs_spread <= hi:
            return lab
    return None


def _cell() -> dict[str, float]:
    return {"su_w": 0.0, "su_n": 0.0, "ats_w": 0.0, "ats_n": 0.0, "miss": 0.0}


def _rate(w: float, n: float) -> float | None:
    return round(w / n, 4) if n else None


def fetch_games() -> list[dict[str, str]]:
    resp = requests.get(GAMES_CSV, timeout=TIMEOUT)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def build_distributions(rows: list[dict[str, str]], min_season: int = MIN_SEASON) -> dict[str, Any]:
    data: dict[tuple, dict] = defaultdict(_cell)   # (wk_bucket, side, label)
    prior: dict[str, dict] = defaultdict(_cell)    # label -> all weeks / all sides
    byweek: dict[int, dict] = defaultdict(_cell)   # week int -> all sides
    byweek_side: dict[tuple, dict] = defaultdict(_cell)  # (week, fav side) -> all bins
    pooled: dict[tuple, dict] = defaultdict(_cell) # (wk_bucket, side) -> big-n summary
    max_season = min_season

    for r in rows:
        if r.get("game_type") != "REG":
            continue
        try:
            season = int(r["season"])
            week = int(r["week"])
        except (KeyError, ValueError):
            continue
        if season < min_season:
            continue
        sl_raw = r.get("spread_line", "")
        if sl_raw in ("", "NA", None):
            continue
        try:
            sl = float(sl_raw)
        except ValueError:
            continue
        # final margin, home perspective
        res_raw = r.get("result", "")
        if res_raw in ("", "NA", None):
            try:
                res = float(r["home_score"]) - float(r["away_score"])
            except (KeyError, ValueError):
                continue
        else:
            try:
                res = float(res_raw)
            except ValueError:
                continue

        max_season = max(max_season, season)
        abs_s = abs(sl)
        lab = _bucket(abs_s)
        if lab is None:
            continue
        home_fav = sl >= 0
        side = "home" if home_fav else "away"
        neutral = (r.get("location", "Home") == "Neutral")
        fav_margin = res if home_fav else -res
        su = 1.0 if fav_margin > 0 else 0.0
        cov = fav_margin - abs_s
        miss = abs(cov)
        is_push = abs(cov) < 1e-9
        ats = None if is_push else (1.0 if cov > 0 else 0.0)
        wkb = "week1" if week == 1 else "rest"

        targets = [(wkb, "all", lab)]
        if not neutral:
            targets.append((wkb, side, lab))
        for key in targets:
            c = data[key]
            c["su_w"] += su; c["su_n"] += 1; c["miss"] += miss
            if ats is not None:
                c["ats_w"] += ats; c["ats_n"] += 1

        p = prior[lab]
        p["su_w"] += su; p["su_n"] += 1
        if ats is not None:
            p["ats_w"] += ats; p["ats_n"] += 1

        b = byweek[week]
        b["su_w"] += su; b["su_n"] += 1; b["miss"] += miss
        if ats is not None:
            b["ats_w"] += ats; b["ats_n"] += 1
        if not neutral:
            bs = byweek_side[(week, side)]
            bs["su_w"] += su; bs["su_n"] += 1
            if ats is not None:
                bs["ats_w"] += ats; bs["ats_n"] += 1

        for pkey in [(wkb, "all")] + ([] if neutral else [(wkb, side)]):
            pc = pooled[pkey]
            pc["su_w"] += su; pc["su_n"] += 1; pc["miss"] += miss
            if ats is not None:
                pc["ats_w"] += ats; pc["ats_n"] += 1

    # shrink each bucket cell toward the all-weeks prior for that bucket
    def shrunk(c: dict, lab: str, field: str) -> float | None:
        w, n = c[f"{field}_w"], c[f"{field}_n"]
        pw, pn = prior[lab][f"{field}_w"], prior[lab][f"{field}_n"]
        if not n and not pn:
            return None
        prior_rate = pw / pn if pn else 0.5
        return round((w + SHRINK_K * prior_rate) / (n + SHRINK_K), 4)

    cells: dict[str, Any] = {"week1": {}, "rest": {}}
    for (wkb, side, lab), c in data.items():
        cells[wkb].setdefault(side, {})[lab] = {
            "su": shrunk(c, lab, "su"),
            "ats": shrunk(c, lab, "ats"),
            "n": int(c["su_n"]),
        }

    summary: dict[str, Any] = {}
    for wkb in ("week1", "rest"):
        allc = pooled.get((wkb, "all"), _cell())
        homec = pooled.get((wkb, "home"), _cell())
        awayc = pooled.get((wkb, "away"), _cell())
        summary[wkb] = {
            "su": _rate(allc["su_w"], allc["su_n"]),
            "ats": _rate(allc["ats_w"], allc["ats_n"]),
            # su for a favored side; 1 - away_su = home-dog upset rate, 1 - home_su = away-dog
            "home_su": _rate(homec["su_w"], homec["su_n"]),
            "away_su": _rate(awayc["su_w"], awayc["su_n"]),
            "home_ats": _rate(homec["ats_w"], homec["ats_n"]),
            "away_ats": _rate(awayc["ats_w"], awayc["ats_n"]),
            "home_n": int(homec["su_n"]),
            "away_n": int(awayc["su_n"]),
            "avg_miss": round(allc["miss"] / allc["su_n"], 2) if allc["su_n"] else None,
            "n": int(allc["su_n"]),
        }

    week_series = [
        {
            "week": w,
            "su": _rate(byweek[w]["su_w"], byweek[w]["su_n"]),
            "ats": _rate(byweek[w]["ats_w"], byweek[w]["ats_n"]),
            "home_su": _rate(byweek_side[(w, "home")]["su_w"], byweek_side[(w, "home")]["su_n"]),
            "away_su": _rate(byweek_side[(w, "away")]["su_w"], byweek_side[(w, "away")]["su_n"]),
            "avg_miss": round(byweek[w]["miss"] / byweek[w]["su_n"], 2) if byweek[w]["su_n"] else None,
            "n": int(byweek[w]["su_n"]),
        }
        for w in sorted(byweek)
    ]

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "seasons": f"{min_season}-{max_season}",
        "shrink_k": SHRINK_K,
        "buckets": BUCKET_LABELS,
        "cells": cells,
        "summary": summary,
        "byweek": week_series,
    }


def refresh(store, force: bool = False) -> str:
    """Rebuild the distribution at most once per day. Returns a status string."""
    today = date.today().isoformat()
    if not force and store.kv_get("hist_date") == today:
        return "cached"
    dist = build_distributions(fetch_games())
    import json

    store.kv_set("hist_distribution", json.dumps(dist))
    store.kv_set("hist_date", today)
    return f"built {dist['seasons']} (week1 n={dist['summary']['week1']['n']})"
