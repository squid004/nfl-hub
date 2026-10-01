"""Historical favorite performance vs. the closing spread, from the nflverse games dataset.

Produces a small blob (stored in kv as `hist_distribution`) that the dashboard joins to
each week's spreads to answer "how many games should I expect to be upsets?":

- per spread-size bucket x home/away favorite: P(favorite wins straight up) and P(favorite
  covers), shrunk toward the all-games rate for that bucket so thin cells don't read as
  0% / 100%. Used for the descriptive History panels (js/history.js render()) — grouping
  games into bins is still the clearest way to *show* this trend, even though it's no longer
  how ranking decisions get made (below).
- a smooth logistic curve P(favorite wins|covers) = sigmoid(a + b*abs_spread), fit per
  home/away favorite via IRLS on the raw per-game data (pure Python, no numpy — 2-parameter
  fit, closed-form per iteration). This is what week_budget() actually uses now: the
  discrete buckets above treat a -3.5 and a -6.0 favorite as identical (same bucket, same
  rate), which was never really true, it just wasn't captured. A single continuous curve
  fixes that everywhere *except* one real effect: NFL final margins cluster hard at 3 and 7
  (a field goal, a touchdown+XP), so favorites laying exactly 3 or 7 cover at a measurably
  different rate than neighboring numbers — a smooth monotonic curve can't represent that
  bump by construction. `key_adjustments` is a small, sample-size-shrunk additive correction
  at exactly 3 and 7 so that real effect survives the move to a continuous model instead of
  getting smoothed away.
- one pooled summary across the whole season (large n, no week-of-season split).

One constant estimate regardless of week, by design, not by omission: this used to split
Week 1 from "the rest," and a later investigation (prompted by the week-over-week table
showing some weeks visibly more upset-heavy than others, e.g. week 10 vs week 17) tested
replacing that with a season-finale split instead. Both were checked against replication on
an independent half of the data (2007-2016 vs. 2017-2025) before being added to production,
same standard as every other model in this project:
  - Week 1 vs. rest (SU): full sample z=-0.78, first half z=-0.33, second half z=-0.79 —
    never significant anywhere, despite "Week 1 underperforms" being the direction every
    slice points. Also true for ATS (|z| < 0.4 everywhere).
  - Season finale vs. rest (SU, finale = each season's actual last REG week — 17 through
    2020, 18 from 2021 on, not hardcoded): full sample z=+2.87 (looks real), but that's
    almost entirely carried by 2007-2016 (z=+2.96) — 2017-2025 alone is z=+1.05, well short
    of significant. Widening to the last 2-4 weeks of the season doesn't fix it either
    (second-half z stays in the 1.0-1.9 range at every window width tested).
  - Week 10's isolated upset-heavy pooled result (z=-3.30 full sample) has the same problem:
    z=-2.82 first half, z=-1.55 second half.
None of these hold up on the more recent half alone, so none of them are in here — pooling
everything is both simpler and the one choice actually supported by the data. If NFL
schedule-related effects strengthen again in a way that replicates on fresh data, worth
revisiting; until then, don't split on week.

nflverse `spread_line` convention: positive = home favored. (The dashboard's own odds use
the opposite sign; the frontend passes a `home_fav` boolean, not a signed number.)
"""

from __future__ import annotations

import csv
import io
import logging
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

GAMES_CSV = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TIMEOUT = 30
MIN_SEASON = 2007
SHRINK_K = 40  # pseudo-count pulling a bucket cell toward its all-games prior
KEY_NUMBERS = (3.0, 7.0)  # NFL final-margin clustering: a field goal, a TD+XP
ELWAY_BLEND_WEIGHT = 0.65  # how far the ranking blend moves from the market toward ELWAY

# (lo, hi, label) on the absolute spread; 0.5-pt increments, so these tile the line cleanly.
# Display/grouping only now (see module docstring) — ranking uses the smooth curve below.
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


def _sigmoid(z: float) -> float:
    # Numerically stable both directions.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _fit_logistic(points: list[tuple[float, float]], iters: int = 25) -> Optional[tuple[float, float]]:
    """1-predictor logistic regression p = sigmoid(a + b*x) via IRLS. `points` = [(x, y)],
    y in {0,1}. Each iteration is a closed-form 2x2 weighted-least-squares solve (Cramer's
    rule) on the IRLS working response — no matrix library needed for a single predictor.
    Returns None if there's nothing to fit (e.g. an empty side/week split)."""
    n = len(points)
    if n < 10:
        return None
    a, b = 0.0, 0.0
    for _ in range(iters):
        Sw = Swx = Swxx = Swz = Swxz = 0.0
        for x, y in points:
            eta = a + b * x
            p = _sigmoid(eta)
            w = max(p * (1 - p), 1e-10)
            z = eta + (y - p) / w
            Sw += w; Swx += w * x; Swxx += w * x * x
            Swz += w * z; Swxz += w * x * z
        det = Sw * Swxx - Swx * Swx
        if abs(det) < 1e-12:
            break
        new_a = (Swxx * Swz - Swx * Swxz) / det
        new_b = (Sw * Swxz - Swx * Swz) / det
        converged = abs(new_a - a) < 1e-9 and abs(new_b - b) < 1e-9
        a, b = new_a, new_b
        if converged:
            break
    return round(a, 6), round(b, 6)


def _key_cell() -> dict[str, float]:
    return {"su_w": 0.0, "su_n": 0.0, "ats_w": 0.0, "ats_n": 0.0}


def fetch_games() -> list[dict[str, str]]:
    resp = requests.get(GAMES_CSV, timeout=TIMEOUT)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def build_distributions(rows: list[dict[str, str]], min_season: int = MIN_SEASON) -> dict[str, Any]:
    data: dict[tuple, dict] = defaultdict(_cell)   # (side, label)
    prior: dict[str, dict] = defaultdict(_cell)    # label -> all sides
    pooled: dict[str, dict] = defaultdict(_cell)   # side -> big-n summary
    # raw (abs_spread, outcome) pairs for the smooth curve fit, keyed by side; "all" gets
    # every game regardless of side, as a fallback for thin side-specific fits.
    raw_su: dict[str, list[tuple[float, float]]] = defaultdict(list)
    raw_ats: dict[str, list[tuple[float, float]]] = defaultdict(list)
    key_data: dict[tuple, dict] = defaultdict(_key_cell)  # (side, key_number)
    max_season = min_season

    for r in rows:
        if r.get("game_type") != "REG":
            continue
        try:
            season = int(r["season"])
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

        targets = [("all", lab)]
        if not neutral:
            targets.append((side, lab))
        for key in targets:
            c = data[key]
            c["su_w"] += su; c["su_n"] += 1; c["miss"] += miss
            if ats is not None:
                c["ats_w"] += ats; c["ats_n"] += 1

        # smooth-curve fit inputs: every game feeds "all", plus its own side unless neutral
        curve_keys = ["all"]
        if not neutral:
            curve_keys.append(side)
        for ck in curve_keys:
            raw_su[ck].append((abs_s, su))
            if ats is not None:
                raw_ats[ck].append((abs_s, ats))
            for kn in KEY_NUMBERS:
                if abs(abs_s - kn) < 0.01:
                    kc = key_data[(ck, kn)]
                    kc["su_w"] += su; kc["su_n"] += 1
                    if ats is not None:
                        kc["ats_w"] += ats; kc["ats_n"] += 1

        p = prior[lab]
        p["su_w"] += su; p["su_n"] += 1
        if ats is not None:
            p["ats_w"] += ats; p["ats_n"] += 1

        for pkey in ["all"] + ([] if neutral else [side]):
            pc = pooled[pkey]
            pc["su_w"] += su; pc["su_n"] += 1; pc["miss"] += miss
            if ats is not None:
                pc["ats_w"] += ats; pc["ats_n"] += 1

    # shrink each bucket cell toward the all-games prior for that bucket
    def shrunk(c: dict, lab: str, field: str) -> float | None:
        w, n = c[f"{field}_w"], c[f"{field}_n"]
        pw, pn = prior[lab][f"{field}_w"], prior[lab][f"{field}_n"]
        if not n and not pn:
            return None
        prior_rate = pw / pn if pn else 0.5
        return round((w + SHRINK_K * prior_rate) / (n + SHRINK_K), 4)

    cells: dict[str, Any] = {}
    for (side, lab), c in data.items():
        cells.setdefault(side, {})[lab] = {
            "su": shrunk(c, lab, "su"),
            "ats": shrunk(c, lab, "ats"),
            "n": int(c["su_n"]),
        }

    # smooth curve: p(field) = sigmoid(a + b*abs_spread), per side
    curves: dict[str, Any] = {}
    curve_coef: dict[str, dict[str, Optional[tuple[float, float]]]] = {}
    for side in set(raw_su) | set(raw_ats):
        coef = {"su": _fit_logistic(raw_su.get(side, [])), "ats": _fit_logistic(raw_ats.get(side, []))}
        curve_coef[side] = coef
        curves[side] = {field: (list(c) if c else None) for field, c in coef.items()}

    # key-number correction: shrink the raw empirical rate at exactly 3 / 7 toward the
    # smooth curve's OWN prediction there (not the bucket prior — the curve already IS
    # the model), so the adjustment is the part the curve structurally can't capture.
    key_adjustments: dict[str, Any] = {}
    for (side, kn), kc in key_data.items():
        coef = curve_coef.get(side) or {}
        out_side = key_adjustments.setdefault(side, {})
        for field in ("su", "ats"):
            ab = coef.get(field)
            w, n = kc[f"{field}_w"], kc[f"{field}_n"]
            if not ab or not n:
                continue
            predicted = _sigmoid(ab[0] + ab[1] * kn)
            shrunk_rate = (w + SHRINK_K * predicted) / (n + SHRINK_K)
            out_side.setdefault(field, {})[str(int(kn))] = round(shrunk_rate - predicted, 4)

    allc = pooled.get("all", _cell())
    homec = pooled.get("home", _cell())
    awayc = pooled.get("away", _cell())
    summary: dict[str, Any] = {
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

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "seasons": f"{min_season}-{max_season}",
        "shrink_k": SHRINK_K,
        "buckets": BUCKET_LABELS,
        "cells": cells,
        "curves": curves,
        "key_adjustments": key_adjustments,
        "summary": summary,
    }


def _lookup(dist: dict, abs_spread: float, home_fav: bool) -> dict | None:
    """Mirror of History.lookup in js/history.js: most specific cell for a game."""
    cells = dist.get("cells") or {}
    lab = _bucket(abs_spread)
    side = "home" if home_fav else "away"
    return cells.get(side, {}).get(lab) or cells.get("all", {}).get(lab)


def predict(dist: dict, abs_spread: float, home_fav: bool, field: str) -> Optional[float]:
    """P(favorite wins SU / covers) from the smooth curve at this exact spread, with the
    key-number correction applied at 3 and 7. Mirror: History.predict in js/history.js."""
    side = "home" if home_fav else "away"
    curves = dist.get("curves") or {}
    coef = (curves.get(side, {}) or {}).get(field) or (curves.get("all", {}) or {}).get(field)
    if not coef:
        return None
    p = _sigmoid(coef[0] + coef[1] * abs_spread)
    for kn in KEY_NUMBERS:
        if abs(abs_spread - kn) < 0.01:
            adj = (dist.get("key_adjustments") or {}).get(side, {}).get(field, {}).get(str(int(kn)))
            if adj is not None:
                p += adj
            break
    return round(min(0.995, max(0.005, p)), 4)


def _predict_signed(dist: dict, home_fav_side: bool, signed_margin: float, field: str) -> Optional[float]:
    """P(the team on `home_fav_side`, home if True else away, wins/covers), from a SIGNED
    margin for that side (positive = favored by that many points; negative = actually the
    worse side by that many, evaluated by flipping to the other side's curve at the
    positive magnitude). Lets a ranking blend go past "toss-up" without a floor: unlike
    clamping the magnitude at 0, which forces every game past that point to the same
    value (the bug an ELWAY-vs-market blend hit in practice — three different games all
    clamped to the identical toss-up number and got tie-broken by list order), this stays
    strictly monotonic and keeps differentiating games no matter how large the blend gets.
    """
    if signed_margin >= 0:
        return predict(dist, signed_margin, home_fav_side, field)
    p_other = predict(dist, -signed_margin, not home_fav_side, field)
    return None if p_other is None else round(1 - p_other, 4)


def week_budget(
    dist: dict, games: list[dict], odds_map: dict[str, dict],
    elway_map: dict[str, dict] | None = None,
) -> dict[str, list[dict]]:
    """Per spread-bin "take N dogs" suggestion for this week, for both modes.

    Server-side twin of History.renderBins in js/history.js so the suggestion can be
    frozen into budget_snapshot for end-of-season analysis.

    Bucket ASSIGNMENT (which row of the table a game displays under) still comes from the
    discrete spread buckets — that's just grouping for readability now. Both the "how many
    to take" count and which SPECIFIC game gets flagged come from the smooth curve
    (predict(), see its docstring): "how many" sums 1-predict() at the game's real market
    spread. "Which one" ranks games within a bucket by predict() at an ELWAY-blended
    margin instead: ELWAY_BLEND_WEIGHT of the way from the market's own favorite-margin
    toward ELWAY's margin for that same team (0.65 -- more trust than an even split,
    since ELWAY is Nate Silver's Silver Bulletin NFL forecasting model (team ratings +
    QBERT, refined for 2026: natesilver.net/i/176207317/2026-changes-to-elway-and-qbert),
    not an untested personal formula, but still short of full weight since a live betting
    market prices in more real-money information than any single outside model reliably
    beats). The blend is a signed margin with no floor — _predict_signed() lets it cross zero and
    keep differentiating games past "toss-up," which matters in practice: an earlier
    version clamped the magnitude at 0 instead, and multiple games whose blend crossed
    zero all landed on the identical clamped value, silently re-tied and broken by list
    order (exactly the arbitrary-order problem this ranking exists to avoid). ELWAY only
    shifts the ranking, never the "how many" count.
    """
    elway_map = elway_map or {}
    out: dict[str, list[dict]] = {}
    for mode, field in (("ml", "su"), ("ats", "ats")):
        by_bin: dict[str, list[dict]] = {lab: [] for lab in BUCKET_LABELS}
        for g in games:
            o = odds_map.get(g["game_id"])
            if not o or o.get("spread") is None:
                continue
            sp = float(o["spread"])
            home_fav = sp <= 0
            v = predict(dist, abs(sp), home_fav, field)
            if v is None:
                continue

            # Blend the market favorite's margin toward ELWAY's margin for that same
            # team (positive = favored by that many points; can go negative, meaning
            # that team is now the modeled underdog).
            eff_margin = abs(sp)
            el = elway_map.get(g["game_id"])
            if el and el.get("spread_home") is not None:
                elway_fav_margin = -el["spread_home"] if home_fav else el["spread_home"]
                eff_margin = (1 - ELWAY_BLEND_WEIGHT) * abs(sp) + ELWAY_BLEND_WEIGHT * elway_fav_margin
            rank_v = _predict_signed(dist, home_fav, eff_margin, field)
            if rank_v is None:
                rank_v = v

            # Display bucket only: BUCKETS assumes clean half-point spreads (true for a
            # single book's line, not necessarily for a cross-book average like
            # avg_spread_home, e.g. 2.75) — round to the nearest half-point so every
            # spread lands in a bucket. predict() above already used the exact value.
            disp_spread = round(abs(sp) * 2) / 2
            lab = _bucket(disp_spread)
            if lab is None:
                continue
            hist_cell = _lookup(dist, disp_spread, home_fav) or {}
            by_bin[lab].append({
                "game_id": g["game_id"],
                "fav": g["home"] if home_fav else g["away"],
                "dog": g["away"] if home_fav else g["home"],
                "dog_home": not home_fav,
                "hist_su": hist_cell.get("su"),
                "hist_ats": hist_cell.get("ats"),
                "_v": v,
                "_rank_v": rank_v,
            })
        rows: list[dict] = []
        tot_n = 0
        tot_exp = 0.0
        for lab in BUCKET_LABELS:
            gs = sorted(by_bin[lab], key=lambda x: x["_rank_v"])  # most live dog (lowest fav prob) first
            n = len(gs)
            exp = sum(1 - x["_v"] for x in gs)
            take = round(exp)
            for i, x in enumerate(gs):
                x["flagged"] = i < take
                x.pop("_v")
                x.pop("_rank_v")
            tot_n += n
            tot_exp += exp
            rows.append({
                "bin": lab, "n_games": n,
                "rate": round(exp / n, 4) if n else None,
                "suggested": take, "games": gs,
            })
        rows.append({
            "bin": "TOTAL", "n_games": tot_n,
            "rate": round(tot_exp / tot_n, 4) if tot_n else None,
            "suggested": round(tot_exp), "games": [],
        })
        out[mode] = rows
    return out


def refresh(store, force: bool = False) -> str:
    """Rebuild the distribution at most once per day. Returns a status string."""
    today = date.today().isoformat()
    if not force and store.kv_get("hist_date") == today:
        return "cached"
    dist = build_distributions(fetch_games())
    import json

    store.kv_set("hist_distribution", json.dumps(dist))
    store.kv_set("hist_date", today)
    return f"built {dist['seasons']} (n={dist['summary']['n']})"
