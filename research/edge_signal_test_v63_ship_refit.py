"""v63 ship refit: production constants with Elo as the 10th composite stat (on top of the v61/v62 settings).

v63 passed the pre-registered ADOPT rule (9 stats + Elo vs 9 stats). This script only produces the
numbers that have to be written into nflhub/sources/team_ratings.py when that ships, using v57's
joint-fit recipe (settings: alpha 0.08, bias-corrected, prior 8, L2 0.03):
  1. POWER_WEIGHTS: non-negative L2 = 0.03 logistic fit on the standardized per-game rating diffs.
  2. delta-scale constants: one joint logistic fit of home_win ~ delta_raw + QB gap + fantasy gap +
     skill out + HFA logit + streak (2009+ games); each correction = its coefficient / delta's.
  3. delta calibration: sigmoid(a + b*|delta|) on clean (no backup-QB) games.
It also prints the production-before vs after walk-forward AUC / log loss for the record.

Run: PYTHONPATH=. python research/edge_signal_test_v63_ship_refit.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
import nflhub.sources.team_ratings as tr  # noqa: E402
from edge_signal_test_v54_pen_presnap_production_refit import fit_logit, fit_nonneg, metrics, walk_forward  # noqa: E402

M9 = tr.POWER_METRICS  # 9 EWMA stats + elo

L2 = 0.03


def game_log(played, current, alpha, prior, bias):
    tr.EWMA_ALPHA, tr.SEASON_PRIOR_GAMES, tr.EWMA_BIAS_CORRECT = alpha, prior, bias
    tr.EWMA_ALPHA_BY_METRIC = {}
    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    return [r for r in log if r["home_score"] != r["away_score"]]


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)

    print("=== walk-forward before vs after (test seasons 2011+, L2 0.03) ===")
    for tag, cols in (("BEFORE 9 stats", [m for m in M9 if m != "elo"]), ("AFTER  9 stats + elo", list(M9))):
        log = game_log(played, current, 0.08, 8, True)
        seasons = np.array([r["season"] for r in log])
        y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
        X = np.column_stack([[r["diffs"][m] for r in log] for m in cols])
        p, yy, ss, _ = walk_forward(seasons, X, y, L2)
        a, l = metrics(p, yy)
        print(f"  {tag}: AUC {a:.4f} LL {l:.4f} acc {np.mean((p > 0.5) == (yy == 1)):.4f} ({len(yy)} games)")

    # final config
    log = game_log(played, current, 0.08, 8, True)
    seasons = np.array([r["season"] for r in log])
    y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
    X = np.column_stack([[r["diffs"][m] for r in log] for m in M9])
    mu, sd = X.mean(0), X.std(0)
    w = fit_nonneg((X - mu) / sd, y, L2)
    d = ((X - mu) / sd) @ w[1:]
    print(f"\n=== POWER_WEIGHTS ({len(log)} decided games {seasons.min()}-{seasons.max()}) ===")
    for m, v in zip(M9, w[1:]):
        print(f'    "{m}": {v:.4f},')
    print(f"  sum={w[1:].sum():.4f}  std(delta_raw)={d.std():.4f}")

    qb_gap, *_ = tr.qb_quality_gap_data(played, current)
    fantasy_gap, _t, _l, _b, fantasy_available = tr.qb_fantasy_gap_data(played, current, 1)
    skill_out = tr.skill_epa_out_by_team_week(current)
    hfa_by_season = tr.home_field_logit_by_season(played)
    streak = tr.team_streak_by_week(played)

    def g(dct, r, side):
        return dct.get((r[side], r["season"], r["week"]), 0.0)

    qb = np.array([g(qb_gap, r, "home") - g(qb_gap, r, "away") for r in log])
    fa = np.array([g(fantasy_gap, r, "home") - g(fantasy_gap, r, "away") for r in log])
    sk = np.array([g(skill_out, r, "home") - g(skill_out, r, "away") for r in log])
    hfa = np.array([0.0 if r.get("neutral") else hfa_by_season.get(r["season"], 0.0) for r in log])
    stk = np.array([g(streak, r, "home") - g(streak, r, "away") for r in log])
    keep = seasons >= 2009
    names = ["qb_gap", "fantasy_gap", "skill_out", "hfa_logit", "streak"]
    wj, se = fit_logit(np.column_stack([d, qb, fa, sk, hfa, stk])[keep], y[keep])
    rt = {nm: wj[i] / wj[1] for i, nm in enumerate(names, start=2)}
    print("\n=== delta-scale constants (ratio of each coefficient to delta's) ===")
    print(f"  delta coef={wj[1]:.4f}; " + "  ".join(f"{nm}={rt[nm]:+.4f} (z{wj[i] / se[i]:+.1f})" for i, nm in enumerate(names, start=2)))
    print("  currently shipped (v62): QB_QUALITY_GAP -1.2518, QB_FANTASY_GAP -0.0598, SKILL -0.0726, HFA 1.0319, MOMENTUM 0.0164, calib 0.0336/0.9921")

    use_fp = np.array([fantasy_available.get((r["season"], r["week"]), False) for r in log])
    qb_out = tr._qb_out_doubtful_by_week(current)
    clean = np.array([not (qb_out.get((r["home"], r["season"], r["week"]), False) or qb_out.get((r["away"], r["season"], r["week"]), False)) for r in log])
    qbh = np.where(use_fp, rt["fantasy_gap"] * fa, rt["qb_gap"] * qb)
    dt = d + qbh + rt["skill_out"] * sk + rt["hfa_logit"] * hfa + rt["streak"] * stk
    fav = np.where(dt > 0, y, 1 - y)
    m = clean & (dt != 0)
    wc, _ = fit_logit(np.abs(dt[m])[:, None], fav[m])
    print("\n=== delta calibration (clean games) ===")
    print(f"  n={m.sum()} DELTA_CALIBRATION_INTERCEPT={wc[0]:.4f} DELTA_CALIBRATION_SLOPE={wc[1]:.4f} fav hit={fav[m].mean():.4f} |delta| std={np.abs(dt[m]).std():.4f}")
    print(f"  hit rate of full delta over all decided games: {np.mean((dt > 0) == (y == 1)):.4f}")


if __name__ == "__main__":
    main()
