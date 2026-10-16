"""v64: how much of the backtest hit rate survives when EVERY fitted piece is walk-forward?

Production's backtest scatter applies constants fit ONCE on all data (all seasons, including the
season being scored) to every historical game: the z-score mean/std of the rating diffs, the
composite weights, the QB / fantasy / skill / home-field / momentum corrections, and the
delta->probability calibration. The v57-v63 ADOPT tests are walk-forward for the composite weights
only. This script rebuilds the whole delta pipeline walk-forward, one piece at a time, so the
backtest number is an honest out-of-sample one:

  V0  production constants, all-data (what the scatter shows today; in-sample)
  V1  composite weights refit walk-forward (train seasons < S), everything else as V0
  V2  V1 + z-score mean/std from the training seasons only
  V3  V2 + QB/fantasy/skill/HFA/momentum constants refit on training seasons only
  V4  V3 + delta->probability calibration refit on training seasons only (the fully honest model)

For each test season S (2011+): training = seasons < S (corrections/calibration use training games
from 2009, when the injury data starts). Metrics on the same test games for every variant: hit rate
of the sign of delta, log loss and Brier of the calibrated P(home win), AUC of delta. Also a
reliability table by |delta| bin (V0 vs V4) and the effect of flooring P(favorite) at 0.5.

No pass/fail rule here: this is a measurement, not a candidate. The question it answers is "what
should the headline hit rate be".

Run: PYTHONPATH=. python research/edge_signal_test_v64_honest_walkforward_delta.py
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
from edge_signal_test_v54_pen_presnap_production_refit import fit_logit, fit_nonneg  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

M = tr.POWER_METRICS
L2 = 0.03
TEST_START = 2011
CORR_FROM = 2009
BINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 99.0)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def p_home(delta, a, b, floor):
    pf = sig(a + b * np.abs(delta))
    if floor:
        pf = np.maximum(pf, 0.5)
    pf = np.minimum(pf, tr.DELTA_PROB_MAX)
    return np.where(delta > 0, pf, 1 - pf)


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)
    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    log = [r for r in log if r["home_score"] != r["away_score"]]
    seasons = np.array([r["season"] for r in log])
    y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
    X = np.column_stack([[r["diffs"][m] for r in log] for m in M])

    qb_gap, *_ = tr.qb_quality_gap_data(played, current)
    fantasy_gap, _t, _l, _b, fantasy_available = tr.qb_fantasy_gap_data(played, current, 1)
    skill_out = tr.skill_epa_out_by_team_week(current)
    hfa_by_season = tr.home_field_logit_by_season(played)
    streak = tr.team_streak_by_week(played)
    qb_out = tr._qb_out_doubtful_by_week(current)

    def g(d, r, side):
        return d.get((r[side], r["season"], r["week"]), 0.0)

    qb = np.array([g(qb_gap, r, "home") - g(qb_gap, r, "away") for r in log])
    fa = np.array([g(fantasy_gap, r, "home") - g(fantasy_gap, r, "away") for r in log])
    sk = np.array([g(skill_out, r, "home") - g(skill_out, r, "away") for r in log])
    hfa = np.array([0.0 if r.get("neutral") else hfa_by_season.get(r["season"], 0.0) for r in log])
    stk = np.array([g(streak, r, "home") - g(streak, r, "away") for r in log])
    use_fp = np.array([fantasy_available.get((r["season"], r["week"]), False) for r in log])
    clean = np.array([not (qb_out.get((r["home"], r["season"], r["week"]), False) or qb_out.get((r["away"], r["season"], r["week"]), False)) for r in log])

    def delta_from(dr, c):
        """c = dict(qb, fantasy, skill, hfa, streak) delta-equivalent constants."""
        qbh = np.where(use_fp, c["fantasy"] * fa, c["qb"] * qb)
        return dr + qbh + c["skill"] * sk + c["hfa"] * hfa + c["streak"] * stk

    prod_c = dict(qb=tr.QB_QUALITY_GAP_WEIGHT, fantasy=tr.QB_FANTASY_GAP_WEIGHT, skill=tr.SKILL_EPA_OUT_WEIGHT,
                  hfa=tr.HFA_WEIGHT, streak=tr.MOMENTUM_WEIGHT)
    prod_w = np.array([tr.POWER_WEIGHTS[m] for m in M])
    mu_all, sd_all = X.mean(0), X.std(0)
    d_prod = delta_from(((X - mu_all) / sd_all) @ prod_w, prod_c)

    test_seasons = [s for s in sorted(set(seasons)) if s >= TEST_START]
    names = ("V0 production, all-data (in-sample)", "V1 + weights walk-forward", "V2 + z-score training-only",
             "V3 + corrections walk-forward", "V4 + calibration walk-forward (honest)")
    delta_v = {n: np.full(len(y), np.nan) for n in names}
    cal_v = {n: np.zeros((len(y), 2)) for n in names}
    prod_cal = (tr.DELTA_CALIBRATION_INTERCEPT, tr.DELTA_CALIBRATION_SLOPE)
    per_season_corr = {}

    for S in test_seasons:
        tri, tei = seasons < S, seasons == S
        mu_t, sd_t = X[tri].mean(0), X[tri].std(0)
        sd_t[sd_t == 0] = 1.0
        w = fit_nonneg((X[tri] - mu_t) / sd_t, y[tri], L2)[1:]
        # V0
        delta_v[names[0]][tei] = d_prod[tei]
        cal_v[names[0]][tei] = prod_cal
        # V1: walk-forward weights, all-data z-score, production corrections
        d1 = delta_from(((X - mu_all) / sd_all) @ w, prod_c)
        delta_v[names[1]][tei] = d1[tei]
        cal_v[names[1]][tei] = prod_cal
        # V2: training-only z-score
        d_raw2 = ((X - mu_t) / sd_t) @ w
        d2 = delta_from(d_raw2, prod_c)
        delta_v[names[2]][tei] = d2[tei]
        cal_v[names[2]][tei] = prod_cal
        # V3: corrections refit on training seasons >= CORR_FROM
        kp = tri & (seasons >= CORR_FROM)
        feats = np.column_stack([d_raw2, qb, fa, sk, hfa, stk])[kp]
        live = feats.std(0) > 0  # a correction with no training data yet (e.g. FantasyPros before its first season) gets weight 0
        live[0] = True
        wj, _se = fit_logit(feats[:, live], y[kp])
        ratio = np.zeros(6)
        ratio[live] = np.concatenate([[1.0], wj[2:] / wj[1]])
        c3 = dict(qb=ratio[1], fantasy=ratio[2], skill=ratio[3], hfa=ratio[4], streak=ratio[5])
        per_season_corr[S] = c3
        d3 = delta_from(d_raw2, c3)
        delta_v[names[3]][tei] = d3[tei]
        cal_v[names[3]][tei] = prod_cal
        # V4: calibration refit on training clean games, using the training delta
        km = kp & clean & (d3 != 0)
        fav = np.where(d3 > 0, y, 1 - y)
        wc, _ = fit_logit(np.abs(d3[km])[:, None], fav[km])
        delta_v[names[4]][tei] = d3[tei]
        cal_v[names[4]][tei] = (wc[0], wc[1])

    test = seasons >= TEST_START
    yt = y[test]
    sel, con = seasons[test] <= 2018, seasons[test] > 2018
    print(f"{test.sum()} decided test games, seasons {TEST_START}-{max(test_seasons)}\n")
    print(f"{'variant':40s} {'hit':>7} {'hit<=18':>8} {'hit>18':>7} {'LL':>7} {'Brier':>7} {'AUC':>7}")
    for n in names:
        d = delta_v[n][test]
        ph = np.array([p_home(d[i], *cal_v[n][test][i], True) for i in range(len(d))]).reshape(-1)
        hit = (d > 0) == (yt == 1)
        print(f"{n:40s} {hit.mean():7.4f} {hit[sel].mean():8.4f} {hit[con].mean():7.4f} "
              f"{log_loss(yt, ph):7.4f} {np.mean((ph - yt) ** 2):7.4f} {roc_auc_score(yt, d):7.4f}")

    print("\n=== per-season hit rate: V0 (in-sample) vs V4 (honest) ===")
    for S in test_seasons:
        m = seasons[test] == S
        h0 = ((delta_v[names[0]][test] > 0) == (yt == 1))[m].mean()
        h4 = ((delta_v[names[4]][test] > 0) == (yt == 1))[m].mean()
        print(f"  {S}: V0 {h0:.3f}  V4 {h4:.3f}  ({m.sum()} games)")

    print("\n=== reliability by |delta| bin (predicted P(favorite) vs actual favorite win rate) ===")
    for n in (names[0], names[4]):
        d = delta_v[n][test]
        fav_y = np.where(d > 0, yt, 1 - yt)
        a = cal_v[n][test][:, 0]
        b = cal_v[n][test][:, 1]
        pf = np.minimum(sig(a + b * np.abs(d)), tr.DELTA_PROB_MAX)
        print(f"  {n}")
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (np.abs(d) >= lo) & (np.abs(d) < hi)
            if m.sum():
                print(f"    |delta| {lo:4.2f}-{min(hi, 9):4.2f}: n={m.sum():4d}  predicted {pf[m].mean():.3f}  actual {fav_y[m].mean():.3f}  gap {fav_y[m].mean() - pf[m].mean():+.3f}")

    print("\n=== floor P(favorite) at 0.5 (V4 calibration) ===")
    d = delta_v[names[4]][test]
    a, b = cal_v[names[4]][test][:, 0], cal_v[names[4]][test][:, 1]
    raw = np.minimum(sig(a + b * np.abs(d)), tr.DELTA_PROB_MAX)
    print(f"  share of test games where the curve gives the favorite < 50%: {np.mean(raw < 0.5):.4f}; min P(fav) {raw.min():.3f}")
    ph_f = np.where(d > 0, np.maximum(raw, 0.5), 1 - np.maximum(raw, 0.5))
    ph_r = np.where(d > 0, raw, 1 - raw)
    print(f"  log loss unfloored {log_loss(yt, ph_r):.4f}  floored {log_loss(yt, ph_f):.4f}")

    print("\n=== constants fit on the final training window (all seasons) vs production ===")
    wj, se = fit_logit(np.column_stack([((X - mu_all) / sd_all) @ prod_w, qb, fa, sk, hfa, stk])[seasons >= CORR_FROM], y[seasons >= CORR_FROM])
    print("  all-data refit:", {k: round(wj[i] / wj[1], 4) for i, k in enumerate(("qb", "fantasy", "skill", "hfa", "streak"), start=2)})
    print("  production    :", {k: round(v, 4) for k, v in prod_c.items()})
    print("  walk-forward constants by test season (qb / fantasy / skill / hfa / streak):")
    for S in test_seasons:
        c = per_season_corr[S]
        print(f"    {S}: {c['qb']:+.3f} {c['fantasy']:+.4f} {c['skill']:+.4f} {c['hfa']:+.3f} {c['streak']:+.4f}")


if __name__ == "__main__":
    main()
