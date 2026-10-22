"""v69: shrink the passer's trailing EPA toward the pooled backup average by sample size (cowork plan B7.1; v38 flagged it untried).

v38 found using a specific backup's raw trailing EPA (50-150 attempts) was no better than the pooled backup average
(46,000 attempts). Empirical Bayes keeps the specific number but pulls it toward the pool by career attempts:
    shrunk = (n * raw + K * backup_avg_epa) / (n + K)
v43 tried shrinkage in the pre-v52 era only to see whether it rescued decay/reinsertion; this is the direct test on today's
production pipeline. `ta` (whoever actually played) is shrunk by default; the "both" variant also shrinks the primary's number.
Implemented behind team_ratings.QB_EPA_SHRINK_K / QB_EPA_SHRINK_BOTH (0 / False = shipped behavior).

Candidates: K in {25, 50, 100, 200, 400} (backup only), and K=100 on both sides. Baseline: K=0 (production).
Pipeline: v64's fully walk-forward V4 (z-score from training seasons, composite weights, corrections incl. QB/fantasy by win/loss
logit, calibration refit), test seasons 2011+. Only the EPA QB gap changes between runs.

Pre-registered ADOPT rule (v57/v61-v63, fixed before running): a candidate replaces the baseline only if dAUC > 0 AND dLogLoss > 0 in
BOTH halves (<=2018 / >2018), paired-bootstrap P(dLL <= 0) <= 0.01 (N_BOOT = 2000), and log loss better in >= 60% of test seasons.
Six candidates are compared against one baseline, so a passing result would still be reported with that multiplicity noted.

Run: PYTHONPATH=. python research/edge_signal_test_v69_backup_qb_epa_shrinkage.py
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
from edge_signal_test_v61_ewma_speed_season_prior import compare  # noqa: E402
from edge_signal_test_v68_qb_weight_ats_residual import fmt, p_home, verdict  # noqa: E402

M = tr.POWER_METRICS
L2 = 0.03
TEST_START = 2011
CORR_FROM = 2009
VARIANTS = [("base", 0.0, False), ("K25", 25.0, False), ("K50", 50.0, False), ("K100", 100.0, False),
            ("K200", 200.0, False), ("K400", 400.0, False), ("K100both", 100.0, True)]


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

    fantasy_gap, _t, _l, _b, fantasy_available = tr.qb_fantasy_gap_data(played, current, 1)
    skill_out = tr.skill_epa_out_by_team_week(current)
    hfa_by_season = tr.home_field_logit_by_season(played)
    streak = tr.team_streak_by_week(played)
    qb_out = tr._qb_out_doubtful_by_week(current)

    def g(d, r, side):
        return d.get((r[side], r["season"], r["week"]), 0.0)

    fa = np.array([g(fantasy_gap, r, "home") - g(fantasy_gap, r, "away") for r in log])
    sk = np.array([g(skill_out, r, "home") - g(skill_out, r, "away") for r in log])
    hfa = np.array([0.0 if r.get("neutral") else hfa_by_season.get(r["season"], 0.0) for r in log])
    stk = np.array([g(streak, r, "home") - g(streak, r, "away") for r in log])
    use_fp = np.array([fantasy_available.get((r["season"], r["week"]), False) for r in log])
    clean = np.array([not (qb_out.get((r["home"], r["season"], r["week"]), False) or qb_out.get((r["away"], r["season"], r["week"]), False)) for r in log])

    test_seasons = [s for s in sorted(set(seasons)) if s >= TEST_START]
    probs = {}
    for name, k, both in VARIANTS:
        tr.QB_EPA_SHRINK_K, tr.QB_EPA_SHRINK_BOTH = k, both
        qb_gap, *_ = tr.qb_quality_gap_data(played, current)
        qb = np.array([g(qb_gap, r, "home") - g(qb_gap, r, "away") for r in log])
        delta_t = np.full(len(y), np.nan)
        cal = np.zeros((len(y), 2))
        for S in test_seasons:
            tri, tei = seasons < S, seasons == S
            mu_t, sd_t = X[tri].mean(0), X[tri].std(0)
            sd_t[sd_t == 0] = 1.0
            w = fit_nonneg((X[tri] - mu_t) / sd_t, y[tri], L2)[1:]
            d_raw = ((X - mu_t) / sd_t) @ w
            kp = tri & (seasons >= CORR_FROM)
            feats = np.column_stack([d_raw, qb, fa, sk, hfa, stk])[kp]
            live = feats.std(0) > 0
            live[0] = True
            wj, _se = fit_logit(feats[:, live], y[kp])
            ratio = np.zeros(6)
            ratio[live] = np.concatenate([[1.0], wj[2:] / wj[1]])
            qbh = np.where(use_fp, ratio[2] * fa, ratio[1] * qb)
            d = d_raw + qbh + ratio[3] * sk + ratio[4] * hfa + ratio[5] * stk
            km = kp & clean & (d != 0)
            fav = np.where(d > 0, y, 1 - y)
            wc, _ = fit_logit(np.abs(d[km])[:, None], fav[km])
            delta_t[tei] = d[tei]
            cal[tei] = (wc[0], wc[1])
        test = seasons >= TEST_START
        probs[name] = np.array([p_home(delta_t[test][i], cal[test][i, 0], cal[test][i, 1]) for i in range(test.sum())]).reshape(-1)
        print(f"  built {name}", flush=True)
    tr.QB_EPA_SHRINK_K, tr.QB_EPA_SHRINK_BOTH = 0.0, False

    test = seasons >= TEST_START
    yt, ss = y[test], seasons[test]
    rng = np.random.default_rng(69)
    print(f"\n{len(yt)} decided test games, seasons {TEST_START}-{max(test_seasons)}\n")
    print("=== ADOPT test vs baseline (K=0, production) ===")
    for name, _k, _b in VARIANTS[1:]:
        c = compare(yt, ss, probs["base"], probs[name], rng)
        ok, rep = verdict(c)
        print(f"{name:9s}: {fmt(c)}; replicated both halves={rep} -> {'ADOPT' if ok else 'DO NOT ADOPT'}")


if __name__ == "__main__":
    main()
