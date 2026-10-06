"""v22: v19-v21 all reused the SAME p85/p95 cut across snap-share, touches, and yards
without rechecking whether that's actually where each metric's signal is strongest -- a real
gap, since the three severity distributions have different shapes/noise levels and there's
no reason the right cutoff would coincide. This scans a full percentile grid per scheme
instead of two fixed points.

Correctness matters here: scanning many thresholds and reporting whichever gives the best
p-value is a textbook multiple-comparisons trap -- with enough cuts tried, SOMETHING will
look significant by chance even in pure noise. So this uses proper train/test discipline:
SELECT the best-looking threshold using ONLY the EARLY half (2009-2017), then CONFIRM it on
the fully held-out LATE half (2018-2025) it was never chosen against -- and the reverse
(select on LATE, confirm on EARLY) as a second, independent check. A threshold only counts
as real if it was picked on one half and still holds on the other; a threshold that only
ever looks good on the half it was chosen from is exactly the artifact this guards against.

Scheme definitions are unchanged from v19 (snap share) and v21 (touches, yards) -- this
script only changes WHICH percentile of each distribution gets tested, not how severity
itself is computed.

Run: python research/edge_signal_test_v22_threshold_scan.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import importlib.util
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v14 = _load("v14", "edge_signal_test_v14_qb_injury_dropout.py")
v16 = _load("v16", "edge_signal_test_v16_unit_health.py")
v17 = _load("v17", "edge_signal_test_v17_unit_health_snapweighted.py")
v21 = _load("v21", "edge_signal_test_v21_skill_touches_weighted.py")

PERCENTILES = [50, 55, 60, 65, 70, 75, 78, 80, 82, 85, 88, 90, 92, 95, 97, 99]
SPLIT_SEASON_SNAP = 2019   # snap share is bounded to 2012-2025 (v17's limit) -- same split v20 used
SPLIT_SEASON_USAGE = 2018  # touches/yards cover 2009-2025 (v21's wider window)


def fit_logistic(X: np.ndarray, y: np.ndarray):
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])

    def nll_grad(w):
        z = Xb @ w
        p_hat = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        nll = -np.sum(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))
        grad = Xb.T @ (p_hat - y)
        return nll, grad

    res = minimize(nll_grad, x0=np.zeros(p + 1), jac=True, method="L-BFGS-B")
    w = res.x
    z = Xb @ w
    p_hat = 1.0 / (1.0 + np.exp(-z))
    W = p_hat * (1 - p_hat)
    cov = np.linalg.inv((Xb * W[:, None]).T @ Xb)
    se = np.sqrt(np.diag(cov))
    return w, se, -res.fun


def outcome_test(games_subset, sev_by_tw, threshold_val):
    y = np.array([g["home_win"] for g in games_subset], dtype=float)
    delta = np.array([g["delta"] for g in games_subset]).reshape(-1, 1)
    diffs = []
    for g in games_subset:
        hv = 1 if sev_by_tw.get((g["home"], g["season"], g["week"]), 0.0) >= threshold_val else 0
        av = 1 if sev_by_tw.get((g["away"], g["season"], g["week"]), 0.0) >= threshold_val else 0
        diffs.append(hv - av)
    dcol = np.array(diffs, dtype=float).reshape(-1, 1)
    if np.std(dcol) == 0:
        return None
    w_r, se_r, ll_r = fit_logistic(delta, y)
    w_f, se_f, ll_f = fit_logistic(np.hstack([delta, dcol]), y)
    lr = 2 * (ll_f - ll_r)
    p = 1 - chi2.cdf(lr, df=1)
    return w_f[2], p


def scan_and_select(games_select, games_confirm, sev_by_tw, pct_values, label_select, label_confirm):
    """Select the percentile with the lowest p-value on `games_select`, then report that
    SAME threshold's result on `games_confirm` -- the only number that actually counts."""
    curve = []
    for pct, val in pct_values:
        r = outcome_test(games_select, sev_by_tw, val)
        if r is None:
            continue
        coef, p = r
        curve.append((pct, val, coef, p))
    if not curve:
        print(f"    (no valid thresholds)")
        return
    best = min(curve, key=lambda t: t[3])
    pct, val, sel_coef, sel_p = best
    confirm = outcome_test(games_confirm, sev_by_tw, val)
    print(f"    {label_select} curve (percentile: p-value): " +
          "  ".join(f"p{pc}={pv:.3f}" for pc, _, _, pv in curve))
    if confirm is None:
        print(f"    best on {label_select}: p{pct} (coef={sel_coef:+.3f}, p={sel_p:.4f}) -- degenerate on {label_confirm}, can't confirm")
        return
    conf_coef, conf_p = confirm
    same_sign = (sel_coef < 0) == (conf_coef < 0)
    verdict = "CONFIRMED" if (same_sign and conf_p < 0.10) else ("same sign, not significant" if same_sign else "SIGN FLIPPED")
    print(f"    best on {label_select}: p{pct} (coef={sel_coef:+.3f}, p={sel_p:.4f})  ->  "
          f"on {label_confirm}: coef={conf_coef:+.3f}, p={conf_p:.4f}  [{verdict}]")


def main():
    print("Building game-level dataset...", file=sys.stderr)
    rows_all = v14.build_dataset()
    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows_all])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS
    for i, r in enumerate(rows_all):
        x = X_all[i]
        r["delta"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))

    # --- scheme 1: snap share (v17/v19's original), 2012-2025 ---
    print("\n### snap share (skill group), 2012-2025 ###", file=sys.stderr)
    player_games, pos_default = v17.load_snap_shares()
    injury_rows = v17.load_injury_rows()
    weighted = v17.weighted_absence_by_team_week_group(injury_rows, player_games, pos_default, None)
    sev_snap = {(team, season, week): v for (team, season, week, grp), v in weighted.items() if grp == "skill"}
    games_snap = [r for r in rows_all if 2012 <= r["season"] <= 2025]
    all_vals_snap = np.array([sev_snap.get((r[s], r["season"], r["week"]), 0.0) for r in games_snap for s in ("home", "away")])
    pct_vals_snap = [(p, np.percentile(all_vals_snap, p)) for p in PERCENTILES]
    early_snap = [g for g in games_snap if g["season"] < SPLIT_SEASON_SNAP]
    late_snap = [g for g in games_snap if g["season"] >= SPLIT_SEASON_SNAP]
    print("  [snap share] select on EARLY, confirm on LATE:")
    scan_and_select(early_snap, late_snap, sev_snap, pct_vals_snap, "EARLY", "LATE")
    print("  [snap share] select on LATE, confirm on EARLY:")
    scan_and_select(late_snap, early_snap, sev_snap, pct_vals_snap, "LATE", "EARLY")

    # --- schemes 2+3: touches, yards (v21's), 2009-2025 ---
    print("\n### touches / yards (skill group), 2009-2025 ###", file=sys.stderr)
    player_usage_games, fallback = v21.load_skill_usage()
    usage_injury_rows = v21.load_skill_injury_rows()
    games_usage = [r for r in rows_all if 2009 <= r["season"] <= 2025]
    early_usage = [g for g in games_usage if g["season"] < SPLIT_SEASON_USAGE]
    late_usage = [g for g in games_usage if g["season"] >= SPLIT_SEASON_USAGE]

    for scheme in ("touches", "yards"):
        sev = v21.weighted_severity(usage_injury_rows, player_usage_games, fallback, scheme)
        all_vals = np.array([sev.get((r[s], r["season"], r["week"]), 0.0) for r in games_usage for s in ("home", "away")])
        pct_vals = [(p, np.percentile(all_vals, p)) for p in PERCENTILES]
        print(f"\n  [{scheme}] select on EARLY, confirm on LATE:")
        scan_and_select(early_usage, late_usage, sev, pct_vals, "EARLY", "LATE")
        print(f"  [{scheme}] select on LATE, confirm on EARLY:")
        scan_and_select(late_usage, early_usage, sev, pct_vals, "LATE", "EARLY")


if __name__ == "__main__":
    main()
