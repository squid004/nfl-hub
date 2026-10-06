"""v20: replicate v19's tail-threshold findings on an out-of-sample split, same discipline
every validated finding in this project goes through (see nflhub/sources/history.py's own
module docstring, which explicitly rejected two patterns for failing exactly this check).

v19 ran ~28 tests and found 6 "significant" hits at p<0.05 -- more than the ~1-2 chance alone
predicts, with a coherent story (right signs, a dose-response from short-handed to decimated),
but not independently confirmed. This script re-runs ONLY those 6 candidate findings (plus
QB as the already-known positive control, and OL/DL as already-null negative controls) on two
non-overlapping season halves of the SAME 2012-2025 data: EARLY (2012-2018) and LATE
(2019-2025). Thresholds stay FIXED at v19's own full-dataset percentiles (not recomputed per
half) so "decimated" means the same severity level in both halves. A finding that holds its
sign and lands anywhere near significance in BOTH halves independently is real; one that
only shows up in one half (or flips sign) is the multiple-comparisons artifact v19 flagged
as the live risk.

Run: python research/edge_signal_test_v20_unit_health_replication.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, ttest_ind

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

FIRST_SNAP_SEASON = 2012
LAST_SEASON = 2025
SPLIT_SEASON = 2019  # EARLY = 2012-2018 (7 seasons), LATE = 2019-2025 (7 seasons)
GROUPS = list(v16.POSITION_GROUPS.keys())
SIDES = {"offense": ["qb", "ol", "skill"], "defense": ["dl", "lb", "secondary"]}

# The specific candidates to replicate: v19's 6 "significant" hits + QB (known positive
# control) + OL/DL (known negative controls) -- NOT re-running the full ~28-test matrix,
# since re-scanning everything on each half would just reintroduce the same multiple-
# comparisons problem this check exists to rule out.
OUTCOME_CANDIDATES = [
    ("decimated", "qb", "positive control"),
    ("short", "skill", "v19 hit, p=0.0007"),
    ("short", "offense", "v19 hit, p=0.013"),
    ("decimated", "offense", "v19 hit, p=0.017"),
    ("decimated", "secondary", "v19 hit, p=0.018"),
    ("decimated", "defense", "v19 hit, p=0.043"),
    ("short", "ol", "negative control (null in v19)"),
    ("decimated", "dl", "negative control (null in v19)"),
]
EPA_CANDIDATE = ("decimated", "lb", "pass_def_epa_allowed", "v19 hit, p=0.029")


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


def main():
    print("Loading snap-weighted injury severity (2012+)...", file=sys.stderr)
    player_games, pos_default = v17.load_snap_shares()
    injury_rows = v17.load_injury_rows()
    severity = v17.weighted_absence_by_team_week_group(injury_rows, player_games, pos_default, None)

    print("Building per-team-game rows (2012+)...", file=sys.stderr)
    all_team_rows = v16.build_team_game_rows()
    team_rows = [r for r in all_team_rows if FIRST_SNAP_SEASON <= r["season"] <= LAST_SEASON]
    for r in team_rows:
        for grp in GROUPS:
            r[f"sev_{grp}"] = severity.get((r["team"], r["season"], r["week"], grp), 0.0)
        for side, grps in SIDES.items():
            r[f"sev_{side}"] = sum(r[f"sev_{g}"] for g in grps)

    # FIXED thresholds from the FULL 2012-2025 dataset (same ones v19 reported) -- not
    # recomputed per half, so "decimated" is the same severity bar in both.
    thresholds = {}
    for key in GROUPS + list(SIDES):
        vals = np.array([r[f"sev_{key}"] for r in team_rows])
        thresholds[key] = (np.percentile(vals, 85), np.percentile(vals, 95))
    for r in team_rows:
        for key in GROUPS + list(SIDES):
            p85, p95 = thresholds[key]
            r[f"short_{key}"] = 1 if r[f"sev_{key}"] >= p85 else 0
            r[f"decimated_{key}"] = 1 if r[f"sev_{key}"] >= p95 else 0

    print("Building game-level dataset (for delta + outcome tests)...", file=sys.stderr)
    rows_all = v14.build_dataset()
    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows_all])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS
    for i, r in enumerate(rows_all):
        x = X_all[i]
        r["delta"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))
    games = [r for r in rows_all if FIRST_SNAP_SEASON <= r["season"] <= LAST_SEASON]

    sev_by_team_week = {(r["team"], r["season"], r["week"]): r for r in team_rows}

    def run_outcome_test(game_subset, tag, key):
        y = np.array([g["home_win"] for g in game_subset], dtype=float)
        delta = np.array([g["delta"] for g in game_subset]).reshape(-1, 1)
        diffs = []
        for g in game_subset:
            hr = sev_by_team_week.get((g["home"], g["season"], g["week"]))
            ar = sev_by_team_week.get((g["away"], g["season"], g["week"]))
            hv = hr[f"{tag}_{key}"] if hr else 0
            av = ar[f"{tag}_{key}"] if ar else 0
            diffs.append(hv - av)
        dcol = np.array(diffs, dtype=float).reshape(-1, 1)
        if np.std(dcol) == 0:
            return None
        w_r, se_r, ll_r = fit_logistic(delta, y)
        X_full = np.hstack([delta, dcol])
        w_f, se_f, ll_f = fit_logistic(X_full, y)
        lr = 2 * (ll_f - ll_r)
        p = 1 - chi2.cdf(lr, df=1)
        return w_f[2], se_f[2], p, int(np.sum(np.abs(dcol) > 0))

    early_games = [g for g in games if g["season"] < SPLIT_SEASON]
    late_games = [g for g in games if g["season"] >= SPLIT_SEASON]
    print(f"\nEARLY = {FIRST_SNAP_SEASON}-{SPLIT_SEASON-1} ({len(early_games)} games)   "
          f"LATE = {SPLIT_SEASON}-{LAST_SEASON} ({len(late_games)} games)\n")

    print("=== game-outcome replication (coef, p-value, each half independently) ===")
    print(f"{'indicator':24s} {'note':32s} {'EARLY coef':>11s} {'EARLY p':>8s}   {'LATE coef':>10s} {'LATE p':>8s}   replicates?")
    for tag, key, note in OUTCOME_CANDIDATES:
        er = run_outcome_test(early_games, tag, key)
        lr_ = run_outcome_test(late_games, tag, key)
        if er is None or lr_ is None:
            print(f"{tag+'_'+key:24s} {note:32s}  (degenerate in one half, skipped)")
            continue
        ec, _, ep, en = er
        lc, _, lp, ln = lr_
        same_sign = (ec < 0) == (lc < 0)
        both_suggestive = ep < 0.15 and lp < 0.15
        verdict = "YES" if (same_sign and both_suggestive) else ("partial" if same_sign else "NO")
        print(f"{tag+'_'+key:24s} {note:32s} {ec:+11.3f} {ep:8.4f}   {lc:+10.3f} {lp:8.4f}   {verdict}")

    print("\n=== EPA-residual replication (LB decimated -> pass_def_epa_allowed) ===")
    tag, key, metric, note = EPA_CANDIDATE
    for label, subset in (("EARLY", [r for r in team_rows if r["season"] < SPLIT_SEASON]),
                           ("LATE", [r for r in team_rows if r["season"] >= SPLIT_SEASON])):
        resid = np.array([r[f"{metric}_actual"] - r[f"{metric}_pregame"] for r in subset])
        flag = np.array([r[f"{tag}_{key}"] for r in subset], dtype=bool)
        if flag.sum() < 10 or (~flag).sum() < 10:
            print(f"  {label}: too few decimated weeks ({flag.sum()}) to test")
            continue
        a, b = resid[flag], resid[~flag]
        t, p = ttest_ind(a, b, equal_var=False)
        print(f"  {label}: n_decimated={flag.sum():3d}  diff={a.mean()-b.mean():+.4f}  t={t:+.2f}  p={p:.4f}")


if __name__ == "__main__":
    main()
