"""v43: tests whether shrinking a passer's trailing EPA/dropback toward backup_avg_epa by
sample size changes the v40/v42 decay/reinsertion verdict.

Motivation: `ta` (whoever's actually playing) uses a passer's RAW, unshrunk career trailing
average with no regard for how few attempts that average is built from. A true debut (zero
tracked attempts) falls back to backup_avg_epa, but a passer with even a handful of tracked
attempts (one earlier relief series) gets his literal small-sample average treated with the
same confidence as a 1000-attempt veteran. That's noise in BOTH the static and decay
formulas, but it plausibly matters MORE for decay/reinsertion specifically: the reinsertion
boost FREEZES whatever gap value was computed on the backup's LAST start before the primary
returns, then projects that single snapshot forward (decaying it) over several subsequent
weeks -- a noisy point estimate at exactly the wrong moment gets amplified across multiple
future predictions, instead of being refreshed from new data every week the way the ongoing
penalty is. If that's what sank v40's decay/reinsertion result, fixing the small-sample noise
first should let the real signal (if any) show through.

Empirical-Bayes shrinkage: shrunk = (attempts * raw + k * backup_avg_epa) / (attempts + k),
applied to BOTH tp (primary) and ta (actual) using career-cumulative ATTEMPT counts tracked
alongside trailing_epa (near-zero shrinkage for an established primary with hundreds of
attempts; substantial for a true backup with a handful of starts). Sweeps k against both
static and decay/reinsertion gap constructions, primary fixed at window5 (already validated).

Run: python research/edge_signal_test_v43_qb_epa_shrinkage.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import importlib.util
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v14 = _load("v14", "edge_signal_test_v14_qb_injury_dropout.py")
v40 = _load("v40", "edge_signal_test_v40_qb_recency_decay.py")

FIRST_SEASON = 2009
TEST_START_SEASON = 2014
EWMA_ALPHA = v40.EWMA_ALPHA
PRIMARY_MODE = "window5"


def build_trailing_epa_with_attempts(game_order, actual_passer, game_team_epa):
    """Same no-lookahead walk as v40.build_trailing_epa, but also tracks each passer's
    CUMULATIVE career attempts as of before each game -- the sample size the shrinkage
    below needs. `backup_epa_sum`/`backup_attempts` also computed here (reused for
    backup_avg_epa, the shrinkage prior) since this walk already touches everything needed."""
    passer_hist: dict[str, list[float]] = defaultdict(list)
    passer_attempts: dict[str, int] = defaultdict(int)
    trailing_epa: dict[tuple[str, str], float | None] = {}
    trailing_attempts: dict[tuple[str, str], int] = {}
    league_avg: list[float] = []
    for gid, team, season, week in game_order:
        ap = actual_passer.get((gid, team))
        if ap is None:
            continue
        hist = passer_hist[ap]
        trailing_epa[(ap, gid)] = (sum(hist) / len(hist)) if hist else None
        trailing_attempts[(ap, gid)] = passer_attempts[ap]
        epa_sum, attempts = game_team_epa[(gid, team)]
        if attempts:
            per_play = epa_sum / attempts
            hist.append(per_play)
            league_avg.append(per_play)
            passer_attempts[ap] += attempts
    league_avg_epa = float(np.mean(league_avg)) if league_avg else 0.0
    return trailing_epa, trailing_attempts, league_avg_epa


def shrink(raw: float, attempts: int, prior: float, k: float) -> float:
    if k <= 0:
        return raw
    return (attempts * raw + k * prior) / (attempts + k)


def build_qb_gap_shrunk(game_order, actual_passer, primary_so_far, trailing_epa,
                         trailing_attempts, league_avg_epa, backup_avg_epa, k: float, dynamic: bool):
    """Same two formulas as v40.build_qb_gap (static vs decay/reinsertion), but tp/ta are
    shrunk toward backup_avg_epa by sample size first -- matching production's own fallback
    convention (tp falls back to league_avg_epa on no history at all, ta falls back to
    backup_avg_epa, since an unknown passer thrust into playing is better modeled as
    backup-average than league-average)."""
    gap: dict[tuple[str, str], float] = {}

    def shrunk_tp(primary, gid):
        raw = trailing_epa.get((primary, gid))
        raw = raw if raw is not None else league_avg_epa
        n = trailing_attempts.get((primary, gid), 0)
        return shrink(raw, n, backup_avg_epa, k)

    def shrunk_ta(actual, gid):
        raw = trailing_epa.get((actual, gid))
        raw = raw if raw is not None else backup_avg_epa
        n = trailing_attempts.get((actual, gid), 0)
        return shrink(raw, n, backup_avg_epa, k)

    if not dynamic:
        for gid, team, season, week in game_order:
            primary = primary_so_far.get((team, gid))
            actual = actual_passer.get((gid, team))
            if primary is None or actual is None:
                gap[(team, gid)] = 0.0
                continue
            gap[(team, gid)] = shrunk_tp(primary, gid) - shrunk_ta(actual, gid)
        return gap

    streak_n: dict[str, int] = defaultdict(int)
    pending_boost: dict[str, float] = defaultdict(float)
    for gid, team, season, week in game_order:
        primary = primary_so_far.get((team, gid))
        actual = actual_passer.get((gid, team))
        if primary is None or actual is None:
            gap[(team, gid)] = 0.0
            continue
        raw_gap = shrunk_tp(primary, gid) - shrunk_ta(actual, gid)

        if actual != primary:
            streak_n[team] += 1
            decay_factor = (1 - EWMA_ALPHA) ** (streak_n[team] - 1)
            gap[(team, gid)] = raw_gap * decay_factor
            pending_boost[team] = raw_gap * decay_factor
        else:
            streak_n[team] = 0
            if pending_boost[team]:
                gap[(team, gid)] = -pending_boost[team]
                pending_boost[team] *= (1 - EWMA_ALPHA)
            else:
                gap[(team, gid)] = 0.0
    return gap


def fit_logistic(X, y):
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
    return res.x


def evaluate(games, gap_diff_key):
    seasons = sorted({g["season"] for g in games})
    all_y, all_p = [], []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [g for g in games if g["season"] < test_season]
        test = [g for g in games if g["season"] == test_season]
        if len(train) < 300 or not test:
            continue
        gap_train = np.array([g[gap_diff_key] for g in train])
        gmu, gsd = gap_train.mean(), (gap_train.std() or 1.0)
        def feat(g):
            return [g["delta"], (g[gap_diff_key] - gmu) / gsd]
        X_train = np.array([feat(g) for g in train])
        y_train = np.array([g["home_win"] for g in train], dtype=float)
        w = fit_logistic(X_train, y_train)
        for g in test:
            f = feat(g)
            z = w[0] + w[1] * f[0] + w[2] * f[1]
            p_hat = 1.0 / (1.0 + np.exp(-z))
            all_y.append(g["home_win"])
            all_p.append(p_hat)
    all_y = np.array(all_y)
    all_p = np.array(all_p)
    acc = float(np.mean((all_p > 0.5).astype(int) == all_y))
    auc = roc_auc_score(all_y, all_p)
    ll = log_loss(all_y, all_p)
    return len(all_y), acc, auc, ll


def main():
    passer_df = v40.load_passer_df()
    print(f"{len(passer_df)} (game, team, passer) rows\n")
    actual_passer, game_team_epa, game_order = v40.build_game_order(passer_df)
    primary_so_far = v40.build_primary_so_far(game_order, actual_passer, game_team_epa, PRIMARY_MODE)
    trailing_epa, trailing_attempts, league_avg_epa = build_trailing_epa_with_attempts(game_order, actual_passer, game_team_epa)

    # backup_avg_epa (the shrinkage prior) -- same definition as production: EPA/play summed
    # over every game where the actual passer differed from that week's window5 primary.
    backup_epa_sum = backup_attempts = 0.0
    for gid, team, season, week in game_order:
        primary = primary_so_far.get((team, gid))
        ap = actual_passer.get((gid, team))
        if primary is not None and ap is not None and ap != primary:
            epa_sum, attempts = game_team_epa[(gid, team)]
            backup_epa_sum += epa_sum
            backup_attempts += attempts
    backup_avg_epa = (backup_epa_sum / backup_attempts) if backup_attempts else league_avg_epa
    print(f"backup_avg_epa (window5 definition) = {backup_avg_epa:+.4f}\n")

    print("Building game dataset + frozen production delta...", file=sys.stderr)
    rows_all = v14.build_dataset()
    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows_all])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS
    for i, r in enumerate(rows_all):
        x = X_all[i]
        r["delta"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))
    games = [r for r in rows_all if r["season"] >= FIRST_SEASON and r["delta"] != 0]

    k_grid = [0, 25, 50, 100, 200, 400]
    gap_modes = [("static", False), ("decay", True)]

    print(f"=== walk-forward comparison, {TEST_START_SEASON}-2025, primary=window5, delta + qb_gap_diff ===")
    print(f"{'k':>5s} {'gap':7s} {'n':>6s} {'acc':>7s} {'AUC':>7s} {'logloss':>8s}")
    results = []
    for k in k_grid:
        for g_label, dynamic in gap_modes:
            gap = build_qb_gap_shrunk(game_order, actual_passer, primary_so_far, trailing_epa,
                                       trailing_attempts, league_avg_epa, backup_avg_epa, k, dynamic)
            key = f"gap_k{k}_{g_label}"
            for g in games:
                g[key] = gap.get((g["home"], g["game_id"]), 0.0) - gap.get((g["away"], g["game_id"]), 0.0)
            n, acc, auc, ll = evaluate(games, key)
            print(f"{k:5d} {g_label:7s} {n:6d} {100*acc:6.2f}% {auc:7.4f} {ll:8.4f}")
            results.append((k, g_label, n, acc, auc, ll))

    print("\n=== best variant by AUC ===")
    best = max(results, key=lambda r: r[4])
    print(f"k={best[0]} gap={best[1]}  AUC={best[4]:.4f}  acc={100*best[3]:.2f}%  logloss={best[5]:.4f}")
    prod = next(r for r in results if r[0] == 0 and r[1] == "static")
    print(f"k=0 static (= v40/v42 baseline) AUC={prod[4]:.4f}  acc={100*prod[3]:.2f}%  logloss={prod[5]:.4f}")
    print(f"delta AUC vs that baseline: {best[4]-prod[4]:+.4f}")


if __name__ == "__main__":
    main()
