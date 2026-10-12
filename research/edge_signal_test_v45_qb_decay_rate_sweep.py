"""v45: sweeps the penalty-decay RATE as a free parameter, instead of assuming it equals
EWMA_ALPHA (0.2, the team-level rolling rating's own half-life) the way v40/v44 did.

v44 decomposed v40's bundled decay+reinsertion mechanism and found the penalty-decay half
alone (no reinsertion boost) looks real: AUC +0.0020, logloss -0.0021 vs static, both metrics
agreeing, isolated to the 1,011 games where an active absence actually applies -- but a
paired bootstrap put the 95% CI at (-0.0005, +0.0046), just barely straddling zero (93.8% of
resamples favor it). The rate was never fit; it was borrowed wholesale from a DIFFERENT
part of the model (the team rolling rating's own decay) on the theory that it should transfer.
If 0.2 isn't the right rate for THIS signal specifically, a properly-swept rate could either
tighten the borderline result into something clearly real, or reveal that decay_rate=0
(static) was close to optimal all along and 0.2 just happened to land near it by luck.

decay_rate=0 reproduces static exactly ((1-0)**(n-1) = 1 for all n -- no decay at all).
decay_rate=1 is the opposite extreme: full penalty on a backup's first start, zero on every
start after that. Reinsertion stays OFF throughout (that half was confirmed to hurt in v44);
only the ongoing-absence penalty's decay rate varies. Primary fixed at window5, evaluation
restricted to the 1,011-game active-absence subset (the lens v44 showed was NOT diluted by
irrelevant zero-everywhere games), same TEST_START_SEASON=2014 walk-forward as every prior
script in this family.

Run: python research/edge_signal_test_v45_qb_decay_rate_sweep.py
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
PRIMARY_MODE = "window5"


def build_qb_gap_decay_rate(game_order, actual_passer, primary_so_far, trailing_epa,
                             league_avg_epa, backup_avg_epa, decay_rate: float):
    """decay-only (no reinsertion), `decay_rate` free instead of fixed at EWMA_ALPHA."""
    gap: dict[tuple[str, str], float] = {}
    streak_n: dict[str, int] = defaultdict(int)
    for gid, team, season, week in game_order:
        primary = primary_so_far.get((team, gid))
        actual = actual_passer.get((gid, team))
        if primary is None or actual is None:
            gap[(team, gid)] = 0.0
            continue
        tp = trailing_epa.get((primary, gid))
        tp = tp if tp is not None else league_avg_epa
        ta = trailing_epa.get((actual, gid))
        ta = ta if ta is not None else backup_avg_epa
        raw_gap = tp - ta

        if actual != primary:
            streak_n[team] += 1
            decay_factor = (1 - decay_rate) ** (streak_n[team] - 1)
            gap[(team, gid)] = raw_gap * decay_factor
        else:
            streak_n[team] = 0
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


def evaluate_and_collect(games, gap_diff_key, restrict_ids=None):
    """Walk-forward fit + score -- returns BOTH summary stats and the raw per-game (y, p)
    pairs in one pass (v44 split these into two functions; folding them together here since
    the rate sweep calls this many times and refitting twice per rate would double the cost)."""
    seasons = sorted({g["season"] for g in games})
    rows = []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [g for g in games if g["season"] < test_season]
        test = [g for g in games if g["season"] == test_season]
        if restrict_ids is not None:
            test = [g for g in test if g["game_id"] in restrict_ids]
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
            rows.append((g["game_id"], g["home_win"], p_hat))
    if not rows:
        return (0, float("nan"), float("nan"), float("nan")), {}
    y = np.array([r[1] for r in rows])
    p = np.array([r[2] for r in rows])
    acc = float(np.mean((p > 0.5).astype(int) == y))
    auc = roc_auc_score(y, p)
    ll = log_loss(y, p)
    by_id = {r[0]: (r[1], r[2]) for r in rows}
    return (len(rows), acc, auc, ll), by_id


def main():
    passer_df = v40.load_passer_df()
    print(f"{len(passer_df)} (game, team, passer) rows\n")
    actual_passer, game_team_epa, game_order = v40.build_game_order(passer_df)
    primary_so_far = v40.build_primary_so_far(game_order, actual_passer, game_team_epa, PRIMARY_MODE)
    trailing_epa, league_avg_epa = v40.build_trailing_epa(game_order, actual_passer, game_team_epa)

    backup_epa_sum = backup_attempts = 0.0
    for gid, team, season, week in game_order:
        primary = primary_so_far.get((team, gid))
        ap = actual_passer.get((gid, team))
        if primary is not None and ap is not None and ap != primary:
            epa_sum, attempts = game_team_epa[(gid, team)]
            backup_epa_sum += epa_sum
            backup_attempts += attempts
    backup_avg_epa = (backup_epa_sum / backup_attempts) if backup_attempts else league_avg_epa

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

    # Static's own nonzero games (an active, currently-known absence) -- same subset v44 used,
    # computed once up front via decay_rate=0 (mathematically identical to static).
    gap0 = build_qb_gap_decay_rate(game_order, actual_passer, primary_so_far, trailing_epa, league_avg_epa, backup_avg_epa, 0.0)
    for g in games:
        g["_gap0"] = gap0.get((g["home"], g["game_id"]), 0.0) - gap0.get((g["away"], g["game_id"]), 0.0)
    test_games = [g for g in games if g["season"] >= TEST_START_SEASON]
    affected_ids = {g["game_id"] for g in test_games if g["_gap0"] != 0}
    print(f"affected games (active absence): {len(affected_ids)} / {len(test_games)} ({100*len(affected_ids)/len(test_games):.1f}%)\n")

    rate_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0]
    print(f"=== decay-rate sweep, RESTRICTED to the {len(affected_ids)} active-absence games, {TEST_START_SEASON}-2025 ===")
    print(f"{'rate':>6s} {'n':>6s} {'acc':>7s} {'AUC':>7s} {'logloss':>8s}")
    results = {}
    preds = {}
    for rate in rate_grid:
        gap = build_qb_gap_decay_rate(game_order, actual_passer, primary_so_far, trailing_epa, league_avg_epa, backup_avg_epa, rate)
        key = f"gap_r{rate}"
        for g in games:
            g[key] = gap.get((g["home"], g["game_id"]), 0.0) - gap.get((g["away"], g["game_id"]), 0.0)
        (n, acc, auc, ll), by_id = evaluate_and_collect(games, key, restrict_ids=affected_ids)
        results[rate] = (n, acc, auc, ll)
        preds[rate] = by_id
        flag = "  <- EWMA_ALPHA (v40/v44's assumed rate)" if rate == 0.2 else ("  <- static (no decay)" if rate == 0.0 else "")
        print(f"{rate:6.2f} {n:6d} {100*acc:6.2f}% {auc:7.4f} {ll:8.4f}{flag}")

    best_rate = max(results, key=lambda r: results[r][2])
    print(f"\nBest rate by AUC: {best_rate} (AUC={results[best_rate][2]:.4f}) vs static/rate=0 (AUC={results[0.0][2]:.4f})")

    if best_rate != 0.0:
        print(f"\n=== paired bootstrap, rate={best_rate} vs static, same {len(affected_ids)}-game subset (2000 resamples) ===")
        common = sorted(set(preds[0.0]) & set(preds[best_rate]))
        y = np.array([preds[0.0][g][0] for g in common])
        p_static = np.array([preds[0.0][g][1] for g in common])
        p_best = np.array([preds[best_rate][g][1] for g in common])
        rng = np.random.default_rng(0)
        n = len(common)
        diffs = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            yi = y[idx]
            if len(set(yi)) < 2:
                continue
            diffs.append(roc_auc_score(yi, p_best[idx]) - roc_auc_score(yi, p_static[idx]))
        diffs = np.array(diffs)
        print(f"{n} paired games, {len(diffs)} valid resamples")
        print(f"AUC(rate={best_rate}) - AUC(static): mean={diffs.mean():+.4f}, 95% CI=({np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f})")
        print(f"fraction of resamples favoring rate={best_rate}: {100*np.mean(diffs>0):.1f}%")


if __name__ == "__main__":
    main()
