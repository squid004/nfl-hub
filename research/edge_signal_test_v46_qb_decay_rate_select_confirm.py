"""v46: fixes v45's optimism-bias problem (it picked the best of 12 decay rates using the
SAME data it then "confirmed" against) with an honest nested select/confirm split -- and
tests the user's own concern about WHICH kind of split is right: a CHRONOLOGICAL one (early
seasons select, late seasons confirm, matching this project's walk-forward convention
everywhere else -- e.g. v36's trap-game select/confirm test) risks picking an early-era
optimum that's already stale if the true best rate has drifted over the NFL's history, while
a RANDOM split (ignore season, split the 1,011 affected games in half regardless of year)
stays representative of the whole era but can't detect that kind of drift at all -- it only
tests whether a stable, time-invariant relationship exists. Runs both and compares.

IMPORTANT: the underlying walk-forward PREDICTIONS (train strictly on seasons before the one
being scored, exactly as this would be deployed) are identical to v45 for every game and
every rate -- never shuffled or leaked across time in EITHER scheme. What differs between
"chronological" and "random" is ONLY which subset of those already-honest predictions gets
used to SELECT the best rate by AUC vs which held-out subset gets used to CONFIRM it. No game
is ever in both the select and confirm subsets, in either scheme.

Chronological has one natural split point (computed from the data -- the season where the
cumulative affected-game count first reaches half the total -- not hardcoded). Random is
repeated 30x with different seeds, since any single random split could be lucky or unlucky;
reports the distribution of selected rates and confirm-set results across all 30.

Run: python research/edge_signal_test_v46_qb_decay_rate_select_confirm.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

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
v45 = _load("v45", "edge_signal_test_v45_qb_decay_rate_sweep.py")

FIRST_SEASON = 2009
TEST_START_SEASON = 2014
PRIMARY_MODE = "window5"
RATE_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0]
N_RANDOM_REPEATS = 30


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


def walk_forward_predictions(games, gap_diff_key, restrict_ids):
    """Same strictly-chronological train-on-the-past procedure as every evaluate() in this
    family -- returns {game_id: (y, p)} for games in `restrict_ids` only. This is the ONE
    thing that must never be randomized: a game's prediction always comes from a model fit
    only on seasons before it, exactly as it would be deployed."""
    seasons = sorted({g["season"] for g in games})
    out = {}
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [g for g in games if g["season"] < test_season]
        test = [g for g in games if g["season"] == test_season and g["game_id"] in restrict_ids]
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
            out[g["game_id"]] = (g["home_win"], p_hat)
    return out


def auc_on(preds: dict, ids) -> float:
    rows = [preds[g] for g in ids if g in preds]
    if len(rows) < 10:
        return float("nan")
    y = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows])
    if len(set(y)) < 2:
        return float("nan")
    return roc_auc_score(y, p)


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

    gap0 = v45.build_qb_gap_decay_rate(game_order, actual_passer, primary_so_far, trailing_epa, league_avg_epa, backup_avg_epa, 0.0)
    for g in games:
        g["_gap0"] = gap0.get((g["home"], g["game_id"]), 0.0) - gap0.get((g["away"], g["game_id"]), 0.0)
    test_games = [g for g in games if g["season"] >= TEST_START_SEASON]
    affected_ids = {g["game_id"] for g in test_games if g["_gap0"] != 0}
    season_of = {g["game_id"]: g["season"] for g in test_games}
    print(f"affected games (active absence): {len(affected_ids)} / {len(test_games)} ({100*len(affected_ids)/len(test_games):.1f}%)")

    print("\nComputing walk-forward predictions for every rate in the grid (this is the slow part)...", file=sys.stderr)
    preds: dict[float, dict] = {}
    for rate in RATE_GRID:
        gap = v45.build_qb_gap_decay_rate(game_order, actual_passer, primary_so_far, trailing_epa, league_avg_epa, backup_avg_epa, rate)
        key = f"gap_r{rate}"
        for g in games:
            g[key] = gap.get((g["home"], g["game_id"]), 0.0) - gap.get((g["away"], g["game_id"]), 0.0)
        preds[rate] = walk_forward_predictions(games, key, affected_ids)
        print(f"  rate={rate} done", file=sys.stderr)

    # --- Chronological split: data-driven cut point (first season where cumulative affected
    # games reaches half the total), not hardcoded.
    counts_by_season = Counter(season_of[g] for g in affected_ids)
    half = len(affected_ids) / 2
    running = 0
    cut_season = None
    for s in sorted(counts_by_season):
        running += counts_by_season[s]
        if running >= half:
            cut_season = s
            break
    select_chrono = {g for g in affected_ids if season_of[g] <= cut_season}
    confirm_chrono = affected_ids - select_chrono
    print(f"\n=== CHRONOLOGICAL split: select <= {cut_season} ({len(select_chrono)} games), confirm > {cut_season} ({len(confirm_chrono)} games) ===")
    print(f"{'rate':>6s} {'select AUC':>11s}")
    for rate in RATE_GRID:
        print(f"{rate:6.2f} {auc_on(preds[rate], select_chrono):11.4f}")
    best_chrono = max(RATE_GRID, key=lambda r: auc_on(preds[r], select_chrono))
    auc_confirm_best = auc_on(preds[best_chrono], confirm_chrono)
    auc_confirm_static = auc_on(preds[0.0], confirm_chrono)
    print(f"selected rate = {best_chrono} (select AUC={auc_on(preds[best_chrono], select_chrono):.4f})")
    print(f"CONFIRM set: AUC(rate={best_chrono})={auc_confirm_best:.4f} vs AUC(static)={auc_confirm_static:.4f}  "
          f"(delta {auc_confirm_best-auc_confirm_static:+.4f})")

    # --- Random split: repeated N_RANDOM_REPEATS times.
    print(f"\n=== RANDOM split: {N_RANDOM_REPEATS} repeats, ~50/50 ({len(affected_ids)} affected games total) ===")
    ids_list = sorted(affected_ids)
    n = len(ids_list)
    half_n = n // 2
    selected_rates = []
    confirm_deltas = []
    for seed in range(N_RANDOM_REPEATS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(ids_list)
        select_ids = set(perm[:half_n].tolist())
        confirm_ids = set(perm[half_n:].tolist())
        best_rate = max(RATE_GRID, key=lambda r: auc_on(preds[r], select_ids))
        auc_c_best = auc_on(preds[best_rate], confirm_ids)
        auc_c_static = auc_on(preds[0.0], confirm_ids)
        selected_rates.append(best_rate)
        if not (np.isnan(auc_c_best) or np.isnan(auc_c_static)):
            confirm_deltas.append(auc_c_best - auc_c_static)

    rate_counts = Counter(selected_rates)
    print("selected-rate distribution across repeats:", dict(sorted(rate_counts.items())))
    deltas = np.array(confirm_deltas)
    print(f"confirm-set AUC(selected rate) - AUC(static): mean={deltas.mean():+.4f}, median={np.median(deltas):+.4f}, "
          f"fraction positive={100*np.mean(deltas>0):.1f}%  (n={len(deltas)} valid repeats)")

    print("\n=== comparison ===")
    print(f"Chronological: selected rate={best_chrono}, confirm-set delta={auc_confirm_best-auc_confirm_static:+.4f}")
    print(f"Random (mean over {N_RANDOM_REPEATS} repeats): confirm-set delta={deltas.mean():+.4f}, "
          f"most-selected rate={rate_counts.most_common(1)[0][0]}")


if __name__ == "__main__":
    main()
