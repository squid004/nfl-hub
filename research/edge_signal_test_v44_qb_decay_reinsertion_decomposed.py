"""v44: decomposes v40's bundled "dynamic" QB-gap variant (decay + reinsertion together)
into its two independent halves, and evaluates each both on the full walk-forward population
AND restricted to just the games where some QB-health signal is actually nonzero.

Why: v40/v43 only ever tested "both decay and reinsertion together" vs "neither" -- never
decay alone or reinsertion alone. If one half is doing something real and the other is pure
noise, a bundled test can't tell them apart; the bad half can mask the good one. Separately,
every comparison so far scored AUC/logloss over the FULL 3,151-game walk-forward population,
but only ~27-36% of those games have a nonzero qb_gap_diff under any formula -- the rest are
identical across every variant being compared and dilute the real effect. Restricting
evaluation (not training -- the fit still uses all data) to the union of games where ANY
variant produces a nonzero signal gives a cleaner read on whether a given mechanism helps or
hurts WHERE it actually applies.

Four variants, window5 primary (already validated) held fixed throughout:
  - static:          no decay, no reinsertion (today's shipped formula, the control)
  - decay-only:       penalty decays by (1-EWMA_ALPHA)**(n-1) while the backup keeps starting,
                      snaps to 0 the instant the primary returns (no boost)
  - reinsertion-only: penalty stays FLAT (undecayed) the whole absence, but credits a
                      decaying boost back once the primary returns, instead of snapping to 0
  - both:             v40's original bundled mechanism (decay AND reinsertion together) --
                      reproducing it here is also a consistency check on this script itself

Run: python research/edge_signal_test_v44_qb_decay_reinsertion_decomposed.py
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


def build_qb_gap_variant(game_order, actual_passer, primary_so_far, trailing_epa,
                          league_avg_epa, backup_avg_epa, decay_penalty: bool, reinsertion_boost: bool):
    gap: dict[tuple[str, str], float] = {}
    streak_n: dict[str, int] = defaultdict(int)
    pending_boost: dict[str, float] = defaultdict(float)
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
            decay_factor = (1 - EWMA_ALPHA) ** (streak_n[team] - 1) if decay_penalty else 1.0
            value = raw_gap * decay_factor
            gap[(team, gid)] = value
            if reinsertion_boost:
                pending_boost[team] = value
        else:
            streak_n[team] = 0
            if reinsertion_boost and pending_boost[team]:
                gap[(team, gid)] = -pending_boost[team]
                pending_boost[team] *= (1 - EWMA_ALPHA)
                # A geometric decay never actually HITS zero in floating point -- left
                # unguarded, a team's first-ever backup start (anywhere in 2009-2025) leaves a
                # vanishingly tiny but still-truthy leftover applied to every one of its games
                # forever after. Caught via the "affected games" sanity check coming back as
                # 99.4% of the whole test population instead of the expected ~30%.
                if abs(pending_boost[team]) < 1e-6:
                    pending_boost[team] = 0.0
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


def evaluate(games, gap_diff_key, restrict_ids=None):
    """Walk-forward fit on ALL training data every fold (unchanged); only the TEST rows
    scored are optionally restricted to `restrict_ids` (game_ids), so the subset analysis
    doesn't bias the fit itself, only which rows the reported metrics are computed over."""
    seasons = sorted({g["season"] for g in games})
    all_y, all_p = [], []
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
            all_y.append(g["home_win"])
            all_p.append(p_hat)
    if not all_y:
        return 0, float("nan"), float("nan"), float("nan")
    all_y = np.array(all_y)
    all_p = np.array(all_p)
    acc = float(np.mean((all_p > 0.5).astype(int) == all_y))
    auc = roc_auc_score(all_y, all_p)
    ll = log_loss(all_y, all_p)
    return len(all_y), acc, auc, ll


def collect_predictions(games, gap_diff_key, restrict_ids=None):
    """Same walk-forward procedure as evaluate(), but returns per-game (game_id, y, p) rows
    instead of summary stats -- needed to PAIR predictions across two variants (same games,
    same train/test splits) for a bootstrap significance test."""
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
    return rows


def bootstrap_auc_diff(games, key_a, key_b, restrict_ids, n_boot=2000, seed=0):
    """Paired bootstrap (resample GAMES with replacement, same resample applied to both
    variants) on the AUC(key_b) - AUC(key_a) difference, restricted to the same subset both
    were evaluated on. Reports the mean difference, a 95% CI, and the fraction of resamples
    where b beats a -- a quick stand-in for a proper significance test."""
    rows_a = {gid: (y, p) for gid, y, p in collect_predictions(games, key_a, restrict_ids)}
    rows_b = {gid: (y, p) for gid, y, p in collect_predictions(games, key_b, restrict_ids)}
    common = sorted(set(rows_a) & set(rows_b))
    y = np.array([rows_a[g][0] for g in common])
    p_a = np.array([rows_a[g][1] for g in common])
    p_b = np.array([rows_b[g][1] for g in common])
    n = len(common)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if len(set(yi)) < 2:
            continue
        diffs.append(roc_auc_score(yi, p_b[idx]) - roc_auc_score(yi, p_a[idx]))
    diffs = np.array(diffs)
    return n, diffs


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

    variants = {
        "static":          (False, False),
        "decay-only":      (True,  False),
        "reinsertion-only":(False, True),
        "both":            (True,  True),
    }
    gaps = {}
    for name, (decay_penalty, reinsertion_boost) in variants.items():
        gap = build_qb_gap_variant(game_order, actual_passer, primary_so_far, trailing_epa,
                                    league_avg_epa, backup_avg_epa, decay_penalty, reinsertion_boost)
        key = f"gap_{name}"
        for g in games:
            g[key] = gap.get((g["home"], g["game_id"]), 0.0) - gap.get((g["away"], g["game_id"]), 0.0)
        gaps[name] = key

    # Defined against STATIC specifically (a currently-known, active absence for at least one
    # team) rather than a union across all 4 variants -- reinsertion-only/both's pending_boost
    # has a long decay tail at EWMA_ALPHA=0.2 (~55 games to fall below 1e-6), and real QB
    # absences recur often enough across 32 teams over 16 seasons that the union was ~97% of
    # the whole population, not a meaningfully tighter subset. Static's own nonzero games are
    # unambiguous: an actual non-primary passer played for at least one team that week.
    test_games = [g for g in games if g["season"] >= TEST_START_SEASON]
    affected_ids = {g["game_id"] for g in test_games if g[gaps["static"]] != 0}
    print(f"affected games (static nonzero -- an active, known absence) in the {TEST_START_SEASON}-2025 test window: "
          f"{len(affected_ids)} / {len(test_games)} ({100*len(affected_ids)/len(test_games):.1f}%)\n")

    print(f"=== FULL population, {TEST_START_SEASON}-2025 ===")
    print(f"{'variant':18s} {'n':>6s} {'acc':>7s} {'AUC':>7s} {'logloss':>8s}")
    full_results = {}
    for name in variants:
        n, acc, auc, ll = evaluate(games, gaps[name])
        full_results[name] = (n, acc, auc, ll)
        print(f"{name:18s} {n:6d} {100*acc:6.2f}% {auc:7.4f} {ll:8.4f}")

    print(f"\n=== RESTRICTED to the {len(affected_ids)} affected games only ===")
    print(f"{'variant':18s} {'n':>6s} {'acc':>7s} {'AUC':>7s} {'logloss':>8s}")
    restricted_results = {}
    for name in variants:
        n, acc, auc, ll = evaluate(games, gaps[name], restrict_ids=affected_ids)
        restricted_results[name] = (n, acc, auc, ll)
        print(f"{name:18s} {n:6d} {100*acc:6.2f}% {auc:7.4f} {ll:8.4f}")

    print("\n=== summary: best variant in each view ===")
    best_full = max(full_results, key=lambda k: full_results[k][2])
    best_restr = max(restricted_results, key=lambda k: restricted_results[k][2])
    print(f"Full population best AUC: {best_full} ({full_results[best_full][2]:.4f}) vs static ({full_results['static'][2]:.4f})")
    print(f"Restricted best AUC:      {best_restr} ({restricted_results[best_restr][2]:.4f}) vs static ({restricted_results['static'][2]:.4f})")

    print("\n=== paired bootstrap, decay-only vs static, restricted to the affected subset (n=2000 resamples) ===")
    n, diffs = bootstrap_auc_diff(games, gaps["static"], gaps["decay-only"], affected_ids)
    print(f"{n} paired games, {len(diffs)} valid resamples")
    print(f"AUC(decay-only) - AUC(static): mean={diffs.mean():+.4f}, 95% CI=({np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f})")
    print(f"fraction of resamples where decay-only beats static: {100*np.mean(diffs > 0):.1f}%")


if __name__ == "__main__":
    main()
