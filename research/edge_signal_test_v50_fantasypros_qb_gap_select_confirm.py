"""v50: honest confirmation of v49's QB fantasy-gap finding, same chronological select/
confirm + bootstrap discipline used on the decay-rate work (v46) before trusting a result
built on just one walk-forward pass.

v49 found that REPLACING the shipped EPA-based QB-health gap with a FantasyPros-projection-
based one improves accuracy/AUC/logloss on every metric, restricted to the 559 games with an
active QB change (fantasy_gap: 64.76%/0.6876/0.6434 vs epa_gap: 62.43%/0.6778/0.6500) -- and
unlike v48's team-total signal, this isn't just a market-spread proxy (corr with -spread_line
is 0.187-0.201 for both, nowhere near v48's 0.956). That clears a real bar, but it's still
one walk-forward pass on a modest sample; this script checks whether it replicates
out-of-sample on data the selection process never touched.

Chronological split: pick the best of {epa_gap, fantasy_gap, both} by AUC using only the
EARLIER half of the 559 affected games (by season, data-driven cutpoint, not hardcoded),
then confirm on the LATER half -- same structure as v46. Reports the confirm-set result for
the selected feature set vs. epa_gap alone (the shipped baseline), plus a bootstrap on that
specific comparison.

Run: python research/edge_signal_test_v50_fantasypros_qb_gap_select_confirm.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
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
v49 = _load("v49", "edge_signal_test_v49_fantasypros_qb_gap.py")

TEST_START_SEASON = 2020
FEATURE_SETS = {
    "epa_gap": ["epa_gap_diff"],
    "fantasy_gap": ["fantasy_gap_diff"],
    "both": ["epa_gap_diff", "fantasy_gap_diff"],
}


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


def walk_forward_predictions(games, feature_keys, restrict_ids):
    """Same strictly-chronological train-on-the-past procedure as v46 -- returns
    {game_id: (y, p)} for games in `restrict_ids` only."""
    seasons = sorted({g["season"] for g in games})
    out = {}
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [g for g in games if g["season"] < test_season]
        test = [g for g in games if g["season"] == test_season and g["game_id"] in restrict_ids]
        if len(train) < 150 or not test:
            continue
        stats = {}
        for k in feature_keys:
            vals = np.array([g[k] for g in train])
            stats[k] = (vals.mean(), vals.std() or 1.0)
        def feat(g):
            return [g["delta"]] + [(g[k] - stats[k][0]) / stats[k][1] for k in feature_keys]
        X_train = np.array([feat(g) for g in train])
        y_train = np.array([g["home_win"] for g in train], dtype=float)
        w = fit_logistic(X_train, y_train)
        for g in test:
            f = feat(g)
            z = w[0] + sum(wi * fi for wi, fi in zip(w[1:], f))
            p_hat = 1.0 / (1.0 + np.exp(-z))
            out[g["game_id"]] = (g["home_win"], p_hat)
    return out


def auc_on(preds, ids):
    rows = [preds[g] for g in ids if g in preds]
    if len(rows) < 10:
        return float("nan")
    y = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows])
    if len(set(y)) < 2:
        return float("nan")
    return roc_auc_score(y, p)


def main():
    print("Rebuilding v49's game/gap data...", file=sys.stderr)
    passer_df = v40.load_passer_df()
    actual_passer, game_team_epa, game_order = v40.build_game_order(passer_df)
    primary_so_far = v40.build_primary_so_far(game_order, actual_passer, game_team_epa, v49.PRIMARY_MODE)
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

    ids = pd.read_parquet(os.path.join(HERE, "cache", "fantasypros_id_map.parquet"))
    ids = ids.dropna(subset=["fantasypros_id", "gsis_id"]).copy()
    ids["fantasypros_id"] = ids["fantasypros_id"].astype(int)
    gsis_to_fpid = dict(zip(ids["gsis_id"], ids["fantasypros_id"]))

    pw = pd.read_parquet(os.path.join(HERE, "cache", "fantasypros_player_week.parquet"))
    pw_qb = pw[pw.position == "QB"]
    player_proj = {(r.season, r.week, r.fpid): r.proj_points for r in pw_qb.itertuples(index=False)}

    trailing_proj, league_avg_proj = v49.build_trailing_fantasy_proj(game_order, actual_passer, gsis_to_fpid, player_proj)
    epa_gap = v45.build_qb_gap_decay_rate(game_order, actual_passer, primary_so_far, trailing_epa, league_avg_epa, backup_avg_epa, v49.DECAY_RATE)
    fantasy_gap = v49.build_fantasy_gap(game_order, actual_passer, primary_so_far, trailing_proj, league_avg_proj, gsis_to_fpid, player_proj)

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

    games = []
    for r in rows_all:
        if r["delta"] == 0 or r["season"] < 2017:
            continue
        r["epa_gap_diff"] = epa_gap.get((r["home"], r["game_id"]), 0.0) - epa_gap.get((r["away"], r["game_id"]), 0.0)
        r["fantasy_gap_diff"] = fantasy_gap.get((r["home"], r["game_id"]), 0.0) - fantasy_gap.get((r["away"], r["game_id"]), 0.0)
        games.append(r)

    test_games = [g for g in games if g["season"] >= TEST_START_SEASON]
    affected_ids = {g["game_id"] for g in test_games if g["epa_gap_diff"] != 0 or g["fantasy_gap_diff"] != 0}
    season_of = {g["game_id"]: g["season"] for g in test_games}
    print(f"{len(affected_ids)} affected games (active QB change), {TEST_START_SEASON}-2025\n")

    print("Computing walk-forward predictions for each feature set...", file=sys.stderr)
    preds = {name: walk_forward_predictions(games, keys, affected_ids) for name, keys in FEATURE_SETS.items()}

    # Chronological split: data-driven cut point (first season where cumulative affected-game
    # count reaches half the total), same convention as v46.
    counts_by_season = Counter(season_of[g] for g in affected_ids)
    half = len(affected_ids) / 2
    running = 0
    cut_season = None
    for s in sorted(counts_by_season):
        running += counts_by_season[s]
        if running >= half:
            cut_season = s
            break
    select_ids = {g for g in affected_ids if season_of[g] <= cut_season}
    confirm_ids = affected_ids - select_ids
    print(f"=== CHRONOLOGICAL split: select <= {cut_season} ({len(select_ids)} games), confirm > {cut_season} ({len(confirm_ids)} games) ===")
    for name in FEATURE_SETS:
        print(f"  {name:12s} select AUC = {auc_on(preds[name], select_ids):.4f}")
    best = max(FEATURE_SETS, key=lambda k: auc_on(preds[k], select_ids))
    auc_confirm_best = auc_on(preds[best], confirm_ids)
    auc_confirm_epa = auc_on(preds["epa_gap"], confirm_ids)
    print(f"selected: {best} (select AUC={auc_on(preds[best], select_ids):.4f})")
    print(f"CONFIRM set: AUC({best})={auc_confirm_best:.4f} vs AUC(epa_gap, shipped baseline)={auc_confirm_epa:.4f}  "
          f"(delta {auc_confirm_best-auc_confirm_epa:+.4f})")

    # The question that actually matters for "keep epa_gap or not": does 'both' beat
    # 'fantasy_gap' ALONE on the same held-out confirm set -- not just vs epa_gap alone.
    print(f"\n=== CONFIRM set, all three (the real question: keep epa_gap, or go fantasy_gap-only?) ===")
    for name in FEATURE_SETS:
        rows = [preds[name][g] for g in confirm_ids if g in preds[name]]
        y_c = np.array([r[0] for r in rows])
        p_c = np.array([r[1] for r in rows])
        acc_c = float(np.mean((p_c > 0.5).astype(int) == y_c))
        from sklearn.metrics import log_loss
        ll_c = log_loss(y_c, p_c)
        print(f"  {name:12s} n={len(rows):4d}  acc={100*acc_c:6.2f}%  AUC={roc_auc_score(y_c,p_c):.4f}  logloss={ll_c:.4f}")

    # Random split, repeated, same purpose as v46: check whether the chronological result is
    # an artifact of how THAT particular split happened to fall.
    print(f"\n=== RANDOM split: 30 repeats, ~50/50 ({len(affected_ids)} affected games total) ===")
    ids_list = sorted(affected_ids)
    n = len(ids_list)
    half_n = n // 2
    selected_names = []
    confirm_deltas = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(ids_list)
        sel = set(perm[:half_n].tolist())
        conf = set(perm[half_n:].tolist())
        best_r = max(FEATURE_SETS, key=lambda k: auc_on(preds[k], sel))
        d = auc_on(preds[best_r], conf) - auc_on(preds["epa_gap"], conf)
        selected_names.append(best_r)
        if not np.isnan(d):
            confirm_deltas.append(d)
    print("selected-feature-set distribution:", dict(Counter(selected_names)))
    deltas = np.array(confirm_deltas)
    print(f"confirm-set AUC(selected) - AUC(epa_gap): mean={deltas.mean():+.4f}, median={np.median(deltas):+.4f}, "
          f"fraction positive={100*np.mean(deltas>0):.1f}%  (n={len(deltas)})")

    # Bootstrap on the chronological confirm set specifically.
    def bootstrap_pair(name_a, name_b, ids, n_boot=2000, seed=0):
        common = sorted(set(preds[name_a]) & set(preds[name_b]) & ids)
        y = np.array([preds[name_a][g][0] for g in common])
        p_a = np.array([preds[name_a][g][1] for g in common])
        p_b = np.array([preds[name_b][g][1] for g in common])
        rng = np.random.default_rng(seed)
        m = len(common)
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, m, m)
            yi = y[idx]
            if len(set(yi)) < 2:
                continue
            diffs.append(roc_auc_score(yi, p_b[idx]) - roc_auc_score(yi, p_a[idx]))
        diffs = np.array(diffs)
        print(f"{m} paired games, {len(diffs)} valid resamples")
        print(f"AUC({name_b}) - AUC({name_a}): mean={diffs.mean():+.4f}, 95% CI=({np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f})")
        print(f"fraction of resamples favoring {name_b}: {100*np.mean(diffs>0):.1f}%")

    print(f"\n=== paired bootstrap, {best} vs epa_gap, chronological confirm set (2000 resamples) ===")
    bootstrap_pair("epa_gap", best, confirm_ids)

    # The actual question this turn is asking: does keeping epa_gap alongside fantasy_gap
    # buy anything over fantasy_gap ALONE, on this same held-out confirm set?
    print(f"\n=== paired bootstrap, both vs fantasy_gap ALONE, chronological confirm set (2000 resamples) ===")
    bootstrap_pair("fantasy_gap", "both", confirm_ids)


if __name__ == "__main__":
    main()
