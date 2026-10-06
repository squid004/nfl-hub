"""v27: does adding OL/DL/LB/secondary pay off under the CALIBRATION framing (confidence
reduced on misses vs. hits) rather than hit-rate? v16-v20 only ever tested hit-rate/EPA-
residual significance for these units; this redoes it properly in two stages:

Stage 1: same select-on-one-half/confirm-on-the-other percentile scan v22 used to find
skill's real threshold (p70, not the originally-assumed p85/p95) -- applied here to OL/DL/
LB/secondary's snap-share-weighted severity (v17's), since that's the exact signal v19's
tail test found LB/secondary effects with. OL/DL already came back null under raw count,
snap-weight, AND a flat p85/p95 tail cut (v16-v19) -- rerun through the proper scan anyway
for symmetry, not skipped on the assumption they'll stay null.

Stage 2 (only for whatever survives Stage 1): fit a final delta-equivalent weight the same
way QB_QUALITY_GAP_WEIGHT/SKILL_EPA_OUT_WEIGHT were derived (research/edge_signal_test_
v26_production_weights.py -- one joint logistic on ALL data, raw/unstandardized features,
coefficient ratio to delta's own), add it on top of the ALREADY-SHIPPED delta (which already
includes QB + skill), and re-run the exact compute_calibration_stats() comparison from
production: does the new unit ALSO show a higher confidence-reduction rate on misses than on
hits, same bar QB and skill already cleared? That's the actual question being asked --- not
"does it flip picks," whether it's honest signal worth reflecting in displayed confidence.

Run: python research/edge_signal_test_v27_defense_calibration.py
"""
from __future__ import annotations

import math
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
v25 = _load("v25", "edge_signal_test_v25_qb_quality_gap.py")
v26 = _load("v26", "edge_signal_test_v26_production_weights.py")

CANDIDATE_GROUPS = ["ol", "dl", "lb", "secondary"]
PERCENTILES = [50, 55, 60, 65, 70, 75, 78, 80, 82, 85, 88, 90, 92, 95, 97, 99]
SPLIT_SEASON = 2019  # snap_counts' own 2012 start bounds this -- same split v20 used


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


def outcome_test(games_subset, sev_by_tw, group, threshold_val, delta_key="delta"):
    y = np.array([g["home_win"] for g in games_subset], dtype=float)
    delta = np.array([g[delta_key] for g in games_subset]).reshape(-1, 1)
    diffs = []
    for g in games_subset:
        hv = 1 if sev_by_tw.get((g["home"], g["season"], g["week"], group), 0.0) >= threshold_val else 0
        av = 1 if sev_by_tw.get((g["away"], g["season"], g["week"], group), 0.0) >= threshold_val else 0
        diffs.append(hv - av)
    dcol = np.array(diffs, dtype=float).reshape(-1, 1)
    if np.std(dcol) == 0:
        return None
    w_r, se_r, ll_r = fit_logistic(delta, y)
    w_f, se_f, ll_f = fit_logistic(np.hstack([delta, dcol]), y)
    lr = 2 * (ll_f - ll_r)
    p = 1 - chi2.cdf(lr, df=1)
    return w_f[2], p, dcol.flatten()


def scan_select_confirm(games_select, games_confirm, sev_by_tw, group, pct_values):
    curve = []
    for pct, val in pct_values:
        r = outcome_test(games_select, sev_by_tw, group, val)
        if r is None:
            continue
        coef, p, _ = r
        curve.append((pct, val, coef, p))
    if not curve:
        return None
    best = min(curve, key=lambda t: t[3])
    pct, val, sel_coef, sel_p = best
    confirm = outcome_test(games_confirm, sev_by_tw, group, val)
    if confirm is None:
        return {"pct": pct, "val": val, "sel_coef": sel_coef, "sel_p": sel_p, "confirmed": None}
    conf_coef, conf_p, _ = confirm
    same_sign = (sel_coef < 0) == (conf_coef < 0)
    ok = same_sign and conf_p < 0.10
    return {"pct": pct, "val": val, "sel_coef": sel_coef, "sel_p": sel_p,
            "conf_coef": conf_coef, "conf_p": conf_p, "confirmed": ok}


def main():
    print("Loading snap-weighted severity for OL/DL/LB/secondary (2012-2025)...", file=sys.stderr)
    player_games, pos_default = v17.load_snap_shares()
    injury_rows = v17.load_injury_rows()
    weighted = v17.weighted_absence_by_team_week_group(injury_rows, player_games, pos_default, None)

    print("Building game dataset + SHIPPED delta (raw + QB + skill, exactly as production)...", file=sys.stderr)
    rows_all = v14.build_dataset()
    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows_all])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS
    for i, r in enumerate(rows_all):
        x = X_all[i]
        r["delta_raw"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))

    # Reproduce the SHIPPED adjustment (QB quality-gap + skill EPA-out) so "delta" here
    # matches production exactly -- testing whether defense adds anything ON TOP of what's
    # already live, not instead of it.
    print("Reproducing shipped QB quality-gap + skill EPA-out...", file=sys.stderr)
    all_passer_rows = []
    import pandas as pd
    for season in range(2009, 2026):
        try:
            df = v25.passer_game_stats(season)
        except Exception:
            continue
        df["season"] = season
        all_passer_rows.append(df)
    passer_df = pd.concat(all_passer_rows, ignore_index=True)
    games_meta = {r["game_id"]: (r["season"], r["week"]) for r in rows_all}
    def week_of(gid, season):
        _, w = games_meta.get(gid, (None, None))
        if w is not None:
            return w
        try:
            return int(gid.split("_")[1])
        except (IndexError, ValueError):
            return None
    passer_df["week"] = [week_of(gid, s) for gid, s in zip(passer_df["game_id"], passer_df["season"])]
    passer_df = passer_df.dropna(subset=["week"])
    passer_df["week"] = passer_df["week"].astype(int)
    actual_passer, game_team_epa = {}, {}
    for (gid, team), grp in passer_df.groupby(["game_id", "team"]):
        row = grp.loc[grp["attempts"].idxmax()]
        actual_passer[(gid, team)] = row["passer_player_id"]
        game_team_epa[(gid, team)] = (row["epa_sum"], row["attempts"])
    gt_meta = passer_df[["game_id", "team", "season", "week"]].drop_duplicates().sort_values(["season", "week", "game_id", "team"])
    game_order = list(gt_meta.itertuples(index=False, name=None))
    from collections import defaultdict
    season_attempts = defaultdict(lambda: defaultdict(int))
    last_season_leader, last_season_num, primary_so_far = {}, {}, {}
    for gid, team, season, week in game_order:
        if last_season_num.get(team) is not None and last_season_num[team] != season and season_attempts[(team, season - 1)]:
            last_season_leader[team] = max(season_attempts[(team, season - 1)].items(), key=lambda kv: kv[1])[0]
        last_season_num[team] = season
        this_season = season_attempts[(team, season)]
        if this_season:
            primary_so_far[(team, gid)] = max(this_season.items(), key=lambda kv: kv[1])[0]
        elif team in last_season_leader:
            primary_so_far[(team, gid)] = last_season_leader[team]
        else:
            primary_so_far[(team, gid)] = actual_passer.get((gid, team))
        ap = actual_passer.get((gid, team))
        if ap is not None:
            epa_sum, attempts = game_team_epa[(gid, team)]
            this_season[ap] += attempts
    passer_hist = defaultdict(list)
    trailing_epa = {}
    league_avg = []
    for gid, team, season, week in game_order:
        ap = actual_passer.get((gid, team))
        if ap is None:
            continue
        hist = passer_hist[ap]
        trailing_epa[(ap, gid)] = np.mean(hist) if hist else None
        epa_sum, attempts = game_team_epa[(gid, team)]
        if attempts:
            per_play = epa_sum / attempts
            hist.append(per_play)
            league_avg.append(per_play)
    league_avg_epa = float(np.mean(league_avg)) if league_avg else 0.0
    def qb_quality_gap(team, gid):
        primary = primary_so_far.get((team, gid))
        actual = actual_passer.get((gid, team))
        if primary is None or actual is None:
            return 0.0
        tp = trailing_epa.get((primary, gid))
        ta = trailing_epa.get((actual, gid))
        tp = tp if tp is not None else league_avg_epa
        ta = ta if ta is not None else league_avg_epa
        return tp - ta

    skill_player_games, skill_fallback = v26.load_skill_epa_history()
    skill_injury_rows = v17.load_injury_rows()  # reuse: has "skill" group already tagged
    # v26's skill_epa_out_by_team_week needs (injury_rows, player_games, fallback) where
    # injury_rows are dicts with "name"/"team"/"season"/"week" for SKILL group only.
    skill_rows_filtered = [r for r in skill_injury_rows if r["group"] == "skill"]
    skill_out = v26.skill_epa_out_by_team_week(skill_rows_filtered, skill_player_games, skill_fallback)

    games = [r for r in rows_all if r["season"] >= 2009 and r["delta_raw"] != 0]
    # production weights
    QB_W, SKILL_W = -0.5903, -0.0362
    for g in games:
        qb_gap_diff = qb_quality_gap(g["home"], g["game_id"]) - qb_quality_gap(g["away"], g["game_id"])
        skill_diff = skill_out.get((g["home"], g["season"], g["week"]), 0.0) - skill_out.get((g["away"], g["season"], g["week"]), 0.0)
        g["delta"] = g["delta_raw"] + QB_W * qb_gap_diff + SKILL_W * skill_diff

    games_2012 = [g for g in games if g["season"] >= 2012]
    early = [g for g in games_2012 if g["season"] < SPLIT_SEASON]
    late = [g for g in games_2012 if g["season"] >= SPLIT_SEASON]
    print(f"{len(games_2012)} games 2012-2025 for the threshold scan\n")

    print("=== Stage 1: select-on-one-half / confirm-on-other, OL/DL/LB/secondary ===")
    results = {}
    for group in CANDIDATE_GROUPS:
        all_vals = np.array([weighted.get((r[s], r["season"], r["week"], group), 0.0) for r in games_2012 for s in ("home", "away")])
        pct_vals = [(p, np.percentile(all_vals, p)) for p in PERCENTILES]
        print(f"\n[{group}]")
        r1 = scan_select_confirm(early, late, weighted, group, pct_vals)
        print(f"  select EARLY/confirm LATE: {r1}")
        r2 = scan_select_confirm(late, early, weighted, group, pct_vals)
        print(f"  select LATE/confirm EARLY: {r2}")
        results[group] = (r1, r2)

    survivors = [g for g, (r1, r2) in results.items() if (r1 and r1["confirmed"]) or (r2 and r2["confirmed"])]
    print(f"\nSurvivors (confirmed in at least one direction): {survivors}")
    if not survivors:
        print("\nNo candidate group survived the select/confirm scan -- stopping here, nothing to test for calibration value.")
        return

    print("\n=== Stage 2: calibration value for survivors ===")
    for group in survivors:
        r1, r2 = results[group]
        best = r1 if (r1 and (not r2 or r1["sel_p"] < r2["sel_p"])) else r2
        threshold_val = best["val"]
        print(f"\n[{group}] using threshold at p{best['pct']} (val={threshold_val:.3f})")

        # final fit: delta (shipped) + this group's indicator diff, ALL games with group data (2012+)
        diffs = []
        for g in games_2012:
            hv = 1 if weighted.get((g["home"], g["season"], g["week"], group), 0.0) >= threshold_val else 0
            av = 1 if weighted.get((g["away"], g["season"], g["week"], group), 0.0) >= threshold_val else 0
            diffs.append(hv - av)
        y = np.array([g["home_win"] for g in games_2012], dtype=float)
        delta_col = np.array([g["delta"] for g in games_2012]).reshape(-1, 1)
        dcol = np.array(diffs, dtype=float).reshape(-1, 1)
        w, se, ll = fit_logistic(np.hstack([delta_col, dcol]), y)
        b_delta, b_group = w[1], w[2]
        unit_weight = b_group / b_delta
        print(f"  delta-equivalent weight: {unit_weight:+.4f} (b_delta={b_delta:.3f}, b_group={b_group:.3f}, se={se[2]:.3f}, z={b_group/se[2]:.2f})")

        for g in games_2012:
            hv = 1 if weighted.get((g["home"], g["season"], g["week"], group), 0.0) >= threshold_val else 0
            av = 1 if weighted.get((g["away"], g["season"], g["week"], group), 0.0) >= threshold_val else 0
            g["delta_with_group"] = g["delta"] + unit_weight * (hv - av)

        # calibration check: old = shipped delta (QB+skill only), new = + this group
        # (v14.build_dataset() rows carry home_win, not raw scores -- ties already collapse
        # into home_win=0 upstream there, same convention every other v14-based script here
        # already relies on; no separate tie filter needed/possible at this level.)
        def pick(d): return "home" if d > 0 else ("away" if d < 0 else None)
        def winner(g): return "home" if g["home_win"] == 1 else "away"
        graded = [g for g in games_2012 if g["delta"] != 0 and g["delta_with_group"] != 0]
        old_correct = [pick(g["delta"]) == winner(g) for g in graded]
        new_correct = [pick(g["delta_with_group"]) == winner(g) for g in graded]
        old_hits, new_hits, n = sum(old_correct), sum(new_correct), len(graded)
        misses = [g for g, oc in zip(graded, old_correct) if not oc]
        hits = [g for g, oc in zip(graded, old_correct) if oc]
        def reduced_rate(rows):
            if not rows:
                return 0, 0
            return sum(1 for g in rows if abs(g["delta_with_group"]) < abs(g["delta"])), len(rows)
        miss_r, miss_n = reduced_rate(misses)
        hit_r, hit_n = reduced_rate(hits)
        print(f"  hit rate: shipped {100*old_hits/n:.2f}% -> +{group} {100*new_hits/n:.2f}%  (n={n})")
        if miss_n >= 10 and hit_n >= 10:
            p1, p2 = miss_r/miss_n, hit_r/hit_n
            p_pool = (miss_r+hit_r)/(miss_n+hit_n)
            se_p = math.sqrt(p_pool*(1-p_pool)*(1/miss_n+1/hit_n))
            z = (p1-p2)/se_p if se_p>0 else 0
            pval = math.erfc(abs(z)/math.sqrt(2))
            print(f"  confidence reduced on misses: {100*p1:.1f}% (n={miss_n})  vs on hits: {100*p2:.1f}% (n={hit_n})  z={z:.2f} p={pval:.4f}")


if __name__ == "__main__":
    main()
