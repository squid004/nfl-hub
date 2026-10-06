"""v26: final step before deploying the QB + skill health adjustment to production. Builds
the skill-position analog of v25's qb_quality_gap (same logic, applied to a multi-player
unit instead of a 1-for-1 swap -- see below), then fits ONE joint logistic regression on
ALL available games (not walk-forward -- we're deploying this now, same "fit on all data"
convention as POWER_WEIGHTS/POINTS_WEIGHTS/WEATHER_ADJUSTMENT_WEIGHTS, each of which was
walk-forward VALIDATED first, then finalized as one fit on everything) to get weights that
convert both gaps into delta-equivalent adjustments.

skill_epa_out: QB has one role and a clean 1-for-1 swap (starter vs. actual passer), so
v25's qb_quality_gap could directly compare "who's playing" to "who should be." Skill
positions don't have a clean swap -- a committee of 3+ players share touches, and who picks
up the slack varies. The direct analog instead sums each Out/Doubtful skill player's own
trailing (career-to-date, no lookahead) rushing_epa+receiving_epa per game -- "how much
known offensive value is unavailable this week," zero in a healthy week, scaling with both
how many are out AND how good they normally are (same EPA units as everything else in this
project, from nflverse's player_stats.csv, already verified live and cached this session).

Conversion to delta units: fits home_win ~ delta + qb_quality_gap_diff + skill_epa_out_diff
with ALL THREE features in their RAW (unstandardized) native units -- the ratio of each
gap's coefficient to delta's own coefficient is then directly "how many delta-equivalent
units is 1 unit of this gap worth," which is what actually gets added to delta in
production (QB_QUALITY_GAP_WEIGHT, SKILL_EPA_OUT_WEIGHT below).

Run: python research/edge_signal_test_v26_production_weights.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import importlib.util
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v14 = _load("v14", "edge_signal_test_v14_qb_injury_dropout.py")
v21 = _load("v21", "edge_signal_test_v21_skill_touches_weighted.py")
v25 = _load("v25", "edge_signal_test_v25_qb_quality_gap.py")

FIRST_SEASON = 2009
CACHE_DIR = os.path.join(HERE, "cache")


def load_skill_epa_history() -> tuple[dict, float]:
    """{name_norm: [(season, week, epa_per_game), ...]} sorted chronologically, using
    player_stats.csv's own rushing_epa+receiving_epa (already cached locally from v21's
    verification) -- same file, just a different column than v21 used (touches/yards)."""
    path = os.path.join(CACHE_DIR, "player_stats.csv")
    player_games: dict[str, list] = defaultdict(list)
    total = n = 0.0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("position") not in v21.SKILL_POSITIONS or row.get("season_type") != "REG":
                continue
            try:
                season = int(row["season"])
            except (ValueError, TypeError):
                continue
            if not (FIRST_SEASON <= season <= 2025):
                continue
            try:
                week = int(row["week"])
                epa = float(row["rushing_epa"] or 0) + float(row["receiving_epa"] or 0)
            except (ValueError, TypeError):
                continue
            name = v21._norm_name(row.get("player_display_name") or row.get("player_name"))
            if not name:
                continue
            player_games[name].append((season, week, epa))
            total += epa
            n += 1
    for name in player_games:
        player_games[name].sort()
    return player_games, (total / n if n else 0.0)


def skill_epa_out_by_team_week(injury_rows, player_games, fallback) -> dict:
    import bisect
    out: dict[tuple, float] = defaultdict(float)
    for r in injury_rows:
        games = player_games.get(r["name"])
        weight = fallback
        if games:
            keys = [(s, w) for s, w, _ in games]
            idx = bisect.bisect_left(keys, (r["season"], r["week"]))
            prior = games[:idx]
            if prior:
                weight = sum(e for _, _, e in prior) / len(prior)
        out[(r["team"], r["season"], r["week"])] += max(weight, 0.0)  # a below-average player's "loss" isn't negative value
    return out


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
    return w, se


def main():
    print("Rebuilding QB quality gap (v25's machinery)...", file=sys.stderr)
    # Re-run v25's own data pipeline up through qb_quality_gap_diff per game -- import and
    # call its main() would print everything again, so re-derive just what's needed here
    # using its already-cached passer_game parquets (fast, no re-download).
    all_rows = []
    for season in range(FIRST_SEASON, 2026):
        try:
            df = v25.passer_game_stats(season)
        except Exception:
            continue
        df["season"] = season
        all_rows.append(df)
    import pandas as pd
    passer_df = pd.concat(all_rows, ignore_index=True)
    games_meta = {r["game_id"]: (r["season"], r["week"]) for r in v14.build_dataset()}
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

    season_attempts = defaultdict(lambda: defaultdict(int))
    last_season_leader, last_season_num = {}, {}
    primary_so_far = {}
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

    print("Building skill EPA-out signal...", file=sys.stderr)
    skill_player_games, skill_fallback = load_skill_epa_history()
    skill_injury_rows = v21.load_skill_injury_rows()
    skill_epa_out = skill_epa_out_by_team_week(skill_injury_rows, skill_player_games, skill_fallback)

    print("Building game dataset + production delta...", file=sys.stderr)
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

    for g in games:
        g["qb_gap_diff"] = qb_quality_gap(g["home"], g["game_id"]) - qb_quality_gap(g["away"], g["game_id"])
        g["skill_out_diff"] = (skill_epa_out.get((g["home"], g["season"], g["week"]), 0.0) -
                                skill_epa_out.get((g["away"], g["season"], g["week"]), 0.0))

    y = np.array([g["home_win"] for g in games], dtype=float)
    X = np.array([[g["delta"], g["qb_gap_diff"], g["skill_out_diff"]] for g in games])

    print(f"\n{len(games)} games, {FIRST_SEASON}-2025 -- fitting FINAL joint model (raw units, all data)...")
    w, se = fit_logistic(X, y)
    names = ["intercept", "delta", "qb_gap_diff", "skill_out_diff"]
    print(f"{'term':16s} {'coef':>10s} {'se':>9s} {'z':>7s}")
    for nm, wi, sei in zip(names, w, se):
        print(f"{nm:16s} {wi:10.4f} {sei:9.4f} {wi/sei:7.2f}")

    b_delta, b_qb, b_skill = w[1], w[2], w[3]
    print(f"\nDelta-equivalent conversion (coef / delta's own coef):")
    print(f"  QB_QUALITY_GAP_WEIGHT   = {b_qb/b_delta:+.4f}")
    print(f"  SKILL_EPA_OUT_WEIGHT    = {b_skill/b_delta:+.4f}")

    # sanity: distribution of the raw signals, for docstring/comment context in production
    for key, label in (("qb_gap_diff", "qb_gap_diff"), ("skill_out_diff", "skill_out_diff")):
        vals = np.array([g[key] for g in games])
        print(f"  {label}: mean={vals.mean():+.4f} std={vals.std():.4f} nonzero={np.mean(vals!=0)*100:.1f}%")


if __name__ == "__main__":
    main()
