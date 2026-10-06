"""v25: two things. (1) directly measure the historical passing-EPA impact of a backup QB
starting, instead of inferring it indirectly through residuals. (2) build a QB QUALITY GAP
signal -- each actual starter's own trailing passing EPA/dropback vs. the team's normal
starter's -- as a more direct correction than the binary Out/Doubtful flag or its EWMA-gap
version (both tested null in v23/v24).

Historical QBERT (Nate Silver's own QB rating, week-by-week since 1950) was checked and
ruled out: it's real and has exactly this kind of archive, but it's locked behind a paid
Substack subscription with no public bulk/API access -- not something this pipeline can
pull. Built an equivalent from data we already fully control instead: nflverse's own
play-by-play `passer_player_id`/`epa` columns (verified live -- same gsis_id format the
injury report already uses, no fuzzy name matching needed this time) give each QB's actual
passing efficiency directly, in the exact same EPA units the rating itself is built from.

Definitions:
  - `actual_passer(team, game)`: whichever passer_player_id had the most attempts for that
    team in that specific game.
  - `primary_so_far(team, season, week)`: whichever passer has the most CUMULATIVE attempts
    for that team THIS SEASON so far (strictly before this game); falls back to last
    season's leader if this team hasn't thrown a pass yet this season (e.g. week 1). No
    lookahead -- never uses full-season hindsight to declare "the starter."
  - `backup_game`: actual_passer != primary_so_far for that team/week.
  - `qb_quality_gap(team, game)`: primary_so_far's own trailing (career-to-date, any team,
    no lookahead) EPA/dropback MINUS actual_passer's own trailing EPA/dropback -- 0 when the
    normal starter is playing, positive when whoever's actually playing is worse than usual.

Stage 1 prints the direct, paired (within team-season) comparison. Stage 2 re-runs the
EXACT v23/v24 walk-forward joint-logistic backtest, swapping gap_qb_diff for
qb_quality_gap_diff (home - away), to see whether a magnitude-aware signal succeeds where
the binary and EWMA-gap versions didn't.

Run: python research/edge_signal_test_v25_qb_quality_gap.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest, ttest_rel

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import importlib.util
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v14 = _load("v14", "edge_signal_test_v14_qb_injury_dropout.py")

CACHE_DIR = os.path.join(HERE, "cache")
FIRST_SEASON = 2009
TEST_START_SEASON = 2014
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
WP_LO, WP_HI = 0.05, 0.95
PASS_COLS = ["game_id", "season", "week", "season_type", "posteam", "play_type", "epa", "wp", "passer_player_id"]


def passer_game_stats(season: int) -> pd.DataFrame:
    """One row per (game_id, team, passer_player_id): attempts, epa_sum that game. Cached
    per season -- NOT reusable from _phase_stats_for_season's own cache, which doesn't carry
    passer identity."""
    cache_path = os.path.join(CACHE_DIR, f"passer_game_{season}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    print(f"  downloading/parsing pbp {season} for passer identity...", file=sys.stderr)
    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=PASS_COLS, low_memory=False)
    df = df[(df["season_type"] == "REG") & (df["play_type"] == "pass")]
    df = df.dropna(subset=["wp", "epa", "posteam", "passer_player_id"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)]
    out = df.groupby(["game_id", "posteam", "passer_player_id"]).agg(
        attempts=("epa", "size"), epa_sum=("epa", "sum")
    ).reset_index().rename(columns={"posteam": "team"})
    os.makedirs(CACHE_DIR, exist_ok=True)
    out.to_parquet(cache_path)
    return out


def main():
    print("Loading passer-level game stats (2009-2025)...", file=sys.stderr)
    all_rows = []
    for season in range(FIRST_SEASON, 2026):
        try:
            df = passer_game_stats(season)
        except Exception as exc:  # noqa: BLE001
            print(f"  {season}: unavailable ({exc}), skipping", file=sys.stderr)
            continue
        df["season"] = season
        all_rows.append(df)
    passer_df = pd.concat(all_rows, ignore_index=True)
    print(f"{len(passer_df)} (game, team, passer) rows\n")

    # game_id -> week (needed below, since passer_game_stats doesn't carry it post-groupby)
    games_meta = {r["game_id"]: (r["season"], r["week"]) for r in v14.build_dataset()}
    # build_dataset() only keeps games with a full pre-game rating -- fall back to parsing
    # game_id itself ("2014_03_NE_KC" -> week 3) for any game it dropped early in a season.
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

    # actual_passer(team, game) = most-attempts passer that specific game
    actual_passer: dict[tuple[str, str], str] = {}
    game_team_epa: dict[tuple[str, str], tuple[float, int]] = {}  # (game_id,team) -> (epa_sum, attempts) for the ACTUAL passer
    for (gid, team), grp in passer_df.groupby(["game_id", "team"]):
        row = grp.loc[grp["attempts"].idxmax()]
        actual_passer[(gid, team)] = row["passer_player_id"]
        game_team_epa[(gid, team)] = (row["epa_sum"], row["attempts"])

    # primary_so_far(team, season, week): cumulative attempts leader THIS season so far,
    # falling back to last season's leader if no attempts yet this season. Processed in
    # chronological order so nothing leaks forward.
    season_attempts: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    last_season_leader: dict[str, str] = {}
    last_season_num: dict[str, int] = {}
    primary_so_far: dict[tuple[str, str], str] = {}  # (team, game_id) -> passer_id

    # Chronological (game_id, team, season, week) list, one entry per team-game -- the
    # single source of order every running/no-lookahead computation below walks through.
    gt_meta = passer_df[["game_id", "team", "season", "week"]].drop_duplicates()
    gt_meta = gt_meta.sort_values(["season", "week", "game_id", "team"])
    game_order = list(gt_meta.itertuples(index=False, name=None))

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
            primary_so_far[(team, gid)] = actual_passer.get((gid, team))  # no history at all yet

        ap = actual_passer.get((gid, team))
        if ap is not None:
            epa_sum, attempts = game_team_epa[(gid, team)]
            this_season[ap] += attempts

    print("=== Stage 1: direct, paired (within team-season) passing-EPA impact of a backup start ===")
    season_of = {(gid, team): season for gid, team, season, week in game_order}
    team_season_starter_epa: dict[tuple[str, int], list] = defaultdict(list)
    team_season_backup_epa: dict[tuple[str, int], list] = defaultdict(list)
    for (gid, team), (epa_sum, attempts) in game_team_epa.items():
        season = season_of[(gid, team)]
        ap = actual_passer[(gid, team)]
        primary = primary_so_far.get((team, gid))
        epa_per_play = epa_sum / attempts if attempts else None
        if epa_per_play is None:
            continue
        if ap == primary:
            team_season_starter_epa[(team, season)].append(epa_per_play)
        else:
            team_season_backup_epa[(team, season)].append(epa_per_play)

    paired_starter, paired_backup = [], []
    for key in team_season_starter_epa:
        if key in team_season_backup_epa and len(team_season_starter_epa[key]) >= 2 and len(team_season_backup_epa[key]) >= 1:
            paired_starter.append(np.mean(team_season_starter_epa[key]))
            paired_backup.append(np.mean(team_season_backup_epa[key]))

    paired_starter, paired_backup = np.array(paired_starter), np.array(paired_backup)
    diff = paired_backup - paired_starter
    t, p = ttest_rel(paired_backup, paired_starter)
    print(f"{len(paired_starter)} team-seasons with BOTH starter and backup games")
    print(f"mean pass EPA/play with normal starter: {paired_starter.mean():+.4f}")
    print(f"mean pass EPA/play with backup(s):       {paired_backup.mean():+.4f}")
    print(f"mean within-team drop:                   {diff.mean():+.4f} EPA/play  (paired t={t:+.2f}, p={p:.6f})")

    all_backup_games = sum(len(v) for v in team_season_backup_epa.values())
    all_starter_games = sum(len(v) for v in team_season_starter_epa.values())
    print(f"\n(unpaired, for scale) {all_starter_games} starter team-games, {all_backup_games} backup team-games total")

    print("\n=== Stage 2: trailing per-passer EPA + qb_quality_gap ===")
    # Each passer's own trailing (career-to-date, cross-team, no lookahead) EPA/dropback.
    passer_hist: dict[str, list[float]] = defaultdict(list)
    trailing_epa: dict[tuple[str, str], float] = {}  # (passer_id, game_id) -> trailing avg BEFORE this game
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
    print(f"league-average pass EPA/play (fallback for a passer's first tracked game): {league_avg_epa:+.4f}")

    print("\nBuilding game dataset + production delta...", file=sys.stderr)
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

    def qb_quality_gap(team, gid):
        primary = primary_so_far.get((team, gid))
        actual = actual_passer.get((gid, team))
        if primary is None or actual is None:
            return 0.0
        t_primary = trailing_epa.get((primary, gid))
        t_actual = trailing_epa.get((actual, gid))
        t_primary = t_primary if t_primary is not None else league_avg_epa
        t_actual = t_actual if t_actual is not None else league_avg_epa
        return t_primary - t_actual  # positive = actual passer worse than normal starter

    for g in games:
        g["qb_quality_gap_diff"] = qb_quality_gap(g["home"], g["game_id"]) - qb_quality_gap(g["away"], g["game_id"])

    vals = np.array([g["qb_quality_gap_diff"] for g in games])
    print(f"qb_quality_gap_diff: mean={vals.mean():+.4f} std={vals.std():.4f} nonzero={np.mean(vals!=0)*100:.1f}%\n")

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

    seasons = sorted({g["season"] for g in games})
    all_old = all_new = all_n = 0
    improvements = regressions = 0
    season_rows = []
    coef_history = []

    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [g for g in games if g["season"] < test_season]
        test = [g for g in games if g["season"] == test_season]
        if len(train) < 300 or not test:
            continue
        gap_train = np.array([g["qb_quality_gap_diff"] for g in train])
        gmu, gsd = gap_train.mean(), (gap_train.std() or 1.0)

        def feat(g):
            return [g["delta"], (g["qb_quality_gap_diff"] - gmu) / gsd]

        X_train = np.array([feat(g) for g in train])
        y_train = np.array([g["home_win"] for g in train], dtype=float)
        w = fit_logistic(X_train, y_train)
        coef_history.append((test_season, *w))

        s_old = s_new = s_n = 0
        for g in test:
            old_home = g["delta"] > 0
            f = feat(g)
            z = w[0] + w[1] * f[0] + w[2] * f[1]
            new_home = z > 0
            actual_home = g["home_win"] == 1
            oh, nh = old_home == actual_home, new_home == actual_home
            s_old += oh; s_new += nh; s_n += 1
            all_old += oh; all_new += nh; all_n += 1
            if old_home != new_home:
                if nh and not oh: improvements += 1
                elif oh and not nh: regressions += 1
        season_rows.append((test_season, s_n, s_old, s_new))

    print(f"{'season':>6s} {'n':>5s} {'old%':>7s} {'new%':>7s} {'delta':>7s}")
    for s, n, o, nw in season_rows:
        print(f"{s:6d} {n:5d} {100*o/n:6.1f}% {100*nw/n:6.1f}% {100*(nw-o)/n:+6.1f}pt")

    print(f"\n=== overall, {TEST_START_SEASON}-{seasons[-1]}, {all_n} games ===")
    print(f"old (delta alone):          {all_old}/{all_n} = {100*all_old/all_n:.2f}%")
    print(f"new (delta + qb_quality_gap): {all_new}/{all_n} = {100*all_new/all_n:.2f}%")
    print(f"net change: {all_new-all_old:+d} games ({100*(all_new-all_old)/all_n:+.2f}pt)")
    n_disc = improvements + regressions
    print(f"flips: {n_disc} (improvements={improvements}, regressions={regressions})")
    if n_disc >= 10:
        p = binomtest(improvements, n_disc, 0.5).pvalue
        print(f"McNemar exact test: p={p:.4f}")

    print("\n=== fitted coefficients by fold ===")
    print(f"{'season':>6s} {'intercept':>10s} {'delta':>8s} {'qb_qual_gap':>12s}")
    for s, b0, b1, b2 in coef_history:
        print(f"{s:6d} {b0:10.3f} {b1:8.3f} {b2:12.3f}")


if __name__ == "__main__":
    main()
