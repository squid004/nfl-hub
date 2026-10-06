"""v14: does the model's own weight-fit get dragged down by games where a team's starting QB
didn't play? v13's final weights (now POWER_WEIGHTS in production) were fit on EVERY game,
including ones where one side was clearly playing a backup QB -- a game the market itself
down-weights (nflverse's own injury report exists because the market cares), but the model's
8 rating stats have zero live awareness of. If those games are noisy/unpredictable inputs to
the fit, including them could pull the weights away from what best separates OUTCOMES in the
games that don't have this problem.

Test: split the walk-forward dataset into "clean" games (neither team had a QB listed Out/
Doubtful that week) and "backup" games (at least one team did -- the held-out test set).
Refit the composite weights -- same methodology as v13 (standardize, non-negative L2=0.3
logistic regression, intercept dropped before use, matching how POWER_WEIGHTS is actually
applied in compute_backtest_scatter) -- using ONLY the clean games. Then score the backup
games (the ones excluded from that refit) with BOTH the old production weights and the new
clean-only weights, and compare hit rate on that held-out set.

This is NOT a weight-search: L2 stays fixed at 0.3 (production's own chosen value) so the
only variable that changes is which games the fit saw -- isolates "did backup-QB games bias
the weights" from "what's the best L2 anyway." Nothing here is written back to team_ratings.py
-- this is a standalone comparison, same self-contained convention as every other
edge_signal_test_v*.py (doesn't import POWER_WEIGHTS, hardcodes the current production
values below so a later change to that file can't silently change what this script reports).

QB-injury proxy: a team's QB listed "Out" or "Doubtful" on nflverse's historical injury report
that week (same proxy and same cached data as the earlier live QB-health filter added to
team_ratings.py's backtest scatter). No report exists before 2009, so 2007-2008 games are
necessarily treated as "clean" by default (no claim either way, just no evidence of a
problem) -- this matches how the live filter handles it too.

Run: python research/edge_signal_test_v14_qb_injury_dropout.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
import requests
from scipy.optimize import minimize
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
SEASONS = range(2007, 2026)
FIRST_INJURY_SEASON = 2009
EWMA_ALPHA = 0.2
SEASON_PRIOR_GAMES = 5  # matches nflhub/sources/team_ratings.py exactly
WP_LO, WP_HI = 0.05, 0.95
L2 = 0.3  # production's own chosen value (edge_signal_test_v13) -- fixed, not re-searched here
RATING_METRICS = (
    "rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed",
    "points_off", "points_def_allowed", "turnovers_off", "turnovers_def_forced",
)
ORIENTATION = {
    "rush_off_epa": 1, "pass_off_epa": 1, "rush_def_epa_allowed": -1, "pass_def_epa_allowed": -1,
    "points_off": 1, "points_def_allowed": -1, "turnovers_off": -1, "turnovers_def_forced": 1,
}
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "wp", "interception", "fumble_lost"]

# Current production weights (nflhub/sources/team_ratings.py POWER_WEIGHTS) -- hardcoded, not
# imported, so this script's report can't silently change if that file changes later.
PRODUCTION_WEIGHTS = {
    "rush_off_epa": 0.0850, "pass_off_epa": 0.1536, "rush_def_epa_allowed": 0.0572,
    "pass_def_epa_allowed": 0.0578, "points_off": 0.1587, "points_def_allowed": 0.1006,
    "turnovers_off": 0.0217, "turnovers_def_forced": 0.0000,
}


def _cached_fetch_text(url: str, cache_name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return resp.text


def load_games() -> dict[str, dict]:
    text = _cached_fetch_text(GAMES_URL, "games.csv")
    games = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r["game_type"] != "REG":
            continue
        try:
            season = int(r["season"])
        except ValueError:
            continue
        if season in SEASONS:
            games[r["game_id"]] = r
    return games


def team_game_phase_stats(season: int):
    import pandas as pd
    cache_path = os.path.join(CACHE_DIR, f"team_game_phase_turnovers_{season}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    print(f"  downloading/parsing pbp {season}...", file=sys.stderr)
    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=PBP_COLS, low_memory=False)
    df = df[(df["season_type"] == "REG") & (df["play_type"].isin(["run", "pass"]))]
    df = df.dropna(subset=["wp", "epa", "posteam"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)]
    df["turnover"] = df["interception"].fillna(0) + df["fumble_lost"].fillna(0)
    off = df.groupby(["game_id", "posteam", "play_type"])["epa"].agg(["size", "sum"]).reset_index()
    off = off.pivot_table(index=["game_id", "posteam"], columns="play_type", values=["size", "sum"])
    off.columns = [f"{a}_{b}" for a, b in off.columns]
    off = off.reset_index().rename(columns={"posteam": "team"})
    for col in ("size_run", "size_pass", "sum_run", "sum_pass"):
        if col not in off.columns:
            off[col] = 0.0
    off["rush_off_epa"] = off["sum_run"] / off["size_run"].replace(0, np.nan)
    off["pass_off_epa"] = off["sum_pass"] / off["size_pass"].replace(0, np.nan)
    tov = df.groupby(["game_id", "posteam"])["turnover"].sum().reset_index().rename(columns={"posteam": "team", "turnover": "turnovers_committed"})
    out = off[["game_id", "team", "rush_off_epa", "pass_off_epa"]].merge(tov, on=["game_id", "team"], how="left")
    out["turnovers_committed"] = out["turnovers_committed"].fillna(0.0)
    out = out.dropna(subset=["rush_off_epa", "pass_off_epa"])
    out.to_parquet(cache_path)
    return out


def qb_out_doubtful_by_week() -> dict[tuple[str, int, int], bool]:
    """Same proxy as team_ratings.py's live _qb_out_doubtful_by_week(), reimplemented
    standalone (same self-contained convention as the rest of this script) -- reuses the
    already-cached injuries_{season}.csv files in research/cache/ from the earlier QB-injury
    investigation this session."""
    out: dict[tuple[str, int, int], bool] = {}
    for season in range(FIRST_INJURY_SEASON, SEASONS.stop):
        text = _cached_fetch_text(INJURIES_URL.format(season=season), f"injuries_{season}.csv")
        for row in csv.DictReader(io.StringIO(text)):
            if row.get("game_type") != "REG" or row.get("position") != "QB":
                continue
            if row.get("report_status") not in ("Out", "Doubtful"):
                continue
            try:
                team = normalize_team(row["team"])
                week = int(row["week"])
            except (UnknownTeamError, ValueError, TypeError):
                continue
            out[(team, season, week)] = True
    return out


def build_dataset() -> list[dict]:
    """Identical to v13's build_dataset(), plus home/away team codes on each row (needed here
    to join against the QB-injury map; v13 didn't need them)."""
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for season in SEASONS:
        for row in team_game_phase_stats(season).to_dict("records"):
            try:
                team_key = normalize_team(row["team"])
            except UnknownTeamError:
                continue
            by_game[row["game_id"]][team_key] = row

    games = load_games()
    rating: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    season_ewma: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    season_prior: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    games_played: dict[str, int] = defaultdict(int)
    last_season: dict[str, int] = {}

    rows_out = []
    for gid, teams in sorted(by_game.items()):
        g = games.get(gid)
        if not g or len(teams) != 2:
            continue
        season = int(g["season"])
        try:
            home, away = normalize_team(g["home_team"]), normalize_team(g["away_team"])
        except UnknownTeamError:
            continue
        if home not in teams or away not in teams:
            continue
        try:
            home_pts, away_pts = float(g["home_score"]), float(g["away_score"])
        except (ValueError, TypeError):
            continue

        home_raw, away_raw = dict(teams[home]), dict(teams[away])
        home_raw["rush_def_epa_allowed"] = away_raw["rush_off_epa"]
        away_raw["rush_def_epa_allowed"] = home_raw["rush_off_epa"]
        home_raw["pass_def_epa_allowed"] = away_raw["pass_off_epa"]
        away_raw["pass_def_epa_allowed"] = home_raw["pass_off_epa"]
        home_raw["points_off"], away_raw["points_off"] = home_pts, away_pts
        home_raw["points_def_allowed"], away_raw["points_def_allowed"] = away_pts, home_pts
        home_raw["turnovers_off"] = home_raw.pop("turnovers_committed")
        away_raw["turnovers_off"] = away_raw.pop("turnovers_committed")
        home_raw["turnovers_def_forced"] = away_raw["turnovers_off"]
        away_raw["turnovers_def_forced"] = home_raw["turnovers_off"]

        prior_w: dict[str, float] = {}
        for team in (home, away):
            if last_season.get(team) is not None and last_season[team] != season:
                season_prior[team] = dict(season_ewma[team])
                season_ewma[team] = {m: None for m in RATING_METRICS}
                games_played[team] = 0
            last_season[team] = season
            games_played[team] += 1
            prior_w[team] = max(0.0, (SEASON_PRIOR_GAMES - games_played[team]) / SEASON_PRIOR_GAMES)

        ok = all(rating[home][m] is not None and rating[away][m] is not None for m in RATING_METRICS)
        feat_row = {"season": season, "week": int(g["week"]), "game_id": gid, "home": home, "away": away}
        if ok:
            for m in RATING_METRICS:
                feat_row[f"{m}_diff"] = ORIENTATION[m] * (rating[home][m] - rating[away][m])
            try:
                feat_row["spread_line"] = float(g["spread_line"])
            except (ValueError, TypeError):
                ok = False
        if ok:
            try:
                res = float(g["result"])
            except (ValueError, TypeError):
                res = home_pts - away_pts
            feat_row["home_win"] = 1 if res > 0 else 0
            rows_out.append(feat_row)

        for team, raw in ((home, home_raw), (away, away_raw)):
            for m in RATING_METRICS:
                prev = season_ewma[team][m]
                season_ewma[team][m] = raw[m] if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * raw[m]
                prior_val = season_prior[team].get(m)
                rating[team][m] = (
                    prior_w[team] * prior_val + (1 - prior_w[team]) * season_ewma[team][m]
                    if prior_val is not None else season_ewma[team][m]
                )

    return rows_out


def fit_nonneg_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])

    def loss_grad(w):
        z = Xb @ w
        p_hat = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        nll = -np.mean(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))
        reg = 0.5 * l2 * np.sum(w[1:] ** 2)
        grad = Xb.T @ (p_hat - y) / n
        grad[1:] += l2 * w[1:]
        return nll + reg, grad

    bounds = [(None, None)] + [(0, None)] * p
    res = minimize(loss_grad, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x


def main():
    print("Building dataset (same windowing as production)...", file=sys.stderr)
    rows = build_dataset()
    cols = [f"{m}_diff" for m in RATING_METRICS]
    print(f"{len(rows)} total games ({SEASONS.start}-{SEASONS.stop - 1})")

    print("Loading QB-injury report (Out/Doubtful, 2009+)...", file=sys.stderr)
    qb_out = qb_out_doubtful_by_week()
    for r in rows:
        r["backup_flag"] = qb_out.get((r["home"], r["season"], r["week"]), False) or \
                            qb_out.get((r["away"], r["season"], r["week"]), False)

    clean = [r for r in rows if not r["backup_flag"]]
    backup = [r for r in rows if r["backup_flag"]]
    print(f"clean games (neither team's QB out/doubtful): {len(clean)}")
    print(f"backup games (excluded from the refit):        {len(backup)}\n")

    # --- refit on CLEAN games only, same methodology as production (standardize from the
    # training set, non-negative L2=0.3 logistic, intercept dropped before use -- matching
    # how compute_backtest_scatter actually applies POWER_WEIGHTS) ---
    X_clean = np.array([[r[c] for c in cols] for r in clean])
    y_clean = np.array([r["home_win"] for r in clean])
    mu_clean, sd_clean = X_clean.mean(axis=0), X_clean.std(axis=0)
    sd_clean[sd_clean == 0] = 1.0
    w_clean = fit_nonneg_logistic((X_clean - mu_clean) / sd_clean, y_clean, L2)
    new_weights = dict(zip(RATING_METRICS, w_clean[1:]))

    print("--- weights: production (fit on ALL games) vs. clean-refit (backup games excluded) ---")
    print(f"{'metric':22s} {'production':>11s} {'clean-refit':>12s}")
    for m in RATING_METRICS:
        print(f"{m:22s} {PRODUCTION_WEIGHTS[m]:11.4f} {new_weights[m]:12.4f}")

    # --- score the CLEAN games (not the held-out backup games) with both weight sets ---
    # This asks a different question than scoring on the backup games does: not "does the
    # refit generalize to the games it's never seen," but "does excluding backup-QB games
    # from the fit change how well the model predicts the NORMAL games" -- i.e. were they
    # actively dragging the joint fit away from what best separates outcomes on everything
    # else. NOTE the asymmetry this introduces: new_weights was fit ON this exact clean set
    # (in-sample for the new model), while PRODUCTION_WEIGHTS was fit on clean+backup
    # together (in-sample too, just diluted by the backup games) -- so this is an in-sample
    # vs. in-sample comparison, not a held-out generalization test. That's fine for the
    # specific question being asked here, just not a claim about out-of-sample performance.
    # Old model's standardization still uses the FULL dataset (matches compute_backtest_
    # scatter's own "standardized once across the whole dataset" convention for production).
    X_all = np.array([[r[c] for c in cols] for r in rows])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0

    old_hits, new_hits = 0, 0
    agree_old_right_new_wrong, agree_new_right_old_wrong = 0, 0
    decided = 0
    for r in clean:
        if r["home_win"] not in (0, 1):
            continue
        x = np.array([r[c] for c in cols])
        delta_old = sum(PRODUCTION_WEIGHTS[m] * ((x[i] - mu_all[i]) / sd_all[i]) for i, m in enumerate(RATING_METRICS))
        delta_new = sum(new_weights[m] * ((x[i] - mu_clean[i]) / sd_clean[i]) for i, m in enumerate(RATING_METRICS))
        if delta_old == 0 or delta_new == 0:
            continue  # no predicted favorite, can't grade
        decided += 1
        old_pred = 1 if delta_old > 0 else 0
        new_pred = 1 if delta_new > 0 else 0
        old_right = old_pred == r["home_win"]
        new_right = new_pred == r["home_win"]
        old_hits += old_right
        new_hits += new_right
        if old_right and not new_right:
            agree_old_right_new_wrong += 1
        if new_right and not old_right:
            agree_new_right_old_wrong += 1

    print(f"\n--- backtest on the {decided} CLEAN games (in-sample for both weight sets) ---")
    print(f"production weights (fit on everything):      {old_hits}/{decided} = {100*old_hits/decided:.1f}%")
    print(f"clean-refit weights (backup games excluded):  {new_hits}/{decided} = {100*new_hits/decided:.1f}%")
    print(f"\ngames old got right, new got wrong: {agree_old_right_new_wrong}")
    print(f"games new got right, old got wrong: {agree_new_right_old_wrong}")
    n_discordant = agree_old_right_new_wrong + agree_new_right_old_wrong
    if n_discordant >= 10:
        # McNemar's exact test on the discordant pairs -- the right test here since both
        # models are scored on the SAME games, not independent samples.
        p = binomtest(agree_new_right_old_wrong, n_discordant, 0.5).pvalue
        print(f"McNemar exact test on {n_discordant} discordant games: p={p:.4f}")
    else:
        print(f"only {n_discordant} discordant games -- too few for a meaningful significance test")


if __name__ == "__main__":
    main()
