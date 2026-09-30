"""v6: solve for every team's offense/defense rating SIMULTANEOUSLY, SRS/Sagarin-style,
instead of the single-pass "adjust by opponent's own noisy rolling estimate" that v5 showed
doesn't help.

The model: each team-game observation is
    y (team's rush or pass EPA/play this game) = mu + Offense[team] - Defense[opponent] + noise
This is a standard two-way fixed-effects regression -- solved for ALL 32+ teams' offense and
defense coefficients AT ONCE via one Ridge regression (one solve for rush, one for pass), not
by adjusting one team's estimate using a single noisy opponent number the way v5 did. Ridge's
L2 penalty also does automatically what v5's naive adjustment got wrong: a team with few
games in the window (an uncertain estimate) gets shrunk toward league-average rather than
given full-strength credit/blame.

Re-solved before every week (walk-forward safe: only uses games strictly before that week,
within a trailing window) and applied to every game that week -- this is a rolling SRS, not a
single end-of-season snapshot.

Run: python research/edge_signal_test_v6_srs.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
SEASONS = range(2007, 2025)
TEST_START_SEASON = 2011
WP_LO, WP_HI = 0.05, 0.95
TRAILING_WEEKS = 51  # ~3 seasons of games feed each week's simultaneous solve
RIDGE_ALPHA = 30.0   # L2 penalty; shrinks thin-sample teams toward league average
RECENCY_DECAY = 0.8  # per week-elapsed sample weight -- matches EWMA_ALPHA=0.2's implied decay
                      # (v2-v5 baseline), so this isn't conflated with "lost recency weighting"
RATING_METRICS = ("rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed")
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "wp"]


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


def team_game_phase_stats(season: int) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"team_game_phase_{season}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"  downloading/parsing pbp {season}...", file=sys.stderr)
    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=PBP_COLS, low_memory=False)
    df = df[(df["season_type"] == "REG") & (df["play_type"].isin(["run", "pass"]))]
    df = df.dropna(subset=["wp", "epa", "posteam"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)]
    off = df.groupby(["game_id", "posteam", "play_type"])["epa"].agg(["size", "sum"]).reset_index()
    off = off.pivot_table(index=["game_id", "posteam"], columns="play_type", values=["size", "sum"])
    off.columns = [f"{a}_{b}" for a, b in off.columns]
    off = off.reset_index().rename(columns={"posteam": "team"})
    for col in ("size_run", "size_pass", "sum_run", "sum_pass"):
        if col not in off.columns:
            off[col] = 0.0
    off["rush_off_epa"] = off["sum_run"] / off["size_run"].replace(0, np.nan)
    off["pass_off_epa"] = off["sum_pass"] / off["size_pass"].replace(0, np.nan)
    out = off[["game_id", "team", "rush_off_epa", "pass_off_epa"]].dropna()
    out.to_parquet(cache_path)
    return out


def build_long_table() -> pd.DataFrame:
    """One row per (team, game): season, week, wk_idx (chronological index for trailing-
    window filtering), team, opponent, rush_off_epa, pass_off_epa."""
    games = load_games()
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for season in SEASONS:
        for row in team_game_phase_stats(season).to_dict("records"):
            by_game[row["game_id"]][row["team"]] = row

    rows = []
    for gid, teams in by_game.items():
        g = games.get(gid)
        if not g or len(teams) != 2:
            continue
        try:
            season, week = int(g["season"]), int(g["week"])
        except ValueError:
            continue
        home, away = g["home_team"], g["away_team"]
        if home not in teams or away not in teams:
            continue
        for team, opp in ((home, away), (away, home)):
            r = teams[team]
            rows.append({
                "season": season, "week": week, "game_id": gid,
                "team": team, "opponent": opp,
                "rush_off_epa": r["rush_off_epa"], "pass_off_epa": r["pass_off_epa"],
            })
    df = pd.DataFrame(rows)
    weeks_sorted = sorted(df[["season", "week"]].drop_duplicates().itertuples(index=False), key=lambda r: (r.season, r.week))
    wk_idx = {(s, w): i for i, (s, w) in enumerate(weeks_sorted)}
    df["wk_idx"] = [wk_idx[(s, w)] for s, w in zip(df["season"], df["week"])]
    return df.sort_values("wk_idx").reset_index(drop=True)


def solve_ratings(window: pd.DataFrame, all_teams: list[str], team_idx: dict[str, int], phase_col: str, alpha: float, weights: np.ndarray | None = None) -> dict[str, tuple[float, float]]:
    """One simultaneous Ridge solve: y = mu + Offense[team] - Defense[opponent].
    Returns {team: (offense_rating, defense_rating_allowed)}; defense sign convention matches
    v2-v5 (negative = good defense, suppresses opponent output), since a good defense's
    coefficient is driven negative by the regression to explain lower observed opponent EPA."""
    n = len(window)
    n_teams = len(all_teams)
    if n == 0:
        return {t: (0.0, 0.0) for t in all_teams}

    off_cols = window["team"].map(team_idx).to_numpy()
    def_cols = window["opponent"].map(team_idx).to_numpy()
    rows = np.arange(n)
    # columns: [0]=intercept, [1..n_teams]=offense dummies, [n_teams+1..2n_teams]=defense dummies
    data = np.ones(3 * n)
    row_idx = np.concatenate([rows, rows, rows])
    col_idx = np.concatenate([np.zeros(n, dtype=int), 1 + off_cols, 1 + n_teams + def_cols])
    X = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, 1 + 2 * n_teams))
    y = window[phase_col].to_numpy()

    model = Ridge(alpha=alpha, fit_intercept=False, solver="sparse_cg")
    model.fit(X, y, sample_weight=weights)
    coef = model.coef_
    return {t: (coef[1 + team_idx[t]], coef[1 + n_teams + team_idx[t]]) for t in all_teams}


def build_dataset() -> list[dict]:
    long_df = build_long_table()
    all_teams = sorted(long_df["team"].unique())
    team_idx = {t: i for i, t in enumerate(all_teams)}
    games = load_games()

    week_keys = sorted(long_df[["season", "week", "wk_idx"]].drop_duplicates().itertuples(index=False), key=lambda r: r.wk_idx)

    rows_out = []
    for wk in week_keys:
        if wk.season < TEST_START_SEASON:
            continue
        lo = max(0, wk.wk_idx - TRAILING_WEEKS)
        window = long_df[(long_df["wk_idx"] >= lo) & (long_df["wk_idx"] < wk.wk_idx)]
        if len(window) < 200:
            continue
        weights = RECENCY_DECAY ** (wk.wk_idx - window["wk_idx"].to_numpy())
        rush = solve_ratings(window, all_teams, team_idx, "rush_off_epa", RIDGE_ALPHA, weights)
        pas = solve_ratings(window, all_teams, team_idx, "pass_off_epa", RIDGE_ALPHA, weights)

        week_games = long_df[long_df["wk_idx"] == wk.wk_idx][["game_id", "team", "opponent"]].drop_duplicates("game_id")
        for _, wg in week_games.iterrows():
            g = games.get(wg["game_id"])
            if not g:
                continue
            home, away = g["home_team"], g["away_team"]
            if home not in team_idx or away not in team_idx:
                continue
            h_ro, h_rd = rush[home]
            a_ro, a_rd = rush[away]
            h_po, h_pd = pas[home]
            a_po, a_pd = pas[away]
            feat = {
                "season": wk.season, "week": wk.week, "game_id": wg["game_id"],
                "rush_off_epa_diff": h_ro - a_ro,
                "pass_off_epa_diff": h_po - a_po,
                "rush_def_epa_allowed_diff": h_rd - a_rd,
                "pass_def_epa_allowed_diff": h_pd - a_pd,
            }
            try:
                feat["spread_line"] = float(g["spread_line"])
            except (ValueError, TypeError):
                continue
            try:
                res = float(g["result"])
            except (ValueError, TypeError):
                res = float(g.get("home_score") or 0) - float(g.get("away_score") or 0)
            feat["home_win"] = 1 if res > 0 else 0
            rows_out.append(feat)

    # dedupe (each game appears twice in week_games, once per team row)
    seen = set()
    dedup = []
    for r in rows_out:
        if r["game_id"] in seen:
            continue
        seen.add(r["game_id"])
        dedup.append(r)
    return dedup


def walk_forward_auc(rows: list[dict], feature_cols: list[str]) -> dict:
    seasons = sorted({r["season"] for r in rows})
    preds, actuals = [], []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        if len(train) < 100 or not test:
            continue
        X_train = np.array([[r[c] for c in feature_cols] for r in train])
        y_train = np.array([r["home_win"] for r in train])
        X_test = np.array([[r[c] for c in feature_cols] for r in test])
        y_test = [r["home_win"] for r in test]
        mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression()
        clf.fit((X_train - mu) / sd, y_train)
        p = clf.predict_proba((X_test - mu) / sd)[:, 1]
        preds.extend(p)
        actuals.extend(y_test)
    return {
        "n": len(actuals),
        "auc": round(roc_auc_score(actuals, preds), 4),
        "log_loss": round(log_loss(actuals, preds), 4),
        "accuracy": round(accuracy_score(actuals, [1 if x > 0.5 else 0 for x in preds]), 4),
    }


def main():
    print(f"Solving rolling SRS ratings (trailing {TRAILING_WEEKS} weeks, ridge alpha={RIDGE_ALPHA})...", file=sys.stderr)
    rows = build_dataset()
    print(f"{len(rows)} games with a solved rating for both teams ({TEST_START_SEASON}-{SEASONS.stop - 1})\n")

    cols = [f"{m}_diff" for m in RATING_METRICS]
    feature_sets = {
        "rush_off_epa_diff (v6 SRS)": ["rush_off_epa_diff"],
        "pass_off_epa_diff (v6 SRS)": ["pass_off_epa_diff"],
        "rush_def_epa_allowed_diff (v6 SRS)": ["rush_def_epa_allowed_diff"],
        "pass_def_epa_allowed_diff (v6 SRS)": ["pass_def_epa_allowed_diff"],
        "all 4 ratings (v6 SRS combined)": cols,
        "market spread_line alone": ["spread_line"],
        "v6 SRS + market spread_line": cols + ["spread_line"],
    }

    print(f"{'feature set':40s} {'n':>6s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    for name, fcols in feature_sets.items():
        r = walk_forward_auc(rows, fcols)
        print(f"{name:40s} {r['n']:6d} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.4f}")


if __name__ == "__main__":
    main()
