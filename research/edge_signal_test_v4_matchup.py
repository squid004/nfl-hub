"""v4: does splitting team ratings by phase (rush vs pass) -- and explicitly testing a
run-offense-vs-run-defense (and pass-vs-pass) matchup interaction -- add anything over v3's
single pooled offense/defense EPA rating per team?

v3 pooled rushing_epa + passing_epa into one number, so a team's rush-specific strength was
invisible (averaged into whatever their pass game was doing). Here each team gets 4 ratings
(rush off, pass off, rush def allowed, pass def allowed) instead of 2. Two things are tested
separately, since they are NOT the same claim:

1. Does phase-splitting the LINEAR features help? (a team's rush-off rating minus the
   opponent's rush-def-allowed rating, as two separate diff terms -- logistic regression can
   already combine two linear diffs additively, so this alone isn't a new "interaction",
   just a finer-grained version of what v3 did.)
2. Does an explicit MULTIPLICATIVE interaction term (rush_off_rating * opponent's
   rush_def_allowed_rating) add anything BEYOND the linear terms -- i.e. is a bad matchup
   worse than the sum of its parts, not just the sum of them? This is the real test of
   whether "these cancel out" understates or overstates what actually happens.

Same leakage discipline / walk-forward evaluation as v2/v3.

Run: python research/edge_signal_test_v4_matchup.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
SEASONS = range(2007, 2025)
TEST_START_SEASON = 2011
EWMA_ALPHA = 0.2
CARRYOVER = 0.65
WP_LO, WP_HI = 0.05, 0.95

# 4 ratings per team instead of v3's 2: phase-split offense/defense-allowed EPA.
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
    """One row per (game_id, team): garbage-time-filtered EPA/play, separately for rush and
    pass plays."""
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


def build_dataset() -> list[dict]:
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for season in SEASONS:
        tg = team_game_phase_stats(season)
        for row in tg.to_dict("records"):
            by_game[row["game_id"]][row["team"]] = row

    class RunningMean:
        def __init__(self):
            self.n, self.mean = 0, 0.0

        def update(self, x):
            self.n += 1
            self.mean += (x - self.mean) / self.n

    running_mean = {"rush_off_epa": RunningMean(), "pass_off_epa": RunningMean()}

    def league_mean(metric: str) -> float:
        base = metric.replace("_def_epa_allowed", "_off_epa")
        return running_mean[base].mean

    rating: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    last_season: dict[str, int] = {}

    rows_out = []
    games = load_games()
    for gid, teams in sorted(by_game.items()):
        if gid not in games or len(teams) != 2:
            continue
        g = games[gid]
        try:
            season, week = int(g["season"]), int(g["week"])
        except ValueError:
            continue
        home, away = g["home_team"], g["away_team"]
        if home not in teams or away not in teams:
            continue

        home_raw, away_raw = dict(teams[home]), dict(teams[away])
        home_raw["rush_def_epa_allowed"] = away_raw["rush_off_epa"]
        away_raw["rush_def_epa_allowed"] = home_raw["rush_off_epa"]
        home_raw["pass_def_epa_allowed"] = away_raw["pass_off_epa"]
        away_raw["pass_def_epa_allowed"] = home_raw["pass_off_epa"]

        for team in (home, away):
            if last_season.get(team) is not None and last_season[team] != season:
                for m in RATING_METRICS:
                    r = rating[team][m]
                    if r is not None:
                        lm = league_mean(m)
                        rating[team][m] = lm + CARRYOVER * (r - lm)
            last_season[team] = season

        ok = all(rating[home][m] is not None and rating[away][m] is not None for m in RATING_METRICS)
        feat_row = {"season": season, "week": week, "game_id": gid}
        if ok:
            hr, ar = rating[home], rating[away]
            for m in RATING_METRICS:
                feat_row[f"{m}_diff"] = hr[m] - ar[m]
            # explicit matchup interactions: is home's rush offense vs away's rush D (and the
            # mirror) worth anything BEYOND the two linear diffs above? Product terms, not sums.
            feat_row["rush_matchup_home"] = hr["rush_off_epa"] * ar["rush_def_epa_allowed"]
            feat_row["rush_matchup_away"] = ar["rush_off_epa"] * hr["rush_def_epa_allowed"]
            feat_row["pass_matchup_home"] = hr["pass_off_epa"] * ar["pass_def_epa_allowed"]
            feat_row["pass_matchup_away"] = ar["pass_off_epa"] * hr["pass_def_epa_allowed"]
            try:
                feat_row["spread_line"] = float(g["spread_line"])
            except (ValueError, TypeError):
                ok = False
        if ok:
            try:
                res = float(g["result"])
            except (ValueError, TypeError):
                res = float(g.get("home_score") or 0) - float(g.get("away_score") or 0)
            feat_row["home_win"] = 1 if res > 0 else 0
            rows_out.append(feat_row)

        for m in ("rush_off_epa", "pass_off_epa"):
            running_mean[m].update(home_raw[m])
            running_mean[m].update(away_raw[m])
        for team, raw in ((home, home_raw), (away, away_raw)):
            for m in RATING_METRICS:
                prev = rating[team][m]
                rating[team][m] = raw[m] if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * raw[m]

    return rows_out


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
    print("Building phase-split pbp dataset (reuses cached raw pbp if present)...", file=sys.stderr)
    rows = build_dataset()
    print(f"{len(rows)} games with an established rating for both teams ({SEASONS.start}-{SEASONS.stop - 1})\n")

    linear_cols = [f"{m}_diff" for m in RATING_METRICS]
    matchup_cols = ["rush_matchup_home", "rush_matchup_away", "pass_matchup_home", "pass_matchup_away"]
    feature_sets = {
        "rush_off_epa_diff alone": ["rush_off_epa_diff"],
        "pass_off_epa_diff alone": ["pass_off_epa_diff"],
        "4 phase-split linear diffs (rush+pass, off+def)": linear_cols,
        "4 linear diffs + explicit matchup products": linear_cols + matchup_cols,
        "explicit matchup products alone (no linear diffs)": matchup_cols,
        "market spread_line alone": ["spread_line"],
        "4 linear diffs + matchups + spread_line": linear_cols + matchup_cols + ["spread_line"],
    }

    print(f"{'feature set':50s} {'n':>6s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    for name, cols in feature_sets.items():
        r = walk_forward_auc(rows, cols)
        print(f"{name:50s} {r['n']:6d} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.4f}")


if __name__ == "__main__":
    main()
