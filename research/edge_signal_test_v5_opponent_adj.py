"""v5: does opponent-adjusting each game's raw EPA before it feeds the rolling rating help?

Raw EPA has zero built-in opponent-strength normalization (nflfastR's model conditions on
down/distance/field position/score/time, never on who's on defense). v2-v4's rolling ratings
inherited that: a team's rush-offense rating after a stretch against elite run defenses looks
worse than their true talent, purely from schedule luck, and the EWMA's short effective
window (~3-game half-life at alpha=0.2) makes this worse, not better.

Single-pass adjustment (SRS/Sagarin-style, not full iterative convergence -- fits the
existing streaming/incremental architecture): before a game's raw EPA updates a team's
rolling rating, adjust it by how far the opponent's CURRENT (pre-game, no look-ahead) rating
on the complementary side sits from the league mean. Faced a defense that's stingier than
average? raw value gets bumped up before crediting it. Faced a bad one? discounted. Same
logic in reverse when crediting a defense for what it held an offense to.

Computes BOTH adjusted and unadjusted ratings in the same pass (identical games, identical
EWMA/carryover/garbage-time-filter) for a clean walk-forward AUC comparison.

Run: python research/edge_signal_test_v5_opponent_adj.py
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


class RunningMean:
    def __init__(self):
        self.n, self.mean = 0, 0.0

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n


def build_dataset() -> list[dict]:
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for season in SEASONS:
        for row in team_game_phase_stats(season).to_dict("records"):
            by_game[row["game_id"]][row["team"]] = row

    games = load_games()
    running_mean = {"rush_off_epa": RunningMean(), "pass_off_epa": RunningMean()}

    def league_mean(metric: str) -> float:
        return running_mean[metric.replace("_def_epa_allowed", "_off_epa")].mean

    # two parallel rating tracks -- unadjusted (v4, baseline) and opponent-adjusted (v5) --
    # over the IDENTICAL game sequence, so any AUC difference is purely the adjustment.
    rating_raw: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    rating_adj: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    last_season: dict[str, int] = {}

    rows_out = []
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
                    for rt in (rating_raw, rating_adj):
                        r = rt[team][m]
                        if r is not None:
                            lm = league_mean(m)
                            rt[team][m] = lm + CARRYOVER * (r - lm)
            last_season[team] = season

        ok = all(rating_raw[home][m] is not None and rating_raw[away][m] is not None for m in RATING_METRICS)
        feat_row = {"season": season, "week": week, "game_id": gid}
        if ok:
            for m in RATING_METRICS:
                feat_row[f"{m}_diff_raw"] = rating_raw[home][m] - rating_raw[away][m]
                feat_row[f"{m}_diff_adj"] = rating_adj[home][m] - rating_adj[away][m]
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

        # opponent-adjust each side's raw phase values using the OTHER team's pre-game
        # adjusted rating on the complementary metric, before either update happens.
        def adj_value(off_val: float, opp_def_rating: float | None, off_metric: str) -> float:
            if opp_def_rating is None:
                return off_val
            return off_val + (league_mean(off_metric) - opp_def_rating)

        def adj_def_value(def_allowed_val: float, opp_off_rating: float | None, off_metric: str) -> float:
            if opp_off_rating is None:
                return def_allowed_val
            return def_allowed_val - (opp_off_rating - league_mean(off_metric))

        home_adj = {
            "rush_off_epa": adj_value(home_raw["rush_off_epa"], rating_adj[away]["rush_def_epa_allowed"], "rush_off_epa"),
            "pass_off_epa": adj_value(home_raw["pass_off_epa"], rating_adj[away]["pass_def_epa_allowed"], "pass_off_epa"),
            "rush_def_epa_allowed": adj_def_value(home_raw["rush_def_epa_allowed"], rating_adj[away]["rush_off_epa"], "rush_off_epa"),
            "pass_def_epa_allowed": adj_def_value(home_raw["pass_def_epa_allowed"], rating_adj[away]["pass_off_epa"], "pass_off_epa"),
        }
        away_adj = {
            "rush_off_epa": adj_value(away_raw["rush_off_epa"], rating_adj[home]["rush_def_epa_allowed"], "rush_off_epa"),
            "pass_off_epa": adj_value(away_raw["pass_off_epa"], rating_adj[home]["pass_def_epa_allowed"], "pass_off_epa"),
            "rush_def_epa_allowed": adj_def_value(away_raw["rush_def_epa_allowed"], rating_adj[home]["rush_off_epa"], "rush_off_epa"),
            "pass_def_epa_allowed": adj_def_value(away_raw["pass_def_epa_allowed"], rating_adj[home]["pass_off_epa"], "pass_off_epa"),
        }

        for m in ("rush_off_epa", "pass_off_epa"):
            running_mean[m].update(home_raw[m])
            running_mean[m].update(away_raw[m])
        for team, raw, adj in ((home, home_raw, home_adj), (away, away_raw, away_adj)):
            for m in RATING_METRICS:
                prev_raw, prev_adj = rating_raw[team][m], rating_adj[team][m]
                rating_raw[team][m] = raw[m] if prev_raw is None else (1 - EWMA_ALPHA) * prev_raw + EWMA_ALPHA * raw[m]
                rating_adj[team][m] = adj[m] if prev_adj is None else (1 - EWMA_ALPHA) * prev_adj + EWMA_ALPHA * adj[m]

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
    print("Building raw vs opponent-adjusted rating dataset...", file=sys.stderr)
    rows = build_dataset()
    print(f"{len(rows)} games with an established rating for both teams ({SEASONS.start}-{SEASONS.stop - 1})\n")

    raw_cols = [f"{m}_diff_raw" for m in RATING_METRICS]
    adj_cols = [f"{m}_diff_adj" for m in RATING_METRICS]
    feature_sets = {
        "rush_off_epa_diff (v4 raw, no opp adj)": ["rush_off_epa_diff_raw"],
        "rush_off_epa_diff (v5 opponent-adjusted)": ["rush_off_epa_diff_adj"],
        "pass_off_epa_diff (v4 raw)": ["pass_off_epa_diff_raw"],
        "pass_off_epa_diff (v5 adjusted)": ["pass_off_epa_diff_adj"],
        "all 4 ratings (v4 raw, baseline)": raw_cols,
        "all 4 ratings (v5 opponent-adjusted)": adj_cols,
        "market spread_line alone": ["spread_line"],
        "v4 raw + market spread_line": raw_cols + ["spread_line"],
        "v5 adjusted + market spread_line": adj_cols + ["spread_line"],
    }

    print(f"{'feature set':45s} {'n':>6s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    for name, cols in feature_sets.items():
        r = walk_forward_auc(rows, cols)
        print(f"{name:45s} {r['n']:6d} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.4f}")


if __name__ == "__main__":
    main()
