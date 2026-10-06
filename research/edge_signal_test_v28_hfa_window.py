"""v28: what's the right way to estimate home-field advantage so it tracks a real regime
shift (the 2019-2021 drop, including 2020's no-fans anomaly) without being whipsawed by
single-season noise (each season is only ~256 games, +/-6pt 95% CI on its own win rate)?

Tests several candidate estimators for "P(home win)" with NO other information (pure
baseline, nothing from the 8-stat composite) via walk-forward log-loss -- same discipline as
every other choice in this project (EWMA_ALPHA, SEASON_PRIOR_GAMES, L2 strength): don't pick
a window by feel, let out-of-sample prediction quality decide.

Candidates:
  - flat: all-seasons-to-date average (no recency at all)
  - trailing-N: simple average over the last N completed seasons (N in {2,3,5,7,10})
  - ewma-a: season-level EWMA with alpha in {0.15, 0.3, 0.5} (NOT the team rating's game-
    level EWMA_ALPHA=0.2 -- a separate, season-granularity smoothing constant)
  - shrink-k: trailing-5-season average, shrunk toward the long-run (all-time-to-date)
    average with pseudo-count k games (same shrinkage convention nflhub/sources/history.py
    already uses for spread-bucket win rates, k in {100, 300, 600})

All walk-forward from TEST_START (needs several seasons of history first), log-loss scored
against actual home_win each test season using ONLY prior seasons' data.

Run: python research/edge_signal_test_v28_hfa_window.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np
import requests
from sklearn.metrics import log_loss

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
FIRST_SEASON = 2007
TEST_START = 2013
EPS = 1e-6


def load_games():
    path = os.path.join(CACHE_DIR, "games.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
    else:
        resp = requests.get(GAMES_URL, timeout=30)
        resp.raise_for_status()
        text = resp.text
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if r["game_type"] != "REG":
            continue
        try:
            season = int(r["season"])
            home_pts, away_pts = float(r["home_score"]), float(r["away_score"])
        except (ValueError, TypeError):
            continue
        if season < FIRST_SEASON or home_pts == away_pts:
            continue
        rows.append({"season": season, "home_win": 1 if home_pts > away_pts else 0})
    return rows


def main():
    games = load_games()
    by_season: dict[int, list[int]] = {}
    for g in games:
        by_season.setdefault(g["season"], []).append(g["home_win"])
    seasons = sorted(by_season)
    print(f"{len(games)} games, seasons {seasons[0]}-{seasons[-1]}\n")

    def season_rate(s):
        wins = by_season.get(s, [])
        return (sum(wins), len(wins))

    candidates = {"flat": None}
    for n in (2, 3, 5, 7, 10):
        candidates[f"trailing-{n}"] = n
    for a in (0.15, 0.3, 0.5):
        candidates[f"ewma-{a}"] = a
    for k in (100, 300, 600):
        candidates[f"shrink5-k{k}"] = k

    results = {}
    for name in candidates:
        preds, actuals = [], []
        for test_season in seasons:
            if test_season < TEST_START:
                continue
            prior_seasons = [s for s in seasons if s < test_season]
            if len(prior_seasons) < 3:
                continue
            if name == "flat":
                w = sum(season_rate(s)[0] for s in prior_seasons)
                n = sum(season_rate(s)[1] for s in prior_seasons)
                p_home = w / n
            elif name.startswith("trailing-"):
                n_seasons = candidates[name]
                window = prior_seasons[-n_seasons:]
                w = sum(season_rate(s)[0] for s in window)
                n = sum(season_rate(s)[1] for s in window)
                p_home = w / n
            elif name.startswith("ewma-"):
                alpha = candidates[name]
                est = None
                for s in prior_seasons:
                    w, n = season_rate(s)
                    rate = w / n
                    est = rate if est is None else (1 - alpha) * est + alpha * rate
                p_home = est
            elif name.startswith("shrink5-"):
                k = candidates[name]
                window = prior_seasons[-5:]
                w5 = sum(season_rate(s)[0] for s in window)
                n5 = sum(season_rate(s)[1] for s in window)
                rate5 = w5 / n5 if n5 else 0.5
                w_all = sum(season_rate(s)[0] for s in prior_seasons)
                n_all = sum(season_rate(s)[1] for s in prior_seasons)
                rate_all = w_all / n_all if n_all else 0.5
                p_home = (n5 * rate5 + k * rate_all) / (n5 + k)
            p_home = min(max(p_home, EPS), 1 - EPS)

            test_games = by_season[test_season]
            preds.extend([p_home] * len(test_games))
            actuals.extend(test_games)

        ll = log_loss(actuals, preds, labels=[0, 1])
        results[name] = (ll, len(actuals))

    print(f"{'estimator':16s} {'n':>5s} {'logloss':>9s}")
    for name, (ll, n) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"{name:16s} {n:5d} {ll:9.5f}")

    best = min(results, key=lambda k: results[k][0])
    print(f"\nbest: {best} (logloss={results[best][0]:.5f})")

    # show the WINNING method's own trailing estimate per season, so the shape is visible
    print(f"\n--- {best}'s own estimate by season (what it would have used going INTO that season) ---")
    for test_season in seasons:
        if test_season < TEST_START:
            continue
        prior_seasons = [s for s in seasons if s < test_season]
        if len(prior_seasons) < 3:
            continue
        if best == "flat":
            w = sum(season_rate(s)[0] for s in prior_seasons); n = sum(season_rate(s)[1] for s in prior_seasons)
            est = w / n
        elif best.startswith("trailing-"):
            window = prior_seasons[-candidates[best]:]
            w = sum(season_rate(s)[0] for s in window); n = sum(season_rate(s)[1] for s in window)
            est = w / n
        elif best.startswith("ewma-"):
            alpha = candidates[best]; est = None
            for s in prior_seasons:
                w, n = season_rate(s); rate = w / n
                est = rate if est is None else (1 - alpha) * est + alpha * rate
        elif best.startswith("shrink5-"):
            k = candidates[best]
            window = prior_seasons[-5:]
            w5 = sum(season_rate(s)[0] for s in window); n5 = sum(season_rate(s)[1] for s in window)
            rate5 = w5 / n5 if n5 else 0.5
            w_all = sum(season_rate(s)[0] for s in prior_seasons); n_all = sum(season_rate(s)[1] for s in prior_seasons)
            rate_all = w_all / n_all if n_all else 0.5
            est = (n5 * rate5 + k * rate_all) / (n5 + k)
        actual_w, actual_n = season_rate(test_season)
        print(f"  {test_season}: predicted {est*100:5.1f}%  (actual that season: {100*actual_w/actual_n:5.1f}%)")


if __name__ == "__main__":
    main()
