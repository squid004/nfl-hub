"""v35: is "momentum" (a team's current win/loss streak) a real signal on top of the
model's own rating, or just a restatement of recent play quality the EWMA rating already
captures? `delta` already recency-weights the 8 underlying stats (EWMA_ALPHA=0.2), so a team
on a hot streak should mostly already show up as a HIGHER delta -- the interesting question
isn't "do streaky teams tend to win" (obviously yes, confounded with being good) but "does
streak length add anything ONCE delta is already in the model." Same joint-fit design as
v30 (rest/travel) and v29 (weather): control for delta, see if the new term survives.

streak_diff = home_streak - away_streak, where each team's own streak is its signed run of
CONSECUTIVE wins/losses immediately before this game (positive = win streak, negative = loss
streak, reset to 0 at the first game of a season and after any tie) -- derived directly from
games.csv's own chronological order, no new data source needed.

Same 3-stage structure as v29/v30:
  1. does the shipped model's hit rate differ in big-streak-mismatch games vs normal ones?
  2. joint logistic fit, home_win ~ delta + streak_diff -- and a second fit using a binary
     "hot streak" (>=3) version, since the momentum CLAIM is usually about streaks, not a
     linear per-game effect.
  3. if real: confidence-calibration check, same bar QB/skill/HFA were held to.

Run: python research/edge_signal_test_v35_momentum.py
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import requests
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub import config, store  # noqa: E402
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


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


def z_test(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return z, math.erfc(abs(z) / math.sqrt(2))


def main():
    print("Loading production backtest_scatter...", file=sys.stderr)
    config.get_config()
    data = json.loads(store.kv_get("historical_power_rankings"))
    scatter = data["backtest_scatter"]
    by_key = {(r["home"], r["away"], r["season"], r["week"]): r for r in scatter}
    print(f"{len(scatter)} backtest rows")

    print("Loading games.csv, deriving streaks...", file=sys.stderr)
    resp = requests.get(GAMES_URL, timeout=30)
    resp.raise_for_status()
    games_raw = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]

    by_team_season: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for g in games_raw:
        if g.get("result") in ("", "NA", None) or not g.get("home_score"):
            continue
        try:
            season, week = int(g["season"]), int(g["week"])
            home, away = normalize_team(g["home_team"]), normalize_team(g["away_team"])
            home_pts, away_pts = float(g["home_score"]), float(g["away_score"])
        except (ValueError, TypeError, UnknownTeamError):
            continue
        if home_pts == away_pts:
            home_result = away_result = "tie"
        else:
            home_result = "win" if home_pts > away_pts else "loss"
            away_result = "loss" if home_result == "win" else "win"
        by_team_season[(home, season)].append({"week": week, "result": home_result})
        by_team_season[(away, season)].append({"week": week, "result": away_result})

    # streak_before[(team, season, week)] = signed streak ENTERING that week's game.
    streak_before: dict[tuple[str, int, int], int] = {}
    for (team, season), games in by_team_season.items():
        games.sort(key=lambda g: g["week"])
        streak = 0
        for g in games:
            streak_before[(team, season, g["week"])] = streak
            if g["result"] == "tie":
                streak = 0
            elif g["result"] == "win":
                streak = streak + 1 if streak >= 0 else 1
            else:
                streak = streak - 1 if streak <= 0 else -1

    rows = []
    for r in scatter:
        if r["delta"] == 0 or r["home_score"] == r["away_score"]:
            continue
        hs = streak_before.get((r["home"], r["season"], r["week"]))
        aws = streak_before.get((r["away"], r["season"], r["week"]))
        if hs is None or aws is None:
            continue
        rows.append({**r, "home_streak": hs, "away_streak": aws, "streak_diff": hs - aws})
    print(f"{len(rows)} games joined to streak data\n")

    def pick(d): return "home" if d > 0 else "away"
    def winner(r): return "home" if r["home_score"] > r["away_score"] else "away"

    print("=== Stage 1: does accuracy differ by streak situation? ===")
    hot_mismatch = [r for r in rows if abs(r["streak_diff"]) >= 4]
    normal = [r for r in rows if abs(r["streak_diff"]) <= 1]

    def hit_rate(subset, label):
        hits = sum(1 for r in subset if pick(r["delta"]) == winner(r))
        print(f"  {label}: {hits}/{len(subset)} = {100 * hits / len(subset):.1f}%" if subset else f"  {label}: n=0")
        return hits, len(subset)

    h_hot, n_hot = hit_rate(hot_mismatch, "big streak mismatch (|diff|>=4)")
    h_norm, n_norm = hit_rate(normal, "near-even streaks (baseline)")
    if n_hot >= 10 and n_norm >= 10:
        z, p = z_test(h_hot, n_hot, h_norm, n_norm)
        print(f"    vs baseline: z={z:.2f}, p={p:.4f}")

    print("\n=== Stage 2: joint fit, home_win ~ delta + streak_diff ===")
    y = np.array([1 if winner(r) == "home" else 0 for r in rows], dtype=float)
    delta = np.array([r["delta"] for r in rows]).reshape(-1, 1)
    streak_diff = np.array([r["streak_diff"] for r in rows]).reshape(-1, 1)

    w_r, se_r = fit_logistic(delta, y)
    print(f"  reduced (delta only): delta={w_r[1]:+.4f}")

    w_f, se_f = fit_logistic(np.hstack([delta, streak_diff]), y)
    print(f"  full (linear streak): delta={w_f[1]:+.4f} (se {se_f[1]:.4f})  "
          f"streak_diff={w_f[2]:+.4f} (se {se_f[2]:.4f}) z={w_f[2] / se_f[2]:.2f}")

    # "Hot streak" binary version -- the momentum CLAIM is usually about being on a real
    # streak (>=3), not a per-game linear effect across the whole range.
    hot_diff = np.array([
        (1 if r["home_streak"] >= 3 else -1 if r["home_streak"] <= -3 else 0)
        - (1 if r["away_streak"] >= 3 else -1 if r["away_streak"] <= -3 else 0)
        for r in rows
    ], dtype=float).reshape(-1, 1)
    w_h, se_h = fit_logistic(np.hstack([delta, hot_diff]), y)
    print(f"  full (binary hot-streak>=3): delta={w_h[1]:+.4f} (se {se_h[1]:.4f})  "
          f"hot_diff={w_h[2]:+.4f} (se {se_h[2]:.4f}) z={w_h[2] / se_h[2]:.2f}")

    best_z, best_w, best_name, best_feat = (
        (w_f[2] / se_f[2], w_f, "linear streak_diff", streak_diff)
        if abs(w_f[2] / se_f[2]) >= abs(w_h[2] / se_h[2])
        else (w_h[2] / se_h[2], w_h, "binary hot_diff", hot_diff)
    )
    if abs(best_z) < 1.96:
        print(f"\nNeither streak version cleared significance (best: {best_name}, "
              f"z={best_z:.2f}) -- stopping here, no Stage 3 calibration check needed.")
        return

    print(f"\n=== Stage 3: confidence-calibration check (using {best_name}, z={best_z:.2f}) ===")
    unit_weight = best_w[2] / best_w[1]
    print(f"  delta-equivalent weight: {unit_weight:+.4f}")
    feat = best_feat.flatten()
    for r, f in zip(rows, feat):
        r["delta_with_momentum"] = r["delta"] + unit_weight * f
    old_correct = [pick(r["delta"]) == winner(r) for r in rows]
    misses = [r for r, oc in zip(rows, old_correct) if not oc]
    hitsg = [r for r, oc in zip(rows, old_correct) if oc]

    def reduced_rate(subset):
        if not subset:
            return 0, 0
        return sum(1 for r in subset if abs(r["delta_with_momentum"]) < abs(r["delta"])), len(subset)

    mr, mn = reduced_rate(misses)
    hr, hn = reduced_rate(hitsg)
    print(f"  confidence reduced on misses: {mr}/{mn} ({100 * mr / mn:.1f}%)  on hits: {hr}/{hn} ({100 * hr / hn:.1f}%)")
    if mn >= 10 and hn >= 10:
        z2, p2 = z_test(mr, mn, hr, hn)
        print(f"  z={z2:.2f}, p={p2:.4f}")


if __name__ == "__main__":
    main()
