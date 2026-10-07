"""v30: does rest differential (short week / bye week / normal mismatch) add real signal
to win probability, same bar QB/skill/HFA were held to (and weather just failed)? Unlike
weather, nflverse's games.csv already ships `home_rest`/`away_rest` directly (verified live:
Thursday games correctly show rest=4, bye-week games show 12+) -- no derivation needed.
`div_game` is sitting in the same file, so tested as a quick bonus (familiarity-breeds-parity
is a well-known claim worth a cheap check while the data's already loaded).

Same 3-stage structure as v29 (weather) and the QB/skill/HFA work before it:
  1. does the SHIPPED model's accuracy differ when there's a real rest mismatch (the
     QB-out-style "does accuracy crater" check)?
  2. joint logistic fit, home_win ~ delta + rest_diff (home_rest - away_rest, raw units) on
     ALL data -- significant and correctly signed, or not?
  3. if real: confidence-calibration check (does it reduce confidence selectively on misses,
     the actual bar used to justify shipping QB/skill/HFA)?

Run: python research/edge_signal_test_v30_rest_travel.py
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys

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


def main():
    print("Loading production backtest_scatter...", file=sys.stderr)
    config.get_config()
    data = json.loads(store.kv_get("historical_power_rankings"))
    scatter = data["backtest_scatter"]
    by_key = {(r["home"], r["away"], r["season"], r["week"]): r for r in scatter}
    print(f"{len(scatter)} backtest rows")

    print("Loading games.csv (home_rest/away_rest/div_game)...", file=sys.stderr)
    resp = requests.get(GAMES_URL, timeout=30)
    resp.raise_for_status()
    games_raw = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]

    rows = []
    for g in games_raw:
        try:
            season, week = int(g["season"]), int(g["week"])
            home, away = normalize_team(g["home_team"]), normalize_team(g["away_team"])
            home_rest, away_rest = int(g["home_rest"]), int(g["away_rest"])
            div_game = int(g["div_game"])
        except (ValueError, TypeError, UnknownTeamError):
            continue
        r = by_key.get((home, away, season, week))
        if not r or r["delta"] == 0 or r["home_score"] == r["away_score"]:
            continue
        rows.append({**r, "home_rest": home_rest, "away_rest": away_rest,
                     "rest_diff": home_rest - away_rest, "div_game": div_game})
    print(f"{len(rows)} games joined to rest/div data\n")

    def pick(d): return "home" if d > 0 else "away"
    def winner(r): return "home" if r["home_score"] > r["away_score"] else "away"

    print("=== Stage 1: does accuracy differ by rest situation? ===")
    short_week = [r for r in rows if r["home_rest"] == 4 or r["away_rest"] == 4]
    bye = [r for r in rows if r["home_rest"] >= 10 or r["away_rest"] >= 10]
    mismatch = [r for r in rows if abs(r["rest_diff"]) >= 3]
    normal = [r for r in rows if abs(r["rest_diff"]) == 0 and r["home_rest"] not in (4,) and r["home_rest"] < 10]

    def hit_rate(subset, label):
        hits = sum(1 for r in subset if pick(r["delta"]) == winner(r))
        print(f"  {label}: {hits}/{len(subset)} = {100*hits/len(subset):.1f}%" if subset else f"  {label}: n=0")
        return hits, len(subset)

    h_short, n_short = hit_rate(short_week, "short week (either team on 4 days rest)")
    h_bye, n_bye = hit_rate(bye, "bye week (either team on 10+ days rest)")
    h_mis, n_mis = hit_rate(mismatch, "big rest mismatch (|diff|>=3 days)")
    h_norm, n_norm = hit_rate(normal, "normal/even rest (baseline)")

    for label, h, n in (("short week", h_short, n_short), ("bye week", h_bye, n_bye), ("big mismatch", h_mis, n_mis)):
        if n < 10 or n_norm < 10:
            continue
        p1, p2 = h/n, h_norm/n_norm
        pp = (h+h_norm)/(n+n_norm)
        se = math.sqrt(pp*(1-pp)*(1/n+1/n_norm))
        z = (p1-p2)/se if se > 0 else 0
        print(f"    {label} vs baseline: z={z:.2f}, p={math.erfc(abs(z)/math.sqrt(2)):.4f}")

    print("\n=== Stage 2: joint fit, home_win ~ delta + rest_diff (+ div_game bonus check) ===")
    y = np.array([1 if winner(r) == "home" else 0 for r in rows], dtype=float)
    delta = np.array([r["delta"] for r in rows]).reshape(-1, 1)
    rest_diff = np.array([r["rest_diff"] for r in rows]).reshape(-1, 1)

    w_r, se_r = fit_logistic(delta, y)
    print(f"  reduced (delta only): delta={w_r[1]:+.4f}")

    w_f, se_f = fit_logistic(np.hstack([delta, rest_diff]), y)
    print(f"  full: delta={w_f[1]:+.4f} (se {se_f[1]:.4f})  rest_diff={w_f[2]:+.4f} (se {se_f[2]:.4f}) z={w_f[2]/se_f[2]:.2f}")

    # bonus: div_game as a VARIANCE check (does it predict LOWER |delta|-implied accuracy,
    # i.e. is it a calibration candidate the way QB/skill are, not a directional one)
    div = [r for r in rows if r["div_game"] == 1]
    nondiv = [r for r in rows if r["div_game"] == 0]
    hd, nd = hit_rate(div, "divisional games")
    hn, nn = hit_rate(nondiv, "non-divisional games")
    pp = (hd+hn)/(nd+nn); se = math.sqrt(pp*(1-pp)*(1/nd+1/nn))
    z = (hd/nd - hn/nn)/se if se > 0 else 0
    print(f"    divisional vs non-divisional: z={z:.2f}, p={math.erfc(abs(z)/math.sqrt(2)):.4f}")

    if abs(w_f[2] / se_f[2]) < 1.96:
        print("\nrest_diff did not clear significance -- stopping here, no Stage 3 calibration check needed.")
        return

    print("\n=== Stage 3: confidence-calibration check ===")
    unit_weight = w_f[2] / w_f[1]
    print(f"  delta-equivalent weight: {unit_weight:+.4f}")
    for r in rows:
        r["delta_with_rest"] = r["delta"] + unit_weight * r["rest_diff"]
    old_correct = [pick(r["delta"]) == winner(r) for r in rows]
    misses = [r for r, oc in zip(rows, old_correct) if not oc]
    hitsg = [r for r, oc in zip(rows, old_correct) if oc]
    def reduced_rate(subset):
        if not subset: return 0, 0
        return sum(1 for r in subset if abs(r["delta_with_rest"]) < abs(r["delta"])), len(subset)
    mr, mn = reduced_rate(misses)
    hr, hn = reduced_rate(hitsg)
    print(f"  confidence reduced on misses: {mr}/{mn} ({100*mr/mn:.1f}%)  on hits: {hr}/{hn} ({100*hr/hn:.1f}%)")
    if mn >= 10 and hn >= 10:
        p1, p2 = mr/mn, hr/hn
        pp = (mr+hr)/(mn+hn)
        se2 = math.sqrt(pp*(1-pp)*(1/mn+1/hn))
        z2 = (p1-p2)/se2 if se2 > 0 else 0
        print(f"  z={z2:.2f}, p={math.erfc(abs(z2)/math.sqrt(2)):.4f}")


if __name__ == "__main__":
    main()
