"""v29: weather was already validated against POINTS (predict_points' WEATHER_ADJUSTMENT_
WEIGHTS, 7.647->7.603 MAE, "real but modest") but never actually tested against WIN
PROBABILITY / delta -- it just never cleared the old "must beat the market" bar for that
question. Given QB/skill/HFA all shipped on "real signal, even if small" instead, this
re-asks the win-probability question properly, same way every other signal this session was
tested: does the model's accuracy degrade in bad weather (the QB-out-style first check), and
does a weather term add real, calibration-worthy signal on top of the SHIPPED delta
(QB+skill+HFA)?

Reuses research/weather_scoring_analysis.py's exact data pipeline (cached per-stadium
Open-Meteo archive data, outdoor/open-roof games only, home-city not neutral-site) joined
against the live production backtest_scatter (already has delta/delta_raw/outcome).

v10's own prior finding is a relevant prior: "is rushing relatively stronger than passing in
high wind?" -- answer was no, passing stays stronger at every wind bucket. That argues
against a team-STYLE-asymmetric (pass-heavy vs. run-heavy) signal and FOR a pure confidence/
variance framing instead (bad weather -> more chaotic game -> less predictable, regardless of
which side is favored) -- tested here as a dampening effect on |delta|, not a directional
home-vs-away shift like QB/skill/HFA.

Run: python research/edge_signal_test_v29_weather_winprob.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import ttest_ind

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import weather_scoring_analysis as wx  # noqa: E402

sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub import config, store  # noqa: E402


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


def main():
    print("Loading production backtest_scatter (delta/delta_raw)...", file=sys.stderr)
    config.get_config()
    data = json.loads(store.kv_get("historical_power_rankings"))
    scatter = data["backtest_scatter"]
    by_key = {(r["home"], r["date"]): r for r in scatter}
    print(f"{len(scatter)} backtest rows")

    print("Loading games.csv + weather (cached from weather_scoring_analysis.py)...", file=sys.stderr)
    games = wx.load_games()
    outdoor = [g for g in games if g["roof"] in ("outdoors", "open") and g["location"] == "Home"
               and g["home_team"] in wx.STADIUM_COORDS and g["temp"] not in ("", "NA")]
    weather_by_team = {}
    for team in sorted({g["home_team"] for g in outdoor}):
        lat, lon = wx.STADIUM_COORDS[team]
        weather_by_team[team] = wx.fetch_stadium_weather(team, lat, lon)

    rows = []
    for g in outdoor:
        wxday = weather_by_team.get(g["home_team"], {}).get(g["gameday"])
        if not wxday:
            continue
        r = by_key.get((g["home_team"], g["gameday"]))
        if not r or r["delta"] == 0 or r["delta_raw"] == 0 or r["home_score"] == r["away_score"]:
            continue
        wind_mph = wxday["wind_kmh"] * 0.621371
        precip_mm = wxday["precip_mm"]
        snow_cm = wxday["snowfall_cm"]
        temp_f = wxday["temp_max_c"] * 9 / 5 + 32
        cold_flag = 1.0 if temp_f < 20 else 0.0
        bad = wind_mph >= 20 or precip_mm >= 10 or snow_cm > 0 or cold_flag
        rows.append({**r, "wind_mph": wind_mph, "precip_mm": precip_mm, "snow_cm": snow_cm,
                     "temp_f": temp_f, "cold_flag": cold_flag, "bad_weather": bad})
    print(f"{len(rows)} games joined to real historical weather\n")

    def pick(d): return "home" if d > 0 else "away"
    def winner(r): return "home" if r["home_score"] > r["away_score"] else "away"

    bad = [r for r in rows if r["bad_weather"]]
    good = [r for r in rows if not r["bad_weather"]]
    print(f"bad weather games: {len(bad)}   good weather games: {len(good)}\n")

    print("=== Stage 1: does the SHIPPED model's accuracy degrade in bad weather? ===")
    for label, subset in (("bad weather", bad), ("good weather", good)):
        hits = sum(1 for r in subset if pick(r["delta"]) == winner(r))
        print(f"  {label}: {hits}/{len(subset)} = {100*hits/len(subset):.1f}%")
    hits_bad = sum(1 for r in bad if pick(r["delta"]) == winner(r))
    hits_good = sum(1 for r in good if pick(r["delta"]) == winner(r))
    p1, n1 = hits_bad / len(bad), len(bad)
    p2, n2 = hits_good / len(good), len(good)
    p_pool = (hits_bad + hits_good) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    print(f"  z={z:.2f}, p={math.erfc(abs(z)/math.sqrt(2)):.4f}")

    print("\n=== Stage 2: does a weather severity term add signal on top of shipped delta? ===")
    # weather severity scalar, same ingredients as production's own WEATHER_ADJUSTMENT_WEIGHTS
    # (wind, precip, cold) -- a single 0+ "how bad is it" score, no sign (applies to both
    # teams/the whole game, not a home-vs-away diff the way QB/skill/HFA are).
    for r in rows:
        r["severity"] = r["wind_mph"] / 20 + r["precip_mm"] / 10 + r["snow_cm"] / 2 + r["cold_flag"]

    y = np.array([1 if pick(r["delta"]) == winner(r) else 0 for r in rows], dtype=float)
    # NOTE: testing severity as a DAMPENER on confidence, not a directional shift -- model as
    # delta * (1 - lambda*severity) and fit lambda via grid search on log-loss, since that's
    # not a linear-in-logit term the way the other three corrections are.
    delta_arr = np.array([r["delta"] for r in rows])
    home_win = np.array([1 if winner(r) == "home" else 0 for r in rows], dtype=float)
    sev = np.array([r["severity"] for r in rows])

    def fit_logistic_1d(x, y):
        def nll_grad(w):
            a, b = w
            z = a + b * x
            p = 1 / (1 + np.exp(-z))
            eps = 1e-12
            nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            ga = np.mean(p - y); gb = np.mean((p - y) * x)
            return nll, np.array([ga, gb])
        res = minimize(nll_grad, x0=np.array([0.0, 1.0]), jac=True, method="L-BFGS-B")
        return res.x

    from sklearn.metrics import log_loss
    baseline_a, baseline_b = fit_logistic_1d(delta_arr, home_win)
    baseline_ll = log_loss(home_win, 1/(1+np.exp(-(baseline_a+baseline_b*delta_arr))))
    print(f"  baseline (delta alone) logloss: {baseline_ll:.5f}")

    best_lambda, best_ll = 0.0, baseline_ll
    for lam in np.linspace(0, 0.3, 31):
        damp = delta_arr * (1 - lam * sev)
        a, b = fit_logistic_1d(damp, home_win)
        ll = log_loss(home_win, 1/(1+np.exp(-(a+b*damp))))
        if ll < best_ll:
            best_ll, best_lambda = ll, lam
    print(f"  best damping lambda={best_lambda:.3f}, logloss={best_ll:.5f} (vs {baseline_ll:.5f} undamped)")

    print("\n=== Stage 3: confidence-reduced-on-misses, severity as a direct |delta| dampener ===")
    lam = best_lambda if best_lambda > 0 else 0.1  # report SOMETHING even if grid search found no improvement
    for r in rows:
        r["delta_damped"] = r["delta"] * (1 - lam * r["severity"])
    old_correct = [pick(r["delta"]) == winner(r) for r in rows]
    misses = [r for r, oc in zip(rows, old_correct) if not oc]
    hits_ = [r for r, oc in zip(rows, old_correct) if oc]
    def reduced_rate(subset):
        if not subset: return 0, 0
        return sum(1 for r in subset if abs(r["delta_damped"]) < abs(r["delta"])), len(subset)
    mr, mn = reduced_rate(misses)
    hr, hn = reduced_rate(hits_)
    print(f"  (lambda={lam:.3f}) confidence reduced on misses: {mr}/{mn} ({100*mr/mn:.1f}%)  on hits: {hr}/{hn} ({100*hr/hn:.1f}%)")
    if mn >= 10 and hn >= 10:
        p1, p2 = mr/mn, hr/hn
        pp = (mr+hr)/(mn+hn)
        se2 = math.sqrt(pp*(1-pp)*(1/mn+1/hn))
        z2 = (p1-p2)/se2 if se2 > 0 else 0
        print(f"  z={z2:.2f}, p={math.erfc(abs(z2)/math.sqrt(2)):.4f}")


if __name__ == "__main__":
    main()
