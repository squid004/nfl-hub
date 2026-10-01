"""v10: (1) does adding wind/precipitation/extreme-cold to the points-prediction model
actually reduce out-of-sample error, walk-forward validated -- not just look good on the
residual chart from weather_scoring_analysis.py. (2) is rushing efficiency a relatively
STRONGER win-probability predictor than passing efficiency specifically in high-wind games
(testing the "establish the run in bad weather" folklore directly, not assuming it).

Reuses: v9's rating computation (own-offense + opp-defense points features) and
weather_scoring_analysis's stadium coordinates + cached daily weather (temp/precip/snow/wind).

Run: python research/edge_signal_test_v10_weather.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import edge_signal_test_v9_points_prediction as v9
import weather_scoring_analysis as wx

TEST_START_SEASON = 2011


def build_weather_matched_rows() -> list[dict]:
    """One row per (team, game) for outdoor/home-city games with matched weather -- same
    'features' (own-offense + opp-defense) as v9, plus wind_mph/precip_mm/cold_flag, plus the
    RAW (not lagged) rush/pass off/def EPA diffs needed for the wind-conditional AUC test."""
    print("Loading games + weather...", file=sys.stderr)
    games = wx.load_games()
    outdoor = {g["game_id"]: g for g in games
               if g["roof"] in ("outdoors", "open") and g["location"] == "Home"
               and g["home_team"] in wx.STADIUM_COORDS and g["temp"] not in ("", "NA")}

    weather_by_team = {}
    for team in sorted({g["home_team"] for g in outdoor.values()}):
        lat, lon = wx.STADIUM_COORDS[team]
        weather_by_team[team] = wx.fetch_stadium_weather(team, lat, lon)

    print("Building team ratings...", file=sys.stderr)
    side_rows = v9.build_dataset()  # per-side rows with 'features' (own-off + opp-def), 'points'
    by_game_side: dict[tuple, dict] = {}
    for r in side_rows:
        by_game_side[(r["game_id"], r["side"])] = r

    rows = []
    for gid, g in outdoor.items():
        wxday = weather_by_team.get(g["home_team"], {}).get(g["gameday"])
        if not wxday:
            continue
        hr = by_game_side.get((gid, "home"))
        ar = by_game_side.get((gid, "away"))
        if not hr or not ar:
            continue
        wind_mph = wxday["wind_kmh"] * 0.621371
        precip_mm = wxday["precip_mm"]
        temp_f = wxday["temp_max_c"] * 9 / 5 + 32
        for side_row, opp_row, is_home in ((hr, ar, True), (ar, hr, False)):
            rows.append({
                "season": side_row["season"], "game_id": gid,
                "features": side_row["features"], "points": side_row["points"],
                "wind_mph": wind_mph, "precip_mm": precip_mm, "cold_flag": 1.0 if temp_f < 20 else 0.0,
            })
        # also keep game-level rush/pass diffs + outcome for the wind-conditional AUC test
        home_off = hr["features"][:2]  # [rush_off_epa, pass_off_epa] (own-offense slice)
        away_off = ar["features"][:2]
        home_win = 1 if hr["points"] > ar["points"] else 0
        rows.append({  # sentinel row, filtered out below by presence of 'is_matchup'
            "is_matchup": True, "season": hr["season"], "game_id": gid,
            "rush_diff": home_off[0] - away_off[0], "pass_diff": home_off[1] - away_off[1],
            "wind_mph": wind_mph, "home_win": home_win,
        })
    return rows


def fit_nonneg(X, y, l2, kind="linear"):
    n, p = X.shape
    def loss_grad(w):
        z = w[0] + X @ w[1:]
        if kind == "linear":
            resid = z - y
            loss = np.mean(resid ** 2)
            grad = np.zeros_like(w)
            grad[0] = 2 * np.mean(resid)
            grad[1:] = 2 * (X.T @ resid) / n + 2 * l2 * w[1:]
        else:
            p_hat = 1 / (1 + np.exp(-z))
            eps = 1e-12
            loss = -np.mean(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))
            grad = np.zeros_like(w)
            grad[0] = np.mean(p_hat - y)
            grad[1:] = X.T @ (p_hat - y) / n + 2 * l2 * w[1:]
        reg = l2 * np.sum(w[1:] ** 2)
        return loss + reg, grad
    bounds = [(None, None)] + [(0, None)] * p
    res = minimize(loss_grad, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x


def walk_forward_points_weather(rows: list[dict], use_weather: bool, l2: float = 0.05) -> dict:
    seasons = sorted({r["season"] for r in rows})
    abs_errs = []
    extra = ["wind_mph", "precip_mm", "cold_flag"] if use_weather else []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        if len(train) < 200 or not test:
            continue
        X_train = np.array([list(r["features"]) + [r[k] for k in extra] for r in train])
        y_train = np.array([r["points"] for r in train])
        X_test = np.array([list(r["features"]) + [r[k] for k in extra] for r in test])
        # orient: wind/precip/cold all hurt scoring -> flip sign so higher=better, matching
        # the non-negative-constraint convention used throughout this project
        orient = np.array([1, 1, 1, -1, 1, 1, 1, -1] + [-1] * len(extra))
        Xo_train, Xo_test = X_train * orient, X_test * orient
        mu, sd = Xo_train.mean(axis=0), Xo_train.std(axis=0)
        sd[sd == 0] = 1.0
        w = fit_nonneg((Xo_train - mu) / sd, y_train, l2, kind="linear")
        preds = w[0] + ((Xo_test - mu) / sd) @ w[1:]
        abs_errs.extend(np.abs(preds - np.array([r["points"] for r in test])))
    return {"n": len(abs_errs), "points_mae": round(float(np.mean(abs_errs)), 4)}


def wind_conditional_auc(rows: list[dict]) -> None:
    matchups = [r for r in rows if r.get("is_matchup")]
    print(f"\n{len(matchups)} outdoor games with wind + rush/pass diffs\n")

    def report(subset, label):
        if len(subset) < 30:
            print(f"{label}: n={len(subset)} (too small, skipping)")
            return
        y = np.array([r["home_win"] for r in subset])
        rush = np.array([[r["rush_diff"]] for r in subset])
        pas = np.array([[r["pass_diff"]] for r in subset])
        def auc_for(X):
            mu, sd = X.mean(axis=0), X.std(axis=0)
            sd[sd == 0] = 1.0
            clf = LogisticRegression()
            clf.fit((X - mu) / sd, y)
            p = clf.predict_proba((X - mu) / sd)[:, 1]
            return roc_auc_score(y, p)
        print(f"{label:22s} n={len(subset):4d}   rush AUC={auc_for(rush):.4f}   pass AUC={auc_for(pas):.4f}")

    report([r for r in matchups if r["wind_mph"] < 10], "wind < 10 mph")
    report([r for r in matchups if 10 <= r["wind_mph"] < 15], "wind 10-15 mph")
    report([r for r in matchups if 15 <= r["wind_mph"] < 20], "wind 15-20 mph")
    report([r for r in matchups if r["wind_mph"] >= 15], "wind >= 15 mph")
    report([r for r in matchups if r["wind_mph"] >= 20], "wind >= 20 mph")
    report(matchups, "all outdoor games")


def main():
    rows = build_weather_matched_rows()
    side_rows = [r for r in rows if not r.get("is_matchup")]
    print(f"\n{len(side_rows)} team-game rows with weather for points-model comparison\n")

    r_base = walk_forward_points_weather(side_rows, use_weather=False)
    r_wx = walk_forward_points_weather(side_rows, use_weather=True)
    print("=== Points-prediction MAE: with vs without weather (walk-forward, same games) ===")
    print(f"without weather: n={r_base['n']}  MAE={r_base['points_mae']}")
    print(f"with weather:    n={r_wx['n']}  MAE={r_wx['points_mae']}")
    delta = r_base["points_mae"] - r_wx["points_mae"]
    print(f"improvement: {delta:+.4f} points MAE ({'HELPS' if delta > 0 else 'no help / hurts'})")

    print("\n=== Does rushing matter relatively more than passing in high wind? ===")
    wind_conditional_auc(rows)


if __name__ == "__main__":
    main()
