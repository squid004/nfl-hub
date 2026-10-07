"""v31: how well does predict_points() actually predict each team's score right now, and
what does the ERROR DISTRIBUTION look like -- mean (bias check), spread (SD), shape (is it
close to Normal, or skewed/fat-tailed)? Nothing in this project has looked at the residual
DISTRIBUTION before, only point-accuracy (MAE) -- this is the groundwork for eventually
attaching real uncertainty (a sigma, or a non-Gaussian shape) to a score prediction instead
of just a single number, which matters for future feature work: knowing WHERE the model is
more/less uncertain (home vs away, favorite vs underdog, by rest/weather/etc.) is exactly the
kind of granularity a point-estimate-only MAE number hides.

Reuses research/edge_signal_test_v9_points_prediction.py's build_dataset() (already
produces per-side own-offense/opp-defense features + actual points, exactly what
predict_points() consumes) and the exact live POINTS_INTERCEPT/POINTS_WEIGHTS from
nflhub/sources/team_ratings.py (hardcoded here too, same convention as every other
"reproduce production, don't import it" research script this session).

Run: python research/edge_signal_test_v31_score_distribution.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import edge_signal_test_v9_points_prediction as v9  # noqa: E402

INTERCEPT = 7.21333
WEIGHTS = np.array([2.472559, 6.259427, 0.371737, -0.0, 2.470530, 0.666745, 0.317523, -0.0])


def main():
    print("Building per-side points dataset (reusing v9's rating walk)...", file=sys.stderr)
    rows = v9.build_dataset()
    for r in rows:
        r["pred"] = INTERCEPT + float(np.dot(WEIGHTS, r["features"]))
        r["resid"] = r["points"] - r["pred"]
    print(f"{len(rows)} team-game rows, {min(r['season'] for r in rows)}-{max(r['season'] for r in rows)}\n")

    resid = np.array([r["resid"] for r in rows])
    actual = np.array([r["points"] for r in rows])
    pred = np.array([r["pred"] for r in rows])

    print("=== overall accuracy ===")
    mae = np.mean(np.abs(resid))
    rmse = np.sqrt(np.mean(resid**2))
    bias = np.mean(resid)
    naive_mae = np.mean(np.abs(actual - actual.mean()))
    print(f"  MAE:  {mae:.3f} points/team-game  (naive league-avg guess: {naive_mae:.3f})")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  bias (mean actual-predicted): {bias:+.3f}  (should be ~0 if unbiased)")
    print(f"  actual points: mean={actual.mean():.2f} sd={actual.std():.2f}")
    print(f"  residual: sd={resid.std():.3f}  skew={stats.skew(resid):.3f}  kurtosis={stats.kurtosis(resid):.3f} (excess, 0=Normal)")

    print("\n=== recent-seasons-only (2020-2025), since 'right now' matters more than 2007 ===")
    recent = [r for r in rows if r["season"] >= 2020]
    rr = np.array([r["resid"] for r in recent])
    print(f"  n={len(recent)}  MAE={np.mean(np.abs(rr)):.3f}  bias={rr.mean():+.3f}  sd={rr.std():.3f}")

    print("\n=== by side (home vs. away -- is there a scoring bias predict_points() doesn't know about, like delta_raw had before HFA?) ===")
    for side in ("home", "away"):
        sub = np.array([r["resid"] for r in rows if r["side"] == side])
        print(f"  {side:5s}: n={len(sub)}  bias={sub.mean():+.3f}  MAE={np.mean(np.abs(sub)):.3f}  sd={sub.std():.3f}")
    home_r = np.array([r["resid"] for r in rows if r["side"] == "home"])
    away_r = np.array([r["resid"] for r in rows if r["side"] == "away"])
    t, p = stats.ttest_ind(home_r, away_r)
    print(f"  home vs away bias difference: t={t:+.2f} p={p:.4f}")

    print("\n=== normality check (Shapiro-ish via skew/kurtosis + percentile comparison) ===")
    for pct in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        emp = np.percentile(resid, pct)
        norm_q = stats.norm.ppf(pct/100, loc=resid.mean(), scale=resid.std())
        print(f"  p{pct:2d}: empirical={emp:+7.2f}  Normal-equivalent={norm_q:+7.2f}  diff={emp-norm_q:+.2f}")

    print("\n=== does residual SPREAD (not just win-prob accuracy) vary by context? ===")
    # favorite vs underdog side, using spread_line already in the row
    fav_rows = [r for r in rows if (r["side"] == "home" and r["spread_line"] < 0) or (r["side"] == "away" and r["spread_line"] > 0)]
    dog_rows = [r for r in rows if r not in fav_rows]
    fr = np.array([r["resid"] for r in fav_rows])
    dr_ = np.array([r["resid"] for r in dog_rows])
    print(f"  favorite side:  n={len(fr)}  sd={fr.std():.3f}  bias={fr.mean():+.3f}")
    print(f"  underdog side:  n={len(dr_)}  sd={dr_.std():.3f}  bias={dr_.mean():+.3f}")

    # blowout vs close games (|spread_line| as proxy for expected game competitiveness)
    close = [r for r in rows if abs(r["spread_line"]) <= 3]
    lopsided = [r for r in rows if abs(r["spread_line"]) >= 10]
    cr = np.array([r["resid"] for r in close])
    lr = np.array([r["resid"] for r in lopsided])
    print(f"  close spread (<=3):    n={len(cr)}  sd={cr.std():.3f}")
    print(f"  lopsided spread (>=10): n={len(lr)}  sd={lr.std():.3f}")


if __name__ == "__main__":
    main()
