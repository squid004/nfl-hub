"""v19: v16-v18 all modeled health as a CONTINUOUS linear effect -- one slope coefficient
fit mostly against the dominant case (0 or 1 player out), which would wash out a real but
rare, concentrated, possibly nonlinear "unit is genuinely wrecked" effect (3+ starters out
at once, or banged up across a whole side of the ball) if that's where the real signal
lives. This tests the TAIL directly instead: binary "decimated" indicators at the top of the
observed severity distribution, both per position group AND per side of the ball (the user's
own framing: "a completely wrecked secondary" / "down to WR3,4,5" / "generally on a certain
side of the ball"), rather than a single slope averaged across every severity level.

Severity here is the RAW weekly snap-share-weighted absence (v17's `weighted_absence_by_
team_week_group`, NOT the EWMA-relative gap v16-v18 used) -- "how many starter-equivalents
are out THIS week," full stop, matching the user's framing directly (a team stuck starting
WR3/4/5 for two straight weeks is decimated in both weeks, gap-to-recent-average or not).
Side-of-ball severity sums the 3 groups on that side (qb+ol+skill / dl+lb+secondary).

Two tests per indicator, same as v16-v18:
  (a) per-metric EPA residual: two-sample comparison (decimated weeks vs. not) on
      actual-vs-pregame-rating residual for the metric(s) that group/side feeds.
  (b) game outcome: does a home-minus-away decimated DUMMY add to the existing composite
      `delta` in a logistic fit on home_win (same joint/LR-test framework as v18)?

Thresholds are picked from the observed severity distribution itself (printed first) rather
than guessed -- a looser ("short-handed", ~85th percentile) and a stricter ("decimated",
~95th percentile) cut, to see whether any effect gets stronger as the cut gets more extreme
(a real dose-response pattern) or stays flat/noisy (more likely just sampling noise).

Run: python research/edge_signal_test_v19_unit_health_tail.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm, ttest_ind

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import importlib.util
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v14 = _load("v14", "edge_signal_test_v14_qb_injury_dropout.py")
v16 = _load("v16", "edge_signal_test_v16_unit_health.py")
v17 = _load("v17", "edge_signal_test_v17_unit_health_snapweighted.py")

FIRST_SNAP_SEASON = 2012
GROUPS = list(v16.POSITION_GROUPS.keys())
SIDES = {"offense": ["qb", "ol", "skill"], "defense": ["dl", "lb", "secondary"]}
SIDE_METRICS = {
    "offense": ["rush_off_epa", "pass_off_epa"],
    "defense": ["rush_def_epa_allowed", "pass_def_epa_allowed"],
}


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
    print("Loading snap-weighted injury severity (2012+)...", file=sys.stderr)
    player_games, pos_default = v17.load_snap_shares()
    injury_rows = v17.load_injury_rows()
    severity = v17.weighted_absence_by_team_week_group(injury_rows, player_games, pos_default, None)

    print("Building per-team-game rating/actual rows (2012+)...", file=sys.stderr)
    all_team_rows = v16.build_team_game_rows()
    team_rows = [r for r in all_team_rows if r["season"] >= FIRST_SNAP_SEASON]
    for r in team_rows:
        for grp in GROUPS:
            r[f"sev_{grp}"] = severity.get((r["team"], r["season"], r["week"], grp), 0.0)
        for side, grps in SIDES.items():
            r[f"sev_{side}"] = sum(r[f"sev_{g}"] for g in grps)

    print("\n--- severity distribution (nonzero values only, since most team-weeks have none) ---")
    thresholds: dict[str, tuple[float, float]] = {}
    for key in GROUPS + list(SIDES):
        vals = np.array([r[f"sev_{key}"] for r in team_rows])
        nz = vals[vals > 0]
        p85, p95 = np.percentile(vals, 85), np.percentile(vals, 95)
        thresholds[key] = (p85, p95)
        print(f"  {key:10s} nonzero={len(nz)/len(vals)*100:5.1f}%  "
              f"median(nz)={np.median(nz) if len(nz) else 0:.2f}  p85={p85:.2f}  p95={p95:.2f}  max={vals.max():.2f}")

    for r in team_rows:
        for key in GROUPS + list(SIDES):
            p85, p95 = thresholds[key]
            r[f"short_{key}"] = 1 if r[f"sev_{key}"] >= p85 else 0
            r[f"decimated_{key}"] = 1 if r[f"sev_{key}"] >= p95 else 0

    print("\n=== (a) per-metric EPA residual: decimated vs. not (two-sample t-test) ===")
    group_metrics: dict[str, list[str]] = {}
    for metric, units in v16.METRIC_UNITS.items():
        for u in units:
            group_metrics.setdefault(u, []).append(metric)
    for side, metrics in SIDE_METRICS.items():
        group_metrics[side] = metrics

    for key in GROUPS + list(SIDES):
        for metric in group_metrics.get(key, []):
            resid = np.array([r[f"{metric}_actual"] - r[f"{metric}_pregame"] for r in team_rows])
            for tag in ("short", "decimated"):
                flag = np.array([r[f"{tag}_{key}"] for r in team_rows], dtype=bool)
                if flag.sum() < 15:
                    continue
                a, b = resid[flag], resid[~flag]
                t, p = ttest_ind(a, b, equal_var=False)
                print(f"  [{metric:22s}] {tag:9s} {key:10s} n={flag.sum():4d}  "
                      f"mean(decimated)={a.mean():+.4f}  mean(normal)={b.mean():+.4f}  "
                      f"diff={a.mean()-b.mean():+.4f}  t={t:+.2f}  p={p:.4f}")

    print("\n=== (b) game outcome: does a decimated DUMMY add to delta? ===")
    rows_all = v14.build_dataset()
    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows_all])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS
    for i, r in enumerate(rows_all):
        x = X_all[i]
        r["delta"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))
    games = [r for r in rows_all if r["season"] >= FIRST_SNAP_SEASON]

    sev_by_team_week = {(r["team"], r["season"], r["week"]): r for r in team_rows}
    y = np.array([g["home_win"] for g in games], dtype=float)
    delta = np.array([g["delta"] for g in games]).reshape(-1, 1)

    w_r, se_r, ll_r = fit_logistic(delta, y)
    print(f"\nreduced model (delta only): loglik={ll_r:.2f}")

    for key in GROUPS + list(SIDES):
        for tag in ("short", "decimated"):
            diffs = []
            for g in games:
                hr = sev_by_team_week.get((g["home"], g["season"], g["week"]))
                ar = sev_by_team_week.get((g["away"], g["season"], g["week"]))
                hv = hr[f"{tag}_{key}"] if hr else 0
                av = ar[f"{tag}_{key}"] if ar else 0
                diffs.append(hv - av)
            dcol = np.array(diffs, dtype=float).reshape(-1, 1)
            if np.std(dcol) == 0:
                continue
            X_full = np.hstack([delta, dcol])
            w_f, se_f, ll_f = fit_logistic(X_full, y)
            lr = 2 * (ll_f - ll_r)
            p = 1 - chi2.cdf(lr, df=1)
            coef, se = w_f[2], se_f[2]
            flag = "  <-- QB (positive control)" if key == "qb" else ""
            print(f"  {tag:9s} {key:10s} coef={coef:+.4f} (se {se:.4f})  LR={lr:6.2f}  p={p:.4f}{flag}")


if __name__ == "__main__":
    main()
