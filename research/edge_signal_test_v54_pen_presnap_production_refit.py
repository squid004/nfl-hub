"""v54: production refit for the 9th power-model stat, pre-snap penalty yards (pen_presnap_yds).

v52/v53 screened pre-snap penalties (persistent, replicated in both halves, stable positive
walk-forward weight, +0.0013 AUC / +0.0008 log-loss over the 8-stat baseline, p_ll=0.002) and the
call was GO on the market-free bar: production never uses the closing spread, so a replicated
positive signal in the market-free model is a true improvement even though the market already
prices it. This script does the production work:

  1. Walks the REAL nflhub.sources.team_ratings.compute_ratings (now 9 stats) to get each game's
     oriented pre-game rating diffs -- the exact object POWER_WEIGHTS is applied to.
  2. Walk-forward (train < season, test season, 2011+) 8-stat vs 9-stat non-negative L2 logistic,
     L2 sweep, select (<=2018) / confirm (>2018) halves, paired bootstrap.
  3. Final all-data fit at the best L2 -> new POWER_WEIGHTS.
  4. Downstream: re-run the joint delta + QB/fantasy/skill/HFA/streak fit under the OLD delta_raw
     (sanity: must reproduce the shipped constants roughly) and the NEW one, so the
     delta-equivalent constants can be updated consistently; then refit the delta calibration.

Run: PYTHONPATH=. python research/edge_signal_test_v54_pen_presnap_production_refit.py
"""
from __future__ import annotations

import csv
import io
import sys

import numpy as np
import requests
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

import nflhub.sources.team_ratings as tr

M8 = tuple(m for m in tr.RATING_METRICS if m != "pen_presnap_yds")
M9 = tr.RATING_METRICS
TEST_START = 2011
SELECT_MAX = 2018
N_BOOT = 2000
OLD_W = {"rush_off_epa": 0.0850, "pass_off_epa": 0.1536, "rush_def_epa_allowed": 0.0572,
         "pass_def_epa_allowed": 0.0578, "points_off": 0.1587, "points_def_allowed": 0.1006,
         "turnovers_off": 0.0217, "turnovers_def_forced": 0.0}


def fit_nonneg(X, y, l2):
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])

    def f(w):
        ph = 1 / (1 + np.exp(-(Xb @ w)))
        nll = -np.mean(y * np.log(ph + 1e-12) + (1 - y) * np.log(1 - ph + 1e-12)) + 0.5 * l2 * np.sum(w[1:] ** 2)
        g = Xb.T @ (ph - y) / n
        g[1:] += l2 * w[1:]
        return nll, g

    return minimize(f, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * p).x


def walk_forward(seasons, X, y, l2):
    """Returns (pred, y, season) over the test seasons, plus the per-fold weights."""
    P, Y, S, W = [], [], [], []
    for ts in sorted(set(seasons)):
        if ts < TEST_START:
            continue
        tri, tei = seasons < ts, seasons == ts
        mu, sd = X[tri].mean(0), X[tri].std(0)
        sd[sd == 0] = 1.0
        w = fit_nonneg((X[tri] - mu) / sd, y[tri], l2)
        P.append(1 / (1 + np.exp(-(w[0] + ((X[tei] - mu) / sd) @ w[1:]))))
        Y.append(y[tei])
        S.append(seasons[tei])
        W.append(w[1:])
    return np.concatenate(P), np.concatenate(Y), np.concatenate(S), np.array(W)


def metrics(p, y):
    return roc_auc_score(y, p), log_loss(y, p)


def fit_logit(X, y):
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])

    def f(w):
        ph = 1 / (1 + np.exp(-(Xb @ w)))
        return -np.sum(y * np.log(ph + 1e-12) + (1 - y) * np.log(1 - ph + 1e-12)), Xb.T @ (ph - y)

    w = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B").x
    ph = 1 / (1 + np.exp(-(Xb @ w)))
    se = np.sqrt(np.diag(np.linalg.inv((Xb * (ph * (1 - ph))[:, None]).T @ Xb)))
    return w, se


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    all_rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in all_rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current_season = max(int(r["season"]) for r in played)

    log: list[dict] = []
    tr.compute_ratings(played, current_season, game_log=log)
    log = [r for r in log if r["home_score"] != r["away_score"]]
    seasons = np.array([r["season"] for r in log])
    y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
    D = {m: np.array([r["diffs"][m] for r in log]) for m in M9}
    X8 = np.column_stack([D[m] for m in M8])
    X9 = np.column_stack([D[m] for m in M9])
    print(f"{len(log)} decided games {seasons.min()}-{seasons.max()} (production compute_ratings game_log)")

    print("\n=== 1. walk-forward 8 vs 9 stats, by L2 ===")
    print(f"{'l2':>6} {'AUC8':>8} {'AUC9':>8} {'dAUC':>8} {'LL8':>8} {'LL9':>8} {'dLL':>8}")
    best = None
    res = {}
    for l2 in (0.3, 1, 3, 10, 30, 100):
        p8, yy, ss, w8 = walk_forward(seasons, X8, y, l2)
        p9, _, _, w9 = walk_forward(seasons, X9, y, l2)
        a8, l8 = metrics(p8, yy)
        a9, l9 = metrics(p9, yy)
        res[l2] = (p8, p9, yy, ss, w9)
        print(f"{l2:6.1f} {a8:8.4f} {a9:8.4f} {a9-a8:+8.4f} {l8:8.4f} {l9:8.4f} {l8-l9:+8.4f}")
        if best is None or a9 > best[1]:
            best = (l2, a9)
    l2 = best[0]
    print(f"best 9-stat L2 = {l2} (walk-forward AUC {best[1]:.4f}); production 8-stat reference AUC 0.6746 at L2=0.3")

    p8, p9, yy, ss, w9 = res[l2]
    p8_0, p9_0, _, _, _ = res[0.3]
    for label, a, b, mask in (("all", p8, p9, np.ones(len(yy), bool)), ("select", p8, p9, ss <= SELECT_MAX),
                              ("confirm", p8, p9, ss > SELECT_MAX)):
        da = roc_auc_score(yy[mask], b[mask]) - roc_auc_score(yy[mask], a[mask])
        dl = log_loss(yy[mask], a[mask]) - log_loss(yy[mask], b[mask])
        print(f"  L2={l2} {label:8s} n={mask.sum():5d} dAUC={da:+.4f}  dLL(+=better)={dl:+.4f}")
    rng = np.random.default_rng(54)
    n = len(yy)
    da, dl = [], []
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        if yy[ix].min() == yy[ix].max():
            continue
        da.append(roc_auc_score(yy[ix], p9[ix]) - roc_auc_score(yy[ix], p8[ix]))
        dl.append(log_loss(yy[ix], p8[ix]) - log_loss(yy[ix], p9[ix]))
    da, dl = np.array(da), np.array(dl)
    print(f"  bootstrap 95% CI: dAUC [{np.percentile(da,2.5):+.5f}, {np.percentile(da,97.5):+.5f}] P(<=0)={np.mean(da<=0):.3f};"
          f" dLL [{np.percentile(dl,2.5):+.5f}, {np.percentile(dl,97.5):+.5f}] P(<=0)={np.mean(dl<=0):.3f}")
    print(f"  per-fold pen_presnap weight: mean={w9[:, -1].mean():.4f} min={w9[:, -1].min():.4f} folds at 0: {(w9[:, -1] == 0).sum()}/{len(w9)}")
    yrs = sorted(set(ss))
    wins = sum(roc_auc_score(yy[ss == s], p9[ss == s]) > roc_auc_score(yy[ss == s], p8[ss == s]) for s in yrs)
    wl = sum(log_loss(yy[ss == s], p9[ss == s]) < log_loss(yy[ss == s], p8[ss == s]) for s in yrs)
    print(f"  seasons where 9-stat beats 8-stat: AUC {wins}/{len(yrs)}, log-loss {wl}/{len(yrs)}")

    print(f"\n=== 2. final all-data non-negative fit, L2={l2} ===")
    out = {}
    for tag, cols, X in (("8-stat", M8, X8), ("9-stat", M9, X9)):
        mu, sd = X.mean(0), X.std(0)
        w = fit_nonneg((X - mu) / sd, y, l2)
        out[tag] = dict(zip(cols, w[1:]))
        print(f"  {tag}: " + ", ".join(f"{c}={v:.4f}" for c, v in out[tag].items()) + f"  (sum={w[1:].sum():.4f})")
    NEW_W = {m: round(float(v), 4) for m, v in out["9-stat"].items()}
    print("  NEW POWER_WEIGHTS =", NEW_W)

    def delta_raw(weights, cols):
        return sum(weights[m] * (D[m] - D[m].mean()) / D[m].std() for m in cols)

    d_old, d_new = delta_raw(OLD_W, M8), delta_raw(NEW_W, M9)
    print(f"\n  delta_raw old vs new: corr={np.corrcoef(d_old, d_new)[0,1]:.4f}  std old={d_old.std():.4f} new={d_new.std():.4f}"
          f"  slope(new on old)={np.polyfit(d_old, d_new, 1)[0]:.4f}")

    print("\n=== 3. downstream joint fit (v51 recipe + HFA + streak) under OLD vs NEW delta_raw ===")
    qb_gap, *_ = tr.qb_quality_gap_data(played, current_season)
    fantasy_gap, _t, _l, _b, fantasy_available = tr.qb_fantasy_gap_data(played, current_season, 1)
    skill_out = tr.skill_epa_out_by_team_week(current_season)
    hfa_by_season = tr.home_field_logit_by_season(played)
    streak = tr.team_streak_by_week(played)

    def g(d, r, side):
        return d.get((r[side], r["season"], r["week"]), 0.0)

    qb = np.array([g(qb_gap, r, "home") - g(qb_gap, r, "away") for r in log])
    fa = np.array([g(fantasy_gap, r, "home") - g(fantasy_gap, r, "away") for r in log])
    sk = np.array([g(skill_out, r, "home") - g(skill_out, r, "away") for r in log])
    hfa = np.array([0.0 if r.get("neutral") else hfa_by_season.get(r["season"], 0.0) for r in log])
    stk = np.array([g(streak, r, "home") - g(streak, r, "away") for r in log])
    keep = seasons >= 2009
    names = ["delta_raw", "qb_gap", "fantasy_gap", "skill_out", "hfa_logit", "streak"]
    fits = {}
    for tag, d in (("OLD", d_old), ("NEW", d_new)):
        Xj = np.column_stack([d, qb, fa, sk, hfa, stk])[keep]
        w, se = fit_logit(Xj, y[keep])
        fits[tag] = w
        print(f"  {tag} delta_raw: delta coef={w[1]:.4f}")
        for i, nm in enumerate(names[1:], start=2):
            print(f"    {nm:12s} coef={w[i]:+.4f} z={w[i]/se[i]:+.2f}  delta-equivalent={w[i]/w[1]:+.4f}")
    print("  shipped: QB_QUALITY_GAP -0.8818, QB_FANTASY_GAP -0.0429, SKILL -0.0363, HFA 0.9169, MOMENTUM 0.0262")
    ratio = {nm: fits["NEW"][i] / fits["NEW"][1] for i, nm in enumerate(names[1:], start=2)}
    ratio_old = {nm: fits["OLD"][i] / fits["OLD"][1] for i, nm in enumerate(names[1:], start=2)}
    print("  NEW/OLD delta-equivalent ratio:", {k: round(ratio[k] / ratio_old[k], 3) for k in ratio})

    print("\n=== 4. delta calibration (clean, non-backup-QB games), same recipe under OLD vs NEW composite ===")
    use_fp = np.array([fantasy_available.get((r["season"], r["week"]), False) for r in log])
    qb_out = tr._qb_out_doubtful_by_week(current_season)
    clean = np.array([not (qb_out.get((r["home"], r["season"], r["week"]), False) or qb_out.get((r["away"], r["season"], r["week"]), False))
                      for r in log])
    for tag, d in (("OLD", d_old), ("NEW", d_new)):
        rt = {nm: fits[tag][i] / fits[tag][1] for i, nm in enumerate(names[1:], start=2)}
        qbh = np.where(use_fp, rt["fantasy_gap"] * fa, rt["qb_gap"] * qb)
        delta_tot = d + qbh + rt["skill_out"] * sk + rt["hfa_logit"] * hfa + rt["streak"] * stk
        fav_wins = np.where(delta_tot > 0, y, 1 - y)
        m = clean & (delta_tot != 0)
        w, se = fit_logit(np.abs(delta_tot[m])[:, None], fav_wins[m])
        print(f"  {tag}: n={m.sum()}  intercept={w[0]:.4f}  slope={w[1]:.4f}  fav hit rate={fav_wins[m].mean():.4f}"
              f"  |delta| std={np.abs(delta_tot[m]).std():.4f}   (shipped: 0.0484 / 1.4497)")


if __name__ == "__main__":
    main()
