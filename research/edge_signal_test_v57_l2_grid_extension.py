"""v57: extend the composite's L2 grid below 0.3 using the PRODUCTION recipe.

POWER_WEIGHTS was fit with L2=0.3, chosen by walk-forward AUC over (0.3, 1, 3, ..., 300) in v8.
0.3 was the lowest value on that grid, so the optimum may sit off the edge. v55/v56 hinted at it:
in the research framework the baseline improved at L2=0.01. This script checks it on the exact
object production fits -- compute_ratings' per-game oriented rating diffs -- for the 9-stat model.

Pre-registered rule (fixed before running): adopt a lower L2 only if, versus L2=0.3 on the same
9 stats, dAUC > 0 and dLogLoss > 0 in BOTH chronological halves (select <=2018, confirm >2018),
paired-bootstrap P(dLogLoss<=0) <= 0.01, and log loss better in >= 60% of test seasons.
The chosen L2 is the largest value within 0.0005 AUC of the best (prefer the more regularized).

Run: PYTHONPATH=. python research/edge_signal_test_v57_l2_grid_extension.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
import nflhub.sources.team_ratings as tr  # noqa: E402
from edge_signal_test_v54_pen_presnap_production_refit import (  # noqa: E402
    M8, M9, N_BOOT, SELECT_MAX, fit_logit, fit_nonneg, metrics, walk_forward,
)
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

GRID = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
RIDGE_FLOOR = 1e-6  # l2=0 is unregularized; keep it numerically defined


def compare(yy, ss, p_base, p_alt, rng):
    out = {}
    for lab, mask in (("all", np.ones(len(yy), bool)), ("select", ss <= SELECT_MAX), ("confirm", ss > SELECT_MAX)):
        out[f"dAUC_{lab}"] = roc_auc_score(yy[mask], p_alt[mask]) - roc_auc_score(yy[mask], p_base[mask])
        out[f"dLL_{lab}"] = log_loss(yy[mask], p_base[mask]) - log_loss(yy[mask], p_alt[mask])
    n = len(yy)
    dl = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        dl.append(log_loss(yy[ix], p_base[ix]) - log_loss(yy[ix], p_alt[ix]))
    out["p_ll_le0"] = float(np.mean(np.array(dl) <= 0))
    yrs = sorted(set(ss))
    out["ll_season_win"] = float(np.mean([log_loss(yy[ss == s], p_alt[ss == s]) < log_loss(yy[ss == s], p_base[ss == s]) for s in yrs]))
    return out


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)
    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    log = [r for r in log if r["home_score"] != r["away_score"]]
    seasons = np.array([r["season"] for r in log])
    y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
    X = {tag: np.column_stack([[r["diffs"][m] for r in log] for m in cols]) for tag, cols in (("8", M8), ("9", M9))}
    print(f"{len(log)} decided games {seasons.min()}-{seasons.max()} (production compute_ratings game_log)\n")

    rng = np.random.default_rng(57)
    res = {}
    print("=== walk-forward by L2 (test seasons 2011+) ===")
    print(f"{'l2':>7} | {'AUC8':>7} {'LL8':>7} | {'AUC9':>7} {'LL9':>7} {'acc9':>6} | dAUC9-vs-0.3 select/confirm | dLL9 select/confirm | P(dLL<=0) season-win")
    for l2 in GRID:
        p8, yy, ss, w8 = walk_forward(seasons, X["8"], y, max(l2, RIDGE_FLOOR))
        p9, _, _, w9 = walk_forward(seasons, X["9"], y, max(l2, RIDGE_FLOOR))
        res[l2] = (p8, p9, yy, ss, w9)
    base = res[0.3][1]
    yy, ss = res[0.3][2], res[0.3][3]
    for l2 in GRID:
        p8, p9, _, _, _ = res[l2]
        a8, l8 = metrics(p8, yy)
        a9, l9 = metrics(p9, yy)
        c = compare(yy, ss, base, p9, rng)
        acc = np.mean((p9 > 0.5) == (yy == 1))
        print(f"{l2:7.4f} | {a8:7.4f} {l8:7.4f} | {a9:7.4f} {l9:7.4f} {acc:6.4f} | "
              f"{c['dAUC_select']:+.4f}/{c['dAUC_confirm']:+.4f} | {c['dLL_select']:+.4f}/{c['dLL_confirm']:+.4f} | "
              f"{c['p_ll_le0']:.3f} {c['ll_season_win']:.2f}")

    aucs = {l2: metrics(res[l2][1], yy)[0] for l2 in GRID}
    best = max(aucs.values())
    chosen = max(l2 for l2 in GRID if aucs[l2] >= best - 0.0005)
    print(f"\nbest AUC {best:.4f}; largest L2 within 0.0005 AUC = {chosen}")
    c = compare(yy, ss, base, res[chosen][1], rng)
    rep = all(c[f"dAUC_{h}"] > 0 and c[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
    ok = rep and c["p_ll_le0"] <= 0.01 and c["ll_season_win"] >= 0.6
    print(f"chosen L2={chosen} vs 0.3: dAUC {c['dAUC_all']:+.4f} dLL {c['dLL_all']:+.4f}; replicated both halves={rep}; "
          f"P(dLL<=0)={c['p_ll_le0']:.3f}; season-win={c['ll_season_win']:.2f}  -> {'ADOPT' if ok else 'DO NOT ADOPT'}")

    print("\n=== does pen_presnap_yds still help at the chosen L2? (9 vs 8 stats) ===")
    p8, p9 = res[chosen][0], res[chosen][1]
    c = compare(yy, ss, p8, p9, rng)
    print(f"  dAUC {c['dAUC_all']:+.4f} (select {c['dAUC_select']:+.4f}, confirm {c['dAUC_confirm']:+.4f}); "
          f"dLL {c['dLL_all']:+.4f} (select {c['dLL_select']:+.4f}, confirm {c['dLL_confirm']:+.4f}); "
          f"P(dLL<=0)={c['p_ll_le0']:.3f}; season-win={c['ll_season_win']:.2f}")

    print("\n=== per-fold weight stability at chosen L2 (mean / min / folds at zero of 15) ===")
    w9 = res[chosen][4]
    for i, m in enumerate(M9):
        print(f"  {m:22s} mean={w9[:, i].mean():.4f} min={w9[:, i].min():.4f} zero={(w9[:, i] == 0).sum()}/{len(w9)}")

    print("\n=== final all-data fits (standardized-diff weights) ===")
    mu, sd = X["9"].mean(0), X["9"].std(0)
    for l2 in sorted({chosen, 0.3}):
        w = fit_nonneg((X["9"] - mu) / sd, y, max(l2, RIDGE_FLOOR))
        d = ((X["9"] - mu) / sd) @ w[1:]
        print(f"  L2={l2}: " + ", ".join(f"{m}={v:.4f}" for m, v in zip(M9, w[1:])) + f"  sum={w[1:].sum():.4f} std(delta_raw)={d.std():.4f}")

    downstream(played, current, log, seasons, y, X["9"], chosen)


def downstream(played, current, log, seasons, y, X9, chosen):
    """Refit the joint delta + QB/fantasy/skill/HFA/streak constants and the delta calibration
    under the shipped (L2=0.3) vs candidate (chosen L2) composite, mirroring v54 sections 3-4."""
    mu, sd = X9.mean(0), X9.std(0)
    D = {}
    for tag, l2 in (("OLD(0.3)", 0.3), (f"NEW({chosen})", chosen)):
        w = fit_nonneg((X9 - mu) / sd, y, max(l2, RIDGE_FLOOR))
        D[tag] = ((X9 - mu) / sd) @ w[1:]
        print(f"  {tag} weights: " + ", ".join(f"{m}={round(float(v), 4)}" for m, v in zip(M9, w[1:])))
    qb_gap, *_ = tr.qb_quality_gap_data(played, current)
    fantasy_gap, _t, _l, _b, fantasy_available = tr.qb_fantasy_gap_data(played, current, 1)
    skill_out = tr.skill_epa_out_by_team_week(current)
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
    names = ["qb_gap", "fantasy_gap", "skill_out", "hfa_logit", "streak"]
    print("=== downstream joint fit (delta-equivalent constants) ===")
    fits = {}
    for tag, d in D.items():
        w, se = fit_logit(np.column_stack([d, qb, fa, sk, hfa, stk])[keep], y[keep])
        fits[tag] = {nm: w[i] / w[1] for i, nm in enumerate(names, start=2)}
        print(f"  {tag}: delta coef={w[1]:.4f}  " + "  ".join(f"{nm}={fits[tag][nm]:+.4f}(z{w[i]/se[i]:+.1f})" for i, nm in enumerate(names, start=2)))
    print("  shipped: QB_QUALITY_GAP -0.8818, QB_FANTASY_GAP -0.0429, SKILL -0.0363, HFA 0.9169, MOMENTUM 0.0262, calib 0.0484/1.4497")

    print("=== delta calibration on clean (non-backup-QB) games ===")
    use_fp = np.array([fantasy_available.get((r["season"], r["week"]), False) for r in log])
    qb_out = tr._qb_out_doubtful_by_week(current)
    clean = np.array([not (qb_out.get((r["home"], r["season"], r["week"]), False) or qb_out.get((r["away"], r["season"], r["week"]), False)) for r in log])
    for tag, d in D.items():
        rt = fits[tag]
        qbh = np.where(use_fp, rt["fantasy_gap"] * fa, rt["qb_gap"] * qb)
        dt = d + qbh + rt["skill_out"] * sk + rt["hfa_logit"] * hfa + rt["streak"] * stk
        fav = np.where(dt > 0, y, 1 - y)
        m = clean & (dt != 0)
        w, se = fit_logit(np.abs(dt[m])[:, None], fav[m])
        print(f"  {tag}: n={m.sum()} intercept={w[0]:.4f} slope={w[1]:.4f} fav hit={fav[m].mean():.4f} |delta| std={np.abs(dt[m]).std():.4f}")


if __name__ == "__main__":
    main()
