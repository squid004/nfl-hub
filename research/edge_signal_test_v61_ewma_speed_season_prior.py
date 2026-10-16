"""v61: tune the rating update speed (EWMA_ALPHA) and the season-prior fade (SEASON_PRIOR_GAMES).

Question: EWMA_ALPHA = 0.2 was "reused convention, not independently tuned". A cowork proxy test on
points-only ratings (docs/cowork-tests/ewma_alpha_sweep.py) found alpha 0.08-0.12 beats 0.2 by about
+0.013 AUC. That proxy used a 0.65 regression-to-mean carryover, not production's linear 5-game
fade, and only 2 of the 9 stats. This re-tests it on the exact production object: the full 9-stat
composite from compute_ratings' per-game oriented rating diffs, POWER_WEIGHTS refit (non-negative,
L2 = 0.03, the production recipe) inside a walk-forward loop for EVERY grid cell.

Data: nflverse games + play-by-play via nflhub.sources.team_ratings (2007+). Walk-forward test
seasons 2011+ (v54's TEST_START); each test season is predicted from weights fit on earlier seasons.
Market-free: the closing spread is never an input.

Grid: EWMA_ALPHA in {0.06, 0.08, 0.10, 0.12, 0.15, 0.20} x SEASON_PRIOR_GAMES in {3, 5, 8, 12}.
Production = (0.20, 5).

Pre-registered rule (fixed before running, v57's ADOPT rule): adopt a candidate over production only
if, on the same 9 stats and same L2, dAUC > 0 AND dLogLoss > 0 in BOTH chronological halves (<=2018
and >2018 test seasons), paired-bootstrap P(dLogLoss <= 0) <= 0.01 (N_BOOT = 2000), and log loss
better in >= 60% of test seasons. The candidate is chosen on the <=2018 half only (best select-half
AUC; ties within 0.0005 go to the cell nearest production), then confirmed on >2018.

AMENDMENT (made after a first run of the 24-cell classic grid, kept in output/v61_run_classic_only.txt):
that run found slower alphas MONOTONICALLY worse (alpha 0.06: AUC 0.626 vs production 0.679), the
opposite of the proxy, while a longer prior fade at alpha 0.2 (prior 8) won. Cause hypothesis:
production seeds each season's EWMA with game 1's raw value, so a slow alpha stays anchored to game
1 (weight (1-alpha)^n) -- an estimator artifact, not evidence that fast is right. The grid is
therefore extended with a second estimator, EWMA_BIAS_CORRECT (geometric-weighted average of the
season's games so far), 48 cells total, same rule, same selection procedure. The decision rule was
not loosened; only the candidate set was widened, so the p <= 0.01 bar matters even more.

Run: PYTHONPATH=. python research/edge_signal_test_v61_ewma_speed_season_prior.py
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
    M9, N_BOOT, SELECT_MAX, metrics, walk_forward,
)
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

L2 = 0.03
ALPHAS = (0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
PRIORS = (3, 5, 8, 12)
BIAS = (False, True)
PROD = (0.20, 5, False)


def build(played, current, alpha, prior_games, bias):
    tr.EWMA_BIAS_CORRECT = bias
    tr.EWMA_ALPHA = alpha
    tr.SEASON_PRIOR_GAMES = prior_games
    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    log = [r for r in log if r["home_score"] != r["away_score"]]
    seasons = np.array([r["season"] for r in log])
    y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
    X = np.column_stack([[r["diffs"][m] for r in log] for m in M9])
    return seasons, y, X


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

    res = {}
    for a in ALPHAS:
        for pg in PRIORS:
            for b in BIAS:
                seasons, y, X = build(played, current, a, pg, b)
                p, yy, ss, w = walk_forward(seasons, X, y, L2)
                res[(a, pg, b)] = (p, yy, ss, w)
                print(f"  built alpha={a} prior={pg} bias_correct={b}: {len(y)} games", flush=True)
    base = res[PROD]
    yy, ss = base[1], base[2]
    for k, v in res.items():
        assert np.array_equal(v[1], yy), "game sets differ across cells"
    print(f"\n{len(yy)} test games, seasons {ss.min()}-{ss.max()}; production = alpha {PROD[0]}, prior {PROD[1]}, bias_correct {PROD[2]}\n")

    sel, con = ss <= SELECT_MAX, ss > SELECT_MAX
    print("=== walk-forward AUC / log loss by cell (all | select <=2018 | confirm >2018) ===")
    print(f"{'alpha':>6} {'prior':>5} {'bc':>5} | {'AUC':>7} {'LL':>7} | {'AUCsel':>7} {'LLsel':>7} | {'AUCcon':>7} {'LLcon':>7}")
    for (a, pg, b), (p, _, _, _) in res.items():
        print(f"{a:6.2f} {pg:5d} {str(b):>5} | {roc_auc_score(yy, p):7.4f} {log_loss(yy, p):7.4f} | "
              f"{roc_auc_score(yy[sel], p[sel]):7.4f} {log_loss(yy[sel], p[sel]):7.4f} | "
              f"{roc_auc_score(yy[con], p[con]):7.4f} {log_loss(yy[con], p[con]):7.4f}")

    # choose on the select half only
    sel_auc = {k: roc_auc_score(yy[sel], v[0][sel]) for k, v in res.items()}
    best = max(sel_auc.values())
    near = [k for k, v in sel_auc.items() if v >= best - 0.0005]
    chosen = min(near, key=lambda k: abs(k[0] - PROD[0]) / PROD[0] + abs(k[1] - PROD[1]) / PROD[1] + (k[2] != PROD[2]))
    print(f"\nbest select-half AUC {best:.4f}; candidates within 0.0005: {near}; chosen (nearest production) = {chosen}")

    rng = np.random.default_rng(61)
    c = compare(yy, ss, base[0], res[chosen][0], rng)
    rep = all(c[f"dAUC_{h}"] > 0 and c[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
    ok = rep and c["p_ll_le0"] <= 0.01 and c["ll_season_win"] >= 0.6
    print(f"chosen {chosen} vs production {PROD}:")
    print(f"  dAUC all {c['dAUC_all']:+.4f} (select {c['dAUC_select']:+.4f}, confirm {c['dAUC_confirm']:+.4f})")
    print(f"  dLL  all {c['dLL_all']:+.4f} (select {c['dLL_select']:+.4f}, confirm {c['dLL_confirm']:+.4f})")
    print(f"  P(dLL<=0)={c['p_ll_le0']:.3f}; season-win={c['ll_season_win']:.2f}; replicated both halves={rep}  -> {'ADOPT' if ok else 'DO NOT ADOPT'}")

    print("\n=== every cell vs production under the same rule (information only; selection is the one above) ===")
    for k, v in res.items():
        if k == PROD:
            continue
        cc = compare(yy, ss, base[0], v[0], rng)
        r2 = all(cc[f"dAUC_{h}"] > 0 and cc[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
        ok2 = r2 and cc["p_ll_le0"] <= 0.01 and cc["ll_season_win"] >= 0.6
        print(f"  {k}: dAUC {cc['dAUC_select']:+.4f}/{cc['dAUC_confirm']:+.4f}  dLL {cc['dLL_select']:+.4f}/{cc['dLL_confirm']:+.4f}  "
              f"p={cc['p_ll_le0']:.3f} win={cc['ll_season_win']:.2f}  {'ADOPT' if ok2 else '-'}")

    print("\n=== production vs chosen: per-season log loss (test seasons) ===")
    for s in sorted(set(ss)):
        m = ss == s
        print(f"  {s}: prod {log_loss(yy[m], base[0][m]):.4f}  chosen {log_loss(yy[m], res[chosen][0][m]):.4f}")

    print("\n=== final all-data weights at chosen cell (L2 = 0.03, standardized-diff) ===")
    seasons, y, X = build(played, current, *chosen)
    from edge_signal_test_v54_pen_presnap_production_refit import fit_nonneg
    mu, sd = X.mean(0), X.std(0)
    w = fit_nonneg((X - mu) / sd, y, L2)
    print("  " + ", ".join(f"{m}={v:.4f}" for m, v in zip(M9, w[1:])) + f"  sum={w[1:].sum():.4f}")
    d = ((X - mu) / sd) @ w[1:]
    print(f"  std(delta_raw)={d.std():.4f}")


if __name__ == "__main__":
    main()
