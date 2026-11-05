"""v74: opponent-adjusted EPA stats, and a composite without the points stats (user request 2026-10-29).

Two ideas, both aimed at a composite whose weights match football intuition:

 A. Opponent adjustment. Each game's EPA is discounted by how generous/tough the opponent had been BEFORE the game: a team's
    rush offense is reduced by K * (opponent's rush defense allowed - league average), its rush defense allowed by K *
    (opponent's rush offense - league average), the same for pass (team_ratings.OPP_ADJUST_K; pre-game ratings only, no
    lookahead). So a bad rush defense against a very strong rush offense counts for less. K = 0 is production.
 B. Drop the points stats. points_off and points_def_allowed (together about 33% of today's weight) are forced to weight 0,
    leaving rush/pass offense and defense EPA, the two turnover stats, pre-snap penalties and Elo free; Elo is unchanged.

Candidates, fixed before running (all vs. production = K 0, all 10 stats), with the weights refit walk-forward exactly as in
v57-v63 (non-negative, L2 = 0.03, seasons < S, test 2011-2026):
   C1 adjust K=1.0              C2 adjust K=0.5
   C3 drop points (K=0)         C4 adjust K=1.0 + drop points      C5 adjust K=0.5 + drop points

Pre-registered ADOPT rule (v57+): AUC and log loss both better in BOTH halves (<=2018 / >2018), paired-bootstrap
P(dLL <= 0) <= 0.01 (N_BOOT = 2000), better log loss in >= 60% of seasons. Five candidates share one baseline, so a passing
result is reported with that multiplicity. B changes WHAT the model says as much as how well it scores, so its numbers are
reported either way: the cost of dropping the points stats is shown next to the benefit, and a candidate that fails the rule
is a NO-GO for shipping on accuracy grounds even if it reads more intuitively.

Run: PYTHONPATH=. python research/edge_signal_test_v74_opponent_adjust_drop_points.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np
import requests
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import nflhub.sources.team_ratings as tr  # noqa: E402
from edge_signal_test_v54_pen_presnap_production_refit import fit_nonneg, walk_forward  # noqa: E402
from edge_signal_test_v61_ewma_speed_season_prior import compare  # noqa: E402
from edge_signal_test_v68_qb_weight_ats_residual import fmt, verdict  # noqa: E402

M = tr.POWER_METRICS
L2 = 0.03
DROP = ("points_off", "points_def_allowed")


def build(played, current, k):
    old = tr.OPP_ADJUST_K
    tr.OPP_ADJUST_K = k
    try:
        log: list[dict] = []
        tr.compute_ratings(played, current, game_log=log)
    finally:
        tr.OPP_ADJUST_K = old
    log = [r for r in log if r["home_score"] != r["away_score"]]
    seasons = np.array([r["season"] for r in log])
    y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
    X = np.column_stack([[r["diffs"][m] for r in log] for m in M])
    ids = [r["game_id"] for r in log]
    return seasons, y, X, ids


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)

    data = {}
    for k in (0.0, 0.5, 1.0):
        data[k] = build(played, current, k)
        print(f"  built K={k}: {len(data[k][1])} games", flush=True)
    for k in (0.5, 1.0):
        assert data[k][3] == data[0.0][3], "game sets differ across K"
    keep = [i for i, m in enumerate(M) if m not in DROP]

    variants = {
        "BASE (K0, 10 stats)": (0.0, list(range(len(M)))),
        "C1 adjust K=1.0": (1.0, list(range(len(M)))),
        "C2 adjust K=0.5": (0.5, list(range(len(M)))),
        "C3 drop points": (0.0, keep),
        "C4 adjust K=1.0 + drop points": (1.0, keep),
        "C5 adjust K=0.5 + drop points": (0.5, keep),
    }
    seasons, y = data[0.0][0], data[0.0][1]
    probs, wts = {}, {}
    for name, (k, cols) in variants.items():
        X = data[k][2][:, cols]
        p, yy, ss, w = walk_forward(seasons, X, y, L2)
        probs[name] = p
        mu, sd = X.mean(0), X.std(0)
        wts[name] = (cols, fit_nonneg((X - mu) / sd, y, L2)[1:])
    sel, con = ss <= 2018, ss > 2018
    print(f"\n{len(yy)} decided test games, seasons {ss.min()}-{ss.max()}\n")
    print(f"{'variant':32s} {'AUC':>7} {'LL':>7} {'acc':>7} | {'AUC<=18':>8} {'AUC>18':>7} | {'acc<=18':>8} {'acc>18':>7}")
    for name, p in probs.items():
        print(f"{name:32s} {roc_auc_score(yy, p):7.4f} {log_loss(yy, p):7.4f} {np.mean((p > .5) == (yy == 1)):7.4f} | "
              f"{roc_auc_score(yy[sel], p[sel]):8.4f} {roc_auc_score(yy[con], p[con]):7.4f} | "
              f"{np.mean(((p > .5) == (yy == 1))[sel]):8.4f} {np.mean(((p > .5) == (yy == 1))[con]):7.4f}")

    print("\n=== ADOPT test vs BASE ===")
    rng = np.random.default_rng(74)
    for name in list(variants)[1:]:
        c = compare(yy, ss, probs["BASE (K0, 10 stats)"], probs[name], rng)
        ok, rep = verdict(c)
        print(f"{name:32s}: {fmt(c)}; replicated both halves={rep} -> {'ADOPT' if ok else 'DO NOT ADOPT'}")

    print("\n=== weights fit on all data (standardized diffs; non-negative, L2 0.03) ===")
    head = [m[:14] for m in M]
    print(f"{'variant':32s} " + " ".join(f"{h:>14s}" for h in head))
    for name, (cols, w) in wts.items():
        full = np.zeros(len(M))
        full[cols] = w
        print(f"{name:32s} " + " ".join(f"{v:14.3f}" for v in full))


if __name__ == "__main__":
    main()
