"""v62: per-group update speeds, then an L2 re-check, on top of v61's winner.

Question: v61 found a bias-corrected EWMA at alpha 0.08, season prior 8 games beats production
(alpha 0.2, classic seed, prior 5). EPA stats are noisier per game than points, so the stat groups
may want different speeds. Then v57 found the best L2 moves with the inputs, so re-check L2 at the
new speeds.

Groups: EPA (rush/pass offense EPA and the two defense-EPA-allowed stats), POINTS (points_off,
points_def_allowed), OTHER (turnovers, pre-snap penalty yards). Season prior fixed at v61's 8 games,
bias-corrected EWMA on. Method: coordinate sweep -- vary one group's alpha over ALPHAS holding the
others at v61's alpha, choose each group's alpha on the <=2018 select half (best select AUC, ties
within 0.0005 go to the alpha nearest v61's), combine the three picks. L2 grid {0.003, 0.01, 0.03,
0.1, 0.3} then runs on the combined configuration.

Pre-registered rule (fixed before running; identical to v57/v61): a candidate is adopted over the
baseline only if dAUC > 0 and dLogLoss > 0 in BOTH halves (<=2018 / >2018), paired-bootstrap
P(dLogLoss <= 0) <= 0.01 (N_BOOT = 2000), and log loss better in >= 60% of test seasons. Baseline for
the group-alpha test = v61's winner (one alpha for every stat); baseline for the L2 test = L2 0.03.
Walk-forward test seasons 2011+, market-free.

Run: PYTHONPATH=. python research/edge_signal_test_v62_ewma_group_alphas_l2.py
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
from edge_signal_test_v54_pen_presnap_production_refit import M9, SELECT_MAX, fit_nonneg, walk_forward  # noqa: E402
from edge_signal_test_v61_ewma_speed_season_prior import compare  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

BASE_ALPHA, PRIOR, L2_BASE = 0.08, 8, 0.03
ALPHAS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
L2_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)
GROUPS = {
    "epa": ("rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed"),
    "points": ("points_off", "points_def_allowed"),
    "other": ("turnovers_off", "turnovers_def_forced", "pen_presnap_yds"),
}


def build(played, current, alphas):
    tr.EWMA_BIAS_CORRECT = True
    tr.EWMA_ALPHA = BASE_ALPHA
    tr.SEASON_PRIOR_GAMES = PRIOR
    tr.EWMA_ALPHA_BY_METRIC = {m: alphas[g] for g, ms in GROUPS.items() for m in ms}
    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    log = [r for r in log if r["home_score"] != r["away_score"]]
    seasons = np.array([r["season"] for r in log])
    y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
    X = np.column_stack([[r["diffs"][m] for r in log] for m in M9])
    return seasons, y, X


def verdict(c):
    rep = all(c[f"dAUC_{h}"] > 0 and c[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
    return rep and c["p_ll_le0"] <= 0.01 and c["ll_season_win"] >= 0.6, rep


def fmt(c):
    return (f"dAUC {c['dAUC_all']:+.4f} (sel {c['dAUC_select']:+.4f}, con {c['dAUC_confirm']:+.4f}); "
            f"dLL {c['dLL_all']:+.4f} (sel {c['dLL_select']:+.4f}, con {c['dLL_confirm']:+.4f}); "
            f"P(dLL<=0)={c['p_ll_le0']:.3f}; season-win={c['ll_season_win']:.2f}")


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)
    rng = np.random.default_rng(62)

    def run(alphas, l2=L2_BASE):
        seasons, y, X = build(played, current, alphas)
        return walk_forward(seasons, X, y, l2)

    base_alphas = {g: BASE_ALPHA for g in GROUPS}
    pb, yy, ss, _ = run(base_alphas)
    sel = ss <= SELECT_MAX
    print(f"baseline (v61 winner: every stat alpha {BASE_ALPHA}, prior {PRIOR}, bias-corrected, L2 {L2_BASE}): "
          f"AUC {roc_auc_score(yy, pb):.4f} LL {log_loss(yy, pb):.4f}  ({len(yy)} test games)\n")

    picks = {}
    for g in GROUPS:
        print(f"=== group '{g}' alpha sweep (others at {BASE_ALPHA}) ===")
        sel_auc = {}
        for a in ALPHAS:
            al = dict(base_alphas)
            al[g] = a
            p, _, _, _ = run(al)
            sel_auc[a] = roc_auc_score(yy[sel], p[sel])
            con = ~sel
            print(f"  alpha {a:.2f}: AUC {roc_auc_score(yy, p):.4f} LL {log_loss(yy, p):.4f} | select AUC {sel_auc[a]:.4f} | confirm AUC {roc_auc_score(yy[con], p[con]):.4f}", flush=True)
        best = max(sel_auc.values())
        picks[g] = min((a for a in ALPHAS if sel_auc[a] >= best - 0.0005), key=lambda a: abs(a - BASE_ALPHA))
        print(f"  -> chosen {picks[g]}\n")

    pc, _, _, _ = run(picks)
    c = compare(yy, ss, pb, pc, rng)
    ok, rep = verdict(c)
    print(f"=== combined per-group alphas {picks} vs single-alpha baseline ===")
    print(f"  AUC {roc_auc_score(yy, pc):.4f} LL {log_loss(yy, pc):.4f}; {fmt(c)}; replicated={rep} -> {'ADOPT' if ok else 'DO NOT ADOPT'}\n")

    final_alphas = picks if ok else base_alphas
    print(f"=== L2 re-check at alphas {final_alphas} ===")
    seasons, y, X = build(played, current, final_alphas)
    res = {l2: walk_forward(seasons, X, y, l2) for l2 in L2_GRID}
    p_base = res[L2_BASE][0]
    for l2 in L2_GRID:
        p = res[l2][0]
        line = f"  L2 {l2:6.3f}: AUC {roc_auc_score(yy, p):.4f} LL {log_loss(yy, p):.4f}"
        if l2 != L2_BASE:
            cc = compare(yy, ss, p_base, p, rng)
            o, _ = verdict(cc)
            line += f" | vs 0.03: {fmt(cc)} -> {'ADOPT' if o else '-'}"
        print(line)

    print("\n=== final all-data weights at the L2 = 0.03 fit (standardized-diff) ===")
    mu, sd = X.mean(0), X.std(0)
    w = fit_nonneg((X - mu) / sd, y, L2_BASE)
    print("  " + ", ".join(f"{m}={v:.4f}" for m, v in zip(M9, w[1:])) + f"  sum={w[1:].sum():.4f}")
    print(f"  std(delta_raw)={(((X - mu) / sd) @ w[1:]).std():.4f}")


if __name__ == "__main__":
    main()
