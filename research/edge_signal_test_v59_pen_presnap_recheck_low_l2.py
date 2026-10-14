"""v59: does pre-snap penalty yards (pen_presnap_yds) still earn its place at the new L2=0.03?

pen_presnap_yds shipped at L2=0.3 (v54), where the composite was over-regularized. v57 lowered L2 to
0.03 and a quick 9-vs-8 check there gave dAUC +0.0012 / dLL +0.0008 / bootstrap p=0.018, i.e. the
evidence got weaker. This script re-judges it properly on the production recipe (compute_ratings
game_log, walk-forward from 2011), at the shipped L2 and its neighbors.

Pre-registered rule (same as the v55 SHIP rule, fixed before running), at L2=0.03:
  KEEP if dAUC > 0 AND dLogLoss > 0 in BOTH chronological halves (select <=2018, confirm >2018),
  paired-bootstrap P(dLogLoss<=0) <= 0.01, pen weight > 0 in >= 90% of folds, and log loss better
  in >= 60% of test seasons. Otherwise: if every criterion except the bootstrap p passes and
  p <= 0.05, it is WATCH (kept, flagged); anything else is DROP.

Extra looks (not part of the rule): era blocks, per-season dLogLoss, L2 sensitivity, and a
leave-one-stat-out check (does pen_presnap survive when each other stat is removed, i.e. is it
substituting for something).

Run: PYTHONPATH=. python research/edge_signal_test_v59_pen_presnap_recheck_low_l2.py
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
    M8, M9, N_BOOT, SELECT_MAX, metrics, walk_forward,
)
from edge_signal_test_v57_l2_grid_extension import compare  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

ERAS = {"2011-15": (2011, 2015), "2016-20": (2016, 2020), "2021-26": (2021, 2026)}
L2S = (0.01, 0.03, 0.1, 0.3)
CHOSEN = 0.03


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
    X = {m: np.array([r["diffs"][m] for r in log]) for m in M9}
    X9 = np.column_stack([X[m] for m in M9])
    X8 = np.column_stack([X[m] for m in M8])
    rng = np.random.default_rng(59)
    print(f"{len(log)} decided games {seasons.min()}-{seasons.max()}\n")

    print("=== 9 stats vs 8 stats by L2 (+ve = pen_presnap helps) ===")
    print(f"{'l2':>6} | dAUC all  sel/conf        | dLL all  sel/conf          | P(dLL<=0) season-win pen>0 folds")
    res = {}
    for l2 in L2S:
        p8, yy, ss, _ = walk_forward(seasons, X8, y, l2)
        p9, _, _, w9 = walk_forward(seasons, X9, y, l2)
        res[l2] = (p8, p9, yy, ss, w9)
        c = compare(yy, ss, p8, p9, rng)
        pen_pos = float(np.mean(w9[:, M9.index("pen_presnap_yds")] > 0))
        print(f"{l2:6.3f} | {c['dAUC_all']:+.4f} {c['dAUC_select']:+.4f}/{c['dAUC_confirm']:+.4f} | "
              f"{c['dLL_all']:+.4f} {c['dLL_select']:+.4f}/{c['dLL_confirm']:+.4f} | "
              f"{c['p_ll_le0']:.3f}     {c['ll_season_win']:.2f}        {pen_pos:.2f}")

    p8, p9, yy, ss, w9 = res[CHOSEN]
    c = compare(yy, ss, p8, p9, rng)
    halves = all(c[f"dAUC_{h}"] > 0 and c[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
    pen_pos = float(np.mean(w9[:, M9.index("pen_presnap_yds")] > 0))
    others = halves and pen_pos >= 0.9 and c["ll_season_win"] >= 0.6
    verdict = "KEEP" if others and c["p_ll_le0"] <= 0.01 else "WATCH" if others and c["p_ll_le0"] <= 0.05 else "DROP"
    print(f"\nRULE at L2={CHOSEN}: halves={halves} pen>0 folds={pen_pos:.2f} season-win={c['ll_season_win']:.2f} "
          f"bootstrap p={c['p_ll_le0']:.4f}  -> {verdict}")

    print("\n=== era blocks at the chosen L2 (+ve = pen_presnap helps) ===")
    for lab, (lo, hi) in ERAS.items():
        m = (ss >= lo) & (ss <= hi)
        a8, l8 = metrics(p8[m], yy[m])
        a9, l9 = metrics(p9[m], yy[m])
        print(f"  {lab}: n={m.sum()} dAUC {a9 - a8:+.4f}  dLL {l8 - l9:+.4f}")

    print("\n=== per-season dLogLoss at the chosen L2 (+ve = pen_presnap helps) ===")
    d = {s: log_loss(yy[ss == s], p8[ss == s]) - log_loss(yy[ss == s], p9[ss == s]) for s in sorted(set(ss))}
    print("  " + "  ".join(f"{s}:{v:+.4f}" for s, v in d.items()))
    print(f"  median {np.median(list(d.values())):+.4f}; largest single-season gain {max(d.values()):+.4f} "
          f"({max(d, key=d.get)}); total without that season {np.sum([v for k, v in d.items() if k != max(d, key=d.get)]):+.4f} over {len(d)-1} seasons")

    print("\n=== leave-one-stat-out: pen_presnap's gain with each other stat removed (chosen L2) ===")
    print(f"{'removed':24s} dAUC     dLL      P(dLL<=0)")
    for drop in [None] + [m for m in M8 if m not in ("turnovers_def_forced",)]:
        keep = [m for m in M8 if m != drop]
        Xa = np.column_stack([X[m] for m in keep])
        Xb = np.column_stack([X[m] for m in keep + ["pen_presnap_yds"]])
        pa, y2, s2, _ = walk_forward(seasons, Xa, y, CHOSEN)
        pb, _, _, _ = walk_forward(seasons, Xb, y, CHOSEN)
        cc = compare(y2, s2, pa, pb, rng)
        print(f"{str(drop):24s} {cc['dAUC_all']:+.4f}  {cc['dLL_all']:+.4f}  {cc['p_ll_le0']:.3f}")

    print("\n=== weight stability across folds at the chosen L2 ===")
    i = M9.index("pen_presnap_yds")
    print(f"  pen_presnap_yds weight: mean {w9[:, i].mean():.4f}  min {w9[:, i].min():.4f}  max {w9[:, i].max():.4f}  "
          f"(folds with weight >= 0.03: {(w9[:, i] >= 0.03).sum()}/{len(w9)})")


if __name__ == "__main__":
    main()
