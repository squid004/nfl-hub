"""v56: robustness follow-up on the two v55 leaders (drive_score_off_c, st_kp_c).

v55's pre-registered rule required dAUC > 0 AND dLogLoss > 0 in both chronological halves.
drive_score_off_c had by far the largest log-loss gain (+0.0027, bootstrap p=0.0005, 93% of seasons)
but missed the rule by dAUC_confirm = -0.0003. This script does NOT relax that rule; it asks whether
the signal is real or an artifact of one split, using looks not used in v55:
  (a) three era blocks (2011-15 / 2016-20 / 2021-25), each judged against the 9-stat baseline
  (b) drive_score_off added vs. SWAPPED in for points_off (they overlap; does it replace it?)
  (c) the two leaders stacked, with weights
  (d) per-season dLogLoss series, to see whether one season carries the result

Run: PYTHONPATH=. python research/edge_signal_test_v56_drive_score_stkp_robustness.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from edge_signal_test_v53_walkforward_extra_terms import auc, logloss, walk_forward, paired_boot, summarize  # noqa: E402
from edge_signal_test_v55_market_free_candidate_sweep import BASE9, build_diffs  # noqa: E402

ERAS = {"2011-15": (2011, 2015), "2016-20": (2016, 2020), "2021-25": (2021, 2025)}


def rows(name: str, base: pd.DataFrame, alt: pd.DataFrame) -> dict:
    out = {"model": name}
    for lab, (lo, hi) in ERAS.items():
        b = base[(base.season >= lo) & (base.season <= hi)]
        a = alt[(alt.season >= lo) & (alt.season <= hi)]
        out[f"dAUC {lab}"] = auc(a.y.to_numpy(float), a.p.to_numpy()) - auc(b.y.to_numpy(float), b.p.to_numpy())
        out[f"dLL {lab}"] = logloss(b.y.to_numpy(float), b.p.to_numpy()) - logloss(a.y.to_numpy(float), a.p.to_numpy())
    return out


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    d = build_diffs()
    rng = np.random.default_rng(56)
    base, _ = walk_forward(d, BASE9)
    print(f"{len(d)} games; baseline (9 stats) AUC={summarize(base)['auc']:.4f} LL={summarize(base)['ll']:.4f}\n")

    swap = [c for c in BASE9 if c != "points_off_b"]
    models = {
        "+drive_score_off": (BASE9 + ["drive_score_off_c"], BASE9),
        "+st_kp": (BASE9 + ["st_kp_c"], BASE9),
        "+both": (BASE9 + ["drive_score_off_c", "st_kp_c"], BASE9),
        "swap points_off->drive_score_off": (swap + ["drive_score_off_c"], BASE9),
    }
    res, tab = {}, []
    for name, (cols, ref) in models.items():
        alt, ws = walk_forward(d, cols)
        res[name] = (alt, ws, cols)
        tab.append(rows(name, base, alt))
    print("=== (a) era blocks vs 9-stat baseline (+ve = better) ===")
    print(pd.DataFrame(tab).set_index("model").to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\n=== (b)/(c) overall, bootstrap, mean weights ===")
    for name, (alt, ws, cols) in res.items():
        b = paired_boot(base, alt, rng)
        sa, sb = summarize(alt), summarize(base)
        w = pd.DataFrame(np.array(ws), columns=cols).mean()
        new = {c: round(float(w[c]), 3) for c in cols if c not in BASE9 or name.startswith("swap")}
        print(f"  {name:34s} dAUC={sa['auc']-sb['auc']:+.4f} dLL={sb['ll']-sa['ll']:+.4f}  P(dLL<=0)={b['p_ll_le0']:.4f} "
              f"P(dAUC<=0)={b['p_auc_le0']:.3f}  weights {new}")
    print("\n  mean walk-forward weights, '+both' model:")
    alt, ws, cols = res["+both"]
    print("   ", pd.DataFrame(np.array(ws), columns=cols).mean().round(3).to_dict())
    print("  mean walk-forward weights, swap model:")
    alt, ws, cols = res["swap points_off->drive_score_off"]
    print("   ", pd.DataFrame(np.array(ws), columns=cols).mean().round(3).to_dict())

    print("\n=== (d) per-season dLogLoss (+ve = better), 9-stat baseline -> +drive_score_off / +st_kp ===")
    ps = pd.DataFrame({
        n: res[n][0].groupby("season").apply(lambda g: logloss(g.y.to_numpy(float), g.p.to_numpy()), include_groups=False)
        for n in ("+drive_score_off", "+st_kp")})
    b0 = base.groupby("season").apply(lambda g: logloss(g.y.to_numpy(float), g.p.to_numpy()), include_groups=False)
    print((-ps.sub(b0, axis=0)).to_string(float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
