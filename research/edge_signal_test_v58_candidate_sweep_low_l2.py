"""v58: re-run the v55 one-at-a-time candidate sweep with the composite at L2=0.03.

v57 showed the production composite (L2=0.3) is over-regularized: L2=0.03 improves walk-forward
AUC +0.0024 / log loss +0.0066, replicated in both halves. v55 judged every candidate at L2=0.3,
where an over-shrunk baseline can make a candidate look better (drive_score_off) or worse than it
is. Same pre-registered SHIP/WATCH rule as v55, same candidates, same 9-stat baseline -- only the
regularization strength changes. Baseline and candidates are always fit at the same L2.

Run: PYTHONPATH=. python research/edge_signal_test_v58_candidate_sweep_low_l2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import edge_signal_test_v53_walkforward_extra_terms as v53  # noqa: E402
from edge_signal_test_v55_market_free_candidate_sweep import BASE9, CANDS, build_diffs, sweep  # noqa: E402

L2 = 0.03


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    v53.L2 = L2
    d = build_diffs()
    rng = np.random.default_rng(58)
    base, _ = v53.walk_forward(d, BASE9)
    sb = v53.summarize(base)
    print(f"BASELINE (9 stats, L2={L2}): AUC={sb['auc']:.4f} logloss={sb['ll']:.4f}\n")
    res = sweep(d, BASE9, CANDS, rng)
    show = ["dAUC_all", "dAUC_select", "dAUC_confirm", "dLL_all", "dLL_select", "dLL_confirm",
            "p_ll_le0", "w_mean", "w_pos_frac", "ll_season_win", "auc_season_win", "tier"]
    print(res[show].to_string(float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else v))
    res.to_csv(os.path.join(os.path.dirname(__file__), "output", "v58_round1.csv"))
    print(f"\n{len(CANDS)} looks; P(at least one false SHIP) <= {1 - 0.99 ** len(CANDS):.2f} before the replication requirement")


if __name__ == "__main__":
    main()
