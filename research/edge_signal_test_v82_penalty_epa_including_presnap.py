"""v82: research model R's rush/pass EPA with ALL penalties folded in, pre-snap ones included (owner direction 2026-11-05).

v81 counted live-ball penalty plays (pass interference, holding, roughing...) in rush/pass EPA and kept pre-snap penalties (false start, delay of
game, offside, encroachment, neutral zone) as separate stats. Here each pre-snap penalty's EPA is also folded into rush or pass EPA: it goes to the
type of the next scrimmage play by that offense in the same game (the replayed down), else the game's more common type (r_penalty_features.py).
Both sides of the ball: a defense's version is the opponent offense's value, so a defense's offside or pass interference counts against its
rush/pass EPA allowed and a false start by the offense against its own rush/pass EPA.

Baseline B = v77's P-A at lambda 300 (points target, R0 stats, offense-vs-defense roles). Variants, fixed before running:
  V1  live-ball penalties folded in (v81's swap, re-run here for a like-for-like comparison)
  V4  ALL penalties folded in, pre-snap included                                  <- R's adopted definition
Pre-registered rule (v57+) vs B as before; the decision to adopt V4 as R's definition is the owner's (principle), so the numbers are reported
either way, together with the persistence of the rush/pass stats under each definition.

Run: PYTHONPATH=. python research/edge_signal_test_v82_penalty_epa_including_presnap.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import edge_signal_test_v53_walkforward_extra_terms as v53  # noqa: E402
import edge_signal_test_v75_research_model_no_points_sweep as v75  # noqa: E402
import edge_signal_test_v77_team_points_target as v77  # noqa: E402
import edge_signal_test_v78_points_model_new_families as v78  # noqa: E402
import r_penalty_features  # noqa: E402
from edge_signal_test_v52_extra_terms_screen import build_team_games  # noqa: E402
from persistence import persistence_report  # noqa: E402

LAM = 300.0
SWAP_OUT = ["own_rush_off_epa_b", "own_pass_off_epa_b", "opp_rush_def_epa_b", "opp_pass_def_epa_b"]
SETS = {  # variant -> (rush offense col, pass offense col, rush defense col, pass defense col)
    "V1": ("rush_off_live_c", "pass_off_live_c", "rush_def_live_c", "pass_def_live_c"),
    "V4": ("rush_off_all_c", "pass_off_all_c", "rush_def_all_c", "pass_def_all_c"),
}


def build_tg() -> pd.DataFrame:
    tg = build_team_games()
    f = r_penalty_features.build()
    tg = tg.merge(f.drop(columns=["season"]), on=["game_id", "team"], how="left")

    def opp(col: str) -> pd.Series:
        return tg.groupby("game_id")[col].transform("sum") - tg[col]

    tg["rush_off_live_c"], tg["pass_off_live_c"] = tg["rush_epa_pen"], tg["pass_epa_pen"]
    tg["rush_def_live_c"], tg["pass_def_live_c"] = -opp("rush_epa_pen"), -opp("pass_epa_pen")
    tg["rush_off_all_c"], tg["pass_off_all_c"] = tg["rush_epa_allpen"], tg["pass_epa_allpen"]
    tg["rush_def_all_c"], tg["pass_def_all_c"] = -opp("rush_epa_allpen"), -opp("pass_epa_allpen")
    return tg


def main() -> None:
    pd.set_option("display.width", 250)
    v53.L2 = v77.L2_WIN
    d = v75.build_diffs()
    d = d[d.margin != 0].reset_index(drop=True)
    base_win, _ = v53.walk_forward(d, v75.R0)
    elo_diff = d.set_index("game_id")["elo_b"]
    tg = build_tg()

    cols = ["rush_off_epa_b", "rush_off_live_c", "rush_off_all_c", "pass_off_epa_b", "pass_off_live_c", "pass_off_all_c",
            "rush_def_epa_b", "rush_def_live_c", "rush_def_all_c", "pass_def_epa_b", "pass_def_live_c", "pass_def_all_c"]
    print("=== persistence of rush / pass EPA: original, live-ball penalties in, all penalties in ===")
    print(persistence_report(tg[["team", "season", "week"] + cols], cols)[["n", "splithalf_r", "reliability", "yoy_r", "ewma_r"]].round(3).to_string())
    print()

    smooth = sorted(set(v77.OWN_OFF + v77.OPP_DEF + v77.DIFFS + [c for v in SETS.values() for c in v]))
    m = v78.team_rows(tg, smooth, [])
    m = m[m.game_id.isin(elo_diff.index)].reset_index(drop=True)

    def run(variant: str):
        X, _ = v77.features(m, [], elo_diff)
        if variant in SETS:
            ro, po, rd, pd_ = SETS[variant]
            X = X.drop(columns=SWAP_OUT)
            X["own_" + ro], X["own_" + po] = m[ro], m[po]
            X["opp_" + rd], X["opp_" + pd_] = m[rd + "_o"], m[pd_ + "_o"]
        ok = X.notna().all(axis=1)
        mm, XX = m[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
        res, rp = v77.walk_points(mm, XX, list(XX.columns), LAM)
        res = res[res.game_id.isin(base_win.game_id)]
        b = base_win[base_win.game_id.isin(res.game_id)].reset_index(drop=True)
        return res.set_index("game_id").loc[b.game_id].reset_index(), rp

    rng = np.random.default_rng(82)
    r0, rp0 = run("B")
    s0 = v53.summarize(r0)
    print(f"B (points target, R0 stats): AUC={s0['auc']:.4f} LL={s0['ll']:.4f} acc={s0['acc']:.4f} MAE={(rp0.pts - rp0.pred).abs().mean():.3f} ({len(r0)} games)\n")
    rows = []
    for name, v in (("V1 live-ball penalties in", "V1"), ("V4 all penalties in (R's definition)", "V4")):
        r, rp = run(v)
        c = v53.compare("x", r0, r, rng)
        sb = r0.groupby("season").apply(lambda g: v53.summarize(g)["ll"], include_groups=False)
        sa = r.groupby("season").apply(lambda g: v53.summarize(g)["ll"], include_groups=False)
        c["seasons"] = float(np.mean(sa < sb))
        rep = all(c[f"dAUC_{h}"] > 0 and c[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
        s = v53.summarize(r)
        rows.append({"variant": name, "AUC": s["auc"], "LL": s["ll"], "acc": s["acc"], "dAUC": c["dAUC_all"], "dLL": c["dLL_all"],
                     "dAUC_sel": c["dAUC_select"], "dAUC_con": c["dAUC_confirm"], "dLL_sel": c["dLL_select"], "dLL_con": c["dLL_confirm"],
                     "p": c["p_ll_le0"], "seasons": c["seasons"], "MAE": (rp.pts - rp.pred).abs().mean(),
                     "verdict": "ADOPT" if rep and c["p_ll_le0"] <= 0.01 and c["seasons"] >= 0.6 else "DO NOT ADOPT"})
    print(pd.DataFrame(rows).set_index("variant").to_string(float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
