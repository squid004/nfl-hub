"""v81: put penalties into EPA (research model R, team-points target).

Owner direction (2026-11-05): a spot foul such as defensive pass interference is, for football purposes, the same outcome as a completed pass, so
penalties should be reflected in EPA. Production's four EPA stats drop every play nflverse labels `no_play`, which is most penalty plays (DPI,
false start, offensive and defensive holding, roughing...). r_penalty_features.py rebuilds the stats with those plays included: a penalty play
whose play was a run (pass) counts toward rush (pass) EPA, pre-snap penalties are separate stats, and every penalty's EPA is also available as a
committed / drawn pair.

Baseline B = v77's P-A at ridge lambda 300 (R0 stat set, team-points target, offense-vs-defense roles). Variants, fixed before running:
  V1 swap       replace the four EPA stats (own rush/pass offense, opponent rush/pass defense) by the penalty-inclusive versions
  V2 add        keep B and add the penalty EPA pairs as extra stats (own: live-ball committed / drawn, pre-snap committed / drawn; opponent defense
                role: the same four from the defense's side)
  V3 swap+pre   V1 plus the four pre-snap EPA stats (live-ball penalties already live inside the swapped rush/pass EPA)
Pre-registered rule (v57+): vs B, AUC and log loss of the win probability both better in BOTH halves (<=2018 / >2018), paired-bootstrap
P(dLL <= 0) <= 0.01, better log loss in >= 60% of seasons. Three looks. Also reported: points MAE, and a persistence screen of the new stats
beside the old EPA stats (split-half reliability), since a penalty-inclusive stat that repeats less cannot help.

Run: PYTHONPATH=. python research/edge_signal_test_v81_penalty_inclusive_epa.py
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
OFF = {"rush": "rush_off_pen_c", "pass": "pass_off_pen_c"}
DEF = {"rush": "rush_def_pen_c", "pass": "pass_def_pen_c"}
PEN_OFF = ["live_comm_c", "live_drawn_c", "pre_comm_c", "pre_drawn_c"]
PEN_DEF = ["def_live_comm_c", "def_live_drawn_c", "def_pre_comm_c", "def_pre_drawn_c"]


def build_tg() -> pd.DataFrame:
    tg = build_team_games()
    f = r_penalty_features.build()
    tg = tg.merge(f.drop(columns=["season"]), on=["game_id", "team"], how="left")

    def opp(col: str) -> pd.Series:
        return tg.groupby("game_id")[col].transform("sum") - tg[col]

    tg["rush_off_pen_c"], tg["pass_off_pen_c"] = tg["rush_epa_pen"], tg["pass_epa_pen"]
    tg["rush_def_pen_c"], tg["pass_def_pen_c"] = -opp("rush_epa_pen"), -opp("pass_epa_pen")
    tg["live_comm_c"], tg["live_drawn_c"] = tg["live_committed"], tg["live_drawn"]
    tg["pre_comm_c"], tg["pre_drawn_c"] = tg["presnap_committed"], tg["presnap_drawn"]
    tg["def_live_comm_c"], tg["def_live_drawn_c"] = -opp("live_drawn"), -opp("live_committed")
    tg["def_pre_comm_c"], tg["def_pre_drawn_c"] = -opp("presnap_drawn"), -opp("presnap_committed")
    return tg


def main() -> None:
    pd.set_option("display.width", 250)
    v53.L2 = v77.L2_WIN
    d = v75.build_diffs()
    d = d[d.margin != 0].reset_index(drop=True)
    base_win, _ = v53.walk_forward(d, v75.R0)
    elo_diff = d.set_index("game_id")["elo_b"]
    tg = build_tg()

    print("=== persistence: original vs penalty-inclusive EPA (offense-role stats) ===")
    cols = ["rush_off_epa_b", "rush_off_pen_c", "pass_off_epa_b", "pass_off_pen_c", "rush_def_epa_b", "rush_def_pen_c", "pass_def_epa_b", "pass_def_pen_c",
            "live_comm_c", "live_drawn_c", "pre_comm_c", "pre_drawn_c"]
    print(persistence_report(tg[["team", "season", "week"] + cols], cols)[["n", "splithalf_r", "reliability", "yoy_r", "ewma_r"]].round(3).to_string())
    print()

    smooth = sorted(set(v77.OWN_OFF + v77.OPP_DEF + v77.DIFFS + list(OFF.values()) + list(DEF.values()) + PEN_OFF + PEN_DEF))
    m = v78.team_rows(tg, smooth, [])
    m = m[m.game_id.isin(elo_diff.index)].reset_index(drop=True)

    def design(variant: str):
        X, cols_ = v77.features(m, [], elo_diff)
        if variant in ("V1", "V3"):
            X = X.drop(columns=SWAP_OUT)
            for s in ("rush", "pass"):
                X["own_" + OFF[s]] = m[OFF[s]]
                X["opp_" + DEF[s]] = m[DEF[s] + "_o"]
        if variant in ("V2", "V3"):
            own = PEN_OFF if variant == "V2" else PEN_OFF[2:]
            opp_ = PEN_DEF if variant == "V2" else PEN_DEF[2:]
            for c in own:
                X["own_" + c] = m[c]
            for c in opp_:
                X["opp_" + c] = m[c + "_o"]
        ok = X.notna().all(axis=1)
        return m[ok].reset_index(drop=True), X[ok].reset_index(drop=True)

    def run(variant: str):
        mm, X = design(variant)
        res, rp = v77.walk_points(mm, X, list(X.columns), LAM)
        res = res[res.game_id.isin(base_win.game_id)]
        b = base_win[base_win.game_id.isin(res.game_id)].reset_index(drop=True)
        return res.set_index("game_id").loc[b.game_id].reset_index(), rp

    rng = np.random.default_rng(81)
    r0, rp0 = run("B")
    s0 = v53.summarize(r0)
    print(f"B (points target, R0 stats): AUC={s0['auc']:.4f} LL={s0['ll']:.4f} acc={s0['acc']:.4f} MAE={(rp0.pts - rp0.pred).abs().mean():.3f} ({len(r0)} games)\n")
    rows = []
    for name, v in (("V1 swap in penalty-inclusive EPA", "V1"), ("V2 add penalty EPA pairs", "V2"), ("V3 swap + pre-snap EPA", "V3")):
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
