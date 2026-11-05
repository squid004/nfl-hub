"""v83: expected points per drive (EPD) as an experimental research metric (model R). Research only: production and the EPA stats are untouched.

Owner direction (2026-11-05): instead of weighting rush and pass plays, look at whole-drive value for each offense and defense; throw out garbage-time
DRIVES (any play snapped with a win probability under 5% or over 95% discards the drive); keep drives that begin with 3:00 or less left in the half
out of the main set and track them as their own "3-minute EPD". r_drive_features.py defines the metric (drive EPD = the summed EPA of the drive's plays,
penalties and the closing punt or field goal included).

What this script does:
  1. persistence screen of EPD (offense and defense, regular and 3-minute) next to the rush/pass EPA stats it might replace
  2. first predictive test in the v77 team-points model (baseline B = R0 stats, lambda 300), four variants fixed before running:
       E1 add    regular EPD (own offense, opponent defense) to B
       E2 add    regular + 3-minute EPD to B
       E3 swap   replace the four rush/pass EPA stats by regular EPD (offense, defense): drive-level instead of play-type
       E4 swap+  E3 plus 3-minute EPD
Pre-registered rule (v57+) vs B: AUC and log loss of the win probability both better in BOTH halves (<=2018 / >2018), paired-bootstrap P(dLL <= 0) <= 0.01,
better log loss in >= 60% of seasons; four looks. Everything is reported either way: this is an experimental metric being tracked.

Run: PYTHONPATH=. python research/edge_signal_test_v83_expected_points_per_drive.py
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
import r_drive_features  # noqa: E402
from edge_signal_test_v52_extra_terms_screen import build_team_games  # noqa: E402
from persistence import persistence_report  # noqa: E402

LAM = 300.0
SWAP_OUT = ["own_rush_off_epa_b", "own_pass_off_epa_b", "opp_rush_def_epa_b", "opp_pass_def_epa_b"]
OFF = ["epd_off_c", "epd3_off_c"]
DEF = ["epd_def_c", "epd3_def_c"]


def build_tg() -> pd.DataFrame:
    tg = build_team_games()
    f = r_drive_features.build()
    tg = tg.merge(f.drop(columns=["season"]), on=["game_id", "team"], how="left")

    def opp(col: str) -> pd.Series:      # the other team's value in the same game, NaN-safe (every game has exactly two rows)
        return tg.groupby("game_id")[col].transform(lambda s: s.iloc[::-1].to_numpy())

    tg["epd_off_c"], tg["epd3_off_c"] = tg["epd_reg"], tg["epd_3min"]
    tg["epd_def_c"], tg["epd3_def_c"] = -opp("epd_reg"), -opp("epd_3min")
    return tg


def main() -> None:
    pd.set_option("display.width", 250)
    v53.L2 = v77.L2_WIN
    d = v75.build_diffs()
    d = d[d.margin != 0].reset_index(drop=True)
    base_win, _ = v53.walk_forward(d, v75.R0)
    elo_diff = d.set_index("game_id")["elo_b"]
    tg = build_tg()

    print(f"drives per team-game (eligible after the garbage-time rule): regular {tg.n_reg.mean():.1f}, 3-minute {tg.n_3min.mean():.2f}; "
          f"team-games with a 3-minute value: {tg.epd_3min.notna().mean():.0%}\n")
    cols = ["rush_off_epa_b", "pass_off_epa_b", "epd_off_c", "epd3_off_c", "rush_def_epa_b", "pass_def_epa_b", "epd_def_c", "epd3_def_c"]
    print("=== persistence: rush/pass EPA vs drive EPD (regular and 3-minute), offense and defense roles ===")
    print(persistence_report(tg[["team", "season", "week"] + cols], cols)[["n", "splithalf_r", "reliability", "yoy_r", "ewma_r"]].round(3).to_string())
    print()

    smooth = sorted(set(v77.OWN_OFF + v77.OPP_DEF + v77.DIFFS + OFF + DEF))
    m = v78.team_rows(tg, smooth, [])
    m = m[m.game_id.isin(elo_diff.index)].reset_index(drop=True)

    def run(variant: str):
        X, _ = v77.features(m, [], elo_diff)
        if variant in ("E3", "E4"):
            X = X.drop(columns=SWAP_OUT)
        use3 = variant in ("E2", "E4")
        X["own_epd_off_c"], X["opp_epd_def_c"] = m["epd_off_c"], m["epd_def_c_o"]
        if use3:
            X["own_epd3_off_c"], X["opp_epd3_def_c"] = m["epd3_off_c"], m["epd3_def_c_o"]
        if variant == "B":
            X = X.drop(columns=[c for c in X.columns if "epd" in c])
        # a 3-minute drive value is missing for most team-games' histories early on: treat missing (no such drive yet) as 0 = league-typical
        for c in [c for c in X.columns if "epd3" in c]:
            X[c] = X[c].fillna(0.0)
        ok = X.notna().all(axis=1)
        mm, XX = m[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
        res, rp = v77.walk_points(mm, XX, list(XX.columns), LAM)
        res = res[res.game_id.isin(base_win.game_id)]
        b = base_win[base_win.game_id.isin(res.game_id)].reset_index(drop=True)
        return res.set_index("game_id").loc[b.game_id].reset_index(), rp

    rng = np.random.default_rng(83)
    r0, rp0 = run("B")
    s0 = v53.summarize(r0)
    print(f"B (points target, R0 stats): AUC={s0['auc']:.4f} LL={s0['ll']:.4f} acc={s0['acc']:.4f} MAE={(rp0.pts - rp0.pred).abs().mean():.3f} ({len(r0)} games)\n")
    rows = []
    for name, v in (("E1 add regular EPD", "E1"), ("E2 add regular + 3-minute EPD", "E2"), ("E3 swap rush/pass EPA -> EPD", "E3"),
                    ("E4 swap + 3-minute EPD", "E4")):
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
