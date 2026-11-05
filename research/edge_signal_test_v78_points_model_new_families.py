"""v78: research model R (team-points target, v77 structure) -- add pressure, pass-game, turnover-quality and injury families.

Baseline B = v77's P-A at ridge lambda 300: each team's points from its own offense (rush/pass EPA, giveaways), the opponent's defense
(rush/pass EPA allowed, takeaways), pre-snap penalties, Elo difference and home/away. Points are only the target (never an input).

Families, each added to B on its own with the same offense-vs-defense structure (a team's offense-role stat enters its own row, the
OPPONENT's defense-role stat enters as a separate coefficient):
  F1 sacks / pressure     off: sack rate allowed, QB-hit rate allowed         def: sack rate forced, QB-hit rate forced
  F2 pass-game shape      off: completion rate, air yards/att, YAC/cmp        def: the same allowed
  F3 turnover quality     off: interception rate, fumble rate (all fumbles)   def: interceptions / fumbles forced rates
                          (recovery luck is dropped by construction: fumbles, not fumbles LOST)
  F4 injuries (points-valued, known pre-game from the nflverse injury report): own and opponent QB gap (EPA gap; FantasyPros gap where
                          available, EPA gap otherwise, exactly production's switch) and skill-position EPA out
All four stack as F1-F4; "F1-F4 + 17" also adds the v52 candidates. Ratings are production-style pre-game EWMAs (alpha 0.08, bias-corrected,
8-game prior); the injury values are raw pre-game numbers. Walk-forward by season from 2011, win probability = Phi(predicted margin /
training sigma), judged on the win metrics.

Pre-registered rule (v57+), per family vs B: AUC and log loss both better in BOTH halves (<=2018 / >2018), paired-bootstrap
P(dLL <= 0) <= 0.01, better log loss in >= 60% of seasons. Four single families plus two stacks: six looks (F4a / F4b, QB-only and skill-only, were added after the first run to locate the F4 gain and are exploratory). Also reported: points MAE/RMSE,
and a persistence screen of every new stat (split-half reliability) so a family that fails can be traced to unrepeatable inputs.

Run: PYTHONPATH=. python research/edge_signal_test_v78_points_model_new_families.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import nflhub.sources.team_ratings as tr  # noqa: E402
import edge_signal_test_v53_walkforward_extra_terms as v53  # noqa: E402
import edge_signal_test_v75_research_model_no_points_sweep as v75  # noqa: E402
import edge_signal_test_v77_team_points_target as v77  # noqa: E402
import r_features  # noqa: E402
from edge_signal_test_v52_extra_terms_screen import build_team_games  # noqa: E402
from edge_signal_test_v55_market_free_candidate_sweep import CANDS  # noqa: E402
from persistence import persistence_report  # noqa: E402

LAM = 300.0
FAM = {
    "F1 sacks/pressure": (["sack_allowed_c", "hit_allowed_c"], ["sack_forced_c", "hit_forced_c"]),
    "F2 pass-game shape": (["comp_c", "adot_c", "yac_c"], ["comp_allowed_c", "adot_allowed_c", "yac_allowed_c"]),
    "F3 turnover quality": (["int_c", "fumble_c"], ["int_forced_c", "fumble_forced_c"]),
}
INJ = ["qb_epa_gap", "qb_fp_gap", "skill_out"]       # raw pre-game values: own in this row, opponent's via the _o columns


def build_frame() -> pd.DataFrame:
    tg = build_team_games()
    tg["sec_per_play_fast_c"] = -tg["sec_per_play_c"]
    f = r_features.build()
    tg = tg.merge(f.drop(columns=["season"]), on=["game_id", "team"], how="left")

    def opp(col: str) -> pd.Series:
        return tg.groupby("game_id")[col].transform("sum") - tg[col]

    # offense-role (own), higher = better
    tg["sack_allowed_c"], tg["hit_allowed_c"] = -tg["sack_rate"], -tg["hit_rate"]
    tg["comp_c"], tg["adot_c"], tg["yac_c"] = tg["comp_rate"], tg["air_yards"], tg["yac"]
    tg["int_c"], tg["fumble_c"] = -tg["int_rate"], -tg["fumble_rate"]
    # defense-role: the opponent offense's value (sign so that higher = better for the defense)
    tg["sack_forced_c"], tg["hit_forced_c"] = opp("sack_rate"), opp("hit_rate")
    tg["comp_allowed_c"], tg["adot_allowed_c"], tg["yac_allowed_c"] = -opp("comp_rate"), -opp("air_yards"), -opp("yac")
    tg["int_forced_c"], tg["fumble_forced_c"] = opp("int_rate"), opp("fumble_rate")

    # injuries: production's own helpers (known before kickoff)
    import csv, io, requests
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)
    qb_gap, *_ = tr.qb_quality_gap_data(played, current)
    fp_gap, _t, _l, _b, fp_avail = tr.qb_fantasy_gap_data(played, current, 1)
    skill = tr.skill_epa_out_by_team_week(current)
    key = list(zip(tg["team"], tg["season"], tg["week"]))
    fp_ok = np.array([fp_avail.get((s, w), False) for _t, s, w in key])
    epa = np.array([qb_gap.get(k, 0.0) for k in key])
    fpg = np.array([fp_gap.get(k, 0.0) for k in key])
    tg["qb_epa_gap"] = np.where(fp_ok, 0.0, epa)           # production's switch: FantasyPros gap where available, EPA gap otherwise
    tg["qb_fp_gap"] = np.where(fp_ok, fpg, 0.0)
    tg["skill_out"] = np.array([skill.get(k, 0.0) for k in key])
    return tg


def team_rows(tg: pd.DataFrame, smooth_cols: list[str], raw_cols: list[str]) -> pd.DataFrame:
    pre = v75.pregame_prod(tg, smooth_cols)
    base = pd.concat([tg[["game_id", "season", "week", "team", "is_home"]], pre, tg[raw_cols]], axis=1)
    pts = tg[["game_id", "is_home", "points_off"]].rename(columns={"points_off": "pts"})
    base = base.merge(pts, on=["game_id", "is_home"])
    o = base.drop(columns=["season", "week", "team", "pts"]).copy()
    o["is_home"] = 1 - o["is_home"]
    return base.merge(o, on=["game_id", "is_home"], suffixes=("", "_o"))


def main() -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    v53.L2 = v77.L2_WIN
    d = v75.build_diffs()
    d = d[d.margin != 0].reset_index(drop=True)
    base_win, _ = v53.walk_forward(d, v75.R0)
    elo_diff = d.set_index("game_id")["elo_b"]
    tg = build_frame()

    new_cols = sorted({c for off, de in FAM.values() for c in off + de})
    print("=== persistence of the new stats (split-half / reliability / year-over-year), offense-role stats ===")
    rep = persistence_report(tg[["team", "season", "week"] + new_cols], new_cols)
    print(rep[["n", "splithalf_r", "reliability", "yoy_r", "ewma_r"]].round(3).to_string())
    print()

    smooth = sorted(set(v77.OWN_OFF + v77.OPP_DEF + v77.DIFFS + new_cols + CANDS))
    m = team_rows(tg, smooth, INJ)
    m = m[m.game_id.isin(elo_diff.index)].reset_index(drop=True)

    def design(fams: list[str], extra_cands: list[str], inj):
        X, cols = v77.features(m, extra_cands, elo_diff)
        for f in fams:
            off, de = FAM[f]
            for c in off:
                X["own_" + c] = m[c]
            for c in de:
                X["opp_" + c] = m[c + "_o"]
        if inj:
            for c in (INJ if inj is True else inj):
                X["own_" + c] = m[c]
                X["opp_" + c] = m[c + "_o"]
        ok = X.notna().all(axis=1)
        return m[ok].reset_index(drop=True), X[ok].reset_index(drop=True)

    def run(fams, cands, inj):
        mm, X = design(fams, cands, inj)
        res, rp = v77.walk_points(mm, X, list(X.columns), LAM)
        res = res[res.game_id.isin(base_win.game_id)]
        b = base_win[base_win.game_id.isin(res.game_id)].reset_index(drop=True)
        res = res.set_index("game_id").loc[b.game_id].reset_index()
        return b, res, rp

    rng = np.random.default_rng(78)
    b0, r0, rp0 = run([], [], False)
    s0 = v53.summarize(r0)
    mae0 = (rp0.pts - rp0.pred).abs().mean()
    print(f"B (v77 P-A, points target, lambda {LAM:.0f}): AUC={s0['auc']:.4f} LL={s0['ll']:.4f} acc={s0['acc']:.4f} MAE={mae0:.3f} ({len(r0)} games)")
    print(f"R0 win/loss logistic on the same games: AUC={v53.summarize(b0)['auc']:.4f} LL={v53.summarize(b0)['ll']:.4f}\n")

    configs = [(name, [name], [], False) for name in FAM] + [
        ("F4 injuries", [], [], True),
        ("F4a QB gaps only", [], [], ["qb_epa_gap", "qb_fp_gap"]),
        ("F4b skill-out only", [], [], ["skill_out"]),
        ("F1-F4", list(FAM), [], True),
        ("F1-F4 + 17 candidates", list(FAM), list(CANDS), True),
    ]
    rows = []
    for name, fams, cands, inj in configs:
        b, r, rp = run(fams, cands, inj)
        c = v53.compare("x", r0, r, rng)                       # candidate vs the points-target baseline B (same games)
        s_b = r0.groupby("season").apply(lambda g: v53.summarize(g)["ll"], include_groups=False)
        s_r = r.groupby("season").apply(lambda g: v53.summarize(g)["ll"], include_groups=False)
        c["seasons"] = float(np.mean(s_r < s_b))
        rep_ok = all(c[f"dAUC_{h}"] > 0 and c[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
        sa = v53.summarize(r)
        rows.append({"candidate": name, "AUC": sa["auc"], "LL": sa["ll"], "acc": sa["acc"], "dAUC": c["dAUC_all"], "dLL": c["dLL_all"],
                     "dAUC_sel": c["dAUC_select"], "dAUC_con": c["dAUC_confirm"], "dLL_sel": c["dLL_select"], "dLL_con": c["dLL_confirm"],
                     "p": c["p_ll_le0"], "seasons": c["seasons"], "MAE": (rp.pts - rp.pred).abs().mean(),
                     "verdict": "ADOPT" if rep_ok and c["p_ll_le0"] <= 0.01 and c["seasons"] >= 0.6 else "DO NOT ADOPT"})
    out = pd.DataFrame(rows).set_index("candidate")
    print(out.to_string(float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
