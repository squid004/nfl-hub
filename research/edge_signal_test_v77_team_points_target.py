"""v77: predict each team's POINTS in a game (continuous target, two rows per game) instead of win/loss, in research model R.

User idea (2026-11-05): use everything we know about a team (offense, defense, special teams, turnovers, injuries) to predict the points each
team scores; that doubles the rows for testing, still yields a clear favorite per matchup, and expresses every component in points.

Structure (batter-vs-pitcher): for the row of team T against opponent O,
    pts_T = b0 + [T's offense: rush EPA, pass EPA, turnover giveaways] + [O's defense: rush EPA allowed, pass EPA allowed, takeaways]
            + pre-snap penalties + (Elo_T - Elo_O) + home/away sign + (T - O) differences of the extra candidate stats
so offense and defense enter with their own coefficients. Ratings are R's production-style pre-game EWMAs (alpha 0.08, bias-corrected,
8-game season prior); points scored/allowed are only ever the TARGET, never an input (Elo is the same Elo the production composite uses).
Ridge regression, walk-forward by season from 2011. Predicted margin = predicted home points - predicted away points; P(home win) =
Phi(margin / sigma) with sigma = the training residual sd of the margin (one number, refit each season).

Candidates (fixed before running), each against baseline R0 = the v75 logistic on the same game set:
    P-A   the R0 stat set as above (same information as R0, only the target/structure changes)
    P-B   P-A + the 17 extra candidate stats of the v52 sweep (explosive plays, penalties, tempo/drive quality, special teams, field position)
at ridge lambda in {30, 300} on standardized features (4 candidates, reported with that multiplicity).

Pre-registered rule (v57+): vs R0, AUC and log loss of the WIN probability both better in BOTH halves (<=2018 / >2018), paired-bootstrap
P(dLL <= 0) <= 0.01, better log loss in >= 60% of seasons. Also reported (not part of the rule): points accuracy per team-game (MAE, RMSE)
against a league-mean baseline, and the implied margin's correlation with the actual margin. Injuries (QB, skill) are not in this first
version; they enter later as points-valued adjustments.

Run: PYTHONPATH=. python research/edge_signal_test_v77_team_points_target.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
import edge_signal_test_v53_walkforward_extra_terms as v53  # noqa: E402
import edge_signal_test_v75_research_model_no_points_sweep as v75  # noqa: E402
from edge_signal_test_v52_extra_terms_screen import build_team_games  # noqa: E402
from edge_signal_test_v55_market_free_candidate_sweep import CANDS  # noqa: E402

L2_WIN = 0.03
LAMBDAS = (30.0, 300.0)
TEST_START, SELECT_END = 2011, 2018
OWN_OFF = ["rush_off_epa_b", "pass_off_epa_b", "turnovers_off_b"]      # T's offense (higher = better)
OPP_DEF = ["rush_def_epa_b", "pass_def_epa_b", "turnovers_def_b"]      # O's defense (higher = better)
DIFFS = ["pen_presnap_c"]


def team_rows(extra: list[str]) -> pd.DataFrame:
    tg = build_team_games()
    tg["sec_per_play_fast_c"] = -tg["sec_per_play_c"]
    cols = sorted(set(OWN_OFF + OPP_DEF + DIFFS + extra))
    pre = v75.pregame_prod(tg, cols)
    base = pd.concat([tg[["game_id", "season", "week", "team", "is_home"]], pre], axis=1)
    pts = tg[["game_id", "is_home", "points_off"]].rename(columns={"points_off": "pts"})
    base = base.merge(pts, on=["game_id", "is_home"])
    opp = base.drop(columns=["season", "week", "team", "pts"]).copy()
    opp["is_home"] = 1 - opp["is_home"]
    m = base.merge(opp, on=["game_id", "is_home"], suffixes=("", "_o"))
    return m


def features(m: pd.DataFrame, extra: list[str], elo_diff: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    X = pd.DataFrame(index=m.index)
    for c in OWN_OFF:
        X["own_" + c] = m[c]
    for c in OPP_DEF:
        X["opp_" + c] = m[c + "_o"]
    for c in DIFFS + extra:
        X["diff_" + c] = m[c] - m[c + "_o"]
    sign = np.where(m["is_home"] == 1, 1.0, -1.0)
    X["elo_diff"] = elo_diff.reindex(m["game_id"]).to_numpy() * sign
    X["hfa"] = sign
    return X, list(X.columns)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    A = np.c_[np.ones(len(X)), X]
    P = np.eye(A.shape[1]) * lam
    P[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + P, A.T @ y)


def walk_points(m: pd.DataFrame, X: pd.DataFrame, cols: list[str], lam: float):
    out, rows_pred = [], []
    d = m.assign(_i=np.arange(len(m)))
    for s in sorted(m.season.unique()):
        if s < TEST_START:
            continue
        tri, tei = (m.season < s).to_numpy(), (m.season == s).to_numpy()
        Xtr, Xte = X.loc[tri, cols].to_numpy(float), X.loc[tei, cols].to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        w = ridge_fit((Xtr - mu) / sd, m.loc[tri, "pts"].to_numpy(float), lam)
        ptr = np.c_[np.ones(tri.sum()), (Xtr - mu) / sd] @ w
        pte = np.c_[np.ones(tei.sum()), (Xte - mu) / sd] @ w
        # sigma of the margin from training residuals: pair the home and away rows of each game
        tr = m.loc[tri, ["game_id", "is_home", "pts"]].assign(pred=ptr)
        h, a = tr[tr.is_home == 1].set_index("game_id"), tr[tr.is_home == 0].set_index("game_id")
        j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
        sig = float(((j.pts_h - j.pts_a) - (j.pred_h - j.pred_a)).std())
        te = m.loc[tei, ["game_id", "is_home", "pts", "season"]].assign(pred=pte)
        rows_pred.append(te)
        h, a = te[te.is_home == 1].set_index("game_id"), te[te.is_home == 0].set_index("game_id")
        j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
        margin = j.pred_h - j.pred_a
        p = norm.cdf(margin / sig)
        out.append(pd.DataFrame({"game_id": j.index, "season": s, "p": np.clip(np.asarray(p), 1e-4, 1 - 1e-4),
                                 "y": (j.pts_h > j.pts_a).astype(int).to_numpy(), "margin_pred": margin.to_numpy(),
                                 "margin": (j.pts_h - j.pts_a).to_numpy()}))
    return pd.concat(out, ignore_index=True), pd.concat(rows_pred, ignore_index=True)


def main() -> None:
    pd.set_option("display.width", 250)
    v53.L2 = L2_WIN
    d = v75.build_diffs()
    d = d[d.margin != 0].reset_index(drop=True)
    base_win, _ = v53.walk_forward(d, v75.R0)
    sb = v53.summarize(base_win)
    print(f"R0 (logistic on win/loss): AUC={sb['auc']:.4f} logloss={sb['ll']:.4f} acc={sb['acc']:.4f}  ({len(base_win)} games)\n")

    elo_diff = d.set_index("game_id")["elo_b"]
    rng = np.random.default_rng(77)
    results = {}
    for label, extra in (("P-A  R0 stat set", []), ("P-B  + 17 candidates", list(CANDS))):
        m = team_rows(extra)
        m = m[m.game_id.isin(elo_diff.index)].reset_index(drop=True)
        X, cols = features(m, extra, elo_diff)
        ok = X.notna().all(axis=1)
        m, X = m[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
        for lam in LAMBDAS:
            res, rp = walk_points(m, X, cols, lam)
            res = res[res.game_id.isin(base_win.game_id)]
            b = base_win[base_win.game_id.isin(res.game_id)].reset_index(drop=True)
            res = res.set_index("game_id").loc[b.game_id].reset_index()
            r = v53.compare("x", b, res, rng)
            sa = b.groupby("season").apply(lambda g: v53.summarize(g)["ll"], include_groups=False)
            sc = res.groupby("season").apply(lambda g: v53.summarize(g)["ll"], include_groups=False)
            r["ll_season_win"] = float(np.mean(sc < sa))
            rep = all(r[f"dAUC_{h}"] > 0 and r[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
            r["verdict"] = "ADOPT" if rep and r["p_ll_le0"] <= 0.01 and r["ll_season_win"] >= 0.6 else "DO NOT ADOPT"
            s_alt = v53.summarize(res)
            mae = (rp.pts - rp.pred).abs().mean()
            rmse = float(np.sqrt(((rp.pts - rp.pred) ** 2).mean()))
            rows = rp.merge(m[["game_id", "is_home"]].assign(_k=1), on=["game_id", "is_home"], how="left")
            league = m.loc[m.season < TEST_START, "pts"].mean()
            mae0 = (rp.pts - league).abs().mean()
            corr = float(np.corrcoef(res.margin_pred, res.margin)[0, 1])
            results[f"{label} lam={lam:.0f}"] = dict(AUC=s_alt["auc"], LL=s_alt["ll"], acc=s_alt["acc"], dAUC=r["dAUC_all"], dLL=r["dLL_all"],
                                                  dAUC_sel=r["dAUC_select"], dAUC_con=r["dAUC_confirm"], dLL_sel=r["dLL_select"],
                                                  dLL_con=r["dLL_confirm"], p=r["p_ll_le0"], seasons=r["ll_season_win"],
                                                  MAE=mae, MAE_league_mean=mae0, RMSE=rmse, margin_r=corr, verdict=r["verdict"])
    out = pd.DataFrame(results).T
    print(out.to_string(float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
