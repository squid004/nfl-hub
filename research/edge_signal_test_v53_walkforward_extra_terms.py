"""v53: Stage 3 walk-forward for the extra-term candidates that survived the v52 screen.

Candidate sets were PRE-REGISTERED after v52 (before any walk-forward was run) so the winner
cannot be cherry-picked from 17 looks:
  ST1   st_kp_c                       (kickoff/punt special-teams EPA; best market-residual z in v52)
  ST2   st_kp_c + fp_net_c            (adds net starting field position)
  EXPL  expl_{rush,pass}_{off,def}_c  (explosive-play rates, all four jointly)
  PEN   pen_presnap_c                 (pre-snap penalty yards: false start / delay / offside)
  ALL   union of the above            (informational only, not a ship candidate)
Plus a totals test of sec_per_play_c (pace) against the market total.

Framework mirrors production (research v8): oriented home-minus-away pre-game EWMA diffs
(alpha 0.2, 5-game prior fade), z-scored on the training seasons, ONE joint non-negative
L2 (0.3) logistic, walk-forward by season from 2011. The baseline is the same 8 stats in the
same framework, so every delta is apples-to-apples. Chronological split: select = test
seasons 2011-2018, confirm = 2019-2025.

Run: PYTHONPATH=. python research/edge_signal_test_v53_walkforward_extra_terms.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(__file__))
from persistence import pregame_ewma  # noqa: E402
from edge_signal_test_v52_extra_terms_screen import BENCH, build_team_games  # noqa: E402

HERE = os.path.dirname(__file__)
L2 = 0.3
TEST_START = 2011
SELECT_END = 2018
N_BOOT = 2000

SETS = {
    "ST1": ["st_kp_c"],
    "ST2": ["st_kp_c", "fp_net_c"],
    "EXPL": ["expl_rush_off_c", "expl_pass_off_c", "expl_rush_def_c", "expl_pass_def_c"],
    "PEN": ["pen_presnap_c"],
}
SETS["ALL"] = sorted({c for v in SETS.values() for c in v})
ALL_CANDS = SETS["ALL"]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float, free: np.ndarray) -> np.ndarray:
    """L2 logistic (mean NLL + 0.5*l2*|w|^2, intercept unpenalized); coefficients >= 0 except
    columns flagged `free` (the market spread), which are unconstrained and unpenalized."""
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    pen = np.concatenate([[0.0], np.where(free, 0.0, 1.0)])

    def f(w):
        pr = 1 / (1 + np.exp(-(Xb @ w)))
        nll = -np.mean(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12))
        g = Xb.T @ (pr - y) / n + l2 * pen * w
        return nll + 0.5 * l2 * np.sum(pen * w ** 2), g

    bounds = [(None, None)] + [(None, None) if fr else (0, None) for fr in free]
    return minimize(f, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=bounds).x


def build_diffs() -> pd.DataFrame:
    """One row per game: home-minus-away pre-game rating diffs for the benchmark and every
    candidate, plus outcome / market fields and the home/away pre-game pace."""
    tg = build_team_games()
    cols = BENCH + ALL_CANDS + ["sec_per_play_c"]
    pre = pregame_ewma(tg, cols)
    pre.columns = [c.removeprefix("pre_") for c in pre.columns]
    side = pd.concat([tg[["game_id", "season", "week", "is_home"]], pre], axis=1)
    h = side[side.is_home == 1].set_index("game_id")
    a = side[side.is_home == 0].set_index("game_id").reindex(h.index)
    d = h[cols] - a[cols]
    d["spp_sum"] = h["sec_per_play_c"] + a["sec_per_play_c"]
    d["season"], d["week"] = h["season"], h["week"]
    gm = tg[tg.is_home == 1].set_index("game_id")[["home_score", "away_score", "spread_line", "total_line"]].reindex(h.index)
    d = d.join(gm)
    d["margin"] = d["home_score"] - d["away_score"]
    d["home_win"] = (d["margin"] > 0).astype(int)  # ties count as not-home-win, as in v8 (res > 0)
    d["total"] = d["home_score"] + d["away_score"]
    return d.dropna(subset=BENCH + ALL_CANDS + ["spp_sum"]).reset_index()


def walk_forward(d: pd.DataFrame, cols: list[str], with_market: bool = False):
    """Returns (DataFrame of out-of-sample probabilities, list of per-season weight vectors)."""
    use = cols + (["spread_line"] if with_market else [])
    free = np.array([c == "spread_line" for c in use])
    out, ws = [], []
    for s in sorted(d.season.unique()):
        if s < TEST_START:
            continue
        tr, te = d[d.season < s], d[d.season == s]
        Xtr, Xte = tr[use].to_numpy(float), te[use].to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        w = fit_logistic((Xtr - mu) / sd, tr["home_win"].to_numpy(float), L2, free)
        p = 1 / (1 + np.exp(-(w[0] + ((Xte - mu) / sd) @ w[1:])))
        out.append(pd.DataFrame({"game_id": te["game_id"].to_numpy(), "season": s, "p": p, "y": te["home_win"].to_numpy()}))
        ws.append(w[1:])
    return pd.concat(out, ignore_index=True), ws


def summarize(res: pd.DataFrame) -> dict:
    y, p = res["y"].to_numpy(float), res["p"].to_numpy()
    return {"auc": auc(y, p), "ll": logloss(y, p), "acc": float(np.mean((p > 0.5) == (y == 1)))}


def paired_boot(base: pd.DataFrame, alt: pd.DataFrame, rng: np.random.Generator) -> dict:
    y = base["y"].to_numpy(float)
    pb, pa = base["p"].to_numpy(), alt["p"].to_numpy()
    n = len(y)
    d_ll, d_auc = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        ix = rng.integers(0, n, n)
        yb = y[ix]
        d_auc[i] = auc(yb, pa[ix]) - auc(yb, pb[ix])
        d_ll[i] = logloss(yb, pb[ix]) - logloss(yb, pa[ix])  # positive = alt better
    return {"d_auc_lo": np.percentile(d_auc, 2.5), "d_auc_hi": np.percentile(d_auc, 97.5),
            "p_auc_le0": float(np.mean(d_auc <= 0)),
            "d_ll_lo": np.percentile(d_ll, 2.5), "d_ll_hi": np.percentile(d_ll, 97.5),
            "p_ll_le0": float(np.mean(d_ll <= 0))}


def half(res: pd.DataFrame, which: str) -> pd.DataFrame:
    return res[res.season <= SELECT_END] if which == "select" else res[res.season > SELECT_END]


def compare(name: str, base: pd.DataFrame, alt: pd.DataFrame, rng) -> dict:
    row = {"model": name}
    for which in ("all", "select", "confirm"):
        b = base if which == "all" else half(base, which)
        a = alt if which == "all" else half(alt, which)
        sb, sa = summarize(b), summarize(a)
        row[f"dAUC_{which}"] = sa["auc"] - sb["auc"]
        row[f"dLL_{which}"] = sb["ll"] - sa["ll"]  # positive = alt better
        row[f"dAcc_{which}"] = sa["acc"] - sb["acc"]
    row.update(paired_boot(base, alt, rng))
    return row


def market_resid_test(d: pd.DataFrame, cols: list[str]) -> dict:
    """Walk-forward OLS of (margin - spread) on candidate diffs; out-of-sample MSE vs predicting 0
    and the OOS correlation. spread_line > 0 means home favored, so resid = margin - spread_line."""
    rows = []
    for s in sorted(d.season.unique()):
        if s < TEST_START:
            continue
        tr, te = d[d.season < s], d[d.season == s]
        Xtr = np.column_stack([np.ones(len(tr)), tr[cols].to_numpy(float)])
        Xte = np.column_stack([np.ones(len(te)), te[cols].to_numpy(float)])
        mu, sd = tr[cols].to_numpy(float).mean(0), tr[cols].to_numpy(float).std(0)
        Xtr[:, 1:], Xte[:, 1:] = (Xtr[:, 1:] - mu) / sd, (Xte[:, 1:] - mu) / sd
        ytr = (tr["margin"] - tr["spread_line"]).to_numpy(float)
        # ridge with a modest penalty on slopes so a flat signal shrinks to zero
        lam = 200.0
        reg = np.diag([0.0] + [lam] * len(cols))
        beta = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ ytr)
        yte = (te["margin"] - te["spread_line"]).to_numpy(float)
        rows.append(pd.DataFrame({"season": s, "y": yte, "pred": Xte @ beta}))
    r = pd.concat(rows, ignore_index=True)
    mse0, mse1 = float(np.mean(r.y ** 2)), float(np.mean((r.y - r.pred) ** 2))
    sq_gain = r.y ** 2 - (r.y - r.pred) ** 2
    z = float(sq_gain.mean() / (sq_gain.std() / np.sqrt(len(r))))
    return {"n": len(r), "oos_corr": float(r.y.corr(r.pred)), "mse_market": mse0, "mse_with": mse1, "z_gain": z,
            "corr_select": float(r[r.season <= SELECT_END].y.corr(r[r.season <= SELECT_END].pred)),
            "corr_confirm": float(r[r.season > SELECT_END].y.corr(r[r.season > SELECT_END].pred))}


def totals_test(d: pd.DataFrame, feats: list[str]) -> dict:
    """Walk-forward ridge of (total - total_line) on feats; does it beat the market total OOS?"""
    dd = d.dropna(subset=["total_line"])
    rows = []
    for s in sorted(dd.season.unique()):
        if s < TEST_START:
            continue
        tr, te = dd[dd.season < s], dd[dd.season == s]
        Xtr_raw, Xte_raw = tr[feats].to_numpy(float), te[feats].to_numpy(float)
        mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)
        Xtr = np.column_stack([np.ones(len(tr)), (Xtr_raw - mu) / sd])
        Xte = np.column_stack([np.ones(len(te)), (Xte_raw - mu) / sd])
        ytr = (tr["total"] - tr["total_line"]).to_numpy(float)
        reg = np.diag([0.0] + [200.0] * len(feats))
        beta = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ ytr)
        rows.append(pd.DataFrame({"season": s, "y": (te["total"] - te["total_line"]).to_numpy(float), "pred": Xte @ beta}))
    r = pd.concat(rows, ignore_index=True)
    ae0, ae1 = r.y.abs(), (r.y - r.pred).abs()
    gain = ae0 - ae1
    return {"n": len(r), "mae_market": float(ae0.mean()), "mae_with": float(ae1.mean()),
            "z_gain": float(gain.mean() / (gain.std() / np.sqrt(len(r)))),
            "oos_corr": float(r.y.corr(r.pred)),
            "corr_select": float(r[r.season <= SELECT_END].y.corr(r[r.season <= SELECT_END].pred)),
            "corr_confirm": float(r[r.season > SELECT_END].y.corr(r[r.season > SELECT_END].pred))}


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    d = build_diffs()
    print(f"{len(d)} games with all candidate diffs ({d.season.min()}-{d.season.max()}); "
          f"test seasons {TEST_START}-{d.season.max()}, select <= {SELECT_END} < confirm\n")

    rng = np.random.default_rng(53)
    base, wb = walk_forward(d, BENCH)
    sb = summarize(base)
    print(f"BASELINE (8 production stats, this framework): n={len(base)} AUC={sb['auc']:.4f} "
          f"logloss={sb['ll']:.4f} acc={sb['acc']:.4f}   (production reference AUC 0.6746)")
    for which in ("select", "confirm"):
        s = summarize(half(base, which))
        print(f"   {which:7s} AUC={s['auc']:.4f} logloss={s['ll']:.4f} acc={s['acc']:.4f}")

    print("\n=== A. Win-probability walk-forward: baseline + candidate set (deltas vs baseline; +ve = better) ===")
    rows, alts = [], {}
    for name, cs in SETS.items():
        alt, ws = walk_forward(d, BENCH + cs)
        alts[name] = alt
        r = compare(name, base, alt, rng)
        wmat = np.array(ws)[:, len(BENCH):]
        r["w_mean"] = ", ".join(f"{c.removesuffix('_c')}={m:.3f}" for c, m in zip(cs, wmat.mean(0)))
        r["w_zero_frac"] = float(np.mean(wmat == 0))
        rows.append(r)
    A = pd.DataFrame(rows).set_index("model")
    show = ["dAUC_all", "dAUC_select", "dAUC_confirm", "dLL_all", "dLL_select", "dLL_confirm", "dAcc_all", "p_auc_le0", "p_ll_le0"]
    print(A[show].to_string(float_format=lambda v: f"{v:+.5f}"))
    print("\n95% paired-bootstrap CIs (games resampled), all test seasons:")
    for name, r in A.iterrows():
        print(f"  {name:5s} dAUC [{r.d_auc_lo:+.5f}, {r.d_auc_hi:+.5f}]  dLogLoss [{r.d_ll_lo:+.5f}, {r.d_ll_hi:+.5f}]  "
              f"mean walk-forward weights: {r.w_mean}  (share of folds at exactly 0: {r.w_zero_frac:.2f})")

    print("\n=== B. Same, with the closing spread in the model (does the candidate add on top of the market?) ===")
    mbase, _ = walk_forward(d, BENCH, with_market=True)
    mono, _ = walk_forward(d, [], with_market=True)
    sm, so = summarize(mbase), summarize(mono)
    print(f"  spread only:            AUC={so['auc']:.4f} logloss={so['ll']:.4f}")
    print(f"  8 stats + spread:       AUC={sm['auc']:.4f} logloss={sm['ll']:.4f}")
    rows = []
    for name, cs in SETS.items():
        alt, _ = walk_forward(d, BENCH + cs, with_market=True)
        rows.append(compare(name, mbase, alt, rng))
    B = pd.DataFrame(rows).set_index("model")
    print(B[show].to_string(float_format=lambda v: f"{v:+.5f}"))

    print("\n=== C. Market-residual test: walk-forward ridge of (margin - spread) on candidate diffs ===")
    rows = []
    for name, cs in SETS.items():
        r = market_resid_test(d, cs)
        r["set"] = name
        rows.append(r)
    C = pd.DataFrame(rows).set_index("set")
    print(C.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("  (z_gain > 0 = beats predicting the market exactly; |z| < 2 = nothing there)")

    print("\n=== D. Totals: walk-forward ridge of (total - market total) on pace / special-teams features ===")
    rows = []
    for name, feats in {"spp_sum (pace)": ["spp_sum"],
                        "spp_sum + st_kp": ["spp_sum", "st_kp_c"],
                        "st_kp only": ["st_kp_c"]}.items():
        r = totals_test(d, feats)
        r["feats"] = name
        rows.append(r)
    print(pd.DataFrame(rows).set_index("feats").to_string(float_format=lambda v: f"{v:+.4f}"))

    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    A.to_csv(os.path.join(HERE, "output", "v53_winprob.csv"))
    B.to_csv(os.path.join(HERE, "output", "v53_with_market.csv"))
    C.to_csv(os.path.join(HERE, "output", "v53_market_resid.csv"))
    print("\nCSV written to research/output/v53_*.csv")


if __name__ == "__main__":
    main()
