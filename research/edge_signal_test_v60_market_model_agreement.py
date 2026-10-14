"""v60: market vs. production model -- does agreement/disagreement carry information?

Questions (user request 2026-10-14):
  1. Does a composite of market + model, with extra weight/confidence when they agree and less when
     they disagree, beat either alone (walk-forward log loss / AUC)?
  2. Are there clear trends in the games where model and market disagree (who wins, who covers,
     and in which segments)?

Inputs: the production compute_ratings game_log + compute_backtest_scatter (the CURRENT production
`delta`: 9-stat composite + QB gap + skill + HFA + momentum; weights fit once on all data, so the
model side has a mild in-sample element -- ~12 parameters on ~5,000 games -- disclosed, and the
market side has none). Market = nflverse closing spread_line (home-favored positive).

Composite tests are walk-forward (train < season, test 2011+) with only 2-5 logistic parameters.
Segment tables are descriptive; with ~20 segments some will look interesting by chance, so each is
shown for both chronological halves (<=2018 / >2018) and only "same side of 50%/52.4% in both
halves" is called a trend.

Run: PYTHONPATH=. python research/edge_signal_test_v60_market_model_agreement.py
"""
from __future__ import annotations

import csv
import io
import math
import sys

import numpy as np
import requests
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

import nflhub.sources.team_ratings as tr

TEST_START = 2011
SELECT_MAX = 2018
N_BOOT = 2000
BREAKEVEN = 0.5238


def fit_logit(X, y, l2=1e-4):
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])

    def f(w):
        ph = 1 / (1 + np.exp(-(Xb @ w)))
        nll = -np.mean(y * np.log(ph + 1e-12) + (1 - y) * np.log(1 - ph + 1e-12)) + 0.5 * l2 * np.sum(w[1:] ** 2)
        g = Xb.T @ (ph - y) / n
        g[1:] += l2 * w[1:]
        return nll, g

    return minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B").x


def predict(w, X):
    return 1 / (1 + np.exp(-(w[0] + X @ w[1:])))


def walk_forward(seasons, X, y):
    P, Y, S = [], [], []
    for ts in sorted(set(seasons)):
        if ts < TEST_START:
            continue
        tri, tei = seasons < ts, seasons == ts
        w = fit_logit(X[tri], y[tri])
        P.append(predict(w, X[tei]))
        Y.append(y[tei])
        S.append(seasons[tei])
    return np.concatenate(P), np.concatenate(Y), np.concatenate(S)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    all_rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    by_id = {r["game_id"]: r for r in all_rows}
    played = [r for r in all_rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)

    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    qb_out = tr._qb_out_doubtful_by_week(current)
    qb_gap, *_ = tr.qb_quality_gap_data(played, current)
    fantasy_gap, _t, _l, _b, fantasy_available = tr.qb_fantasy_gap_data(played, current, 1)
    skill_out = tr.skill_epa_out_by_team_week(current)
    hfa = tr.home_field_logit_by_season(played)
    streak = tr.team_streak_by_week(played)
    scatter = tr.compute_backtest_scatter(log, qb_out, qb_gap, fantasy_gap, fantasy_available, skill_out, hfa, None, streak)
    assert len(scatter) == len(log)

    keep = [i for i, r in enumerate(log) if r["home_score"] != r["away_score"]]
    L = [log[i] for i in keep]
    Sx = [scatter[i] for i in keep]
    G = [by_id.get(r["game_id"], {}) for r in L]
    n = len(L)
    season = np.array([r["season"] for r in L])
    week = np.array([r["week"] for r in L])
    margin = np.array([r["home_score"] - r["away_score"] for r in L], float)
    y = (margin > 0).astype(float)
    spread = np.array([r["spread_line"] for r in L])  # home-favored positive (checked below)
    delta = np.array([s["delta"] for s in Sx])
    print(f"{n} decided games {season.min()}-{season.max()};  corr(spread, margin)={np.corrcoef(spread, margin)[0, 1]:.3f} "
          f"(positive => home-favored positive)")

    # Market + model expressed in points: model margin from an all-data OLS of margin on delta.
    b1, b0 = np.polyfit(delta, margin, 1)
    model_margin = b0 + b1 * delta
    gap = model_margin - spread  # >0: model likes the home side more than the market
    print(f"model margin = {b0:+.2f} + {b1:.2f}*delta;  corr(model_margin, spread)={np.corrcoef(model_margin, spread)[0, 1]:.3f};"
          f" gap sd={gap.std():.2f} pts")

    mk_side = np.sign(spread)
    md_side = np.sign(model_margin)  # model pick via its own margin (delta>0 equivalent up to intercept)
    md_side_delta = np.sign(delta)
    nonpk = mk_side != 0
    agree = (mk_side == md_side_delta) & nonpk
    disagree = (mk_side != md_side_delta) & nonpk
    actual = np.sign(margin)
    cover = margin - spread  # >0 home covered
    print(f"pick'ems (spread 0): {(~nonpk).sum()};  agree={agree.sum()} ({agree.sum() / nonpk.sum():.1%});  disagree={disagree.sum()}")

    hi = season > SELECT_MAX
    halves = {"<=2018": ~hi, ">2018": hi}

    def acc(side, mask):
        m = mask & nonpk
        return (side[m] == actual[m]).mean(), m.sum()

    print("\n=== 1. straight-up accuracy by agreement ===")
    for lab, mask in (("all non-pickem", nonpk), ("AGREE", agree), ("DISAGREE", disagree)):
        am, nn = acc(mk_side, mask)
        ad, _ = acc(md_side_delta, mask)
        print(f"  {lab:15s} n={nn:5d}  market {am:.1%}  model {ad:.1%}")
        for hl, hm in halves.items():
            am, nn = acc(mk_side, mask & hm)
            ad, _ = acc(md_side_delta, mask & hm)
            print(f"      {hl:7s} n={nn:5d}  market {am:.1%}  model {ad:.1%}")

    # --- composite -----------------------------------------------------------------------------
    print("\n=== 2. walk-forward composite (train < season, test 2011+) ===")
    sm = spread / 6.0  # scale only to keep the optimizer happy
    dm = model_margin / 6.0
    ag = (np.sign(spread) == np.sign(delta)).astype(float)
    feats = {
        "A market only": np.column_stack([sm]),
        "B model only": np.column_stack([dm]),
        "C market+model": np.column_stack([sm, dm]),
        "D C + (market*model)": np.column_stack([sm, dm, sm * dm]),
        "E separate slopes agree/disagree": np.column_stack([sm * ag, dm * ag, sm * (1 - ag), dm * (1 - ag)]),
        "F C + agree flag*(sm+dm)": np.column_stack([sm, dm, ag * (sm + dm)]),
    }
    preds = {}
    for name, X in feats.items():
        p, yy, ss = walk_forward(season, X, y)
        preds[name] = p
    rng = np.random.default_rng(60)
    ix_all = [rng.integers(0, len(yy), len(yy)) for _ in range(N_BOOT)]

    def boot_dll(pa, pb):  # +ve = pb better than pa
        d = []
        for ix in ix_all:
            if yy[ix].min() == yy[ix].max():
                continue
            d.append(log_loss(yy[ix], pa[ix]) - log_loss(yy[ix], pb[ix]))
        return np.mean(np.array(d) <= 0)

    print(f"{'model':34s} {'AUC':>7} {'LL':>7} | sel LL   conf LL  | vs A: dLL     P(<=0) | vs C: dLL     P(<=0)")
    base_ll = log_loss(yy, preds["A market only"])
    for name, p in preds.items():
        a = roc_auc_score(yy, p)
        ll = log_loss(yy, p)
        sel, con = ss <= SELECT_MAX, ss > SELECT_MAX
        lls, llc = log_loss(yy[sel], p[sel]), log_loss(yy[con], p[con])
        dA = base_ll - ll
        pA = boot_dll(preds["A market only"], p) if name != "A market only" else float("nan")
        dC = log_loss(yy, preds["C market+model"]) - ll
        pC = boot_dll(preds["C market+model"], p) if name not in ("A market only", "B model only", "C market+model") else float("nan")
        print(f"{name:34s} {a:7.4f} {ll:7.4f} | {lls:.4f}  {llc:.4f}  | {dA:+.5f} {pA:8.3f} | {dC:+.5f} {pC:8.3f}")

    print("\nall-data coefficients (per 6 pts), for reading only:")
    for name in ("C market+model", "E separate slopes agree/disagree"):
        w = fit_logit(feats[name], y)
        print(f"  {name}: intercept {w[0]:+.3f}, slopes " + ", ".join(f"{v:+.3f}" for v in w[1:]))
    print("  (E order: market|agree, model|agree, market|disagree, model|disagree)")

    # --- confidence calibration under composite C ---------------------------------------------
    print("\n=== 3. composite confidence vs. actual hit rate (walk-forward C, test seasons) ===")
    mask_t = season >= TEST_START
    p_home = preds["C market+model"]
    ag_t, dis_t = agree[mask_t], disagree[mask_t]
    pick_conf = np.where(p_home >= 0.5, p_home, 1 - p_home)
    pick_hit = ((p_home >= 0.5) == (y[mask_t] == 1)).astype(float)
    mk_hit = (mk_side[mask_t] == actual[mask_t]).astype(float)
    md_hit = (md_side_delta[mask_t] == actual[mask_t]).astype(float)
    print(f"{'group':26s} {'n':>5} {'avg conf':>9} {'composite':>10} {'market':>8} {'model':>7}")
    for lab, m in (("AGREE (all)", ag_t), ("DISAGREE (all)", dis_t)):
        print(f"{lab:26s} {m.sum():5d} {pick_conf[m].mean():9.1%} {pick_hit[m].mean():10.1%} {mk_hit[m].mean():8.1%} {md_hit[m].mean():7.1%}")
    for lab, m0 in (("AGREE", ag_t), ("DISAGREE", dis_t)):
        qs = np.quantile(pick_conf[m0], [0, 1 / 3, 2 / 3, 1])
        for i, nm in enumerate(("low conf", "mid conf", "high conf")):
            m = m0 & (pick_conf >= qs[i]) & (pick_conf <= qs[i + 1] if i == 2 else pick_conf < qs[i + 1])
            print(f"  {lab:9s} {nm:9s} {m.sum():5d} {pick_conf[m].mean():9.1%} {pick_hit[m].mean():10.1%} {mk_hit[m].mean():8.1%} {md_hit[m].mean():7.1%}")

    # --- disagreement: who is right, and where ------------------------------------------------
    print("\n=== 4a. does the model-minus-market gap predict the market's error? ===")
    resid = margin - spread  # actual margin beyond market
    for lab, m in (("all", np.ones(n, bool)), ("<=2018", ~hi), (">2018", hi)):
        A = np.column_stack([np.ones(m.sum()), gap[m]])
        coef, *_ = np.linalg.lstsq(A, resid[m], rcond=None)
        e = resid[m] - A @ coef
        se = math.sqrt(e.var(ddof=2) / (np.sum((gap[m] - gap[m].mean()) ** 2)))
        print(f"  {lab:7s} n={m.sum():5d} slope of (actual margin - spread) on gap = {coef[1]:+.3f} (se {se:.3f}, t={coef[1] / se:+.2f})"
              "   [>0 means model adds info beyond market]")

    print("\n=== 4b. bet the model's side vs. the spread (ATS), by size of disagreement ===")
    model_home = gap > 0
    ats_win = np.where(model_home, cover > 0, cover < 0)
    push = cover == 0

    def ats_row(lab, mask):
        m = mask & ~push
        k, nn = int(ats_win[m].sum()), int(m.sum())
        lo, hi_ = wilson(k, nn)
        cells = []
        for hl, hm in halves.items():
            mm = m & hm
            cells.append(f"{hl} {ats_win[mm].mean():.1%} ({mm.sum()})" if mm.sum() else f"{hl} n/a")
        flag = ""
        vals = [ats_win[m & hm].mean() for hm in halves.values() if (m & hm).sum() >= 30]
        if len(vals) == 2:
            if all(v > BREAKEVEN for v in vals):
                flag = " <-- model side profitable both halves"
            elif all(v < 1 - BREAKEVEN for v in vals):
                flag = " <-- FADE model both halves"
        print(f"  {lab:34s} n={nn:5d} model-side ATS {k / nn:.1%} [{lo:.1%},{hi_:.1%}]  " + "  ".join(cells) + flag)

    ag_ = np.abs(gap)
    ats_row("all games", np.ones(n, bool))
    for lo_, hi_ in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 99)):
        ats_row(f"|gap| {lo_}-{hi_ if hi_ < 99 else '+'} pts", (ag_ >= lo_) & (ag_ < hi_))
    ats_row("|gap| >= 3 pts", ag_ >= 3)

    print("\n=== 4c. side-disagreement games (model picks the market's dog), segments ===")
    print("  columns: n | model-pick SU | market-pick SU | model-pick ATS | halves (SU model-pick)")
    dog_model_home = disagree & (md_side_delta > 0)  # model picks home, market favors away
    dog_model_away = disagree & (md_side_delta < 0)

    def dis_row(lab, base_mask):
        m = base_mask & disagree
        nn = int(m.sum())
        if nn == 0:
            print(f"  {lab:34s} n=0")
            return
        su_md = (md_side_delta[m] == actual[m]).mean()
        su_mk = (mk_side[m] == actual[m]).mean()
        mm = m & ~push
        ats = ats_win[mm].mean() if mm.sum() else float("nan")
        hv = []
        for hl, hm in halves.items():
            q = m & hm
            hv.append(f"{hl} {(md_side_delta[q] == actual[q]).mean():.1%} ({q.sum()})" if q.sum() else f"{hl} n/a")
        print(f"  {lab:34s} {nn:5d} | {su_md:.1%} | {su_mk:.1%} | {ats:.1%} | " + "  ".join(hv))

    def g_(col, f):
        return np.array([bool(f(g.get(col, ""))) for g in G])

    def num(col):
        out = []
        for g in G:
            try:
                out.append(float(g.get(col, "")))
            except (ValueError, TypeError):
                out.append(float("nan"))
        return np.array(out)

    allm = np.ones(n, bool)
    dis_row("ALL disagreements", allm)
    dis_row("model picks HOME dog", md_side_delta > 0)
    dis_row("model picks AWAY dog", md_side_delta < 0)
    asp = np.abs(spread)
    for lo_, hi_ in ((0, 2), (2, 3.5), (3.5, 6.5), (6.5, 10), (10, 99)):
        dis_row(f"market |spread| {lo_}-{hi_ if hi_ < 99 else '+'}", (asp >= lo_) & (asp < hi_))
    for lo_, hi_ in ((1, 4), (5, 9), (10, 14), (15, 99)):
        dis_row(f"week {lo_}-{hi_ if hi_ < 99 else '+'}", (week >= lo_) & (week <= hi_))
    for lo_, hi_ in ((2007, 2012), (2013, 2018), (2019, 2023), (2024, 2030)):
        dis_row(f"seasons {lo_}-{hi_}", (season >= lo_) & (season <= hi_))
    div = g_("div_game", lambda v: v == "1")
    dis_row("divisional", div)
    dis_row("non-divisional", ~div)
    roof = [g.get("roof", "") for g in G]
    dome = np.array([r in ("dome", "closed") for r in roof])
    dis_row("dome/closed roof", dome)
    dis_row("outdoors/open", ~dome)
    prime = np.array([(g.get("weekday", "") in ("Thursday", "Monday")) or (g.get("weekday", "") == "Sunday" and g.get("gametime", "") >= "19:00") for g in G])
    dis_row("primetime (Thu/Mon/SNF)", prime)
    dis_row("non-primetime", ~prime)
    hq = np.array([s["home_qb_out"] for s in Sx]); aq = np.array([s["away_qb_out"] for s in Sx])
    dis_row("a QB Out/Doubtful involved", hq | aq)
    dis_row("no QB flags", ~(hq | aq))
    hr, ar = num("home_rest"), num("away_rest")
    rest_adv = np.nan_to_num(hr - ar)
    dis_row("home rest edge >=2d", rest_adv >= 2)
    dis_row("away rest edge >=2d", rest_adv <= -2)
    dis_row("neutral site", np.array([bool(s["neutral"]) for s in Sx]))
    temp, wind = num("temp"), num("wind")
    dis_row("cold (<=35F)", np.nan_to_num(temp, nan=99) <= 35)
    dis_row("windy (>=15mph)", np.nan_to_num(wind, nan=0) >= 15)
    hs = np.array([s["home_streak"] for s in Sx]); as_ = np.array([s["away_streak"] for s in Sx])
    pick_streak = np.where(md_side_delta > 0, hs, as_)
    opp_streak = np.where(md_side_delta > 0, as_, hs)
    dis_row("model pick on win streak >=2", pick_streak >= 2)
    dis_row("model pick on losing streak <=-2", pick_streak <= -2)

    print("\n=== 4d. same segments but looking at ALL games with a >=3 pt gap (ATS of model side) ===")
    big = ag_ >= 3
    for lab, m in (("home-side model bets", model_home), ("away-side model bets", ~model_home),
                   ("model on market FAVORITE (more)", (np.sign(model_margin) == mk_side) & (np.sign(gap) == mk_side)),
                   ("model on market DOG (less fav)", (np.sign(gap) == -mk_side)),
                   ("divisional", div), ("primetime", prime), ("dome", dome), ("QB flag", hq | aq)):
        ats_row(lab, big & m)

    print("\n=== 5. where does the model add to the market? all games, walk-forward test seasons (2011+) ===")
    print("  dLL = LL(market only) - LL(market+model); + means the model helped in that segment")
    tm = season >= TEST_START
    pA, pC = preds["A market only"], preds["C market+model"]
    ys, ssn = y[tm], season[tm]
    llA = -(ys * np.log(pA) + (1 - ys) * np.log(1 - pA))
    llC = -(ys * np.log(pC) + (1 - ys) * np.log(1 - pC))
    mkh = (mk_side[tm] == actual[tm]); mdh = (md_side_delta[tm] == actual[tm])
    segs = [("all", allm), ("agree", agree), ("disagree", disagree), ("QB flag", hq | aq), ("no QB flag", ~(hq | aq)),
            ("divisional", div), ("non-divisional", ~div), ("dome", dome), ("outdoors", ~dome), ("primetime", prime),
            ("cold <=35F", np.nan_to_num(temp, nan=99) <= 35), ("windy >=15", np.nan_to_num(wind, nan=0) >= 15),
            ("week 1-4", week <= 4), ("week 5-9", (week >= 5) & (week <= 9)), ("week 10-14", (week >= 10) & (week <= 14)), ("week 15+", week >= 15),
            ("|spread|<3", asp < 3), ("|spread| 3-7", (asp >= 3) & (asp < 7)), ("|spread|>=7", asp >= 7),
            ("home rest edge>=2", rest_adv >= 2), ("away rest edge>=2", rest_adv <= -2)]
    print(f"  {'segment':20s} {'n':>5} {'mkt SU':>7} {'model SU':>8} {'dLL':>9}  halves dLL (<=2018 / >2018)")
    for lab, m0 in segs:
        m = m0[tm]
        if m.sum() < 40:
            continue
        d = llA[m] - llC[m]
        hs_ = []
        for hm in (ssn[m] <= SELECT_MAX, ssn[m] > SELECT_MAX):
            hs_.append(f"{(llA[m][hm] - llC[m][hm]).mean():+.4f}" if hm.sum() else "n/a")
        se = d.std(ddof=1) / math.sqrt(len(d))
        both = all(float(v) > 0 for v in hs_ if v != "n/a") and len(hs_) == 2
        print(f"  {lab:20s} {m.sum():5d} {mkh[m & (mk_side[tm] != 0)].mean() if False else mkh[m].mean():7.1%} {mdh[m].mean():8.1%} {d.mean():+9.5f}  {hs_[0]} / {hs_[1]}  (t={d.mean() / se:+.2f}){' <-- both halves +' if both else ''}")


if __name__ == "__main__":
    main()
