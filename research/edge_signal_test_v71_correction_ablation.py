"""v71: does each shipped correction still earn its place? (cowork plan A8: re-check the v14-v51 ships against the v57 rule;
includes the momentum drop test flagged when its weight fell to z=0.7 after Elo.)

Model: v64's fully walk-forward pipeline (z-score from training seasons, composite weights at L2 = 0.03, corrections refit by win/loss
logit on training seasons >= 2009, calibration refit), test seasons 2011-2026. FULL = every shipped correction. For each correction, the
ABLATED model removes just that piece and refits everything else the same way:
  no_momentum   streak term removed (the z = 0.7 question)
  no_skill      skill-out (EPA of injured skill players) term removed
  no_hfa        home-field term removed
  no_qb_epa     EPA-based QB gap removed (affects games without FantasyPros projections, i.e. before 2017)
  no_fantasy    FantasyPros QB gap removed (the EPA gap is used for every season instead)
  no_qb_any     both QB gaps removed
  no_decay      QB gap decay rate set to 0 (the gap stays at full size however long a backup has started)
  none          every correction removed (composite alone)

Pre-registered rule (v57+, applied to "FULL over ABLATED"): the correction earns its place only if AUC and log loss are both better
with it in BOTH halves (<=2018 / >2018), paired-bootstrap P(no improvement) <= 0.01 (N_BOOT = 2000), and log loss is better in
>= 60% of test seasons. A correction that fails is a CANDIDATE FOR REMOVAL, not removed here: dropping something is itself a
shipping decision and gets its own note. v14-v51's structural choices (primary window of 5 starts, mid-game QB injury filter,
weather in the points model only) are not re-run here; this covers the delta corrections and the QB decay constant.

Run: PYTHONPATH=. python research/edge_signal_test_v71_correction_ablation.py
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
from edge_signal_test_v54_pen_presnap_production_refit import fit_logit, fit_nonneg  # noqa: E402
from edge_signal_test_v61_ewma_speed_season_prior import compare  # noqa: E402
from edge_signal_test_v68_qb_weight_ats_residual import fmt, p_home, verdict  # noqa: E402

M = tr.POWER_METRICS
L2 = 0.03
TEST_START = 2011
CORR_FROM = 2009


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
    X = np.column_stack([[r["diffs"][m] for r in log] for m in M])

    skill_out = tr.skill_epa_out_by_team_week(current)
    hfa_by_season = tr.home_field_logit_by_season(played)
    streak = tr.team_streak_by_week(played)
    qb_out = tr._qb_out_doubtful_by_week(current)

    def g(d, r, side):
        return d.get((r[side], r["season"], r["week"]), 0.0)

    def gaps(decay):
        old = tr.QB_GAP_DECAY_RATE
        tr.QB_GAP_DECAY_RATE = decay
        try:
            qg, *_ = tr.qb_quality_gap_data(played, current)
            fg, _t, _l, _b, avail = tr.qb_fantasy_gap_data(played, current, 1)
        finally:
            tr.QB_GAP_DECAY_RATE = old
        qb = np.array([g(qg, r, "home") - g(qg, r, "away") for r in log])
        fa = np.array([g(fg, r, "home") - g(fg, r, "away") for r in log])
        use = np.array([avail.get((r["season"], r["week"]), False) for r in log])
        return qb, fa, use

    qb, fa, use_fp = gaps(tr.QB_GAP_DECAY_RATE)
    qb0, fa0, use_fp0 = gaps(0.0)
    sk = np.array([g(skill_out, r, "home") - g(skill_out, r, "away") for r in log])
    hfa = np.array([0.0 if r.get("neutral") else hfa_by_season.get(r["season"], 0.0) for r in log])
    stk = np.array([g(streak, r, "home") - g(streak, r, "away") for r in log])
    clean = np.array([not (qb_out.get((r["home"], r["season"], r["week"]), False) or qb_out.get((r["away"], r["season"], r["week"]), False)) for r in log])

    zeros = np.zeros(len(y))
    variants = {
        "FULL": dict(qb=qb, fa=fa, fp=use_fp, sk=sk, hfa=hfa, stk=stk),
        "no_momentum": dict(qb=qb, fa=fa, fp=use_fp, sk=sk, hfa=hfa, stk=zeros),
        "no_skill": dict(qb=qb, fa=fa, fp=use_fp, sk=zeros, hfa=hfa, stk=stk),
        "no_hfa": dict(qb=qb, fa=fa, fp=use_fp, sk=sk, hfa=zeros, stk=stk),
        "no_qb_epa": dict(qb=zeros, fa=fa, fp=use_fp, sk=sk, hfa=hfa, stk=stk),
        "no_fantasy": dict(qb=qb, fa=fa, fp=np.zeros(len(y), bool), sk=sk, hfa=hfa, stk=stk),
        "no_qb_any": dict(qb=zeros, fa=zeros, fp=use_fp, sk=sk, hfa=hfa, stk=stk),
        "no_decay": dict(qb=qb0, fa=fa0, fp=use_fp0, sk=sk, hfa=hfa, stk=stk),
        "none": dict(qb=zeros, fa=zeros, fp=use_fp, sk=zeros, hfa=zeros, stk=zeros),
    }
    test_seasons = [s for s in sorted(set(seasons)) if s >= TEST_START]
    probs, ratios = {}, {}
    for name, v in variants.items():
        delta_t = np.full(len(y), np.nan)
        cal = np.zeros((len(y), 2))
        last_ratio = None
        for S in test_seasons:
            tri, tei = seasons < S, seasons == S
            mu_t, sd_t = X[tri].mean(0), X[tri].std(0)
            sd_t[sd_t == 0] = 1.0
            w = fit_nonneg((X[tri] - mu_t) / sd_t, y[tri], L2)[1:]
            d_raw = ((X - mu_t) / sd_t) @ w
            kp = tri & (seasons >= CORR_FROM)
            feats = np.column_stack([d_raw, v["qb"], v["fa"], v["sk"], v["hfa"], v["stk"]])[kp]
            live = feats.std(0) > 0
            live[0] = True
            wj, _se = fit_logit(feats[:, live], y[kp])
            ratio = np.zeros(6)
            ratio[live] = np.concatenate([[1.0], wj[2:] / wj[1]])
            last_ratio = ratio
            qbh = np.where(v["fp"], ratio[2] * v["fa"], ratio[1] * v["qb"])
            d = d_raw + qbh + ratio[3] * v["sk"] + ratio[4] * v["hfa"] + ratio[5] * v["stk"]
            km = kp & clean & (d != 0)
            fav = np.where(d > 0, y, 1 - y)
            wc, _ = fit_logit(np.abs(d[km])[:, None], fav[km])
            delta_t[tei] = d[tei]
            cal[tei] = (wc[0], wc[1])
        test = seasons >= TEST_START
        probs[name] = np.array([p_home(delta_t[test][i], cal[test][i, 0], cal[test][i, 1]) for i in range(test.sum())]).reshape(-1)
        ratios[name] = last_ratio
        print(f"  built {name}", flush=True)

    test = seasons >= TEST_START
    yt, ss = y[test], seasons[test]
    rng = np.random.default_rng(71)
    print(f"\n{len(yt)} decided test games, seasons {TEST_START}-{max(test_seasons)}\n")
    print("=== does the correction earn its place? FULL (with) vs ABLATED (without), v57 rule ===")
    for name in list(variants)[1:]:
        # compare(yy, ss, p_base, p_alt): dAUC/dLL positive when ALT is better; ALT = FULL, base = ablated
        c = compare(yt, ss, probs[name], probs["FULL"], rng)
        ok, rep = verdict(c)
        print(f"{name:12s}: {fmt(c)}; replicated both halves={rep} -> {'EARNS ITS PLACE' if ok else 'CANDIDATE FOR REMOVAL (fails the rule)'}")

    print("\n=== final-fold delta-equivalent constants in the FULL model (qb / fantasy / skill / hfa / momentum) ===")
    r = ratios["FULL"]
    print(f"  qb {r[1]:+.4f}  fantasy {r[2]:+.4f}  skill {r[3]:+.4f}  hfa {r[4]:+.4f}  momentum {r[5]:+.4f}")


if __name__ == "__main__":
    main()
