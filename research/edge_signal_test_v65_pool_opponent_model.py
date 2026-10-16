"""v65: what does the pool field actually do? Calibrating the opponent model for the pool Monte Carlo.

The Monte Carlo (B4) has to simulate 26 opponents' cards. The plan's first-cut model: each opponent
picks each game independently, taking the favorite with probability f = the pool's pick share. This
script checks, on the 2026 weeks 1-5 sheet tabs (research/cache/pool_weeks.json, from
research/pool_data_extract.py), (1) which predictor of f is best, (2) whether the independence
assumption holds (do some entrants chalk more than others, persistently?), and (3) how the Monday-
night tiebreaker guesses are distributed.

Opponents = everyone except the user. The favorite = the team with de-vigged closing-moneyline
probability > 0.5 (nflverse), falling back to the spread's sign.

Part 1 predictors of f (per game), scored by the log loss of each opponent's actual pick (so a game
with 26 picks counts 26 times), walk-forward over weeks 2-5 (fit on earlier weeks only) AND
leave-one-week-out over weeks 1-5:
  M0 constant pool favorite rate; M1 the app's current estimate (national pick % + learned per-team
  bias, bias from earlier weeks only); M2 logistic on logit(market P(fav)); M3 M2 + logit(national %).
No ADOPT rule: with 5 weeks (~75 games) this is a calibration check to pick a sensible model, not a
production test; the choice is the lowest log loss with a simplicity tie-break (within 0.002 nats
prefers the fewer-parameter model).

Run: PYTHONPATH=. python research/edge_signal_test_v65_pool_opponent_model.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import minimize

CACHE = os.path.join(os.path.dirname(__file__), "cache", "pool_weeks.json")


def american(o):
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)


def market_fav_prob(g):
    """(favorite team, P(favorite wins)) de-vigged from the closing moneylines."""
    if g["ml_home"] is not None and g["ml_away"] is not None:
        ph, pa = american(g["ml_home"]), american(g["ml_away"])
        p_home = ph / (ph + pa)
    else:
        p_home = 0.5 + 0.03 * (g["spread_line"] or 0)  # crude fallback, only if a moneyline is missing
    return (g["home"], p_home) if p_home >= 0.5 else (g["away"], 1 - p_home)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def load():
    D = json.load(open(CACHE))
    rows = []  # one per game
    for w, d in D.items():
        w = int(w)
        opps = [e for e in d["entrants"] if d["my_name"].lower() not in e["name"].lower()]
        for g in d["games"]:
            fav, pf = market_fav_prob(g)
            picks = [e for e in opps if g["home"] in e["picks"] or g["away"] in e["picks"]]
            k = sum(fav in e["picks"] for e in picks)
            nat = d["national_pct"].get(fav)
            rows.append({"week": w, "game": g, "fav": fav, "pf": pf, "n": len(picks), "k": k, "nat": nat,
                         "fav_won": (g["home_score"] > g["away_score"]) == (fav == g["home"]) if g["home_score"] is not None else None})
    return D, rows


def nll(f, rows):
    f = np.clip(f, 1e-3, 1 - 1e-3)
    n = np.array([r["n"] for r in rows])
    k = np.array([r["k"] for r in rows])
    return -(k * np.log(f) + (n - k) * np.log(1 - f)).sum() / n.sum()


def fit_logistic(rows, feats):
    n = np.array([r["n"] for r in rows], float)
    k = np.array([r["k"] for r in rows], float)
    X = np.column_stack([np.ones(len(rows))] + [feats(r) for r in rows] if False else [np.ones(len(rows))] + [np.array([f(r) for r in rows]) for f in feats])

    def obj(w):
        z = X @ w
        p = np.clip(sig(z), 1e-6, 1 - 1e-6)
        return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum() / n.sum() + 1e-4 * np.sum(w[1:] ** 2)

    return minimize(obj, np.zeros(X.shape[1]), method="BFGS").x


def predict_logistic(w, rows, feats):
    X = np.column_stack([np.ones(len(rows))] + [np.array([f(r) for r in rows]) for f in feats])
    return sig(X @ w)


FEATS = {
    "M2 logit(market)": [lambda r: logit(r["pf"])],
    "M3 + logit(national)": [lambda r: logit(r["pf"]), lambda r: logit(r["nat"] if r["nat"] is not None else 0.5)],
}


def main():
    D, rows = load()
    print(f"{len(rows)} games, weeks {sorted({r['week'] for r in rows})}; opponents per game ~{np.mean([r['n'] for r in rows]):.1f}")
    print(f"overall opponent favorite-pick rate {sum(r['k'] for r in rows) / sum(r['n'] for r in rows):.3f}; "
          f"mean market P(fav) {np.mean([r['pf'] for r in rows]):.3f}; favorites won {np.mean([r['fav_won'] for r in rows if r['fav_won'] is not None]):.3f}\n")

    print("=== Part 1: predictors of the pool's favorite-pick share ===")
    for name in ("walk-forward weeks 2-5", "leave-one-week-out weeks 1-5"):
        print(f"  [{name}]")
        res = {m: [] for m in ("M0 constant", "M1 national+bias", *FEATS)}
        scored = []
        for w in range(1, 6):
            test = [r for r in rows if r["week"] == w]
            train = [r for r in rows if (r["week"] < w if name.startswith("walk") else r["week"] != w)]
            if not train:
                continue
            scored += test
            p0 = sum(r["k"] for r in train) / sum(r["n"] for r in train)
            res["M0 constant"] += [p0] * len(test)
            # M1: national pct + per-team learned bias (mean pool - national over the TRAIN weeks' games of that team)
            bias = {}
            for r in train:
                if r["nat"] is not None:
                    bias.setdefault(r["fav"], []).append(r["k"] / r["n"] - r["nat"])
            b = {t: np.mean(v) * min(len(v), 3) / 3 for t, v in bias.items()}  # production damping (full weight at >= 3 obs)
            res["M1 national+bias"] += [min(0.98, max(0.02, (r["nat"] if r["nat"] is not None else p0) + b.get(r["fav"], 0.0))) for r in test]
            for m, feats in FEATS.items():
                res[m] += list(predict_logistic(fit_logistic(train, feats), test, feats))
        for m, f in res.items():
            f = np.array(f)
            n = np.array([r["n"] for r in scored])
            share = np.array([r["k"] / r["n"] for r in scored])
            print(f"    {m:22s} pick log loss {nll(f, scored):.4f}   share MAE {np.average(np.abs(f - share), weights=n):.4f}   share RMSE {np.sqrt(np.average((f - share) ** 2, weights=n)):.4f}")
        base = nll(np.array([r["k"] for r in scored], float) / np.array([r["n"] for r in scored]), scored)
        print(f"    (floor: each game's own realized share = {base:.4f})")

    w_all = fit_logistic(rows, FEATS["M2 logit(market)"])
    print(f"\n  M2 all-data fit: logit(f) = {w_all[0]:+.3f} + {w_all[1]:.3f} * logit(market P(fav))")
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)):
        sub = [r for r in rows if lo <= r["pf"] < hi]
        if sub:
            print(f"    market P(fav) {lo:.1f}-{hi:.1f}: {len(sub):2d} games, pool picks fav {sum(r['k'] for r in sub) / sum(r['n'] for r in sub):.3f}, market says {np.mean([r['pf'] for r in sub]):.3f}")

    print("\n=== Part 2: independence -- do entrants differ persistently in how chalky they are? ===")
    ent = {}
    for w, d in D.items():
        by_team_fav = {}
        for r in [x for x in rows if x["week"] == int(w)]:
            by_team_fav[r["game"]["home"]] = r
            by_team_fav[r["game"]["away"]] = r
        for e in d["entrants"]:
            if d["my_name"].lower() in e["name"].lower():
                continue
            res = []
            for t in e["picks"]:
                r = by_team_fav.get(t)
                if r is None:
                    continue
                f_pred = float(sig(w_all[0] + w_all[1] * logit(r["pf"])))
                res.append(((1.0 if t == r["fav"] else 0.0) - f_pred, f_pred * (1 - f_pred)))
            if res:
                ent.setdefault(e["name"], {})[int(w)] = (np.mean([x[0] for x in res]), np.mean([x[1] for x in res]) / len(res), len(res))
    names = [n for n, v in ent.items() if len(v) >= 3]
    mean_res = np.array([np.mean([v[0] for v in ent[n].values()]) for n in names])
    exp_var = np.array([np.mean([v[1] for v in ent[n].values()]) / len(ent[n]) for n in names])
    print(f"  {len(names)} entrants with >=3 weeks. Spread of per-entrant mean chalk residual: sd {mean_res.std():.4f}; "
          f"expected sd if all picked independently from the same f: {np.sqrt(exp_var.mean()):.4f}")
    a = {n: np.mean([ent[n][w][0] for w in (1, 2) if w in ent[n]]) for n in names if any(w in ent[n] for w in (1, 2))}
    b = {n: np.mean([ent[n][w][0] for w in (3, 4, 5) if w in ent[n]]) for n in names if any(w in ent[n] for w in (3, 4, 5))}
    common = [n for n in a if n in b]
    print(f"  persistence: corr(chalk residual weeks 1-2, weeks 3-5) across {len(common)} entrants = {np.corrcoef([a[n] for n in common], [b[n] for n in common])[0, 1]:+.3f}")
    # method of moments for entrant-level logit-scale sd (tau)
    tau_resid = max(0.0, mean_res.var() - exp_var.mean())
    avg_f = np.mean([r["k"] / r["n"] for r in rows])
    print(f"  excess variance -> entrant propensity sd on the probability scale ~ {np.sqrt(tau_resid):.4f} "
          f"(logit scale ~ {np.sqrt(tau_resid) / (avg_f * (1 - avg_f)):.3f})")

    print("\n=== Part 3: Monday-night tiebreaker guesses ===")
    ratios, raw, resid = [], [], []
    for w, d in D.items():
        mnf = [g for g in d["games"] if g["weekday"] == "Monday"]
        if not mnf:
            continue
        line = mnf[-1]["total_line"]
        actual = (mnf[-1]["home_score"] or 0) + (mnf[-1]["away_score"] or 0)
        gs = [e["tiebreak"] for e in d["entrants"] if e["tiebreak"] is not None and d["my_name"].lower() not in e["name"].lower()]
        raw += gs
        ratios += [g / line for g in gs]
        resid += [g - line for g in gs]
        print(f"  week {w}: posted {line}, actual {actual:.0f}; opponents' guesses mean {np.mean(gs):.1f} median {np.median(gs):.1f} sd {np.std(gs):.1f} "
              f"min {min(gs):.0f} max {max(gs):.0f}; mine {[e['tiebreak'] for e in d['entrants'] if d['my_name'].lower() in e['name'].lower()]}")
    print(f"  pooled: guess - posted total: mean {np.mean(resid):+.1f}, sd {np.std(resid):.1f}; guess/posted: mean {np.mean(ratios):.3f}, sd {np.std(ratios):.3f}")
    vals, cnts = np.unique(np.array(raw).astype(int), return_counts=True)
    print("  most common guesses: " + ", ".join(f"{v}x{c}" for v, c in sorted(zip(vals, cnts), key=lambda t: -t[1])[:8]))


if __name__ == "__main__":
    main()
