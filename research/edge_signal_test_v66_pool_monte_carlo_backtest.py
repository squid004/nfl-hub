"""v66: backtest of the pool Monte Carlo (nflhub/sources/pool_mc.py) on 2026 weeks 1-5.

Question: does the expected-dollar optimizer produce a different, better card than what the user played,
than the app's FADE/CHALK/NO_PLAY card, and than "always take the favorite" -- and, run sequentially
(re-solved after each game day with that day's results and the field's revealed picks), would a late pick
or tiebreaker change have changed the user's weekly result?

Data: research/cache/pool_weeks.json (from research/pool_data_extract.py): every entrant's final picks,
tiebreak guess, weekly/season correct counts, the week's games (closing moneylines, posted total, scores,
game day) and the app's recommendation log. Market probability = de-vigged closing moneyline.

Leakage control: the opponent model is fit leave-one-week-out (pick-share parameters, each entrant's chalk
offset and skill come from the OTHER weeks), because 5 weeks cannot support a strict walk-forward. That is
still a mild optimism for the opponent model, never for the market or for the realized results.

Windows (sequential mode), per week, by game day: the state after each day except the last is "every game of
that day is final and its picks are revealed; later days' games are open". For a normal week that is after
Thursday (the Sunday card, decided before Sunday's first kickoff) and after Sunday (the Monday pick and the
tiebreaker, decided before Monday kickoff).

Reported, per week: model P(win week) and E[correct] for the user's actual card, all-favorites, the app's
card and the optimizer's; REALIZED correct picks and weekly-win share for each of those under the real
opponents' picks, results and tiebreakers; then for each sequential window the picks/tiebreak the optimizer
would change and the realized result of following it. Five weeks is a tiny sample: nothing here is a test of
significance, it is a sanity check that the engine behaves and a record of what it would have said.

Run: PYTHONPATH=. python research/edge_signal_test_v66_pool_monte_carlo_backtest.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nflhub.sources import pool_mc as pm  # noqa: E402

CACHE = os.path.join(os.path.dirname(__file__), "cache", "pool_weeks.json")
N = 20000


def american(o):
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)


def sig(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def build_week(d, w):
    """-> games (PoolGame, all unlocked/unresolved), my entrant row, opponents rows (dicts)."""
    games = []
    for g in d["games"]:
        ph = american(g["ml_home"]) / (american(g["ml_home"]) + american(g["ml_away"]))
        pg = pm.PoolGame(game_id=g["game_id"], home=g["home"], away=g["away"], p_home=ph, lock_key=g["gameday"],
                         total_line=g["total_line"])
        pg.national_fav_pct = d["national_pct"].get(pg.fav)
        pg._home_won = g["home_score"] > g["away_score"]
        pg._total = g["home_score"] + g["away_score"]
        games.append(pg)
    me = next(e for e in d["entrants"] if d["my_name"].lower() in e["name"].lower())
    opps = [e for e in d["entrants"] if e is not me]
    return games, me, opps


def picks_dict(entrant, games):
    by_team = {}
    for g in games:
        by_team[g.home] = g.game_id
        by_team[g.away] = g.game_id
    out = {g.game_id: None for g in games}
    for t in entrant["picks"]:
        if t in by_team:
            out[by_team[t]] = t
    return out


def realized_share(games, opps, my_correct, my_guess, total):
    ocor = [e["weekly_correct"] for e in opps]
    best = max(ocor)
    if my_correct > best:
        return 1.0
    if my_correct < best:
        return 0.0
    dists = [abs(e["tiebreak"] - total) if e["tiebreak"] is not None else np.inf for e, c in zip(opps, ocor) if c == best]
    dm = min(dists)
    md = abs(my_guess - total)
    if md < dm:
        return 1.0
    if md == dm:
        return 1.0 / (dists.count(dm) + 1)
    return 0.0


def card_correct(games, card):
    return sum(1 for g in games if card.get(g.game_id) == (g.home if g._home_won else g.away))


def fit_pick_params(weeks_data, exclude_week=None):
    rows = []
    for w, (games, me, opps) in weeks_data.items():
        if w == exclude_week:
            continue
        for g in games:
            k = sum(1 for e in opps if g.fav in e["picks"])
            n = sum(1 for e in opps if g.home in e["picks"] or g.away in e["picks"])
            rows.append((g.p_fav, g.national_fav_pct if g.national_fav_pct is not None else g.p_fav, k, n))
    a = np.array(rows, float)
    X = np.column_stack([np.ones(len(a)), logit(a[:, 0]), logit(a[:, 1])])

    def obj(w):
        p = np.clip(sig(X @ w), 1e-6, 1 - 1e-6)
        return -(a[:, 2] * np.log(p) + (a[:, 3] - a[:, 2]) * np.log(1 - p)).sum() / a[:, 3].sum() + 1e-4 * np.sum(w[1:] ** 2)

    return tuple(minimize(obj, np.zeros(3), method="BFGS").x)


def main():
    D = json.load(open(CACHE))
    weeks_data = {int(w): build_week(d, int(w)) for w, d in D.items()}
    allp = fit_pick_params(weeks_data)
    print(f"all-data pick-share fit (pool_mc.PICK_A/B/C): A={allp[0]:+.2f} B={allp[1]:.2f} C={allp[2]:.2f}\n")

    tot = {k: [] for k in ("p_win_actual", "p_win_allfav", "p_win_app", "p_win_opt", "r_actual", "r_allfav", "r_app", "r_opt",
                            "c_actual", "c_allfav", "c_app", "c_opt")}
    seq_rows = []
    calib_rows = []
    for w, (games, me_row, opps_rows) in sorted(weeks_data.items()):
        d = D[str(w)]
        params = fit_pick_params(weeks_data, exclude_week=w)
        # leave-one-week-out entrant offsets/skills
        names = [e["name"] for e in opps_rows]
        hist = []
        for w2, (g2, me2, o2) in weeks_data.items():
            if w2 == w:
                continue
            hist.append((g2, {e["name"]: picks_dict(e, g2) for e in o2}))
        offsets = pm.estimate_offsets(hist, names, a=params[0], b=params[1], c=params[2])
        skill = {}
        for n_ in names:
            vals = [next((e["weekly_correct"] for e in weeks_data[w2][2] if e["name"] == n_), None) for w2 in weeks_data if w2 != w]
            vals = [v for v in vals if v is not None]
            skill[n_] = float(np.mean(vals)) if vals else None
        pool_mean = float(np.mean([v for v in skill.values() if v is not None] or [10.0]))
        mine_before = (me_row["season_correct"] or 0) - (me_row["weekly_correct"] or 0)
        my_skill = float(np.mean([weeks_data[w2][1]["weekly_correct"] for w2 in weeks_data if w2 != w]))
        total = games[[g.lock_key for g in games].index(max(g.lock_key for g in games))]._total if False else None
        mnf = max((g for g in games if g.total_line is not None), key=lambda g: g.lock_key)
        total = mnf._total

        def mk_state(day_cut):
            """Games with lock_key <= day_cut are final + revealed; later ones open."""
            gs = []
            for g in games:
                c = pm.PoolGame(g.game_id, g.home, g.away, g.p_home, g.lock_key, g.total_line, g.national_fav_pct)
                if day_cut is not None and g.lock_key <= day_cut:
                    c.locked, c.home_won = True, g._home_won
                gs.append(c)
            opps = []
            for e in opps_rows:
                pd_ = picks_dict(e, games)
                revealed = {gid: t for gid, t in pd_.items() if day_cut is not None and next(x for x in games if x.game_id == gid).lock_key <= day_cut}
                opps.append(pm.Entrant(e["name"], offset=offsets[e["name"]], picks=revealed))
            me_pd = picks_dict(me_row, games)
            me_rev = {gid: t for gid, t in me_pd.items() if day_cut is not None and next(x for x in games if x.game_id == gid).lock_key <= day_cut}
            return gs, opps, pm.Entrant(me_row["name"], picks=me_rev)

        # ---- static (pre-week) ----
        gs, opps, me = mk_state(None)
        sim_chk = pm.Sim(gs, opps, me, N, w, params)
        act_cor = np.array([e["weekly_correct"] for e in opps_rows])
        calib_rows.append((w, sim_chk.opp_correct.mean(), act_cor.mean(), sim_chk.max_opp.mean(), act_cor.max(),
                           float(np.mean(sim_chk.max_opp >= act_cor.max())), sim_chk.opp_correct.std(1).mean(), act_cor.std()))
        res = pm.optimize(gs, opps, me, n=N, seed=w, params=params)
        actual_card = {gid: t for gid, t in picks_dict(me_row, games).items() if t is not None}
        allfav = {g.game_id: g.fav for g in games}
        rec_by_team = {}
        for r_ in d["recommendation_log"].values():
            rec_by_team[r_["favorite_team"]] = r_
            rec_by_team[r_["underdog_team"]] = r_
        app_card = {}
        for g in games:
            r_ = rec_by_team.get(g.fav)
            fade = bool(r_) and str(r_.get("recommendation", "")).upper() == "FADE"
            app_card[g.game_id] = (g.home if g.fav == g.away else g.away) if fade else g.fav
        my_guess = me_row["tiebreak"]
        ev = lambda c, gu: pm.evaluate_card(gs, opps, me, c, gu, n=N, seed=w, params=params)  # noqa: E731
        e_act, e_all, e_app = ev(actual_card, my_guess), ev(allfav, my_guess), ev(app_card, my_guess)
        e_opt = ev(res.card, res.guess)
        r = {"actual": (actual_card, my_guess), "allfav": (allfav, my_guess), "app": (app_card, my_guess), "opt": (res.card, res.guess)}
        out = {}
        for k, (c, gu) in r.items():
            cc = card_correct(games, c)
            out[k] = (cc, realized_share(games, opps_rows, cc, gu, total))
        print(f"=== Week {w}: {len(games)} games, {len(opps_rows)} opponents, optimizer flips {len(res.steps)} ===")
        print(f"  {'card':10s} {'model P(win)':>12} {'model E[corr]':>13} | {'realized correct':>16} {'realized win share':>18}")
        for k, e in (("actual", e_act), ("allfav", e_all), ("app", e_app), ("opt", {"p_win": res.p_win, "expected_correct": res.expected_correct})):
            print(f"  {k:10s} {e['p_win']:12.3f} {e['expected_correct']:13.2f} | {out[k][0]:16d} {out[k][1]:18.3f}")
            tot["p_win_" + k].append(e["p_win"]); tot["r_" + k].append(out[k][1]); tot["c_" + k].append(out[k][0])
        flips = [(g.home + "/" + g.away, res.card[g.game_id]) for g in gs if res.card[g.game_id] != g.fav]
        print(f"  optimizer's contrarian picks vs favorite: {flips}; tiebreak {res.guess} (user guessed {my_guess}, posted {mnf.total_line}, actual {total})")

        # ---- sequential windows ----
        days = sorted({g.lock_key for g in games})
        for di, day in enumerate(days[:-1]):
            gs2, opps2, me2 = mk_state(day)
            if not [g for g in gs2 if not g.locked]:
                continue
            nxt = days[di + 1]
            res2 = pm.optimize(gs2, opps2, me2, n=N, seed=100 + w, params=params, start=None)
            # user's actual card, evaluated by the model from the same state
            cur = {g.game_id: actual_card.get(g.game_id, g.fav) for g in gs2}
            e_cur = pm.evaluate_card(gs2, opps2, me2, cur, my_guess, n=N, seed=100 + w, params=params)
            changed = [(g.home + "/" + g.away, actual_card.get(g.game_id), res2.card[g.game_id]) for g in gs2
                       if not g.locked and actual_card.get(g.game_id) != res2.card[g.game_id]]
            cc_act = card_correct(games, actual_card)
            cc_rec = card_correct(games, res2.card)
            r_act = realized_share(games, opps_rows, cc_act, my_guess, total)
            r_rec = realized_share(games, opps_rows, cc_rec, res2.guess, total)
            seq_rows.append((w, day, len(changed), e_cur["p_win"], res2.p_win, r_act, r_rec, cc_act, cc_rec))
            print(f"  [window after {day}] open games {len([g for g in gs2 if not g.locked])}; model P(win): actual {e_cur['p_win']:.3f} -> re-optimized {res2.p_win:.3f}; "
                  f"changes {len(changed)} {changed[:6]}; tiebreak {res2.guess} (user {my_guess}); realized correct {cc_act}->{cc_rec}, win share {r_act:.2f}->{r_rec:.2f}")
        print()

    print("=== Opponent/outcome model check: simulated vs actual opponent scores (pre-week model, leave-one-week-out opponent params) ===")
    print(f"  {'week':>4} {'mean score sim/act':>20} {'max score sim/act':>19} {'P(sim max >= actual max)':>26} {'within-week sd sim/act':>24}")
    for w, ms, ma, xs, xa, pge, ss_, sa in calib_rows:
        print(f"  {w:4d} {ms:10.2f}/{ma:<9.2f} {xs:9.2f}/{xa:<9.0f} {pge:26.3f} {ss_:12.2f}/{sa:<11.2f}")
    print(f"  mean over weeks: score {np.mean([r[1] for r in calib_rows]):.2f} vs {np.mean([r[2] for r in calib_rows]):.2f}; "
          f"winning score {np.mean([r[3] for r in calib_rows]):.2f} vs {np.mean([r[4] for r in calib_rows]):.2f}")
    print()
    print("=== Summary over weeks 1-5 (model values are each week's own simulation) ===")
    for k in ("actual", "allfav", "app", "opt"):
        print(f"  {k:8s} mean model P(win) {np.mean(tot['p_win_' + k]):.3f}   mean realized correct {np.mean(tot['c_' + k]):.2f}   realized weekly wins {sum(tot['r_' + k]):.2f} of {len(tot['r_' + k])}")
    print("\n=== Sequential windows ===")
    print(f"  {len(seq_rows)} windows; windows where the optimizer would change at least one open pick or the tiebreaker call: "
          f"{sum(1 for r in seq_rows if r[2] > 0)}")
    print(f"  mean model P(win): actual card {np.mean([r[3] for r in seq_rows]):.3f} -> re-optimized {np.mean([r[4] for r in seq_rows]):.3f}")
    print(f"  realized weekly-win share summed over windows' weeks: actual {sum(r[5] for r in seq_rows):.2f} vs re-optimized {sum(r[6] for r in seq_rows):.2f}")


if __name__ == "__main__":
    main()
