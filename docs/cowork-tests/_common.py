"""Shared helpers for the Cowork proxy tests (2026-10-15).

Proxy data only: nflverse games.csv (research/cache/games.csv) plus the older
research/output/team_ratings.csv export (4 EPA stats, 0.65 carryover). These are
NOT the production compute_ratings game_log; treat results as leads to confirm.
Run from the repo root:  python docs/cowork-tests/<script>.py
"""
import os
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GAMES = os.path.join(REPO, "research", "cache", "games.csv")
RATINGS = os.path.join(REPO, "research", "output", "team_ratings.csv")
RELOC = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def load_games(first_season=2002):
    g = pd.read_csv(GAMES)
    g = g[(g.season >= first_season) & g.result.notna()].sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return g.replace({"home_team": RELOC, "away_team": RELOC})


def points_ewma(g, alpha, carry=0.65, league_mean=22.0):
    """Pre-game EWMA of points for/against per team. Returns home-minus-away
    [points_for diff, points_allowed diff (sign flipped so higher = better)]."""
    st, out = {}, np.full((len(g), 4), np.nan)
    for i, x in enumerate(g.itertuples()):
        for j, t in enumerate((x.home_team, x.away_team)):
            s = st.get(t)
            if s and s[2] != x.season:
                s[0] = carry * s[0] + (1 - carry) * league_mean
                s[1] = carry * s[1] + (1 - carry) * league_mean
                s[2] = x.season
            if s:
                out[i, 2 * j:2 * j + 2] = s[:2]
        for t, pf, pa in ((x.home_team, x.home_score, x.away_score), (x.away_team, x.away_score, x.home_score)):
            s = st.get(t)
            if not s:
                st[t] = [pf, pa, x.season]
            else:
                s[0] = alpha * pf + (1 - alpha) * s[0]
                s[1] = alpha * pa + (1 - alpha) * s[1]
    return np.c_[out[:, 0] - out[:, 2], out[:, 3] - out[:, 1]]


def elo(g, K=20, HFA=48, REV=1 / 3):
    """FiveThirtyEight-style Elo with margin-of-victory multiplier; returns pre-game home edge in Elo points."""
    r, last, pre = {}, {}, np.zeros(len(g))
    for i, x in enumerate(g.itertuples()):
        for t in (x.home_team, x.away_team):
            r.setdefault(t, 1500.0)
            if last.get(t, x.season) != x.season:
                r[t] = r[t] * (1 - REV) + 1505 * REV
            last[t] = x.season
        d = r[x.home_team] + (0 if x.location == "Neutral" else HFA) - r[x.away_team]
        pre[i] = d
        e = 1 / (1 + 10 ** (-d / 400))
        s = 1 if x.result > 0 else (0 if x.result < 0 else 0.5)
        mov = np.log(abs(x.result) + 1) * 2.2 / ((d if x.result > 0 else -d) * 0.001 + 2.2)
        r[x.home_team] += K * mov * (s - e)
        r[x.away_team] -= K * mov * (s - e)
    return pre


def walk_forward(g, X, ok, first_test=2011, last_test=2025, train_from=2007):
    """Train logistic on seasons < s, predict season s. Returns (p, y, season)."""
    from sklearn.linear_model import LogisticRegression
    y = (g.result > 0).values
    ps, ys, ss = [], [], []
    for s in range(first_test, last_test + 1):
        tr = ok & (g.season < s).values & (g.season >= train_from).values
        te = ok & (g.season == s).values
        if te.sum() == 0:
            continue
        m = LogisticRegression(C=10).fit(X[tr], y[tr])
        ps.append(m.predict_proba(X[te])[:, 1]); ys.append(y[te]); ss.append(np.full(te.sum(), s))
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(ss)


def report(name, p, y, s, split=2018):
    from sklearn.metrics import roc_auc_score, log_loss
    h = s <= split
    print(f"{name:36s} AUC {roc_auc_score(y, p):.4f}  LL {log_loss(y, p):.4f} | "
          f"<= {split} {roc_auc_score(y[h], p[h]):.4f}  > {split} {roc_auc_score(y[~h], p[~h]):.4f}")


def boot_dll(y, p_base, p_new, n=2000, seed=0):
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = ll(p_base) - ll(p_new)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), (bs <= 0).mean()
