"""Totals model vs. the closing over/under (EPA + points ratings, optional actual wind).
Result on 2026-10-15: 49.9% O/U hit rate over 3,628 games; no edge bin reached 52.4% in either half -> dropped."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from _common import load_games, RATINGS, GAMES

tr = pd.read_csv(RATINGS); gm = pd.read_csv(GAMES).set_index("game_id"); g = load_games(2006)
# raw (non-differenced) points levels per side
st, lv = {}, {}
for x in g.itertuples():
    for t in (x.home_team, x.away_team):
        s = st.get(t)
        if s and s[2] != x.season: s[0] = .65 * s[0] + .35 * 22; s[1] = .65 * s[1] + .35 * 22; s[2] = x.season
    lv[x.game_id] = [*(st.get(x.home_team, [np.nan] * 2)[:2]), *(st.get(x.away_team, [np.nan] * 2)[:2])]
    for t, pf, pa in ((x.home_team, x.home_score, x.away_score), (x.away_team, x.away_score, x.home_score)):
        s = st.get(t)
        if not s: st[t] = [pf, pa, x.season]
        else: s[0] = .2 * pf + .8 * s[0]; s[1] = .2 * pa + .8 * s[1]
L = pd.DataFrame.from_dict(lv, orient="index", columns=["h_pf", "h_pa", "a_pf", "a_pa"])
d = tr.join(L, on="game_id")
for c in ("total_line", "total", "wind", "roof"): d[c] = d.game_id.map(gm[c])
F = [c for c in tr.columns if c.endswith("_epa") or c.endswith("_allowed")] + ["h_pf", "h_pa", "a_pf", "a_pa"]
d = d.dropna(subset=F + ["total_line", "total"])
d["windx"] = np.where(d.roof.isin(["outdoors", "open"]), d.wind.fillna(0), 0)
d["ou"] = np.sign(d.total - d.total_line)
for feats, nm in ((F, "stats"), (F + ["windx"], "stats+wind")):
    rows = []
    for s in range(2011, d.season.max() + 1):
        a, b = d[d.season < s], d[d.season == s]
        w = np.linalg.lstsq(np.c_[np.ones(len(a)), a[feats].values], a.total.values, rcond=None)[0]
        pred = np.c_[np.ones(len(b)), b[feats].values] @ w
        rows.append(pd.DataFrame({"season": s, "edge": pred - b.total_line.values, "ou": b.ou.values}))
    o = pd.concat(rows); o = o[o.ou != 0]
    for th in (0, 2, 4, 6):
        mk = o.edge.abs() >= th; h1 = mk & (o.season <= 2018); h2 = mk & (o.season > 2018)
        hit = lambda m_: (np.sign(o.edge[m_]) == o.ou[m_]).mean()
        print(f"{nm:10s} |edge|>={th}: n={mk.sum():4d} hit {hit(mk):.3f}  <=2018 {hit(h1):.3f}  >2018 {hit(h2):.3f}")
