"""Margin target vs. win/loss target for the composite (4 EPA stats from research/output/team_ratings.csv + points EWMA).
Result on 2026-10-15: tie (LL 0.6351 vs 0.6351, AUC 0.6863 vs 0.6855) -> dropped."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, log_loss
from _common import load_games, points_ewma, RATINGS

tr = pd.read_csv(RATINGS)
g = load_games(2006)
P = pd.DataFrame(points_ewma(g, 0.2, 0.65), columns=["pf", "pa"]); P["game_id"] = g.game_id
m = ["rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed"]
X = pd.DataFrame({k: tr["home_" + k] - tr["away_" + k] for k in m})
for k in m[2:]: X[k] = -X[k]
d = pd.concat([tr[["game_id", "season", "result"]], X], axis=1).merge(P, on="game_id").dropna()
d = d[d.result != 0]; d["y"] = (d.result > 0).astype(int); F = m + ["pf", "pa"]
def fit_logit(A, y, l2=0.003):
    f = lambda w: np.mean(np.logaddexp(0, w[0] + A @ w[1:]) - y * (w[0] + A @ w[1:])) + l2 * np.sum(w[1:] ** 2)
    return minimize(f, np.zeros(A.shape[1] + 1), method="L-BFGS-B").x
pl, pm, ys = [], [], []
for s in range(2011, d.season.max() + 1):
    a, b = d[d.season < s], d[d.season == s]
    if len(b) == 0: continue
    mu, sd = a[F].mean(), a[F].std(); A, B = ((a[F] - mu) / sd).values, ((b[F] - mu) / sd).values
    w = fit_logit(A, a.y.values); pl.append(1 / (1 + np.exp(-(w[0] + B @ w[1:]))))
    Aa = np.c_[np.ones(len(A)), A]; beta = np.linalg.solve(Aa.T @ Aa + 1e-3 * np.eye(Aa.shape[1]), Aa.T @ a.result.values)
    pm.append(norm.cdf((np.c_[np.ones(len(B)), B] @ beta) / (a.result.values - Aa @ beta).std())); ys.append(b.y.values)
pl, pm, ys = map(np.concatenate, (pl, pm, ys))
print(f"win/loss logit: AUC {roc_auc_score(ys, pl):.4f} LL {log_loss(ys, pl):.4f}")
print(f"margin OLS    : AUC {roc_auc_score(ys, pm):.4f} LL {log_loss(ys, pm):.4f}")
