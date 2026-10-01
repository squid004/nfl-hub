"""v11: did the v8 convex fit (log-loss + L2, non-negative) leave a better AUC on the table?

v8 solves a convex problem exactly (L-BFGS-B on a convex objective can't get stuck in a local
optimum), so the weight space is already fully spanned FOR THE OBJECTIVE IT FITS: regularized
cross-entropy. But we display/care about win-prediction AUC, not log-loss, and those optima
aren't guaranteed to coincide. This is the one check a convex solver can't give us for free.

Method: sample thousands of non-negative weight vectors from the 8-dim simplex (Dirichlet,
alpha=1 = uniform), apply each directly to the SAME walk-forward-standardized, pre-oriented
features v8 uses (precomputed once -- this is why thousands of samples cost almost nothing:
no per-sample refit, just a matmul + roc_auc_score), and see where the production weights
(nflhub/sources/team_ratings.py POWER_WEIGHTS) land in that distribution. Also perturbs the
production weights with small multiplicative noise to check it sits on a broad plateau rather
than a fragile peak (AUC is scale-invariant to a linear decision function, so no renormalizing
is needed to compare against the simplex samples).

Run: python research/edge_signal_test_v11_weight_search.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import edge_signal_test_v8_power_weights as v8

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nflhub.sources import team_ratings

N_SAMPLES = 5000
SEED = 42


def build_walkforward_matrix(rows: list[dict], cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Standardize each walk-forward split with its own train mean/std (no leakage), then
    concatenate all test splits into one matrix/vector pair. A FIXED weight vector's decision
    score is then a single matmul over the whole thing -- no per-candidate refit needed."""
    seasons = sorted({r["season"] for r in rows})
    X_parts, y_parts = [], []
    for test_season in seasons:
        if test_season < v8.TEST_START_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        if len(train) < 100 or not test:
            continue
        X_train = np.array([[r[c] for c in cols] for r in train])
        X_test = np.array([[r[c] for c in cols] for r in test])
        y_test = np.array([r["home_win"] for r in test])
        mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
        sd[sd == 0] = 1.0
        X_parts.append((X_test - mu) / sd)
        y_parts.append(y_test)
    return np.vstack(X_parts), np.concatenate(y_parts)


def auc_for_weights(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    scores = X @ w  # no intercept: AUC/ranking is invariant to a constant shift
    return roc_auc_score(y, scores)


def main():
    print("Building dataset (cached from v8)...", file=sys.stderr)
    rows = v8.build_dataset()
    cols = [f"{m}_diff" for m in v8.RATING_METRICS]
    X, y = build_walkforward_matrix(rows, cols)
    print(f"{X.shape[0]} walk-forward test-season rows, {X.shape[1]} oriented features\n")

    prod_w = np.array([team_ratings.POWER_WEIGHTS[m] for m in v8.RATING_METRICS])
    prod_auc = auc_for_weights(X, y, prod_w)
    print(f"Production weights (v8 convex fit, L2=0.3): walk-forward AUC = {prod_auc:.4f}")
    print("  " + ", ".join(f"{m}={w:.4f}" for m, w in zip(v8.RATING_METRICS, prod_w)))

    rng = np.random.default_rng(SEED)
    samples = rng.dirichlet(np.ones(len(cols)), size=N_SAMPLES)  # uniform over the simplex
    aucs = np.array([auc_for_weights(X, y, w) for w in samples])

    print(f"\n{N_SAMPLES} random non-negative weight vectors (Dirichlet, uniform over simplex):")
    print(f"  min={aucs.min():.4f}  mean={aucs.mean():.4f}  median={np.median(aucs):.4f}  "
          f"max={aucs.max():.4f}  std={aucs.std():.4f}")
    pct_below_prod = float((aucs < prod_auc).mean() * 100)
    print(f"  production weights beat {pct_below_prod:.1f}% of random samples")

    best_idx = np.argsort(aucs)[-5:][::-1]
    print("\nTop 5 random samples by walk-forward AUC:")
    for i in best_idx:
        w_str = ", ".join(f"{m}={w:.3f}" for m, w in zip(v8.RATING_METRICS, samples[i]))
        print(f"  AUC={aucs[i]:.4f}  {w_str}")

    gap = aucs.max() - prod_auc
    print(f"\nBest random sample vs. production: {gap:+.4f} AUC "
          f"({'meaningfully better -- worth investigating' if gap > 0.003 else 'no meaningful gap'})")

    print("\n=== plateau / robustness check: perturb production weights with multiplicative noise ===")
    for noise_scale in (0.1, 0.25, 0.5, 1.0):
        trial_aucs = []
        for _ in range(200):
            perturbed = np.clip(prod_w * (1 + rng.normal(0, noise_scale, size=len(prod_w))), 0, None)
            trial_aucs.append(auc_for_weights(X, y, perturbed))
        trial_aucs = np.array(trial_aucs)
        print(f"  noise={noise_scale:4.2f}  mean AUC={trial_aucs.mean():.4f}  "
              f"min={trial_aucs.min():.4f}  max={trial_aucs.max():.4f}  "
              f"(vs. production {prod_auc:.4f})")


if __name__ == "__main__":
    main()
