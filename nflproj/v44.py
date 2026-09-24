"""V4.4 — decision-usefulness measurement. No model changes.

Threshold probabilities are read off the V4.3 predictive distribution rather
than fitted: the five quantiles define points on the CDF, which is interpolated
piecewise-linearly between them, with a 0.5 continuity correction for the
integer support of receptions. No new model is trained, so nothing here can
leak — the probabilities are a transformation of walk-forward predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QP = [0.10, 0.25, 0.50, 0.75, 0.90]


def cdf_from_quantiles(qvals: np.ndarray, x: float) -> float:
    """P(Y <= x) implied by a quantile curve.

    Ties (repeated quantile values, common at zero) take the highest
    probability attached to that value. Below the lowest quantile the CDF is
    interpolated toward 0 at the support floor; above the highest it decays
    toward 1 over one inter-quantile width.
    """
    v = np.asarray(qvals, dtype=float)
    if np.isnan(v).any():
        return np.nan
    if x >= v[-1]:
        span = max(v[-1] - v[-2], 1e-6)
        return float(min(1.0, 0.90 + 0.10 * (x - v[-1]) / span))
    if x < v[0]:
        return float(max(0.0, 0.10 * (x + 0.5) / max(v[0] + 0.5, 1e-6)))
    for i in range(len(v) - 1):
        if v[i] <= x < v[i + 1]:
            if v[i + 1] - v[i] < 1e-9:
                return QP[i + 1]
            frac = (x - v[i]) / (v[i + 1] - v[i])
            return float(QP[i] + frac * (QP[i + 1] - QP[i]))
    return float(QP[-1])


def threshold_prob(df: pd.DataFrame, stat: str, k: float,
                   discrete: bool = True) -> np.ndarray:
    """P(Y >= k) for each row."""
    cols = [f"q{int(q*100)}_{stat}" for q in QP]
    M = df[cols].to_numpy(dtype=float)
    x = (k - 0.5) if discrete else k
    return np.array([1.0 - cdf_from_quantiles(row, x) for row in M])


def reliability(p: np.ndarray, y: np.ndarray, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)):
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    d["bin"] = pd.cut(d.p, bins=list(bins), include_lowest=True)
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), pred=("p", "mean"),
                                            obs=("y", "mean")).reset_index()
    return g[g.n >= 50]


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    ok = ~np.isnan(p)
    p, y = p[ok], y[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * (~pos).sum()))
