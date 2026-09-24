"""V4.3 — discrete-aware calibration.

Receptions
----------
Receptions are a small integer count bounded by targets, and continuous
quantile offsets cannot represent that support. The empirical conditional
distribution was tested before choosing a family:

    targets   1     2     3     4     5     6     7     8     9    10    11
    var/binom 1.00  1.00  1.08  1.07  1.12  1.14  1.23  1.23  1.23  1.11  1.21

Binomial is right at one or two targets and progressively too tight above
that, so a **beta-binomial** is used — binomial with an estimated
overdispersion, which nests the binomial at rho = 0 rather than assuming it.

The predictive distribution is a mixture over the existing target quantiles
(the target uncertainty from V3.1 survives untouched) of BetaBinom(t, p, rho),
with p the calibrated catch rate. Quantiles are read off the mixture CDF on
the integer support, so:

  * receptions are integer-valued
  * receptions <= targets holds by construction, with no clipping backstop
  * no negative values are representable

Carries
-------
Rolling per-quantile conformal, the method validated on rushing yards in V4.2,
with offsets estimated only from weeks strictly before the one being predicted.
An integer-aware variant is evaluated alongside it and selected on evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import betabinom

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
MAX_REC = 20


def estimate_rho(train: pd.DataFrame) -> float:
    """Method-of-moments overdispersion for receptions given targets.

    var = n p (1-p) [1 + (n-1) rho]; solved per target count and pooled by
    sample size. rho = 0 recovers the binomial.
    """
    d = train[(train.targets > 0) & train.receptions.notna()]
    num, den = 0.0, 0.0
    for t, g in d.groupby(d.targets.clip(1, 15)):
        if len(g) < 60 or t < 2:
            continue
        p = g.receptions.sum() / (t * len(g))
        binom_var = t * p * (1 - p)
        if binom_var <= 0:
            continue
        ratio = g.receptions.var() / binom_var
        rho_t = (ratio - 1.0) / (t - 1)
        num += rho_t * len(g)
        den += len(g)
    if den == 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 0.5))


def _target_support(row) -> tuple[np.ndarray, np.ndarray]:
    """Discretised target distribution from the existing target quantiles.

    Each quantile stands for one 20% band of the target distribution, so the
    five values carry equal weight.
    """
    ts = np.array([row[f"q{int(q*100)}_targets"] for q in QUANTILES], dtype=float)
    ts = np.rint(np.clip(ts, 0, None)).astype(int)
    return ts, np.full(len(ts), 1.0 / len(ts))


def reception_pmf(t_vals: np.ndarray, t_wts: np.ndarray, p: float,
                  rho: float, kmax: int = MAX_REC) -> np.ndarray:
    """Mixture of beta-binomials over the target distribution."""
    p = float(np.clip(p, 0.01, 0.99))
    if rho <= 1e-6:
        conc = 1e6
    else:
        conc = max((1.0 - rho) / rho, 1.0)
    a, b = p * conc, (1 - p) * conc
    k = np.arange(0, kmax + 1)
    pmf = np.zeros(kmax + 1)
    for t, w in zip(t_vals, t_wts):
        if t <= 0:
            pmf[0] += w
            continue
        sup = k[k <= t]
        pmf[: len(sup)] += w * betabinom.pmf(sup, int(t), a, b)
    s = pmf.sum()
    return pmf / s if s > 0 else pmf


def discrete_quantiles(pmf: np.ndarray, quantiles=QUANTILES) -> dict:
    """Smallest integer k with CDF(k) >= q."""
    cdf = np.cumsum(pmf)
    out = {}
    for q in quantiles:
        idx = int(np.searchsorted(cdf, q, side="left"))
        out[q] = float(min(idx, len(pmf) - 1))
    return out


def receptions_v43(df: pd.DataFrame, rho: float) -> pd.DataFrame:
    """Integer reception quantiles for one week's rows."""
    out = df.copy()
    m = (out.get("proj_targets", pd.Series(0, index=out.index)) >= 1)
    if not m.any():
        return out
    cols = [f"q{int(q*100)}_receptions" for q in QUANTILES]
    res = {c: out[c].to_numpy(copy=True) for c in cols}
    idx = np.where(m.to_numpy())[0]
    for i in idx:
        row = out.iloc[i]
        t_vals, t_wts = _target_support(row)
        denom = row.get("q50_targets", np.nan)
        p = (row.get("q50_receptions", np.nan) / denom) if denom and denom > 0 else np.nan
        if not np.isfinite(p):
            p = 0.66
        pmf = reception_pmf(t_vals, t_wts, p, rho)
        qs = discrete_quantiles(pmf)
        for q in QUANTILES:
            res[f"q{int(q*100)}_receptions"][i] = qs[q]
    for c in cols:
        out[c] = res[c]
    return out


def rolling_conformal_offsets(history: pd.DataFrame, stat: str,
                              pop_mask: pd.Series) -> dict:
    """Per-quantile offsets from weeks strictly before the target week."""
    d = history[pop_mask & history[stat].notna()]
    out = {}
    for q in QUANTILES:
        col = f"q{int(q*100)}_{stat}"
        if col not in d.columns:
            out[q] = 0.0
            continue
        v = d[[stat, col]].dropna()
        if len(v) < 300:
            out[q] = 0.0
            continue
        out[q] = float(np.quantile(v[stat] - v[col], q))
    return out


def apply_offsets(df: pd.DataFrame, stat: str, offsets: dict,
                  mask: pd.Series, integer: bool = False) -> pd.DataFrame:
    out = df.copy()
    for q in QUANTILES:
        col = f"q{int(q*100)}_{stat}"
        if col not in out.columns:
            continue
        vals = out.loc[mask, col] + offsets.get(q, 0.0)
        vals = np.clip(vals, 0, None)
        if integer:
            vals = np.rint(vals)
        out.loc[mask, col] = vals
    return out
