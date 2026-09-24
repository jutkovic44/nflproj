"""V4.5 — conditional dispersion calibration for the reception distribution.

Offline only. V3.1 and V4.3 are untouched; every transformation here acts on
already-produced V4.3 quantiles.

Dispersion scaling keeps the centre fixed:

    q'_p = P50 + s * (q_p - P50)

s is the *scaled conformal* factor: for each calibration row the minimal scale
that would have contained the outcome is

    s_i = (y - P50) / (q90 - P50)   if y > P50
          (P50 - y) / (P50 - q10)   otherwise

and s for a group is the empirical quantile of s_i at the nominal level. s = 1
means the interval is already calibrated, s < 1 tightens, s > 1 widens — the
direction is learned, never imposed. Estimated strictly on weeks before the
one being predicted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QP = [0.10, 0.25, 0.50, 0.75, 0.90]


def minimal_scale(y, lo, mid, hi):
    up = np.maximum(hi - mid, 1e-6)
    dn = np.maximum(mid - lo, 1e-6)
    return np.where(y > mid, (y - mid) / up, (mid - y) / dn)


def group_factors(cal: pd.DataFrame, stat: str, group_col: str,
                  nominal_outer=0.80, nominal_inner=0.50, min_n=150) -> dict:
    """Outer (P10-P90) and inner (P25-P75) scale factors per group."""
    out = {}
    lo, mid, hi = (cal[f"q10_{stat}"], cal[f"q50_{stat}"], cal[f"q90_{stat}"])
    lo2, hi2 = cal[f"q25_{stat}"], cal[f"q75_{stat}"]
    s_out = minimal_scale(cal[stat].to_numpy(), lo.to_numpy(), mid.to_numpy(), hi.to_numpy())
    s_in = minimal_scale(cal[stat].to_numpy(), lo2.to_numpy(), mid.to_numpy(), hi2.to_numpy())
    for g, idx in cal.groupby(group_col, observed=True).groups.items():
        m = cal.index.isin(idx)
        if m.sum() < min_n:
            out[g] = (1.0, 1.0)
            continue
        out[g] = (float(np.quantile(s_out[m], nominal_outer)),
                  float(np.quantile(s_in[m], nominal_inner)))
    return out


def apply_scaling(df: pd.DataFrame, stat: str, group_col: str, factors: dict,
                  cap=(0.5, 2.0)) -> pd.DataFrame:
    out = df.copy()
    mid = out[f"q50_{stat}"]
    so = out[group_col].map(lambda g: factors.get(g, (1.0, 1.0))[0]).astype(float).clip(*cap)
    si = out[group_col].map(lambda g: factors.get(g, (1.0, 1.0))[1]).astype(float).clip(*cap)
    for q, s in [(0.10, so), (0.90, so), (0.25, si), (0.75, si)]:
        c = f"q{int(q*100)}_{stat}"
        out[c] = np.clip(mid + s * (out[c] - mid), 0, None)
    # structural: receptions can never exceed targets (established V4.3 rule)
    for q in QP:
        rc, tc = f"q{int(q*100)}_{stat}", f"q{int(q*100)}_targets"
        if stat == "receptions" and tc in out.columns:
            out[rc] = np.minimum(out[rc], out[tc])
    cols = [f"q{int(q*100)}_{stat}" for q in QP]
    v = out[cols].to_numpy(copy=True)
    ok = ~np.isnan(v).any(axis=1)
    v[ok] = np.sort(v[ok], axis=1)
    out[cols] = v
    return out


def continuous_factor(cal: pd.DataFrame, stat: str, xcol: str,
                      nominal=0.80, nbins=10) -> tuple[float, float]:
    """Linear s(x) fitted to binned scaled-conformal factors. Deliberately a
    two-parameter fit — no search, no ML."""
    x = cal[xcol].to_numpy()
    s = minimal_scale(cal[stat].to_numpy(), cal[f"q10_{stat}"].to_numpy(),
                      cal[f"q50_{stat}"].to_numpy(), cal[f"q90_{stat}"].to_numpy())
    qs = pd.qcut(pd.Series(x), nbins, duplicates="drop")
    d = pd.DataFrame({"x": x, "s": s, "b": qs}).groupby("b", observed=True).agg(
        x=("x", "mean"), s=("s", lambda v: np.quantile(v, nominal)), n=("s", "size"))
    d = d[d.n >= 100]
    if len(d) < 3:
        return (1.0, 0.0)
    A = np.vstack([np.ones(len(d)), d.x.to_numpy()]).T
    coef, *_ = np.linalg.lstsq(A, d.s.to_numpy(), rcond=None)
    return float(coef[0]), float(coef[1])


def apply_continuous(df: pd.DataFrame, stat: str, xcol: str, coef, cap=(0.5, 2.0),
                     inner_coef=None) -> pd.DataFrame:
    out = df.copy()
    mid = out[f"q50_{stat}"]
    so = np.clip(coef[0] + coef[1] * out[xcol].to_numpy(), *cap)
    si = so if inner_coef is None else np.clip(inner_coef[0] + inner_coef[1] * out[xcol].to_numpy(), *cap)
    for q, s in [(0.10, so), (0.90, so), (0.25, si), (0.75, si)]:
        c = f"q{int(q*100)}_{stat}"
        out[c] = np.clip(mid + s * (out[c] - mid), 0, None)
    for q in QP:
        rc, tc = f"q{int(q*100)}_{stat}", f"q{int(q*100)}_targets"
        if stat == "receptions" and tc in out.columns:
            out[rc] = np.minimum(out[rc], out[tc])
    cols = [f"q{int(q*100)}_{stat}" for q in QP]
    v = out[cols].to_numpy(copy=True)
    ok = ~np.isnan(v).any(axis=1)
    v[ok] = np.sort(v[ok], axis=1)
    out[cols] = v
    return out
