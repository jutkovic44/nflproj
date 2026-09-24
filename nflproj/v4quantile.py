"""V4 — conditional outcome distributions on top of the V3.1 point estimate.

V3.1 is not rebuilt or forked. Its pregame state, role selection, team
opportunity, allocation and efficiency stages produce the point projection
exactly as validated; V4 adds a quantile layer whose features are the V3.1
feature set *plus* the V3.1 projections themselves. The distribution is
therefore conditioned on the validated forecast rather than derived from a
parallel model.

    V3.1 point projection ─┐
    V3.1 features ─────────┴─► quantile GBMs ─► P10 P25 P50 P75 P90 + mean

Each quantile is a separate gradient booster trained with pinball loss. No
distributional assumption is imposed: the spread is learned from the features,
which is what allows an honest test of whether role transitions and returning
players genuinely carry more uncertainty (they are never told to).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

SKILL_TARGETS = ["targets", "carries", "receptions", "receiving_yards", "rushing_yards"]
QB_TARGETS = ["attempts", "passing_yards"]

PROJ_COLS = ["proj_targets", "proj_carries", "proj_receptions", "proj_receiving_yards",
             "proj_rushing_yards", "proj_attempts", "proj_passing_yards",
             "proj_total_tds", "share_target", "share_rush",
             "team_targets_budget", "team_rush_budget", "team_att_budget"]

QGB = dict(max_iter=120, learning_rate=0.09, max_depth=5, min_samples_leaf=40,
           l2_regularization=1.0, max_features=0.6, early_stopping=False,
           random_state=7)


def feature_list(df: pd.DataFrame) -> list[str]:
    """V3.1 features plus its own projections. Everything here was computed at
    or before the prediction cutoff by the V3.1 harness."""
    from .features import feature_columns
    from .hierarchy import eb_columns
    base = [c for c in feature_columns(df, enforce=False) if c in df.columns]
    base += [c for c in eb_columns() if c in df.columns]
    base += [c for c in PROJ_COLS if c in df.columns]
    return list(dict.fromkeys(base))


def _fit(X, y, quantile: float | None):
    kw = dict(QGB)
    if quantile is None:
        kw["loss"] = "squared_error"
    else:
        kw["loss"] = "quantile"
        kw["quantile"] = quantile
    return HistGradientBoostingRegressor(**kw).fit(X, y)


def fit_models(train: pd.DataFrame, feats: list[str], targets: list[str],
               mask: pd.Series) -> dict:
    tr = train[mask]
    models = {}
    for t in targets:
        ok = tr[t].notna()
        if ok.sum() < 500:
            continue
        X, y = tr.loc[ok, feats], tr.loc[ok, t]
        for q in QUANTILES:
            models[(t, q)] = _fit(X, y, q)
        models[(t, "mean")] = _fit(X, y, None)
    return models


def predict(models: dict, test: pd.DataFrame, feats: list[str],
            targets: list[str], mask: pd.Series) -> pd.DataFrame:
    out = test.copy()
    for t in targets:
        for q in QUANTILES + ["mean"]:
            col = f"q{int(q*100)}_{t}" if q != "mean" else f"mean_{t}"
            if col not in out.columns:
                out[col] = np.nan
    if not mask.any():
        return out
    X = out.loc[mask, feats]
    for t in targets:
        if (t, "mean") not in models:
            continue
        for q in QUANTILES:
            out.loc[mask, f"q{int(q*100)}_{t}"] = np.clip(models[(t, q)].predict(X), 0, None)
        out.loc[mask, f"mean_{t}"] = np.clip(models[(t, "mean")].predict(X), 0, None)
    return out


def crossing_report(df: pd.DataFrame, targets: list[str]) -> dict:
    """Quantile crossing must be reported, never silently sorted away."""
    rep = {}
    for t in targets:
        cols = [f"q{int(q*100)}_{t}" for q in QUANTILES]
        if not all(c in df.columns for c in cols):
            continue
        v = df[cols].dropna()
        if not len(v):
            continue
        diffs = np.diff(v.to_numpy(), axis=1)
        crossed = (diffs < -1e-9).any(axis=1)
        rep[t] = {"rows": int(len(v)), "crossings": int(crossed.sum()),
                  "pct": round(100 * crossed.mean(), 2)}
    return rep


def enforce_monotone(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Monotone rearrangement (Chernozhukov, Fernandez-Val & Galichon 2010).

    Sorting the estimated quantile curve is a principled correction with a
    proven guarantee: the rearranged curve is never a worse estimate of the
    true quantile function than the crossed one. Applied only after the
    crossing rate has been measured and reported.
    """
    out = df.copy()
    for t in targets:
        cols = [f"q{int(q*100)}_{t}" for q in QUANTILES]
        if not all(c in out.columns for c in cols):
            continue
        vals = out[cols].to_numpy(copy=True)
        ok = ~np.isnan(vals).any(axis=1)
        vals[ok] = np.sort(vals[ok], axis=1)
        out[cols] = vals
    return out


def pinball(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    d = y - yhat
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage_report(df: pd.DataFrame, targets: list[str],
                    group: pd.Series | None = None) -> pd.DataFrame:
    """Empirical coverage per quantile — the honest test of a distribution."""
    rows = []
    g = group if group is not None else pd.Series("all", index=df.index)
    for key, sub in df.groupby(g):
        for t in targets:
            for q in QUANTILES:
                col = f"q{int(q*100)}_{t}"
                if col not in sub.columns:
                    continue
                v = sub[[t, col]].dropna()
                if len(v) < 100:
                    continue
                cov = float((v[t] <= v[col]).mean())
                rows.append({"group": key, "stat": t, "nominal": q,
                             "actual": round(cov, 4),
                             "error_pp": round(100 * (cov - q), 2),
                             "pinball": round(pinball(v[t].to_numpy(),
                                                      v[col].to_numpy(), q), 4),
                             "n": int(len(v))})
    return pd.DataFrame(rows)


def interval_report(df: pd.DataFrame, targets: list[str],
                    group: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    g = group if group is not None else pd.Series("all", index=df.index)
    for key, sub in df.groupby(g):
        for t in targets:
            for lo, hi, nominal in [(10, 90, 0.80), (25, 75, 0.50)]:
                lc, hc = f"q{lo}_{t}", f"q{hi}_{t}"
                if lc not in sub.columns or hc not in sub.columns:
                    continue
                v = sub[[t, lc, hc]].dropna()
                if len(v) < 100:
                    continue
                inside = ((v[t] >= v[lc]) & (v[t] <= v[hc])).mean()
                width = (v[hc] - v[lc])
                rows.append({"group": key, "stat": t,
                             "interval": f"P{lo}-P{hi}", "nominal": nominal,
                             "coverage": round(float(inside), 4),
                             "error_pp": round(100 * (float(inside) - nominal), 2),
                             "avg_width": round(float(width.mean()), 2),
                             "median_width": round(float(width.median()), 2),
                             "n": int(len(v))})
    return pd.DataFrame(rows)


# ============================================================================
# V4.1 — calibration fixes. V4 functions above are left intact so that
# V4-PRE-FIX remains reproducible.
# ============================================================================

# statistics whose zeros are structural rather than informative, with the
# population in which the statistic is a live quantity. The mask uses V3.1's
# own allocation, which is available at the prediction cutoff.
CONDITIONAL_POP = {
    "carries": lambda d: d["proj_carries"] >= 2,
    "rushing_yards": lambda d: d["proj_carries"] >= 2,
    "targets": lambda d: d["proj_targets"] >= 1,
    "receiving_yards": lambda d: d["proj_targets"] >= 1,
    "attempts": lambda d: d["proj_attempts"] >= 10,
    "passing_yards": lambda d: d["proj_attempts"] >= 10,
}

CQR_PAIRS = [(0.10, 0.90, 0.80), (0.25, 0.75, 0.50)]
CALIB_WEEKS = 6   # most recent training weeks held out for conformal calibration


def _conformity_offset(y, lo, hi, nominal: float) -> float:
    """Split-conformal offset (Romano, Patterson & Candès 2019, CQR).

    The score is how far the outcome falls outside the predicted interval;
    its (1-alpha) empirical quantile is added symmetrically to the bounds.
    Estimated only on held-out rows inside the training window, so no future
    observation is used, and it self-cancels when the interval is already
    calibrated (offset -> 0 or negative, tightening rather than widening).
    """
    score = np.maximum(lo - y, y - hi)
    if not len(score):
        return 0.0
    # finite-sample corrected level, capped at 1
    level = min(1.0, nominal * (1 + 1.0 / len(score)))
    return float(np.quantile(score, level))


def fit_models_v41(train: pd.DataFrame, feats: list[str], targets: list[str],
                   mask: pd.Series) -> dict:
    """Conditional-population quantile models plus conformal offsets.

    Three changes from V4:
      1. each statistic is trained on the population where it is live
      2. receptions are modelled as catch rate conditional on targets
      3. interval offsets are conformally calibrated inside the training window
    """
    tr = train[mask]
    models: dict = {}
    for t in targets:
        pop = CONDITIONAL_POP.get(t)
        sub = tr[pop(tr)] if pop is not None else tr
        sub = sub[sub[t].notna()]
        if len(sub) < 400:
            continue
        # chronological split: the most recent weeks calibrate, the rest fit
        cut = sub.gorder.quantile(0.8)
        fit_part, cal_part = sub[sub.gorder <= cut], sub[sub.gorder > cut]
        if len(cal_part) < 150:
            fit_part, cal_part = sub, sub.iloc[:0]
        X, y = fit_part[feats], fit_part[t]
        for q in QUANTILES:
            models[(t, q)] = _fit(X, y, q)
        models[(t, "mean")] = _fit(X, y, None)
        models[(t, "pop")] = pop

        offsets = {}
        if len(cal_part):
            Xc, yc = cal_part[feats], cal_part[t].to_numpy()
            for lo_q, hi_q, nominal in CQR_PAIRS:
                lo = models[(t, lo_q)].predict(Xc)
                hi = models[(t, hi_q)].predict(Xc)
                offsets[(lo_q, hi_q)] = _conformity_offset(yc, lo, hi, nominal)
        models[(t, "cqr")] = offsets
    return models


def fit_catch_rate(train: pd.DataFrame, feats: list[str], mask: pd.Series) -> dict:
    """Catch rate conditional on targets, as quantiles.

    receptions = targets x catch rate, and catch rate lies in [0, 1], so
    building receptions from the matching target quantile guarantees
    receptions <= targets by construction rather than by clipping. Target
    uncertainty and catch uncertainty both survive.
    """
    tr = train[mask & (train.targets > 0)]
    if len(tr) < 400:
        return {}
    y = (tr.receptions / tr.targets).clip(0, 1)
    X = tr[feats]
    m = {q: _fit(X, y, q) for q in QUANTILES}
    m["mean"] = _fit(X, y, None)
    return m


def fit_v42_conformal(train: pd.DataFrame, feats: list[str], mask: pd.Series,
                      sk_models: dict, catch_models: dict) -> dict:
    """Calibration offsets for the two statistics V4.1 diagnostics flagged."""
    tr = train[mask]
    cut = tr.gorder.quantile(0.8)
    cal = tr[tr.gorder > cut]
    if len(cal) < 200:
        return {}
    out = {}

    rec = cal[(cal.proj_targets >= 1) & cal.receptions.notna()]
    if len(rec) > 200 and catch_models:
        X = rec[feats]
        preds = {}
        for q in QUANTILES:
            tq = np.clip(sk_models[("targets", q)].predict(X), 0, None)
            cr = np.clip(catch_models[q].predict(X), 0.0, 1.0)
            preds[q] = tq * cr
        out["receptions"] = fit_quantile_conformal(rec.receptions.to_numpy(), preds)

    run = cal[(cal.proj_carries >= 2) & cal.rushing_yards.notna()]
    if len(run) > 200 and ("rushing_yards", 0.5) in sk_models:
        X = run[feats]
        preds = {q: np.clip(sk_models[("rushing_yards", q)].predict(X), 0, None)
                 for q in QUANTILES}
        out["rushing_yards"] = fit_quantile_conformal(run.rushing_yards.to_numpy(), preds)
    return out


def apply_v42(out: pd.DataFrame, offsets: dict, mask: pd.Series) -> tuple[pd.DataFrame, int]:
    """Apply the offsets, then re-assert receptions <= targets.

    The structural constraint is enforced AFTER calibration and any violation
    it has to correct is counted and reported — it is not a substitute for
    calibration.
    """
    violations = 0
    for stat, off in offsets.items():
        pop = (out["proj_targets"] >= 1) if stat == "receptions" else (out["proj_carries"] >= 2)
        m = mask & pop
        if not m.any():
            continue
        for q in QUANTILES:
            col = f"q{int(q*100)}_{stat}"
            if col in out.columns:
                out.loc[m, col] = np.clip(out.loc[m, col] + off.get(q, 0.0), 0, None)
    if "receptions" in offsets:
        m = mask & (out["proj_targets"] >= 1)
        for q in QUANTILES:
            rc, tc = f"q{int(q*100)}_receptions", f"q{int(q*100)}_targets"
            if rc in out.columns and tc in out.columns:
                bad = m & (out[rc] > out[tc] + 1e-9)
                violations += int(bad.sum())
                out.loc[bad, rc] = out.loc[bad, tc]
    return out, violations


def predict_v41(models: dict, test: pd.DataFrame, feats: list[str],
                targets: list[str], mask: pd.Series,
                catch_models: dict | None = None) -> pd.DataFrame:
    out = test.copy()
    for t in targets + (["receptions"] if catch_models else []):
        for q in QUANTILES + ["mean"]:
            col = f"q{int(q*100)}_{t}" if q != "mean" else f"mean_{t}"
            if col not in out.columns:
                out[col] = np.nan
    if not mask.any():
        return out

    for t in targets:
        if (t, "mean") not in models:
            continue
        pop = models.get((t, "pop"))
        m = mask & (pop(out) if pop is not None else True)
        if not m.any():
            continue
        X = out.loc[m, feats]
        raw = {q: models[(t, q)].predict(X) for q in QUANTILES}
        off = models.get((t, "cqr"), {})
        for lo_q, hi_q, _ in CQR_PAIRS:
            d = off.get((lo_q, hi_q), 0.0)
            raw[lo_q] = raw[lo_q] - d
            raw[hi_q] = raw[hi_q] + d
        for q in QUANTILES:
            out.loc[m, f"q{int(q*100)}_{t}"] = np.clip(raw[q], 0, None)
        out.loc[m, f"mean_{t}"] = np.clip(models[(t, "mean")].predict(X), 0, None)

    if catch_models:
        m = mask & (out["proj_targets"] >= 1)
        if m.any():
            X = out.loc[m, feats]
            for q in QUANTILES:
                cr = np.clip(catch_models[q].predict(X), 0.0, 1.0)
                tq = out.loc[m, f"q{int(q*100)}_targets"]
                out.loc[m, f"q{int(q*100)}_receptions"] = tq.to_numpy() * cr
            out.loc[m, "mean_receptions"] = (
                out.loc[m, "mean_targets"].to_numpy()
                * np.clip(catch_models["mean"].predict(X), 0.0, 1.0))
    return out


# ============================================================================
# V4.2 — targeted per-quantile conformal recalibration.
# Applied ONLY where V4.1 diagnostics showed a systematic marginal miss:
#   receptions    — conservative product of two quantiles (over-covering)
#   rushing_yards — heavy upper tail from breakaway runs (under-covering)
# QB keeps its paired CQR unchanged. Targets, carries and receiving yards are
# left alone: their V4.1 calibration is already within ~2pp.
# ============================================================================

V42_STATS = ["receptions", "rushing_yards"]


def fit_quantile_conformal(y: np.ndarray, preds: dict) -> dict:
    """Per-quantile conformal offsets (marginal quantile recalibration).

    For quantile q the offset is the empirical q-quantile of the residual
    y - qhat on held-out calibration rows. Adding it makes marginal coverage
    exact on that sample, and the offset is naturally signed: negative where
    the interval is too wide, positive where it is too thin. Estimated only
    inside the training window.
    """
    out = {}
    for q, yhat in preds.items():
        resid = y - yhat
        if not len(resid):
            out[q] = 0.0
            continue
        out[q] = float(np.quantile(resid, q))
    return out
