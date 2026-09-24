"""Feature-group ablation.

Permutation importance misleads when features are correlated, so group
membership is tested by removing whole groups and rerunning the identical
walk-forward evaluation. Every variant sees the same weeks, the same refit
cadence and the same rows.

The 2026 live sample is never evaluated here.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import availability as AV
from .features import candidate_columns
from .models import HGB_KW, TD_TARGET

SKILL_TARGETS = ["targets", "receptions", "receiving_yards", "carries", "rushing_yards"]
QB_TARGETS = ["attempts", "passing_yards"]

# V1 kept market + observed weather; the contract now blocks them, so the
# leaked variant is reconstructed explicitly for comparison only.
VARIANTS = {
    "ALL_LEAKED":            {"leaked": True,  "exclude": set()},
    "NO_MARKET":             {"leaked": True,  "exclude": {"market"}},
    "NO_WEATHER":            {"leaked": True,  "exclude": {"weather"}},
    "NO_MARKET_NO_WEATHER":  {"leaked": False, "exclude": set()},
    "CLEAN_NO_TEAM_ENV":     {"leaked": False, "exclude": {"team_env"}},
}


def features_for(df: pd.DataFrame, variant: str) -> list[str]:
    spec = VARIANTS[variant]
    cand = candidate_columns(df)
    AV.register_team_env([c for c in cand if c.startswith(("team_", "opp_team_"))])
    # candidate_columns already filters to shifted windows; this is a second
    # gate so an ablation variant cannot reintroduce a same-week aggregate
    cand = [c for c in cand if not (c.startswith(("team_", "opp_team_"))
                                    and not c.endswith(("_r3", "_r8")))]
    AV.register_rollings([c for c in cand if any(
        c.endswith(s) for s in ("_r3", "_r8", "_exp"))])
    feats = AV.allowed(cand, exclude_groups=spec["exclude"])
    if spec["leaked"]:
        for name in ("spread_line", "total_line", "implied_team_total", "temp", "wind", "is_dome"):
            f = AV.REGISTRY.get(name)
            if name in cand and f and f.group not in spec["exclude"]:
                feats.append(name)
    return list(dict.fromkeys(feats))


def _eligible(df: pd.DataFrame) -> pd.Series:
    return (df["games_played_prior"] >= 1) & (df["status_out"] == 0)


def _fit(X, y, poisson=False):
    kw = dict(HGB_KW)
    if poisson:
        kw["loss"] = "poisson"
    return HistGradientBoostingRegressor(**kw).fit(X, y)


def walk_forward(df: pd.DataFrame, feats: list[str], group: str,
                 seasons: tuple[int, ...], refit_every: int = 6) -> pd.DataFrame:
    """group: 'skill' or 'qb'. Returns per-row predictions for the eval seasons."""
    targets = SKILL_TARGETS if group == "skill" else QB_TARGETS
    mask_pos = df.position.isin(["RB", "WR", "TE"]) if group == "skill" else df.position.eq("QB")
    weeks = (df[df.season.isin(seasons)][["season", "week", "gorder"]]
             .drop_duplicates().sort_values("gorder"))
    out, models, since = [], None, 10 ** 9

    for _, row in weeks.iterrows():
        train = df[(df.gorder < row.gorder) & mask_pos & _eligible(df)]
        test = df[(df.gorder == row.gorder) & mask_pos & _eligible(df)]
        if len(train) < 1500 or not len(test):
            continue
        if models is None or since >= refit_every:
            models = {t: _fit(train[feats], train[t]) for t in targets}
            models[TD_TARGET] = _fit(train[feats], train[TD_TARGET], poisson=True)
            since = 0
        since += 1
        p = test.copy()
        for t in targets:
            p[f"proj_{t}"] = np.clip(models[t].predict(test[feats]), 0, None)
        lam = np.clip(models[TD_TARGET].predict(test[feats]), 0, None)
        p["proj_anytime_td_prob"] = 1 - np.exp(-lam)
        for t in targets + [TD_TARGET]:
            p[f"base_{t}"] = test[f"{t}_r3"]
        p["base_anytime_td_prob"] = 1 - np.exp(-test[f"{TD_TARGET}_r3"].fillna(0))
        out.append(p)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(preds: pd.DataFrame, group: str, min_snap: float = 0.4) -> dict:
    """Contributors only — the population V1 was scored on."""
    if not len(preds):
        return {}
    sub = preds[preds["snap_pct_r3"].fillna(0) >= min_snap]
    targets = SKILL_TARGETS if group == "skill" else QB_TARGETS
    rows, gains, corrs = [], [], []
    for t in targets:
        ok = sub[[t, f"proj_{t}", f"base_{t}"]].notna().all(axis=1)
        if ok.sum() < 50:
            continue
        y, p, b = sub.loc[ok, t], sub.loc[ok, f"proj_{t}"], sub.loc[ok, f"base_{t}"]
        mae, bmae = float(np.abs(y - p).mean()), float(np.abs(y - b).mean())
        corr = float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 else np.nan
        rows.append({"stat": t, "n": int(ok.sum()), "mae": round(mae, 2),
                     "baseline_mae": round(bmae, 2),
                     "gain_pct": round(100 * (bmae - mae) / bmae, 2),
                     "corr": round(corr, 3), "bias": round(float((p - y).mean()), 2)})
        gains.append((bmae - mae) / bmae)
        if not np.isnan(corr):
            corrs.append(corr)

    cal = None
    if "proj_anytime_td_prob" in sub.columns:
        d = sub.dropna(subset=["proj_anytime_td_prob"]).copy()
        d["hit"] = (d[TD_TARGET] > 0).astype(int)
        d["b"] = pd.cut(d["proj_anytime_td_prob"], [0, .1, .2, .3, .5, 1.0], include_lowest=True)
        g = d.groupby("b", observed=True).agg(n=("hit", "size"),
                                              pred=("proj_anytime_td_prob", "mean"),
                                              act=("hit", "mean"))
        g = g[g.n >= 15]
        if len(g):
            cal = round(float((g.pred - g.act).abs().mul(g.n).sum() / g.n.sum()) * 100, 2)

    return {"per_stat": rows, "n_rows": int(len(sub)),
            "mean_gain_pct": round(float(np.mean(gains)) * 100, 2) if gains else None,
            "mean_corr": round(float(np.mean(corrs)), 3) if corrs else None,
            "td_calibration_error_pts": cal}


def run_variant(df: pd.DataFrame, variant: str, group: str,
                seasons=(2022, 2023, 2024, 2025), out_dir: str = "ablation") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{variant}__{group}.json")
    if os.path.exists(path):
        return json.load(open(path))
    feats = features_for(df, variant)
    preds = walk_forward(df, feats, group, seasons)
    res = {"variant": variant, "group": group, "n_features": len(feats),
           "eval_seasons": list(seasons), **evaluate(preds, group)}
    json.dump(res, open(path, "w"), indent=1)
    return res
