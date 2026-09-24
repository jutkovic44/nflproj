"""Model scorecard.

Turns accuracy into one number out of 100, then shows every component that went
into it so the number can be argued with.

The score is deliberately NOT "percent correct" — that would be meaningless for
continuous projections. It is a skill score: how much better than a naive
trailing average, how well calibrated the touchdown probabilities are, and how
much week-to-week variance the projections actually track.

  55 points  skill vs the naive baseline   (15% MAE reduction scores full marks)
  30 points  touchdown calibration          (mean bucket error, 10pts = zero)
  15 points  rank correlation               (0.70 scores full marks)

A score of 0 means the model is no better than a trailing 3-game average.
Roughly 55-70 is what an honest NFL projection system looks like. Anything
above 85 means something is leaking.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from . import data as D

GRADED_STATS = ["receiving_yards", "rushing_yards", "passing_yards",
                "receptions", "targets", "carries"]
SKILL_CEILING = 0.15      # 15% MAE reduction vs baseline = full marks
CAL_FLOOR = 0.10          # 10 percentage points of calibration error = zero
CORR_CEILING = 0.70


def _actuals(seasons) -> pd.DataFrame:
    pw = D.load_player_weeks(seasons)
    pw["total_tds"] = pw["rushing_tds"] + pw["receiving_tds"]
    pw["gorder"] = pw.season * 100 + pw.week
    pw = pw.sort_values(["player_id", "gorder"])
    g = pw.groupby("player_id", sort=False)
    for stat in GRADED_STATS + ["total_tds"]:
        pw[f"base_{stat}"] = (g[stat].shift(1).groupby(pw["player_id"], sort=False)
                              .rolling(3, min_periods=1).mean().reset_index(level=0, drop=True))
    return pw


def grade_archive(pattern: str = "runs/projections_*.csv", seasons=(2025, 2026)) -> pd.DataFrame:
    """Join archived live projections to what actually happened.

    This is the real track record — projections frozen before kickoff, not
    refitted after the fact.
    """
    files = sorted(glob.glob(pattern)) + sorted(glob.glob("projections_*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if {"player_display_name", "season", "week"} <= set(d.columns):
                d["_src"] = os.path.basename(f)
                frames.append(d)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    proj = pd.concat(frames, ignore_index=True)
    proj = proj.drop_duplicates(subset=["season", "week", "player_display_name"], keep="last")

    act = _actuals(seasons)
    keep = ["season", "week", "player_display_name", "position", "total_tds",
            "base_total_tds"] + GRADED_STATS + [f"base_{s}" for s in GRADED_STATS]
    merged = proj.merge(act[[c for c in keep if c in act.columns]],
                        on=["season", "week", "player_display_name"],
                        how="inner", suffixes=("", "_act"))
    return merged


def contributors(df: pd.DataFrame, min_snap: float = 0.4) -> pd.DataFrame:
    """Score on players who actually play. Deep-bench zeros are trivially easy
    to predict and flatter every model that includes them."""
    if "snap_pct_r3" in df.columns:
        return df[df["snap_pct_r3"].fillna(0) >= min_snap]
    return df


def score(df: pd.DataFrame, min_rows: int = 40) -> dict:
    """Compute the 0-100 score and its components from graded rows."""
    if df is None or len(df) < min_rows:
        return {"score": None, "n": 0 if df is None else len(df),
                "note": "not enough graded games yet"}

    per_stat, gains, corrs = [], [], []
    for stat in GRADED_STATS:
        pcol, bcol = f"proj_{stat}", f"base_{stat}"
        if pcol not in df.columns or bcol not in df.columns or stat not in df.columns:
            continue
        ok = df[[stat, pcol, bcol]].notna().all(axis=1)
        if ok.sum() < 30:
            continue
        y, p, b = df.loc[ok, stat], df.loc[ok, pcol], df.loc[ok, bcol]
        mae, bmae = float(np.abs(y - p).mean()), float(np.abs(y - b).mean())
        gain = (bmae - mae) / bmae if bmae > 0 else 0.0
        corr = float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else np.nan
        per_stat.append({"stat": stat, "n": int(ok.sum()), "mae": round(mae, 2),
                         "baseline_mae": round(bmae, 2), "gain_pct": round(gain * 100, 1),
                         "corr": None if np.isnan(corr) else round(corr, 3)})
        gains.append(gain)
        if not np.isnan(corr):
            corrs.append(corr)

    cal = td_calibration_error(df)
    skill_idx = float(np.clip(np.mean(gains) / SKILL_CEILING, 0, 1)) if gains else 0.0
    cal_idx = float(np.clip(1 - (cal["mean_abs_error"] / CAL_FLOOR), 0, 1)) if cal else 0.0
    corr_idx = float(np.clip(np.mean(corrs) / CORR_CEILING, 0, 1)) if corrs else 0.0

    total = round(55 * skill_idx + 30 * cal_idx + 15 * corr_idx)
    return {
        "score": int(total), "grade": _grade(total), "n": int(len(df)),
        "components": {
            "skill_vs_baseline": {"points": round(55 * skill_idx, 1), "max": 55,
                                  "mean_gain_pct": round(float(np.mean(gains)) * 100, 1) if gains else 0.0},
            "td_calibration": {"points": round(30 * cal_idx, 1), "max": 30,
                               "mean_abs_error_pts": round(cal["mean_abs_error"] * 100, 1) if cal else None},
            "correlation": {"points": round(15 * corr_idx, 1), "max": 15,
                            "mean_corr": round(float(np.mean(corrs)), 3) if corrs else None},
        },
        "per_stat": per_stat,
    }


def _grade(total: int) -> str:
    if total >= 80: return "Suspiciously strong — check for leakage"
    if total >= 65: return "Strong"
    if total >= 50: return "Solid"
    if total >= 35: return "Marginal"
    if total >= 20: return "Weak"
    return "No better than a trailing average"


def td_calibration_error(df: pd.DataFrame) -> dict | None:
    col = "proj_anytime_td_prob"
    if col not in df.columns or "total_tds" not in df.columns:
        return None
    d = df.dropna(subset=[col]).copy()
    if len(d) < 50:
        return None
    d["hit"] = (d["total_tds"] > 0).astype(int)
    d["b"] = pd.cut(d[col], [0, .1, .2, .3, .5, 1.0], include_lowest=True)
    g = d.groupby("b", observed=True).agg(n=("hit", "size"), pred=(col, "mean"), act=("hit", "mean"))
    g = g[g.n >= 15]
    if not len(g):
        return None
    err = float((g.pred - g.act).abs().mul(g.n).sum() / g.n.sum())
    return {"mean_abs_error": err,
            "buckets": [{"range": str(i), "n": int(r.n), "predicted": round(r.pred, 3),
                         "actual": round(r.act, 3)} for i, r in g.iterrows()]}


def by_week(df: pd.DataFrame) -> list[dict]:
    """Score each completed week separately, for the trend line."""
    out = []
    for (season, week), sub in df.groupby(["season", "week"]):
        s = score(sub, min_rows=30)
        if s.get("score") is not None:
            out.append({"season": int(season), "week": int(week),
                        "score": s["score"], "n": s["n"]})
    return sorted(out, key=lambda r: (r["season"], r["week"]))


def write_json(df: pd.DataFrame, path: str = "accuracy.json") -> dict:
    graded = contributors(df)
    payload = score(graded)
    payload["by_week"] = by_week(graded) if payload.get("score") is not None else []
    payload["calibration"] = td_calibration_error(graded)
    payload["population"] = ("players at 40%+ snap share" if "snap_pct_r3" in df.columns
                             else "all projected players")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return payload
