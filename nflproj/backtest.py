"""Walk-forward backtest.

For each target week, the model is trained only on games played before that
week, then asked to predict it cold. Retraining happens every REFIT_EVERY weeks
to keep runtime sane; between refits the most recent model is reused, which is
if anything slightly pessimistic about accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as M
from .models import SKILL_TARGETS, QB_TARGETS, TD_TARGET

REFIT_EVERY = 4
MIN_TRAIN_ROWS = 4000


def walk_forward(df: pd.DataFrame, start_season: int = 2022, refit_every: int = REFIT_EVERY,
                 verbose: bool = True) -> pd.DataFrame:
    weeks = (df[df.season >= start_season][["season", "week", "gorder"]]
             .drop_duplicates().sort_values("gorder"))
    preds, fitted, since = [], None, 10 ** 9

    for _, row in weeks.iterrows():
        gorder = row.gorder
        train = df[df.gorder < gorder]
        test = df[(df.gorder == gorder) & M.eligible(df)]
        if len(train) < MIN_TRAIN_ROWS or not len(test):
            continue
        if fitted is None or since >= refit_every:
            fitted = M.fit_models(train)
            since = 0
            if verbose:
                print(f"  refit @ {int(row.season)} wk {int(row.week)} "
                      f"({len(train):,} rows)", flush=True)
        since += 1
        p = M.predict(fitted, test)
        p = M.baseline_predict(p)
        preds.append(p)

    return pd.concat(preds, ignore_index=True)


def _metrics(sub: pd.DataFrame, stat: str) -> dict | None:
    y = sub[stat]
    p = sub[f"proj_{stat}"]
    b = sub[f"base_{stat}"]
    ok = y.notna() & p.notna() & b.notna()
    if ok.sum() < 50:
        return None
    y, p, b = y[ok], p[ok], b[ok]
    mae, bmae = np.abs(y - p).mean(), np.abs(y - b).mean()
    return {
        "stat": stat, "n": int(ok.sum()),
        "model_mae": round(mae, 2), "baseline_mae": round(bmae, 2),
        "improvement_%": round(100 * (bmae - mae) / bmae, 1),
        "model_rmse": round(float(np.sqrt(((y - p) ** 2).mean())), 2),
        "corr": round(float(np.corrcoef(y, p)[0, 1]), 3) if y.std() > 0 else np.nan,
        "bias": round(float((p - y).mean()), 2),
    }


def accuracy_report(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "QB": (preds[preds.position == "QB"], QB_TARGETS + ["passing_tds"]),
        "RB": (preds[preds.position == "RB"], ["carries", "rushing_yards", "targets", "receptions", "receiving_yards"]),
        "WR": (preds[preds.position == "WR"], ["targets", "receptions", "receiving_yards"]),
        "TE": (preds[preds.position == "TE"], ["targets", "receptions", "receiving_yards"]),
    }
    for pos, (sub, stats) in groups.items():
        for stat in dict.fromkeys(stats):
            m = _metrics(sub, stat)
            if m:
                rows.append({"position": pos, **m})
    return pd.DataFrame(rows)


def volume_filtered_report(preds: pd.DataFrame, min_r3_snaps: float = 0.4) -> pd.DataFrame:
    """Accuracy restricted to real contributors, which is what you actually bet.
    Deep bench players are easy to predict (zeros) and inflate the headline."""
    sub = preds[preds["snap_pct_r3"].fillna(0) >= min_r3_snaps]
    return accuracy_report(sub)


def td_calibration(preds: pd.DataFrame, bins=(0, .05, .1, .15, .2, .25, .3, .4, .5, 1.0)) -> pd.DataFrame:
    p = preds.dropna(subset=["proj_anytime_td_prob"]).copy()
    p["actual_td"] = (p[TD_TARGET] > 0).astype(int)
    p["bucket"] = pd.cut(p["proj_anytime_td_prob"], bins=list(bins), include_lowest=True)
    out = p.groupby("bucket", observed=True).agg(
        n=("actual_td", "size"),
        predicted=("proj_anytime_td_prob", "mean"),
        actual=("actual_td", "mean"),
    ).reset_index()
    out["predicted"] = out["predicted"].round(3)
    out["actual"] = out["actual"].round(3)
    out["gap"] = (out["actual"] - out["predicted"]).round(3)
    return out


def season_summary(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, sub in preds.groupby("season"):
        for stat in ["receiving_yards", "rushing_yards", "passing_yards"]:
            m = _metrics(sub, stat)
            if m:
                rows.append({"season": int(season), **m})
    return pd.DataFrame(rows)
