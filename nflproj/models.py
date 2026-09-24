"""Model layer.

Design: one gradient-boosted model per target stat, trained separately for
pass-catchers/runners (RB/WR/TE) and quarterbacks. Touchdowns are modelled as a
Poisson rate, which converts cleanly to P(anytime TD) = 1 - exp(-lambda).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from .features import feature_columns

SKILL_TARGETS = ["targets", "receptions", "receiving_yards", "carries", "rushing_yards"]
QB_TARGETS = ["attempts", "passing_yards", "passing_tds", "passing_interceptions",
              "carries", "rushing_yards"]
TD_TARGET = "total_tds"

HGB_KW = dict(max_iter=160, learning_rate=0.09, max_depth=5, min_samples_leaf=40,
              l2_regularization=1.0, max_features=0.7, early_stopping=False, random_state=7)


def _fit_one(X, y, poisson=False):
    kw = dict(HGB_KW)
    if poisson:
        kw["loss"] = "poisson"
    m = HistGradientBoostingRegressor(**kw)
    m.fit(X, y)
    return m


def eligible(df: pd.DataFrame) -> pd.Series:
    """Rows worth modelling: player has prior history and was not ruled out."""
    return (df["games_played_prior"] >= 1) & (df["status_out"] == 0)


def fit_models(train: pd.DataFrame) -> dict:
    feats = feature_columns(train)
    tr = train[eligible(train)]
    models = {"features": feats}

    skill = tr[tr.position.isin(["RB", "WR", "TE"])]
    Xs = skill[feats]
    for t in SKILL_TARGETS:
        models[("skill", t)] = _fit_one(Xs, skill[t])
    models[("skill", TD_TARGET)] = _fit_one(Xs, skill[TD_TARGET], poisson=True)

    qb = tr[tr.position == "QB"]
    Xq = qb[feats]
    for t in QB_TARGETS:
        models[("qb", t)] = _fit_one(Xq, qb[t])
    models[("qb", TD_TARGET)] = _fit_one(Xq, qb[TD_TARGET], poisson=True)

    models["td_calibrator"] = _fit_td_calibrator(tr, feats)
    return models


def _fit_td_calibrator(tr: pd.DataFrame, feats: list[str]):
    """Isotonic map from raw Poisson probability to observed TD rate.

    Fitted on a chronological holdout (last 25% of the training window) so the
    calibration is learned from predictions the TD model did not train on.
    """
    skill = tr[tr.position.isin(["RB", "WR", "TE"])].sort_values("gorder")
    if len(skill) < 3000:
        return None
    cut = int(len(skill) * 0.75)
    fit_part, hold = skill.iloc[:cut], skill.iloc[cut:]
    if len(hold) < 500:
        return None
    m = _fit_one(fit_part[feats], fit_part[TD_TARGET], poisson=True)
    raw = 1 - np.exp(-np.clip(m.predict(hold[feats]), 0, None))
    actual = (hold[TD_TARGET] > 0).astype(int)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raw, actual)
    return iso


def predict(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    feats = models["features"]
    out = df.copy()
    for t in set(SKILL_TARGETS + QB_TARGETS + [TD_TARGET]):
        out[f"proj_{t}"] = np.nan

    for group, targets in (("skill", SKILL_TARGETS), ("qb", QB_TARGETS)):
        mask = out.position.isin(["RB", "WR", "TE"]) if group == "skill" else out.position.eq("QB")
        if not mask.any():
            continue
        X = out.loc[mask, feats]
        for t in targets + [TD_TARGET]:
            key = (group, t)
            if key in models:
                out.loc[mask, f"proj_{t}"] = np.clip(models[key].predict(X), 0, None)

    # anytime-TD probability from the Poisson rate, isotonically calibrated
    raw = 1 - np.exp(-out["proj_total_tds"].fillna(0))
    out["proj_anytime_td_prob_raw"] = raw
    iso = models.get("td_calibrator")
    out["proj_anytime_td_prob"] = iso.predict(raw) if iso is not None else raw
    # keep receptions coherent with targets
    out["proj_receptions"] = np.minimum(out["proj_receptions"], out["proj_targets"])
    return out


def baseline_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Naive baseline: the player's trailing 4-game (r3+r8 blend) average.
    Any model that cannot beat this is not worth running."""
    out = df.copy()
    for t in set(SKILL_TARGETS + QB_TARGETS + [TD_TARGET]):
        src = f"{t}_r3"
        out[f"base_{t}"] = out[src] if src in out.columns else np.nan
    out["base_anytime_td_prob"] = 1 - np.exp(-out["base_total_tds"].fillna(0))
    return out
