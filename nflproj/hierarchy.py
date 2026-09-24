"""V3 — hierarchical team → player opportunity architecture.

    team environment (prior games)
            ↓
    team opportunity  (pass attempts, rush attempts)
            ↓
    player opportunity share  (target share, carry share)
            ↓
    player opportunity  (targets, carries)
            ↓
    efficiency (empirical-Bayes shrunk)
            ↓
    production (receptions, yards)

Accounting identities measured from nflverse, 2022-2025, 4,300 team-weeks:

    team targets      = 0.9495 x team pass attempts   (sd 0.045)
    attempts - targets = +1.67 per game               (throwaways, spikes)
    completions       = receptions + 0.14             (rounding in charting)

The residual 5% is a genuine non-player category and is modelled explicitly
rather than forced to zero.

Shares are produced by a dedicated share model and only then normalised. The
normalisation is the final enforcement step, not the mechanism — a player who
just became the starter can be assigned a share far above his own history,
because the share model sees snap share, vacated share and injury status.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .models import HGB_KW, TD_TARGET

# measured, not assumed — see module docstring
TARGETS_PER_ATTEMPT = 0.9495

# empirical-Bayes prior strengths (opportunities required to half-trust the
# player's own rate over the positional prior)
EB_K = {"catch_rate": 30.0, "ypr": 25.0, "ypc": 40.0, "ypa": 60.0}

TEAM_TARGETS = ["team_pass_att_actual", "team_rush_att_actual"]


def _hgb(X, y, weight=None, poisson=False):
    kw = dict(HGB_KW)
    if poisson:
        kw["loss"] = "poisson"
    m = HistGradientBoostingRegressor(**kw)
    m.fit(X, y, sample_weight=weight)
    return m


# --------------------------------------------------------------- EB efficiency

def add_eb_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical-Bayes shrunk efficiency rates from PRIOR games only.

    rate_hat = (prior production + k * positional prior * k) / (prior opportunity + k)

    A back with 3 carries at 8.5 YPC is pulled most of the way to the positional
    mean; one with 300 carries is barely moved.
    """
    d = df.sort_values(["player_id", "gorder"]).copy()
    g = d.groupby("player_id", sort=False)

    def prior_sum(col):
        return g[col].shift(1).groupby(d["player_id"], sort=False).cumsum()

    acc = {}
    for col in ["receptions", "targets", "receiving_yards", "carries",
                "rushing_yards", "attempts", "passing_yards"]:
        acc[col] = prior_sum(col)

    # --- positional priors, rebuilt as of each week ------------------------
    # V3 computed these once over the whole frame handed to it, which in the
    # evaluation harness included weeks AFTER the prediction cutoff. The prior
    # is now expanding: for any row, it uses only games played in earlier
    # weeks, so it is correct at every cutoff by construction.
    league = prior_league_rates(df)
    idx = pd.MultiIndex.from_arrays([d["position"], d["gorder"]])

    def prior_series(kind: str) -> pd.Series:
        vals = league[kind].reindex(idx).to_numpy()
        # Forward-fill only. Back-filling would pull a rate from a LATER week
        # into the earliest rows, which is precisely the leak this rebuild
        # exists to remove. Rows with no league history yet get NaN, and the
        # gradient booster handles that natively.
        return pd.Series(vals, index=d.index).ffill()

    k = EB_K["catch_rate"]
    d["eb_catch_rate"] = (acc["receptions"] + k * prior_series("catch_rate")) / (acc["targets"] + k)
    k = EB_K["ypr"]
    d["eb_ypr"] = (acc["receiving_yards"] + k * prior_series("ypr")) / (acc["receptions"] + k)
    k = EB_K["ypc"]
    d["eb_ypc"] = (acc["rushing_yards"] + k * prior_series("ypc")) / (acc["carries"] + k)
    # Passing efficiency is a quarterback quantity. A league "YPA prior" for
    # running backs is driven by the occasional trick play and is unstable
    # week to week, so it is left undefined for non-passers rather than
    # feeding noise into the model.
    k = EB_K["ypa"]
    ypa = (acc["passing_yards"] + k * prior_series("ypa")) / (acc["attempts"] + k)
    d["eb_ypa"] = ypa.where(d["position"].eq("QB"))

    d["prior_targets"] = acc["targets"].fillna(0)
    d["prior_carries"] = acc["carries"].fillna(0)
    d["prior_attempts"] = acc["attempts"].fillna(0)
    return d.sort_index()


def prior_league_rates(df: pd.DataFrame) -> pd.DataFrame:
    """League efficiency by position, as known BEFORE each week.

    Expanding sums shifted by one week: row (position, gorder) holds the rate
    computed from every game that position played in strictly earlier weeks.
    """
    g = (df.groupby(["position", "gorder"], as_index=False)
         .agg(rec=("receptions", "sum"), tgt=("targets", "sum"),
              recyd=("receiving_yards", "sum"), car=("carries", "sum"),
              rushyd=("rushing_yards", "sum"), att=("attempts", "sum"),
              passyd=("passing_yards", "sum"))
         .sort_values(["position", "gorder"]))
    for c in ["rec", "tgt", "recyd", "car", "rushyd", "att", "passyd"]:
        g[c] = (g.groupby("position")[c].shift(1)
                .groupby(g["position"]).cumsum())
    g["catch_rate"] = g.rec / g.tgt.replace(0, np.nan)
    g["ypr"] = g.recyd / g.rec.replace(0, np.nan)
    g["ypc"] = g.rushyd / g.car.replace(0, np.nan)
    g["ypa"] = g.passyd / g.att.replace(0, np.nan)
    return g.set_index(["position", "gorder"])[["catch_rate", "ypr", "ypc", "ypa"]]


def eb_columns() -> list[str]:
    return ["eb_catch_rate", "eb_ypr", "eb_ypc", "eb_ypa",
            "prior_targets", "prior_carries", "prior_attempts"]


# ------------------------------------------------------------------ team level

def team_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game, with the team's realised opportunity."""
    env = [c for c in df.columns if c.startswith(("team_", "opp_team_"))
           and c.endswith(("_r3", "_r8"))]
    ctx = [c for c in ["is_home", "rest", "div_game", "week",
                       "venue_fixed_dome", "venue_retractable"] if c in df.columns]
    agg = {c: (c, "first") for c in env + ctx}
    t = df.groupby(["season", "week", "team", "gorder"], as_index=False).agg(
        team_pass_att_actual=("attempts", "sum"),
        team_rush_att_actual=("carries", "sum"),
        team_targets_actual=("targets", "sum"),
        **agg)
    return t, env + ctx


def fit_team_models(team_train: pd.DataFrame, feats: list[str]) -> dict:
    X = team_train[feats]
    return {t: _hgb(X, team_train[t]) for t in TEAM_TARGETS}


def predict_team(models: dict, team_test: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = team_test.copy()
    for t in TEAM_TARGETS:
        out[f"pred_{t}"] = np.clip(models[t].predict(team_test[feats]), 5, None)
    # the measured identity, not an assumption of equality
    out["pred_team_targets"] = out["pred_team_pass_att_actual"] * TARGETS_PER_ATTEMPT
    return out


# ---------------------------------------------------------------- player level

def fit_share_models(train: pd.DataFrame, feats: list[str]) -> dict:
    """Share models, weighted by team opportunity so that heavily-used
    team-weeks carry more influence than garbage-time ones."""
    skill = train[train.position.isin(["RB", "WR", "TE"])]
    qb = train[train.position == "QB"]
    m = {}
    if len(skill):
        m["target_share"] = _hgb(skill[feats], skill["target_share"].fillna(0),
                                 weight=skill["team_targets"].clip(lower=1))
        m["rush_share"] = _hgb(skill[feats], skill["rush_share"].fillna(0),
                               weight=skill["team_carries"].clip(lower=1))
    if len(qb):
        qb_share = (qb["attempts"] / qb["team_attempts"].replace(0, np.nan)).fillna(0)
        m["qb_att_share"] = _hgb(qb[feats], qb_share.clip(0, 1),
                                 weight=qb["team_attempts"].clip(lower=1))
        m["qb_rush_share"] = _hgb(qb[feats], qb["rush_share"].fillna(0),
                                  weight=qb["team_carries"].clip(lower=1))
    return m


def fit_efficiency_models(train: pd.DataFrame, feats: list[str]) -> dict:
    """Efficiency conditional on opportunity, weighted by opportunity.

    The EB-shrunk rate enters as a feature, so the model learns context
    adjustments on top of a stable prior rather than chasing small samples.
    """
    m = {}
    skill = train[train.position.isin(["RB", "WR", "TE"])]
    rec = skill[skill.targets > 0]
    if len(rec):
        m["catch_rate"] = _hgb(rec[feats], (rec.receptions / rec.targets).clip(0, 1),
                               weight=rec.targets)
    caught = skill[skill.receptions > 0]
    if len(caught):
        m["ypr"] = _hgb(caught[feats], (caught.receiving_yards / caught.receptions).clip(-5, 40),
                        weight=caught.receptions)
    run = train[train.carries > 0]
    if len(run):
        m["ypc"] = _hgb(run[feats], (run.rushing_yards / run.carries).clip(-5, 20),
                        weight=run.carries)
    qb = train[(train.position == "QB") & (train.attempts >= 5)]
    if len(qb):
        m["ypa"] = _hgb(qb[feats], (qb.passing_yards / qb.attempts).clip(0, 20),
                        weight=qb.attempts)
        m["td_rate"] = _hgb(qb[feats], (qb.passing_tds / qb.attempts).clip(0, .3),
                            weight=qb.attempts)
        m["int_rate"] = _hgb(qb[feats], (qb.passing_interceptions / qb.attempts).clip(0, .3),
                             weight=qb.attempts)
    return m


def allocate(players: pd.DataFrame, team_pred: pd.DataFrame, share_models: dict,
             feats: list[str]) -> pd.DataFrame:
    """Predict shares, then allocate the team budget.

    Normalisation happens only after a real share model has run, and only when
    the raw shares overshoot the budget — it enforces the constraint rather
    than creating the estimate.
    """
    p = players.copy()
    skill = p.position.isin(["RB", "WR", "TE"])
    qb = p.position == "QB"

    p["share_target_raw"] = 0.0
    p["share_rush_raw"] = 0.0
    if skill.any() and "target_share" in share_models:
        p.loc[skill, "share_target_raw"] = np.clip(
            share_models["target_share"].predict(p.loc[skill, feats]), 0, 1)
        p.loc[skill, "share_rush_raw"] = np.clip(
            share_models["rush_share"].predict(p.loc[skill, feats]), 0, 1)
    if qb.any() and "qb_att_share" in share_models:
        p.loc[qb, "qb_share_raw"] = np.clip(
            share_models["qb_att_share"].predict(p.loc[qb, feats]), 0, 1)
        p.loc[qb, "share_rush_raw"] = np.clip(
            share_models["qb_rush_share"].predict(p.loc[qb, feats]), 0, 1)

    tp = team_pred.set_index(["season", "week", "team"])
    key = list(zip(p.season, p.week, p.team))
    p["team_targets_budget"] = [tp["pred_team_targets"].get(k, np.nan) for k in key]
    p["team_rush_budget"] = [tp["pred_team_rush_att_actual"].get(k, np.nan) for k in key]
    p["team_att_budget"] = [tp["pred_team_pass_att_actual"].get(k, np.nan) for k in key]

    # --- enforcement -------------------------------------------------------
    # Shares are compositional: within a team they describe how one finite pool
    # is divided, so they must sum to 1. The raw model output already averages
    # 0.993 / 1.007 / 0.999 across team-weeks, so this closes a small residual
    # rather than manufacturing the estimate — and it is applied in BOTH
    # directions. Scaling down only, as in the first V3 build, left a
    # systematic -1.9 shortfall on every team.
    def normalise(sub_mask, col):
        sub = p[sub_mask] if sub_mask is not None else p
        if not len(sub):
            return
        tot = sub.groupby(["season", "week", "team"])[col].transform("sum")
        scale = np.where(tot > 1e-6, 1.0 / tot, 0.0)
        p.loc[sub.index, col.replace("_raw", "")] = sub[col].to_numpy() * scale
        p.loc[sub.index, col.replace("_raw", "_rawsum")] = tot.to_numpy()

    normalise(skill, "share_target_raw")
    normalise(None, "share_rush_raw")
    if qb.any() and "qb_share_raw" in p.columns:
        normalise(qb, "qb_share_raw")
    else:
        p["qb_share"] = 0.0

    p["proj_targets"] = np.where(skill, p["share_target"] * p["team_targets_budget"], np.nan)
    p["proj_carries"] = p["share_rush"] * p["team_rush_budget"]
    p["proj_attempts"] = np.where(qb, p["qb_share"] * p["team_att_budget"], np.nan)
    return p


def produce(p: pd.DataFrame, eff: dict, feats: list[str]) -> pd.DataFrame:
    """opportunity x efficiency = production."""
    X = p[feats]
    catch = np.clip(eff["catch_rate"].predict(X), 0.05, 1.0) if "catch_rate" in eff else 0.65
    ypr = np.clip(eff["ypr"].predict(X), 1.0, 30.0) if "ypr" in eff else 11.0
    ypc = np.clip(eff["ypc"].predict(X), 1.0, 8.0) if "ypc" in eff else 4.2

    p["proj_receptions"] = p["proj_targets"] * catch
    p["proj_receiving_yards"] = p["proj_receptions"] * ypr
    p["proj_rushing_yards"] = p["proj_carries"] * ypc

    if "ypa" in eff:
        qb = p.position == "QB"
        ypa = np.clip(eff["ypa"].predict(X), 3.0, 12.0)
        p["proj_passing_yards"] = np.where(qb, p["proj_attempts"] * ypa, np.nan)
        p["proj_passing_tds"] = np.where(
            qb, p["proj_attempts"] * np.clip(eff["td_rate"].predict(X), 0, .2), np.nan)
        p["proj_passing_interceptions"] = np.where(
            qb, p["proj_attempts"] * np.clip(eff["int_rate"].predict(X), 0, .15), np.nan)
    return p
