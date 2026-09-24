"""V3 walk-forward evaluation and structural tests.

Identical chronological windows, refit cadence and contributor filter as V2, so
the comparison is like for like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import hierarchy as H
from .features import feature_columns
from .models import TD_TARGET, HGB_KW


def _eligible(df: pd.DataFrame) -> pd.Series:
    return (df["games_played_prior"] >= 1) & (df["status_out"] == 0)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add EB efficiency columns and the team-attempt denominator."""
    d = H.add_eb_efficiency(df)
    ta = d.groupby(["season", "week", "team"])["attempts"].transform("sum")
    d["team_attempts"] = ta
    return d


def walk_forward(df: pd.DataFrame, seasons=(2022, 2023, 2024, 2025),
                 refit_every: int = 6, verbose: bool = True) -> pd.DataFrame:
    df = prepare(df)
    base_feats = feature_columns(df, enforce=False)
    feats = base_feats + H.eb_columns()
    team_all, team_feats = H.team_frame(df)

    weeks = (df[df.season.isin(seasons)][["season", "week", "gorder"]]
             .drop_duplicates().sort_values("gorder"))
    out, fitted, since = [], None, 10 ** 9

    for _, row in weeks.iterrows():
        train = df[(df.gorder < row.gorder) & _eligible(df)]
        test = df[(df.gorder == row.gorder) & _eligible(df)]
        if len(train) < 4000 or not len(test):
            continue
        team_train = team_all[team_all.gorder < row.gorder]
        team_test = team_all[team_all.gorder == row.gorder]

        if fitted is None or since >= refit_every:
            fitted = {
                "team": H.fit_team_models(team_train, team_feats),
                "share": H.fit_share_models(train, feats),
                "eff": H.fit_efficiency_models(train, feats),
                "td": _fit_td(train, feats),
            }
            since = 0
            if verbose:
                print(f"  refit @ {int(row.season)} wk {int(row.week)} "
                      f"({len(train):,} player rows, {len(team_train):,} team rows)",
                      flush=True)
        since += 1

        tp = H.predict_team(fitted["team"], team_test, team_feats)
        p = H.allocate(test, tp, fitted["share"], feats)
        p = H.produce(p, fitted["eff"], feats)
        lam = np.clip(fitted["td"].predict(p[feats]), 0, None)
        p["proj_total_tds"] = lam
        p["proj_anytime_td_prob"] = 1 - np.exp(-lam)
        for t in ["targets", "receptions", "receiving_yards", "carries", "rushing_yards",
                  "attempts", "passing_yards", "passing_tds", "passing_interceptions",
                  TD_TARGET]:
            p[f"base_{t}"] = test[f"{t}_r3"] if f"{t}_r3" in test.columns else np.nan
        p["base_anytime_td_prob"] = 1 - np.exp(-test[f"{TD_TARGET}_r3"].fillna(0))
        out.append(p)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _fit_td(train: pd.DataFrame, feats: list[str]):
    from sklearn.ensemble import HistGradientBoostingRegressor
    kw = dict(HGB_KW); kw["loss"] = "poisson"
    return HistGradientBoostingRegressor(**kw).fit(train[feats], train[TD_TARGET])


# ----------------------------------------------------------- structural tests

def structural_report(preds: pd.DataFrame) -> dict:
    """The checks V3 exists to satisfy."""
    p = preds.copy()
    team = p.groupby(["season", "week", "team"]).agg(
        tgt=("proj_targets", "sum"), car=("proj_carries", "sum"),
        att=("proj_attempts", "sum"),
        budget_tgt=("team_targets_budget", "first"),
        budget_att=("team_att_budget", "first"),
        budget_rush=("team_rush_budget", "first")).dropna(subset=["budget_tgt"])

    # a team-week with no projectable QB (rookie debut, returning suspension)
    # has no attempt projection to compare against — counted, not hidden
    qb_count = p.assign(is_qb=(p.position == "QB").astype(int)).groupby(
        ["season", "week", "team"])["is_qb"].sum()
    no_qb = int((qb_count == 0).sum())

    rec_gt_tgt = int((p.proj_receptions > p.proj_targets + 1e-6).sum())
    neg = int(((p.proj_targets < 0) | (p.proj_carries < 0) |
               (p.get("share_target", pd.Series(0, index=p.index)) < 0)).sum())
    over_budget = int((team.tgt > team.budget_tgt + 1e-6).sum())
    # Measured identity: team targets = 0.9495 x team pass attempts. The model's
    # own assertion of team opportunity is the attempt budget, so that is what
    # the identity is checked against; the sum of projected QB attempts is
    # reported separately because it is zero when no QB is projectable.
    budget_att = team.get("budget_att")
    impossible = int((team.tgt > team["budget_att"] + 1e-6).sum())
    qb_sum_short = int((team.tgt > team.att + 1e-6).sum())
    recon_t = (team.tgt - team.budget_tgt).abs()
    recon_r = (team.car - team.budget_rush).abs()

    return {
        "team_weeks": int(len(team)),
        "receptions>targets": rec_gt_tgt,
        "negative_opportunity_or_share": neg,
        "targets>team_budget": over_budget,
        "targets>pass_attempt_budget (impossible)": impossible,
        "team_weeks_with_no_projectable_qb": no_qb,
        "targets>sum_of_projected_qb_attempts": qb_sum_short,
        "mean_abs_target_reconciliation": round(float(recon_t.mean()), 3),
        "max_abs_target_reconciliation": round(float(recon_t.max()), 2),
        "mean_abs_carry_reconciliation": round(float(recon_r.mean()), 3),
        "max_abs_carry_reconciliation": round(float(recon_r.max()), 2),
        "team_carries_outside_15_40": int(((team.car < 15) | (team.car > 40)).sum()),
    }


TESTS = [
    ("receptions <= targets", lambda s: s["receptions>targets"] == 0),
    ("no negative opportunity or share", lambda s: s["negative_opportunity_or_share"] == 0),
    ("targets within team budget", lambda s: s["targets>team_budget"] == 0),
    ("targets <= pass-attempt budget",
     lambda s: s["targets>pass_attempt_budget (impossible)"] == 0),
    ("carry reconciliation under 0.5",
     lambda s: s["mean_abs_carry_reconciliation"] < 0.5),
]


def run_structural_tests(preds: pd.DataFrame, verbose: bool = True) -> bool:
    s = structural_report(preds)
    ok = True
    for name, fn in TESTS:
        passed = bool(fn(s))
        ok &= passed
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return ok
