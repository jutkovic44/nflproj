"""Automated leakage tests.

Run as part of the pipeline. A failure stops the run rather than letting a
compromised model reach the dashboard.

    python3 -m nflproj.cli audit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import availability as AV


def test_contract(df: pd.DataFrame) -> tuple[bool, str]:
    """Every model feature must be registered and pregame-available."""
    from .features import feature_columns
    try:
        feats = feature_columns(df, enforce=True)
    except AV.LeakageError as e:
        return False, str(e)
    return True, f"{len(feats)} features, all pregame-available"


def test_forbidden_absent(df: pd.DataFrame) -> tuple[bool, str]:
    """Closing lines and observed weather must not reach the model."""
    from .features import feature_columns, leaked_columns
    feats = set(feature_columns(df, enforce=False))
    blocked = leaked_columns(df)
    present = sorted(feats & set(blocked))
    if present:
        return False, f"forbidden features in model set: {present}"
    return True, f"{len(blocked)} forbidden columns retained for research, none in model"


def test_rolling_windows(df: pd.DataFrame, samples: int = 400) -> tuple[bool, str]:
    """A rolling feature at game t must use only games before t."""
    checks = [("targets", "targets_r3", 3), ("carries", "carries_r3", 3),
              ("receiving_yards", "receiving_yards_r8", 8)]
    bad, checked = 0, 0
    for stat, feat, window in checks:
        if feat not in df.columns:
            continue
        for _, sub in df.groupby("player_id"):
            if checked >= samples:
                break
            sub = sub.sort_values("gorder")
            if len(sub) < window + 2:
                continue
            row = sub.iloc[window + 1]
            prior = sub.iloc[1:window + 1][stat].mean()
            if pd.isna(row[feat]):
                continue
            checked += 1
            if abs(row[feat] - prior) > 0.01:
                bad += 1
    if bad:
        return False, f"{bad}/{checked} rolling values include the current game"
    return True, f"{checked} rolling values verified to exclude the current game"


def test_no_future_rows(df: pd.DataFrame) -> tuple[bool, str]:
    """Training rows must never post-date the row being predicted. Verified by
    construction in walk_forward; this asserts the ordering key is monotone."""
    if "gorder" not in df.columns:
        return False, "gorder missing — chronological ordering cannot be verified"
    expected = df["season"] * 100 + df["week"]
    if not (df["gorder"] == expected).all():
        return False, "gorder does not match season/week"
    return True, "chronological ordering key intact"


def test_target_not_in_features(df: pd.DataFrame) -> tuple[bool, str]:
    """No raw outcome column may appear as a feature."""
    from .features import feature_columns
    from .models import SKILL_TARGETS, QB_TARGETS, TD_TARGET
    feats = set(feature_columns(df, enforce=False))
    raw = set(SKILL_TARGETS + QB_TARGETS + [TD_TARGET])
    overlap = sorted(feats & raw)
    if overlap:
        return False, f"raw outcomes used as features: {overlap}"
    return True, "no raw outcome columns in the feature set"


def test_v3_efficiency_prior_only(df: pd.DataFrame) -> tuple[bool, str]:
    """EB efficiency must be computable from prior games alone, and a player's
    first-ever game must have no value at all."""
    from .hierarchy import add_eb_efficiency
    cols = ["eb_ypc", "eb_catch_rate", "eb_ypr", "eb_ypa"]
    d = df if all(c in df.columns for c in cols) else add_eb_efficiency(df)
    first = d.sort_values("gorder").groupby("player_id").head(1)
    leaked = int(first[cols].notna().any(axis=1).sum())
    if leaked:
        return False, f"{leaked} first-ever games carry an efficiency prior"
    return True, "efficiency rates use prior games only; first games are null"


def test_no_same_week_aggregates(df: pd.DataFrame) -> tuple[bool, str]:
    """Every derived feature must be a shifted window or an explicitly declared
    static/schedule/injury field. Catches same-week team or player aggregates
    that slip in through a prefix match — the failure mode that put
    team_pass_yards (corr 0.67 with the outcome) into an early V2 build."""
    from .features import feature_columns
    feats = feature_columns(df, enforce=False)
    declared_static = {n for n, f in AV.REGISTRY.items()
                       if f.availability in (AV.STATIC, AV.SCHEDULE, AV.GAMEDAY_AM, AV.FORECAST)}
    # counters over prior games carry no current-game information
    declared_static |= {"games_played_prior", "prior_targets", "prior_carries",
                        "prior_attempts", "eb_catch_rate", "eb_ypr", "eb_ypc", "eb_ypa"}
    offenders = [f for f in feats
                 if f not in declared_static and not f.endswith(AV.WINDOW_SUFFIXES)]
    if offenders:
        return False, f"unshifted derived features: {offenders}"
    return True, "all derived features are shifted windows"


def test_eb_prior_cutoff_invariant(df: pd.DataFrame) -> tuple[bool, str]:
    """The EB positional prior must not depend on data after the cutoff.

    Preparing the full frame and a truncated frame must yield identical values
    for every shared row. If any prior observation with source_date > cutoff
    leaked in, the two would differ — this is what V3 failed.
    """
    from .hierarchy import add_eb_efficiency
    if "gorder" not in df.columns or not len(df):
        return False, "no gorder column"
    cut = int(df.gorder.quantile(0.7))
    full = add_eb_efficiency(df).set_index(["gorder", "player_id"])
    trunc = add_eb_efficiency(df[df.gorder < cut]).set_index(["gorder", "player_id"])
    cols = ["eb_ypc", "eb_catch_rate", "eb_ypr", "eb_ypa"]
    idx = full.index.intersection(trunc.index)
    d = (full.loc[idx, cols].astype(float) - trunc.loc[idx, cols].astype(float)).abs()
    worst = float(d.max().max())
    if worst > 1e-9:
        return False, f"EB prior depends on post-cutoff data (max diff {worst:.6f})"
    return True, f"EB prior identical across cutoffs ({len(idx):,} rows checked)"


def test_pregame_state_cutoff(df: pd.DataFrame) -> tuple[bool, str]:
    """No pregame-state evidence may carry available_at after the cutoff."""
    from . import pregame_state as PS
    import pandas as _pd
    from datetime import datetime as _dt
    cut = _dt(2026, 9, 24, 17, 15)
    dc = PS.depth_state(2026, 3, cut)
    if not len(dc):
        return True, "no timestamped depth-chart evidence for the probe week"
    ts = _pd.to_datetime(dc.available_at, utc=True)
    late = int((ts > _pd.Timestamp(cut).tz_localize("UTC")).sum())
    if late:
        return False, f"{late} depth-chart rows published after the cutoff"
    return True, f"{len(dc)} depth-chart rows, all published before the cutoff"


TESTS = [
    ("availability contract", test_contract),
    ("forbidden features blocked", test_forbidden_absent),
    ("rolling windows exclude current game", test_rolling_windows),
    ("chronological ordering", test_no_future_rows),
    ("targets absent from features", test_target_not_in_features),
    ("no same-week aggregates", test_no_same_week_aggregates),
    ("V3 efficiency uses prior games only", test_v3_efficiency_prior_only),
    ("EB prior is cutoff-invariant", test_eb_prior_cutoff_invariant),
    ("pregame state respects the cutoff", test_pregame_state_cutoff),
]


def run(df: pd.DataFrame | None = None, seasons=(2024, 2025, 2026),
        verbose: bool = True) -> bool:
    if df is None:
        from .features import build
        df = build(seasons=seasons)
    passed = True
    for name, fn in TESTS:
        try:
            ok, detail = fn(df)
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        passed &= ok
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return passed
