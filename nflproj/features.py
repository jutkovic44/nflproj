"""Feature engineering.

Leakage rule enforced everywhere: every player/team/defense feature is built
from games STRICTLY BEFORE the game being predicted. The only same-week inputs
are things genuinely known before kickoff: the injury report, the betting
line/total, rest days, venue and the weather forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

ROLL_WINDOWS = (3, 8)

PLAYER_ROLL_COLS = [
    "targets", "receptions", "receiving_yards", "target_share", "air_yards_share",
    "receiving_air_yards", "carries", "rushing_yards", "attempts", "passing_yards",
    "passing_tds", "passing_interceptions", "total_tds", "rush_share", "snap_pct",
    "yards_per_target", "yards_per_carry", "routes_proxy",
]

DEF_COLS = ["def_rec_yds_allowed", "def_rush_yds_allowed", "def_tds_allowed",
            "def_targets_allowed", "def_carries_allowed"]


def _order(df: pd.DataFrame) -> pd.DataFrame:
    df["gorder"] = df["season"] * 100 + df["week"]
    return df.sort_values(["player_id", "gorder"])


def build_player_base(seasons) -> pd.DataFrame:
    pw = D.load_player_weeks(seasons)
    snaps = D.load_snaps(seasons)

    pw["total_tds"] = pw["rushing_tds"] + pw["receiving_tds"]

    # team-week volume for share calculations
    tw = pw.groupby(["season", "week", "team"], as_index=False).agg(
        team_carries=("carries", "sum"),
        team_targets=("targets", "sum"),
        team_pass_yards=("passing_yards", "sum"),
        team_rush_yards=("rushing_yards", "sum"),
    )
    pw = pw.merge(tw, on=["season", "week", "team"], how="left")
    pw["rush_share"] = pw["carries"] / pw["team_carries"].replace(0, np.nan)
    pw["yards_per_target"] = pw["receiving_yards"] / pw["targets"].replace(0, np.nan)
    pw["yards_per_carry"] = pw["rushing_yards"] / pw["carries"].replace(0, np.nan)

    if len(snaps):
        snaps = snaps.rename(columns={"player": "player_display_name"})
        s = snaps.groupby(["season", "week", "team", "player_display_name"], as_index=False)[
            "offense_pct"].max().rename(columns={"offense_pct": "snap_pct"})
        pw = pw.merge(s, on=["season", "week", "team", "player_display_name"], how="left")
    else:
        pw["snap_pct"] = np.nan

    # crude route proxy: snap share scaled by whether the player saw targets
    pw["routes_proxy"] = pw["snap_pct"].fillna(0) * (pw["targets"] > 0).astype(float)
    return pw


def add_player_rollings(pw: pd.DataFrame) -> pd.DataFrame:
    pw = _order(pw)
    g = pw.groupby("player_id", sort=False)
    out = {}
    for col in PLAYER_ROLL_COLS:
        prior = g[col].shift(1)
        base = prior.groupby(pw["player_id"], sort=False)
        for w in ROLL_WINDOWS:
            out[f"{col}_r{w}"] = base.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"{col}_exp"] = base.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    roll = pd.DataFrame(out, index=pw.index)
    pw = pd.concat([pw, roll], axis=1)
    pw["games_played_prior"] = g.cumcount()
    return pw


def add_team_environment(pw: pd.DataFrame) -> pd.DataFrame:
    """Team pace and pass/rush tendency from PRIOR games only.

    This is the non-market replacement for the closing-line game-script
    signal: how many plays this offence runs and how often it throws, plus the
    same for the opponent it faces.
    """
    tw = pw.groupby(["season", "week", "team"], as_index=False).agg(
        team_pass_att=("attempts", "sum"), team_rush_att=("carries", "sum"))
    tw["team_plays"] = tw["team_pass_att"] + tw["team_rush_att"]
    tw["team_pass_rate"] = tw["team_pass_att"] / tw["team_plays"].replace(0, np.nan)
    tw["gorder"] = tw["season"] * 100 + tw["week"]
    tw = tw.sort_values(["team", "gorder"])
    g = tw.groupby("team", sort=False)
    for col in ["team_plays", "team_pass_att", "team_rush_att", "team_pass_rate"]:
        prior = g[col].shift(1)
        for w in (3, 8):
            tw[f"{col}_r{w}"] = (prior.groupby(tw["team"], sort=False)
                                 .rolling(w, min_periods=1).mean()
                                 .reset_index(level=0, drop=True))
    keep = ["season", "week", "team"] + [f"{c}_r{w}" for c in
            ["team_plays", "team_pass_att", "team_rush_att", "team_pass_rate"] for w in (3, 8)]
    pw = pw.merge(tw[keep], on=["season", "week", "team"], how="left")

    opp = tw[keep].rename(columns={"team": "opponent_team", **{
        c: f"opp_{c}" for c in keep if c not in ("season", "week", "team")}})
    return pw.merge(opp, on=["season", "week", "opponent_team"], how="left")


def add_defense_features(pw: pd.DataFrame) -> pd.DataFrame:
    """Opponent's rolling yards/TDs allowed to each position group."""
    dfa = pw.groupby(["season", "week", "opponent_team", "position"], as_index=False).agg(
        def_rec_yds_allowed=("receiving_yards", "sum"),
        def_rush_yds_allowed=("rushing_yards", "sum"),
        def_tds_allowed=("total_tds", "sum"),
        def_targets_allowed=("targets", "sum"),
        def_carries_allowed=("carries", "sum"),
    ).rename(columns={"opponent_team": "def_team"})
    dfa["gorder"] = dfa["season"] * 100 + dfa["week"]
    dfa = dfa.sort_values(["def_team", "position", "gorder"])
    g = dfa.groupby(["def_team", "position"], sort=False)
    for col in DEF_COLS:
        prior = g[col].shift(1)
        dfa[f"{col}_r8"] = prior.groupby([dfa["def_team"], dfa["position"]], sort=False) \
            .rolling(8, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    keep = ["season", "week", "def_team", "position"] + [f"{c}_r8" for c in DEF_COLS]
    return pw.merge(dfa[keep], left_on=["season", "week", "opponent_team", "position"],
                    right_on=["season", "week", "def_team", "position"], how="left").drop(columns=["def_team"])


def add_game_context(pw: pd.DataFrame, games: pd.DataFrame | None = None,
                     backfill_weather: bool = False) -> pd.DataFrame:
    if games is None:
        games = D.load_games()
    g = games.copy()
    if backfill_weather:
        g = fill_missing_weather(g)
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        r = pd.DataFrame({
            "season": g["season"], "week": g["week"], "team": g[f"{side}_team"],
            "is_home": 1 if side == "home" else 0,
            "spread_line": g["spread_line"] if side == "home" else -g["spread_line"],
            "total_line": g["total_line"],
            "rest": g[f"{side}_rest"], "div_game": g["div_game"],
            "roof": g["roof"], "temp": g["temp"], "wind": g["wind"],
        })
        rows.append(r)
    ctx = pd.concat(rows, ignore_index=True)
    # implied team total: the single most useful game-script feature
    ctx["implied_team_total"] = ctx["total_line"] / 2 + ctx["spread_line"] / 2

    # Venue geometry, not game-day roof position. "closed" is a decision made
    # on the day at retractable stadiums and is not knowable at the prediction
    # cutoff, so only the building's permanent properties are used.
    venue = venue_types()
    ctx["venue_fixed_dome"] = ctx["team"].map(
        lambda t: int(venue.get(t, "outdoor") == "fixed_dome"))
    ctx["venue_retractable"] = ctx["team"].map(
        lambda t: int(venue.get(t, "outdoor") == "retractable"))
    # is_dome is retained in the frame for research/reporting but the
    # availability contract blocks it from entering any model.
    ctx["is_dome"] = ctx["roof"].isin(["dome", "closed"]).astype(int)
    ctx = ctx.drop(columns=["roof"])
    return pw.merge(ctx, on=["season", "week", "team"], how="left")


def venue_types() -> dict[str, str]:
    """Permanent venue class per home team: fixed_dome, retractable, outdoor.

    Taken from stadium metadata rather than from any single game's roof state.
    This is a property of the building and is known indefinitely in advance.
    """
    from .weather import STADIUMS
    RETRACTABLE = {"ARI", "ATL", "DAL", "HOU", "IND", "LV"}
    out = {}
    for team, (_, _, _, fixed_roof) in STADIUMS.items():
        if team in RETRACTABLE:
            out[team] = "retractable"
        elif fixed_roof:
            out[team] = "fixed_dome"
        else:
            out[team] = "outdoor"
    return out


def add_injury_features(pw: pd.DataFrame, injuries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Own game status + the vacated-share features (the 'Nacua is out' signal)."""
    if injuries is None:
        injuries = D.load_injuries(sorted(pw.season.unique()))
    inj = injuries.copy()
    if not len(inj):
        for c in ["status_out", "status_doubtful", "status_questionable",
                  "vacated_target_share", "vacated_rush_share"]:
            pw[c] = 0.0
        return pw

    inj["report_status"] = inj["report_status"].fillna("").str.lower()
    inj["dnp"] = inj["practice_status"].fillna("").str.contains("Did Not", case=False).astype(int)
    status = inj[["season", "week", "team", "player_id", "report_status", "dnp"]].drop_duplicates(
        subset=["season", "week", "player_id"])

    pw = pw.merge(status, on=["season", "week", "team", "player_id"], how="left")
    pw["report_status"] = pw["report_status"].fillna("")
    pw["dnp"] = pw["dnp"].fillna(0)
    pw["status_out"] = (pw["report_status"] == "out").astype(int)
    pw["status_doubtful"] = (pw["report_status"] == "doubtful").astype(int)
    pw["status_questionable"] = (pw["report_status"] == "questionable").astype(int)

    # Vacated share: for each team-week, sum the recent usage of teammates who
    # are ruled out or doubtful. Uses each absentee's PRIOR 4-game usage only.
    usage = pw[["season", "week", "team", "player_id", "target_share_r3", "rush_share_r3"]].copy()
    absent = pw[(pw.status_out == 1) | (pw.status_doubtful == 1)][
        ["season", "week", "team", "player_id"]]
    absent = absent.merge(usage, on=["season", "week", "team", "player_id"], how="left")
    vac = absent.groupby(["season", "week", "team"], as_index=False).agg(
        vacated_target_share=("target_share_r3", "sum"),
        vacated_rush_share=("rush_share_r3", "sum"),
    )
    pw = pw.merge(vac, on=["season", "week", "team"], how="left")
    pw["vacated_target_share"] = pw["vacated_target_share"].fillna(0.0)
    pw["vacated_rush_share"] = pw["vacated_rush_share"].fillna(0.0)
    return pw


def candidate_columns(df: pd.DataFrame) -> list[str]:
    """Every column the pipeline can offer, before the availability filter."""
    roll = [c for c in df.columns if any(
        c.endswith(sfx) for sfx in [f"_r{w}" for w in ROLL_WINDOWS] + ["_exp", "_r8"])]
    # only SHIFTED windows qualify. build_player_base also creates same-week
    # aggregates (team_pass_yards, team_rush_yards) which are current-game
    # values and must never reach a model.
    team_env = [c for c in df.columns
                if c.startswith(("team_plays_", "team_pass_", "team_rush_", "opp_team_"))
                and c.endswith(("_r3", "_r8"))]
    ctx = team_env + ["is_home", "spread_line", "total_line", "implied_team_total", "rest",
           "div_game", "is_dome", "temp", "wind", "games_played_prior",
           "venue_fixed_dome", "venue_retractable",
           "status_doubtful", "status_questionable", "dnp",
           "vacated_target_share", "vacated_rush_share",
           "pos_QB", "pos_RB", "pos_WR", "pos_TE", "week"]
    return [c for c in dict.fromkeys(roll + ctx) if c in df.columns]


def feature_columns(df: pd.DataFrame, exclude_groups: set[str] | None = None,
                    enforce: bool = True) -> list[str]:
    """Contract-compliant model features.

    Rolling windows are registered on sight (they are shifted by construction
    and verified by the leakage test); everything else must already be declared
    in availability.REGISTRY or it is dropped as unregistered.
    """
    from . import availability as AV
    cand = candidate_columns(df)
    AV.register_team_env([c for c in cand if c.startswith(("team_", "opp_team_"))])
    AV.register_rollings([c for c in cand if any(
        c.endswith(sfx) for sfx in [f"_r{w}" for w in ROLL_WINDOWS] + ["_exp", "_r8"])])
    feats = AV.allowed(cand, exclude_groups=exclude_groups)
    if enforce:
        AV.audit(feats, strict=True)
    return feats


def leaked_columns(df: pd.DataFrame) -> list[str]:
    """Columns present in the frame but blocked from models. Kept available for
    research (market analysis, CLV) and never passed to a fit."""
    from . import availability as AV
    return [c for c in candidate_columns(df)
            if (f := AV.REGISTRY.get(c)) and f.availability in AV.FORBIDDEN]


def fill_missing_weather(games: pd.DataFrame, limit: int = 400) -> pd.DataFrame:
    """Backfill observed temp/wind for outdoor games from ERA5 reanalysis.

    The schedule file leaves weather blank for a meaningful slice of games. ERA5
    is free, open, and is the same reanalysis product climate research runs on,
    so the gaps get filled with measured conditions rather than left as nulls
    for the model to guess around.
    """
    from . import weather as WX
    from datetime import datetime as _dt

    g = games.copy()
    outdoor = ~g["roof"].isin(["dome", "closed"])
    played = g["result"].notna() if "result" in g.columns else g["gameday"].notna()
    need = g[outdoor & played & (g["temp"].isna() | g["wind"].isna())].head(limit)
    filled = 0
    for idx, row in need.iterrows():
        try:
            t = str(row.get("gametime") or "13:00")[:5]
            kick = _dt.strptime(f"{row['gameday']} {t}", "%Y-%m-%d %H:%M")
        except Exception:
            continue
        obs = WX.historical(row["home_team"], kick)
        if not obs:
            continue
        if pd.isna(g.at[idx, "temp"]) and obs.get("temp") is not None:
            g.at[idx, "temp"] = obs["temp"]
        if pd.isna(g.at[idx, "wind"]) and obs.get("wind") is not None:
            g.at[idx, "wind"] = obs["wind"]
        filled += 1
    if filled:
        print(f"  backfilled weather for {filled} games from ERA5")
    return g


def build(seasons=(2021, 2022, 2023, 2024, 2025, 2026)) -> pd.DataFrame:
    pw = build_player_base(seasons)
    pw = add_player_rollings(pw)
    pw = add_team_environment(pw)
    pw = add_defense_features(pw)
    pw = add_game_context(pw)
    pw = add_injury_features(pw)
    for p in ["QB", "RB", "WR", "TE"]:
        pw[f"pos_{p}"] = (pw["position"] == p).astype(int)
    return pw.sort_values("gorder").reset_index(drop=True)
