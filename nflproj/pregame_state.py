"""Pregame player state — who is available, who starts, and how we know.

Architecturally separate from the projection model. V3 answers "how is a finite
team opportunity divided among the eligible players"; this layer answers "who
is eligible, and who is designated to start", which is a roster question that
historical statistics cannot reliably infer.

    pregame state  →  eligibility  →  role  →  team budget  →  player share

Position-agnostic: quarterbacks are the loudest case, not a special case.

Evidence classes and what each source can actually support
----------------------------------------------------------
depth_chart_designation   2025+ nflverse depth charts. Daily snapshots with a
                          real ISO timestamp, so the information state at any
                          historical cutoff is exactly reconstructable.
                          <=2024 files carry no timestamp and are therefore
                          NOT treated as timestamped evidence.
roster_status             weekly rosters: ACT / INA / RES / DEV / CUT. Week
                          level only; no transaction timestamp exists. INA is
                          a game-day decision, conventionally published ~90
                          minutes before kickoff.
injury_report             official report_status. Week level, no timestamp.
                          Final game status is published ~90 minutes out.
historical_role           V3's learned usage. Always available, weakest
                          evidence for a role that has just changed.
unavailable               no source spoke to this player's role by the cutoff.

Nothing here scrapes news or social media. A coaching announcement that never
reached one of the sources above is classified `unavailable`, never guessed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import data as D

# conventional publication offsets relative to kickoff, used only for sources
# that carry no timestamp of their own
FINAL_STATUS_LEAD = timedelta(minutes=90)      # inactives, final injury status
REPORT_LEAD = timedelta(days=2)                # Friday practice report

DECISION_TYPES = ["depth_chart_designation", "roster_status", "injury_report",
                  "historical_role", "unavailable"]

OUT_STATUSES = {"out", "doubtful"}
INACTIVE_ROSTER = {"INA", "RES", "CUT", "RET", "TRD"}


def _norm(name: str) -> str:
    return str(name or "").lower().strip()


def depth_state(season: int, week: int, cutoff: datetime,
                depth: pd.DataFrame | None = None) -> pd.DataFrame:
    """Latest depth-chart snapshot published at or before the cutoff.

    Only timestamped rows qualify. Untimestamped (<=2024) rows are dropped
    rather than assumed available, so a historical backtest cannot silently use
    a designation that may have been published after kickoff.
    """
    if depth is None:
        depth = D.load_depth_charts([season])
    if not len(depth):
        return pd.DataFrame()
    d = depth[depth.get("timestamped", False) == True].copy()  # noqa: E712
    if not len(d):
        return pd.DataFrame()
    cut = pd.Timestamp(cutoff).tz_localize("UTC") if pd.Timestamp(cutoff).tzinfo is None \
        else pd.Timestamp(cutoff)
    d = d[d.available_at <= cut]
    if not len(d):
        return pd.DataFrame()
    # a player appears once per formation slot; keep his best (lowest) rank in
    # the most recent snapshot, otherwise he arrives duplicated
    d = d.sort_values("available_at")
    last_ts = d.groupby(["team", "player_id"])["available_at"].transform("max")
    d = d[d.available_at == last_ts]
    latest = (d.sort_values("depth_rank")
              .groupby(["team", "player_id"], as_index=False).head(1))
    latest["season"], latest["week"] = season, week
    return latest


def roster_state(season: int, week: int, kickoff: datetime, cutoff: datetime,
                 rosters: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weekly roster status, admitted only if the cutoff is late enough.

    Inactive designations are game-day decisions; treating them as known days
    in advance would be leakage, so they are withheld until the cutoff reaches
    the conventional 90-minute publication point.
    """
    if rosters is None:
        rosters = D.load_weekly_rosters([season])
    if not len(rosters):
        return pd.DataFrame()
    r = rosters[(rosters.season == season) & (rosters.week == week)].copy()
    if not len(r):
        return pd.DataFrame()
    publish = pd.Timestamp(kickoff) - FINAL_STATUS_LEAD
    r["available_at"] = publish
    # before the publication point only the fact of being rostered is known
    if pd.Timestamp(cutoff) < publish:
        r["status_known"] = False
        r.loc[r.status.isin(INACTIVE_ROSTER), "status"] = "UNKNOWN"
    else:
        r["status_known"] = True
    return r


def build(season: int, week: int, kickoff: datetime, cutoff: datetime,
          history: pd.DataFrame, injuries: pd.DataFrame | None = None,
          depth: pd.DataFrame | None = None,
          rosters: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per candidate player with an explicit, sourced role state."""
    dc = depth_state(season, week, cutoff, depth)
    rs = roster_state(season, week, kickoff, cutoff, rosters)

    if injuries is None:
        injuries = D.load_injuries([season])
    inj = injuries[(injuries.season == season) & (injuries.week == week)].copy() \
        if len(injuries) else pd.DataFrame()
    if len(inj):
        inj["report_status"] = inj["report_status"].fillna("").str.lower()

    # candidate pool: anyone with history, plus anyone on the current roster or
    # depth chart — this is what lets a returning player with zero current-season
    # games become eligible
    gorder = season * 100 + week
    recent = history[history.gorder < gorder]
    hist_players = (recent.sort_values("gorder").groupby("player_id").tail(1)
                    [["player_id", "player_display_name", "position", "team"]]
                    .rename(columns={"player_display_name": "full_name"}))
    hist_players["from_history"] = True

    frames = [hist_players]
    if len(dc):
        d = dc[["player_id", "full_name", "position", "team"]].copy()
        d["from_history"] = False
        frames.append(d)
    if len(rs):
        r = rs[rs.status == "ACT"][["player_id", "full_name", "position", "team"]].copy()
        r["from_history"] = False
        frames.append(r)
    pool = pd.concat(frames, ignore_index=True).dropna(subset=["player_id"])
    pool = pool.drop_duplicates(subset=["player_id"], keep="first")

    st = pool.copy()
    st["season"], st["week"] = season, week

    # --- depth chart -------------------------------------------------------
    if len(dc):
        st = st.merge(dc[["player_id", "depth_rank", "available_at"]]
                      .rename(columns={"available_at": "depth_available_at"}),
                      on="player_id", how="left")
    else:
        st["depth_rank"] = np.nan
        st["depth_available_at"] = pd.NaT

    # --- roster ------------------------------------------------------------
    if len(rs):
        st = st.merge(rs[["player_id", "status", "status_known", "available_at"]]
                      .rename(columns={"available_at": "roster_available_at"}),
                      on="player_id", how="left")
    else:
        st["status"] = np.nan
        st["status_known"] = False
        st["roster_available_at"] = pd.NaT

    # --- injury ------------------------------------------------------------
    if len(inj):
        # the nflverse injury feed has carried the player key as either
        # gsis_id or player_id across releases; accept whichever is present
        idcol = "player_id" if "player_id" in inj.columns else "gsis_id"
        st = st.merge(inj[[idcol, "report_status"]].rename(columns={idcol: "player_id"}),
                      on="player_id", how="left")
    else:
        st["report_status"] = np.nan
    st["injury_status"] = st["report_status"].fillna("")

    # --- resolve -----------------------------------------------------------
    ruled_out = st.injury_status.isin(OUT_STATUSES)
    inactive = st.status.isin(INACTIVE_ROSTER)
    st["active_status"] = np.where(inactive, "inactive",
                                   np.where(st.status.eq("ACT"), "active", "unknown"))
    st["expected_to_play"] = ~(ruled_out | inactive)

    st["starter_status"] = np.where(st.depth_rank == 1, "designated_starter",
                                    np.where(st.depth_rank.notna(), "backup", "unknown"))

    st["decision_type"] = np.select(
        [ruled_out | inactive,
         st.depth_rank.notna(),
         st.status.notna(),
         st.from_history],
        ["injury_report" , "depth_chart_designation", "roster_status", "historical_role"],
        default="unavailable")

    st["role_confidence"] = np.select(
        [st.decision_type.eq("injury_report"),
         st.decision_type.eq("depth_chart_designation") & st.depth_rank.eq(1),
         st.decision_type.eq("depth_chart_designation"),
         st.decision_type.eq("roster_status"),
         st.decision_type.eq("historical_role")],
        [0.95, 0.85, 0.70, 0.45, 0.35], default=0.0)

    st["role_source"] = st["decision_type"]
    st["role_source_timestamp"] = st["depth_available_at"].fillna(st["roster_available_at"])
    st["prediction_cutoff"] = pd.Timestamp(cutoff)
    st["available_at"] = st["role_source_timestamp"]
    return st


def enforce_cutoff(state: pd.DataFrame) -> pd.DataFrame:
    """Drop any evidence published after the prediction cutoff.

    Rows whose evidence fails the check fall back to `unavailable` rather than
    being silently kept.
    """
    if not len(state):
        return state
    s = state.copy()
    cut = pd.Timestamp(s["prediction_cutoff"].iloc[0])
    if cut.tzinfo is None:
        cut = cut.tz_localize("UTC")
    ts = pd.to_datetime(s["available_at"], utc=True, errors="coerce")
    late = ts.notna() & (ts > cut)
    s.loc[late, ["depth_rank", "starter_status", "role_source_timestamp", "available_at"]] = \
        [np.nan, "unknown", pd.NaT, pd.NaT]
    s.loc[late, "decision_type"] = "unavailable"
    s.loc[late, "role_confidence"] = 0.0
    return s


def choose_starters(state: pd.DataFrame, position: str,
                    fallback_rank: pd.Series | None = None) -> pd.DataFrame:
    """One designated player per team for `position`.

    Preference order: depth-chart rank 1 published before the cutoff, then any
    ranked player, then the statistical fallback. The chosen row records which
    of those applied, so "the system did not know" stays distinguishable from
    "the system chose wrongly".
    """
    pos = state[(state.position == position) & state.expected_to_play].copy()
    if not len(pos):
        return pos
    pos["_fallback"] = fallback_rank.reindex(pos.index).fillna(0) if fallback_rank is not None else 0.0
    pos["_pref"] = np.select(
        [pos.depth_rank.eq(1), pos.depth_rank.notna()],
        [2.0, 1.0], default=0.0)
    pos = pos.sort_values(["_pref", "_fallback"], ascending=False)
    chosen = pos.groupby("team", as_index=False).head(1).copy()
    chosen["selection_basis"] = np.where(
        chosen._pref == 2.0, "depth_chart_rank_1",
        np.where(chosen._pref == 1.0, "depth_chart_ranked", "historical_fallback"))
    return chosen


def coverage_summary(state: pd.DataFrame) -> dict:
    """How much of the pool is backed by real pregame evidence."""
    if not len(state):
        return {"players": 0}
    counts = state.decision_type.value_counts().to_dict()
    return {"players": int(len(state)),
            "by_decision_type": {k: int(v) for k, v in counts.items()},
            "timestamped_evidence": int(state.available_at.notna().sum()),
            "information_unavailable": int((state.decision_type == "unavailable").sum()),
            "expected_to_play": int(state.expected_to_play.sum())}
