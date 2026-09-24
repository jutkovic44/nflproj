"""Project an upcoming, unplayed week.

Synthetic rows are appended for every expected active player, then pushed
through the exact same feature pipeline as the backtest, so a projection is
built from identical inputs to the ones that were validated.

LIVE MODEL: V3-TEAM-COHERENT.

Every piece of V3 is imported from `hierarchy` and `v3eval`; none of it is
reimplemented here. This module only assembles the inputs V3 needs for an
unplayed week (synthetic player rows plus a team frame) and calls it.
""" 
from __future__ import annotations

from datetime import timedelta, datetime

import numpy as np
import pandas as pd

from . import data as D, features as F, models as M
from . import hierarchy as H, v3eval as V3
from . import pregame_state as PS, slates as S

RECENT_WEEKS = 4

LIVE_MODEL_VERSION = "V3.1-PREGAME-STATE"

# populated by project_week; records how the eligible set was decided
LAST_STATE_INFO: dict | None = None

# how far before the earliest kickoff a live run is assumed to happen
CUTOFF_LEAD_HOURS = 3

# populated by project_week; the dashboard uses it for injury/usage context
LAST_HISTORY: pd.DataFrame | None = None


def _active_players(pw: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    gorder = season * 100 + week
    recent = pw[(pw.gorder < gorder) & (pw.gorder >= gorder - RECENT_WEEKS)]
    if not len(recent):
        recent = pw[pw.gorder < gorder]
    latest = recent.sort_values("gorder").groupby("player_id").tail(1)
    played = latest[(latest.snap_pct.fillna(0) > 0.05) |
                    (latest.targets + latest.carries + latest.attempts > 0)]
    return played[["player_id", "player_display_name", "position", "team"]].copy()


def _depth_chart_candidates(pw: pd.DataFrame, season: int, week: int,
                            cutoff, teams) -> pd.DataFrame:
    """Skill players designated on a depth chart published before the cutoff.

    Only players who already have prior NFL games in the frame are added: a
    true debutant has no statistical basis for a projection, and that gap is
    reported rather than papered over.
    """
    try:
        dc = PS.depth_state(season, week, cutoff) if cutoff is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if not len(dc):
        return pd.DataFrame()
    dc = dc[dc.position.isin(["QB", "RB", "WR", "TE"]) & dc.depth_rank.le(3)
            & dc.team.isin(teams)]
    known = set(pw.player_id.unique())
    dc = dc[dc.player_id.isin(known)]
    if not len(dc):
        return pd.DataFrame()
    out = dc[["player_id", "full_name", "position", "team"]].rename(
        columns={"full_name": "player_display_name"})
    return out.drop_duplicates(subset=["player_id"])


def build_upcoming(season: int, week: int, seasons=(2021, 2022, 2023, 2024, 2025, 2026),
                   status_overrides: dict | None = None,
                   weather_overrides: dict | None = None,
                   slate: str | None = None,
                   cutoff=None) -> pd.DataFrame:
    pw = F.build_player_base(seasons)
    pw["gorder"] = pw.season * 100 + pw.week
    games = D.load_games()
    from . import slates as S
    if slate and slate != "all":
        sched = S.slate_games(games, season, week, slate)
        if not len(sched):
            raise ValueError(f"No {slate} games for {season} week {week}")
    else:
        sched = games[(games.season == season) & (games.week == week)]
        if not len(sched):
            raise ValueError(f"No scheduled games for {season} week {week}")

    if weather_overrides:
        sched = sched.copy()
        for gid, vals in weather_overrides.items():
            for k, v in vals.items():
                sched.loc[sched.game_id == gid, k] = v

    opp = {}
    for _, g in sched.iterrows():
        opp[g.home_team] = g.away_team
        opp[g.away_team] = g.home_team

    roster = _active_players(pw, season, week)

    # Pregame evidence, not recent usage, controls eligibility. A player who
    # has not appeared in weeks — returning from injury, activated, or newly
    # designated starter — is added here so the allocation layer can consider
    # him. He still needs prior NFL games for his statistical features.
    dc_extra = _depth_chart_candidates(pw, season, week, cutoff, opp)
    if len(dc_extra):
        roster = pd.concat([roster, dc_extra], ignore_index=True)
        roster = roster.drop_duplicates(subset=["player_id"], keep="first")
    roster = roster[roster.team.isin(opp)]
    synth = roster.copy()
    synth["season"] = season
    synth["week"] = week
    synth["opponent_team"] = synth.team.map(opp)
    synth["player_name"] = synth["player_display_name"]
    synth["season_type"] = "REG"
    for col in pw.select_dtypes("number").columns:
        if col not in synth.columns:
            synth[col] = np.nan

    # If the week has already been played for these teams, drop the real rows so
    # the synthetic ones cannot inherit post-game stats through the rolling
    # windows. This keeps a projection strictly pre-game.
    played = (pw.season == season) & (pw.week == week) & (pw.team.isin(opp))
    if played.any():
        print(f"  note: {int(played.sum())} played rows for {season} wk{week} excluded (pre-game simulation)")
    pw = pw[~played]

    full = pd.concat([pw, synth], ignore_index=True)
    full["gorder"] = full.season * 100 + full.week
    full = F.add_player_rollings(full)
    # V3 needs team-level opportunity context; same function the evaluation
    # harness uses, called here so synthetic rows carry it too
    full = F.add_team_environment(full)
    full = F.add_defense_features(full)
    full = F.add_game_context(full, games=sched if False else games)
    # all seasons, not just the current one: the V3 models train on history and
    # the evaluation harness sees injury features across every season. Loading
    # only the current season left historical training rows with blank injury
    # status, which made the live-fitted share models differ from evaluation.
    inj = D.load_injuries(seasons)

    if status_overrides:
        rows = []
        name_to_id = dict(zip(roster.player_display_name, roster.player_id))
        for name, status in status_overrides.items():
            pid = name_to_id.get(name)
            if pid is None:
                print(f"  ! override skipped, player not found: {name}")
                continue
            team = roster.loc[roster.player_id == pid, "team"].iloc[0]
            rows.append({"season": season, "week": week, "team": team, "player_id": pid,
                         "report_status": status, "practice_status": "Did Not Participate In Practice"})
        if rows:
            ov = pd.DataFrame(rows)
            inj = inj[~((inj.week == week) & (inj.player_id.isin(ov.player_id)))]
            inj = pd.concat([inj, ov], ignore_index=True)

    full = F.add_injury_features(full, injuries=inj)
    for p in ["QB", "RB", "WR", "TE"]:
        full[f"pos_{p}"] = (full["position"] == p).astype(int)

    upcoming = full[(full.season == season) & (full.week == week)].copy()
    history = full[full.gorder < season * 100 + week].copy()
    return upcoming, history


def _apply_pregame_state(live: pd.DataFrame, state: pd.DataFrame,
                         one_qb_per_team: bool = True) -> tuple[pd.DataFrame, dict]:
    """Restrict the allocation set using the pregame state layer.

    Position-agnostic. Players the state says will not play are removed
    whatever their history; for positions carrying a single-starter policy the
    designated starter is kept in preference to the statistically favoured
    player. V3's allocation and normalisation are untouched — this only decides
    which players are eligible before the budget is divided.
    """
    info = {"removed_not_expected_to_play": 0, "starter_overrides": [],
            "selection_basis": {}}
    if state is None or not len(state):
        info["state"] = "unavailable"
        return live, info

    st = state.set_index("player_id")
    not_playing = st.index[~st.expected_to_play.astype(bool)]
    drop = live.index[live.player_id.isin(not_playing)]
    info["removed_not_expected_to_play"] = int(len(drop))
    live = live.drop(index=drop)

    if one_qb_per_team:
        qbs = live[live.position == "QB"]
        if len(qbs):
            fallback = (qbs["snap_pct_r3"].fillna(0) * 100
                        + qbs["attempts_r3"].fillna(0))
            cand = st.loc[st.index.intersection(qbs.player_id)].reset_index()
            cand = cand[cand.position == "QB"] if "position" in cand.columns else cand
            fb = pd.Series(fallback.values,
                           index=qbs.player_id.values).reindex(cand.player_id).values
            cand = cand.assign(_fb=fb)
            chosen = PS.choose_starters(
                cand.set_index(pd.RangeIndex(len(cand))), "QB",
                fallback_rank=pd.Series(cand._fb.values, index=range(len(cand))))
            keep_ids = set(chosen.player_id) if len(chosen) else set()
            # any team with no state-backed choice keeps its statistical leader
            for team, grp in qbs.groupby("team"):
                if not keep_ids & set(grp.player_id):
                    best = fallback.loc[grp.index].idxmax()
                    keep_ids.add(grp.loc[best, "player_id"])
                    info["selection_basis"][team] = "historical_fallback"
            for _, r in (chosen.iterrows() if len(chosen) else []):
                info["selection_basis"][r.team] = r.selection_basis
                stat_pick = qbs[qbs.team == r.team]
                if len(stat_pick):
                    top = fallback.loc[stat_pick.index].idxmax()
                    if stat_pick.loc[top, "player_id"] != r.player_id:
                        info["starter_overrides"].append(
                            {"team": r.team, "designated": r.full_name,
                             "statistical_pick": stat_pick.loc[top, "player_display_name"],
                             "basis": r.selection_basis})
            drop_qb = qbs.index[~qbs.player_id.isin(keep_ids)]
            live = live.drop(index=drop_qb)
    return live, info


def _project_v3(upcoming: pd.DataFrame, history: pd.DataFrame,
                season: int, week: int,
                one_qb_per_team: bool = True,
                pregame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Call the frozen V3 architecture on an unplayed week.

    Every step below is the V3 implementation imported from hierarchy/v3eval:
    EB preparation, team opportunity, share allocation (including its
    normalisation), efficiency and the Poisson TD rate. Nothing is duplicated
    or re-derived.
    """
    gorder = season * 100 + week
    full = pd.concat([history, upcoming], ignore_index=True)

    prepared = V3.prepare(full)                       # EB efficiency + team_attempts
    feats = F.feature_columns(prepared, enforce=False) + H.eb_columns()

    hist_p = prepared[prepared.gorder < gorder]
    up_p = prepared[(prepared.season == season) & (prepared.week == week)]

    team_all, team_feats = H.team_frame(prepared)
    team_train = team_all[team_all.gorder < gorder]
    team_test = team_all[team_all.gorder == gorder]

    train = hist_p[V3._eligible(hist_p)]
    if len(train) < 4000 or not len(team_test):
        raise RuntimeError(
            f"V3 cannot run for {season} wk{week}: {len(train)} training rows, "
            f"{len(team_test)} team rows. No fallback to V2 is performed.")

    team_models = H.fit_team_models(team_train, team_feats)
    share_models = H.fit_share_models(train, feats)
    eff_models = H.fit_efficiency_models(train, feats)
    td_model = V3._fit_td(train, feats)

    tp = H.predict_team(team_models, team_test, team_feats)
    live = up_p[V3._eligible(up_p)]
    live, state_info = _apply_pregame_state(live, pregame, one_qb_per_team)
    global LAST_STATE_INFO
    LAST_STATE_INFO = state_info
    p = H.allocate(live, tp, share_models, feats)
    p = H.produce(p, eff_models, feats)

    lam = np.clip(td_model.predict(p[feats]), 0, None)
    p["proj_total_tds"] = lam
    p["proj_anytime_td_prob"] = 1 - np.exp(-lam)
    p["model_version"] = LIVE_MODEL_VERSION
    return p


def project_week(season: int, week: int, seasons=(2021, 2022, 2023, 2024, 2025, 2026),
                 status_overrides: dict | None = None,
                 weather_overrides: dict | None = None,
                 slate: str | None = None,
                 model_version: str = "v3",
                 one_qb_per_team: bool = True,
                 use_pregame_state: bool = True,
                 cutoff: "datetime | None" = None,
                 min_snap: float = 0.15) -> pd.DataFrame:
    """Live projection. Defaults to V3-TEAM-COHERENT.

    model_version="v2" still runs the old flat per-player path for comparison,
    but it is no longer the default and is never selected automatically.
    """
    eff_cutoff = cutoff
    if eff_cutoff is None and use_pregame_state:
        try:
            _g = D.load_games()
            _sched = S.slate_games(_g, season, week, slate or "all")
            eff_cutoff = min(S.kickoff(r) for _, r in _sched.iterrows()) - \
                timedelta(hours=CUTOFF_LEAD_HOURS)
        except Exception:
            eff_cutoff = None

    upcoming, history = build_upcoming(season, week, seasons, status_overrides,
                                       weather_overrides, slate, cutoff=eff_cutoff)
    global LAST_HISTORY
    LAST_HISTORY = history

    if model_version == "v3":
        pregame = None
        if use_pregame_state:
            try:
                games = D.load_games()
                sched = S.slate_games(games, season, week, slate or "all")
                kick = min(S.kickoff(r) for _, r in sched.iterrows())
                cut = eff_cutoff or (kick - timedelta(hours=CUTOFF_LEAD_HOURS))
                pregame = PS.enforce_cutoff(
                    PS.build(season, week, kick, cut, history))
                pregame = pregame[pregame.team.isin(
                    set(sched.home_team) | set(sched.away_team))]
            except Exception as e:
                print(f"  ! pregame state unavailable ({type(e).__name__}: {e}); "
                      "falling back to historical role only")
                pregame = None
        out = _project_v3(upcoming, history, season, week,
                          one_qb_per_team=one_qb_per_team, pregame=pregame)
    elif model_version == "v2":
        fitted = M.fit_models(history)
        out = M.predict(fitted, upcoming[M.eligible(upcoming)])
        out["model_version"] = "V2-CLEAN"
    else:
        raise ValueError(f"unknown model_version {model_version!r}")

    cols = ["season", "week", "team", "opponent_team", "player_display_name", "position",
            "snap_pct_r3", "attempts_r3",
            "implied_team_total", "spread_line", "wind", "status_questionable",
            "vacated_target_share",
            "proj_attempts", "proj_passing_yards", "proj_passing_tds",
            "proj_passing_interceptions", "proj_carries", "proj_rushing_yards",
            "proj_targets", "proj_receptions", "proj_receiving_yards",
            "proj_total_tds", "proj_anytime_td_prob",
            # V3 requires these: the team budgets the allocation reconciles to
            "share_target", "share_rush", "qb_share",
            "team_targets_budget", "team_rush_budget", "team_att_budget",
            "model_version"]
    out = out[[c for c in cols if c in out.columns]]

    if model_version == "v2":
        # V2-only post-filters. Under V3 these are removed: dropping players
        # after allocation would silently break team reconciliation, which is
        # the whole point of V3.
        out = out[(out.proj_targets.fillna(0) + out.proj_carries.fillna(0) +
                   out.proj_attempts.fillna(0)) >= 1.0]
        qbs = out[out.position == "QB"].copy()
        if len(qbs):
            qbs["_rank"] = qbs["snap_pct_r3"].fillna(0) * 100 + qbs["attempts_r3"].fillna(0)
            starters = qbs.sort_values("_rank", ascending=False).groupby("team").head(1)
            out = pd.concat([out[out.position != "QB"],
                             out.loc[out.index.isin(starters.index)]])

    out = out.drop_duplicates(subset=["team", "player_display_name"], keep="first")
    return out.sort_values(["team", "proj_receiving_yards"], ascending=[True, False])
