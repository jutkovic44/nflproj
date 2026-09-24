"""2025 role-layer validation — a measurement pass, not a tuning pass.

For every team-week of 2025 the information state is rebuilt as it stood at the
prediction cutoff (kickoff minus 3 hours), using only depth-chart snapshots
published at or before that moment. The pregame role selection is then compared
against the statistical selection V2/V3 would have made, and both are compared
against what actually happened.

Definitions are the existing ones; nothing new is invented:

  primary QB   most pass attempts among that team's QBs that week
  primary RB   most carries
  primary WR   most targets among WRs
  meaningful   at least 10 attempts (QB), 5 carries (RB), 3 targets (WR/TE)

"Correct" means the selected player is the one who actually led that team in
the position's opportunity measure. Cases where no evidence existed by the
cutoff are reported separately and are never counted as model errors.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from . import data as D, pregame_state as PS, slates as S

CUTOFF_LEAD_HOURS = 3
MEANINGFUL = {"QB": ("attempts", 10), "RB": ("carries", 5),
              "WR": ("targets", 3), "TE": ("targets", 3)}


def _kickoffs(season: int) -> pd.DataFrame:
    g = D.load_games()
    g = g[(g.season == season) & g.gameday.notna()]
    rows = []
    for _, r in g.iterrows():
        try:
            k = S.kickoff(r)
        except Exception:
            continue
        for team in (r.home_team, r.away_team):
            rows.append({"season": season, "week": int(r.week), "team": team, "kickoff": k})
    return pd.DataFrame(rows)


def run(season: int = 2025, feats_path: str = "feats_v2.parquet") -> pd.DataFrame:
    df = pd.read_parquet(feats_path)
    df = df[df.season <= season]
    actual = df[df.season == season]
    depth = D.load_depth_charts([season])
    rosters = D.load_weekly_rosters([season])
    injuries = D.load_injuries([season])
    kicks = _kickoffs(season)

    out = []
    for week in sorted(kicks.week.unique()):
        wk = kicks[kicks.week == week]
        cutoff = wk.kickoff.min() - timedelta(hours=CUTOFF_LEAD_HOURS)
        gorder = season * 100 + week
        hist = df[df.gorder < gorder]
        if not len(hist):
            continue

        dc = PS.depth_state(season, week, cutoff, depth)
        played = actual[actual.week == week]
        prior4 = hist[hist.gorder >= gorder - 4]
        prior_ids = set(prior4.player_id)
        season_ids = set(hist[hist.season == season].player_id)

        # last week's actual primary, for transition detection
        prev = actual[actual.week == week - 1] if week > 1 else pd.DataFrame()

        for team in wk.team.unique():
            tp = played[played.team == team]
            if not len(tp):
                continue
            for pos, (measure, floor) in MEANINGFUL.items():
                pool = tp[tp.position == pos]
                if not len(pool) or pool[measure].max() < floor:
                    continue
                truth = pool.loc[pool[measure].idxmax()]

                # --- statistical selection (V2/V3 behaviour) ---------------
                cand = hist[(hist.team == team) & (hist.position == pos) &
                            (hist.gorder >= gorder - 4)]
                stat_pick, stat_name = None, None
                if len(cand):
                    latest = cand.sort_values("gorder").groupby("player_id").tail(1)
                    score = (latest["snap_pct_r3"].fillna(0) * 100 +
                             latest[f"{measure}_r3"].fillna(0))
                    stat_pick = latest.loc[score.idxmax(), "player_id"]
                    stat_name = latest.loc[score.idxmax(), "player_display_name"]

                # --- pregame role selection --------------------------------
                role_pick, role_name, basis = None, None, "unavailable"
                if len(dc):
                    d = dc[(dc.team == team) & (dc.position == pos)]
                    if len(d):
                        top = d.loc[d.depth_rank.idxmin()]
                        role_pick, role_name = top.player_id, top.full_name
                        basis = ("depth_chart_rank_1" if top.depth_rank == 1
                                 else "depth_chart_ranked")
                if role_pick is None:
                    role_pick, role_name, basis = stat_pick, stat_name, "historical_fallback"

                prev_primary = None
                if len(prev):
                    pp = prev[(prev.team == team) & (prev.position == pos)]
                    if len(pp) and pp[measure].max() >= floor:
                        prev_primary = pp.loc[pp[measure].idxmax(), "player_id"]

                out.append({
                    "season": season, "week": week, "team": team, "position": pos,
                    "cutoff": cutoff,
                    "actual_player": truth.player_id, "actual_name": truth.player_display_name,
                    "actual_opportunity": float(truth[measure]),
                    "role_pick": role_pick, "role_name": role_name, "role_basis": basis,
                    "stat_pick": stat_pick, "stat_name": stat_name,
                    "role_correct": role_pick == truth.player_id,
                    "stat_correct": stat_pick == truth.player_id,
                    "methods_agree": role_pick == stat_pick,
                    "override": (role_pick is not None and stat_pick is not None
                                 and role_pick != stat_pick),
                    "info_available": basis.startswith("depth_chart"),
                    "transition": (prev_primary is not None
                                   and prev_primary != truth.player_id),
                    "actual_returning": truth.player_id not in prior_ids,
                    "actual_zero_season_games": truth.player_id not in season_ids,
                })
    return pd.DataFrame(out)


def summarise(r: pd.DataFrame) -> dict:
    avail = r[r.info_available]
    unavail = r[~r.info_available]
    return {
        "cases": int(len(r)),
        "prediction_states": int(r.groupby(["week"]).ngroups),
        "info_available": int(len(avail)),
        "info_unavailable": int(len(unavail)),
        "info_unavailable_rate_pct": round(100 * len(unavail) / max(len(r), 1), 1),
        "A_available_wrong": int((avail.role_correct == False).sum()),   # noqa: E712
        "B_unavailable": int(len(unavail)),
        "C_available_correct": int(avail.role_correct.sum()),
        "role_accuracy_pct": round(100 * r.role_correct.mean(), 1),
        "stat_accuracy_pct": round(100 * r.stat_correct.mean(), 1),
        "role_accuracy_when_available_pct": round(100 * avail.role_correct.mean(), 1) if len(avail) else None,
        "methods_agree": int(r.methods_agree.sum()),
        "overrides": int(r.override.sum()),
        "override_correct": int(r[r.override].role_correct.sum()),
        "override_wrong": int((r[r.override].role_correct == False).sum()),  # noqa: E712
        "override_stat_would_have_been_right": int(r[r.override].stat_correct.sum()),
    }
