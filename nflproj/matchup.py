"""Matchup context: who is hurt, and who to watch.

Both lists are ranked by recent usage rather than name recognition, so a WR3
with a 22% target share outranks a household name who is barely playing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATUS_RANK = {"out": 0, "doubtful": 1, "questionable": 2}
SKILL = ("QB", "RB", "WR", "TE")
DEF_POS = {"DE", "DT", "NT", "DL", "EDGE", "LB", "OLB", "ILB", "MLB",
           "CB", "S", "SAF", "FS", "SS", "DB"}
OFF_POS = {"QB", "RB", "FB", "WR", "TE", "T", "G", "C", "OL", "OT", "OG"}


def _side(pos: str | None, off_pct, def_pct) -> str:
    """Position is the reliable signal; snap counts only break ties for
    players with no listed position or genuine two-way usage."""
    p = (pos or "").upper()
    if p in DEF_POS:
        return "DEF"
    if p in OFF_POS:
        return "OFF"
    o = 0 if off_pct is None or pd.isna(off_pct) else float(off_pct)
    d = 0 if def_pct is None or pd.isna(def_pct) else float(def_pct)
    return "DEF" if d > o else "OFF"


def injuries_for_slate(inj: pd.DataFrame, history: pd.DataFrame, season: int,
                       week: int, teams, snaps: pd.DataFrame | None = None,
                       per_team: int = 14) -> dict[str, list[dict]]:
    """Injury report rows for the slate, enriched with recent usage.

    Covers both sides of the ball. Skill players are ranked by target/rush
    share; defenders and linemen by defensive snap share, which is the only
    honest availability signal for a position with no box-score volume. A
    starting corner who is ruled out therefore outranks a backup receiver.
    """
    if inj is None or not len(inj):
        return {}
    w = inj[(inj.season == season) & (inj.week == week) & (inj.team.isin(teams))].copy()
    w["report_status"] = w["report_status"].fillna("").str.lower()
    w = w[w.report_status.isin(STATUS_RANK)]
    if not len(w):
        return {}

    gorder = season * 100 + week
    recent = history[(history.gorder < gorder) & (history.gorder >= gorder - 5)]
    usage = (recent.sort_values("gorder").groupby("player_id")
             .agg(tgt_share=("target_share", "mean"),
                  rush_share=("rush_share", "mean"),
                  snap=("snap_pct", "mean"),
                  pos=("position", "last")).reset_index())
    w = w.merge(usage, on="player_id", how="left")

    # snap share covers everyone the box score cannot: defenders, linemen,
    # rotational players. Snap counts key on display name, so match name+team.
    w["snap_share"] = np.nan
    w["side"] = [_side(p, None, None) for p in w["position"].fillna(w.get("pos"))]
    if snaps is not None and len(snaps):
        sn = snaps[(snaps.season == season) & (snaps.week < week) &
                   (snaps.week >= week - 5) & (snaps.team.isin(teams))]
        if len(sn):
            agg = (sn.groupby(["team", "player"], as_index=False)
                   .agg(off=("offense_pct", "mean"), deff=("defense_pct", "mean")))
            agg["player_key"] = agg["player"].str.lower().str.strip()
            w["player_key"] = w["full_name"].fillna("").str.lower().str.strip()
            w = w.merge(agg[["team", "player_key", "off", "deff"]],
                        on=["team", "player_key"], how="left")
            w["side"] = [_side(p, o, d) for p, o, d in
                         zip(w["position"].fillna(w["pos"]), w["off"], w["deff"])]
            w["snap_share"] = np.where(w["side"] == "DEF",
                                       w["deff"].fillna(w["off"]),
                                       w["off"].fillna(w["deff"]))

    w["is_skill"] = w["pos"].isin(SKILL).fillna(False)
    w["usage"] = w[["tgt_share", "rush_share"]].max(axis=1).fillna(0)
    # one comparable importance number across both sides of the ball
    w["weight"] = w[["usage", "snap_share"]].max(axis=1).fillna(0)
    w["rank"] = w["report_status"].map(STATUS_RANK)
    w = w.sort_values(["rank", "weight"], ascending=[True, False])

    out: dict[str, list[dict]] = {}
    for team, sub in w.groupby("team"):
        rows = []
        for _, r in sub.head(per_team).iterrows():
            share = float(r.usage or 0)
            snap = r.snap_share
            rows.append({
                "name": r.get("full_name") or r.get("player_id"),
                "pos": (r.get("position") or r.get("pos") or "—"),
                "side": r.get("side") or "OFF",
                "status": str(r.report_status).title(),
                "injury": (r.get("report_primary_injury") or "").strip() or None,
                "share": round(share * 100, 1) if share > 0.02 else None,
                "snap": None if pd.isna(snap) else round(float(snap) * 100),
                "skill": bool(r.is_skill),
            })
        out[team] = rows
    return out


def watch_list(proj: pd.DataFrame, per_team: int = 3) -> dict[str, list[dict]]:
    """Players whose projection carries the most signal for this specific game.

    Scored on projected volume, TD equity, and how much vacated share they are
    inheriting — not on season-long reputation.
    """
    df = proj.copy()
    def col(name, default=0.0):
        if name in df.columns:
            return df[name].fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    yards = (col("proj_receiving_yards") + col("proj_rushing_yards")
             + col("proj_passing_yards") * 0.35)
    td = col("proj_anytime_td_prob")
    vac = col("vacated_target_share")
    df["_score"] = (yards / max(yards.max(), 1) * 0.55 + td * 0.30 + vac.clip(0, .4) * 1.2)

    out: dict[str, list[dict]] = {}
    for team, sub in df.groupby("team"):
        picks = sub.sort_values("_score", ascending=False).head(per_team)
        rows = []
        for _, r in picks.iterrows():
            rows.append({
                "name": r["player_display_name"], "pos": r["position"],
                "line": _statline(r), "why": _reason(r),
            })
        out[team] = rows
    return out


def _statline(r) -> str:
    def n(v, nd=0):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)
    if r["position"] == "QB":
        parts = [f"{n(r.get('proj_passing_yards'))} pass yds",
                 f"{n(r.get('proj_passing_tds'),1)} pass TD"]
        if (r.get("proj_rushing_yards") or 0) >= 15:
            parts.append(f"{n(r.get('proj_rushing_yards'))} rush yds")
        return " · ".join(p for p in parts if "None" not in p)
    parts = []
    if (r.get("proj_carries") or 0) >= 3:
        parts.append(f"{n(r.get('proj_carries'),1)} car / {n(r.get('proj_rushing_yards'))} yds")
    if (r.get("proj_targets") or 0) >= 1.5:
        parts.append(f"{n(r.get('proj_receptions'),1)} rec / {n(r.get('proj_receiving_yards'))} yds")
    parts.append(f"{round(float(r.get('proj_anytime_td_prob') or 0) * 100)}% TD")
    return " · ".join(parts)


def _reason(r) -> str:
    vac = float(r.get("vacated_target_share") or 0)
    td = float(r.get("proj_anytime_td_prob") or 0)
    if vac >= 0.12:
        return f"inheriting {round(vac*100)}% vacated target share from absent teammates"
    if td >= 0.45:
        return "highest touchdown equity on his team"
    if (r.get("proj_targets") or 0) >= 7:
        return "primary volume, target share concentrated here"
    if (r.get("proj_carries") or 0) >= 14:
        return "carries the ground game and the goal-line work"
    if r["position"] == "QB":
        return "sets the ceiling for every pass catcher in this game"
    return "leading projected producer for this team"
