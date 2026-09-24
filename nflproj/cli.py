"""Command line entry point.

  python -m nflproj.cli refresh
  python -m nflproj.cli project --season 2026 --week 3
  python -m nflproj.cli project --season 2026 --week 3 --out "Puka Nacua" --out "Tyler Higbee"
  python -m nflproj.cli backtest --start 2022
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)


def cmd_refresh(args):
    from . import data as D
    D.refresh_all(SEASONS)
    print("data refreshed")


def cmd_project(args):
    from . import project as P, data as D, slates as S, dashboard as DB
    from . import matchup as MU, weather as WX
    from datetime import datetime
    overrides = {name: "out" for name in (args.out or [])}
    overrides.update({name: "doubtful" for name in (args.doubtful or [])})
    weather = None
    if args.game and (args.wind is not None or args.temp is not None):
        vals = {}
        if args.wind is not None:
            vals["wind"] = args.wind
        if args.temp is not None:
            vals["temp"] = args.temp
        weather = {args.game: vals}

    season, week, slate = args.season, args.week, args.slate
    games_all = D.load_games()
    if args.next or season is None or week is None:
        season, week, slate = S.next_slate(games_all)
        print(f"next slate: {season} week {week} — {S.SLATES[slate]['label']}")

    df = P.project_week(season, week, status_overrides=overrides or None,
                        weather_overrides=weather, slate=slate)
    if args.team:
        df = df[df.team.isin(args.team)]

    sched = S.slate_games(games_all, season, week, slate or "all")
    pairs = {}
    for _, g in sched.iterrows():
        pairs[g.home_team] = f"{g.away_team}@{g.home_team}"
        pairs[g.away_team] = f"{g.away_team}@{g.home_team}"
    df["game_key"] = df.team.map(pairs)
    df["slate"] = slate or "all"
    tag = f"{season}_wk{week}" + (f"_{slate}" if slate and slate != "all" else "")
    path = args.output or f"projections_{tag}.csv"
    df.to_csv(path, index=False)

    if not args.no_dashboard:
        meta = S.slate_window(games_all, season, week, slate or "all") or {}
        meta.update({"season": season, "week": week,
                     "label": meta.get("label") or f"Week {week} — full slate",
                     "games": int(len(sched))})

        teams = set(sched.home_team) | set(sched.away_team)
        hist = P.LAST_HISTORY
        inj_raw = D.load_injuries([season])
        snaps_all = D.load_snaps([season])
        inj = (MU.injuries_for_slate(inj_raw, hist, season, week, teams, snaps=snaps_all)
               if hist is not None else {})
        has_reports = bool(len(inj_raw[(inj_raw.season == season) & (inj_raw.week == week)])) \
            if len(inj_raw) else False
        meta["injury_reports"] = has_reports
        watch = MU.watch_list(df)

        wx = {}
        if not args.no_weather:
            print("fetching kickoff forecasts...")
            for _, g in sched.iterrows():
                key = f"{g.away_team}@{g.home_team}"
                wx[key] = WX.forecast(g.home_team, S.kickoff(g), g.get("roof"))

        accuracy = None
        try:
            import json as _json
            with open(args.accuracy or "accuracy.json") as fh:
                accuracy = _json.load(fh)
        except Exception:
            pass

        out_html = args.dashboard or "dashboard.html"
        DB.build(df, sched, meta, out_html, injuries=inj, watch=watch, weather=wx,
                 accuracy=accuracy)
        print(f"dashboard: {out_html}")

    show = ["team", "player_display_name", "position", "proj_passing_yards",
            "proj_rushing_yards", "proj_receptions", "proj_receiving_yards",
            "proj_anytime_td_prob"]
    print(df[show].round(1).head(25).to_string(index=False))
    print(f"\nwrote {path}  ({len(df)} players)")


def cmd_backtest(args):
    from . import features as F, backtest as B
    df = F.build(seasons=SEASONS)
    preds = B.walk_forward(df, start_season=args.start, refit_every=args.refit_every)
    preds.to_parquet(args.output or "backtest_preds.parquet")
    print("\n=== ACCURACY (contributors) ===")
    print(B.volume_filtered_report(preds).to_string(index=False))
    print("\n=== TD CALIBRATION ===")
    print(B.td_calibration(preds).to_string(index=False))


def cmd_grade(args):
    """Score every completed game the model has projected, write accuracy.json."""
    from . import scorecard as SC
    import pandas as pd

    frames = []
    live = SC.grade_archive()
    if len(live):
        live["_source"] = "live"
        frames.append(live)
        print(f"graded {len(live):,} archived live projections")
    if args.include_backtest:
        try:
            # the live model is V3, so the displayed score must come from V3's
            # evaluation artifact. V2's remains on disk for comparison.
            bt = pd.read_parquet(args.backtest or "backtest_preds_v3.parquet")
            bt["_source"] = "backtest"
            frames.append(bt)
            print(f"added {len(bt):,} walk-forward backtest rows")
        except Exception as e:
            print(f"  no backtest file ({e})")
    if not frames:
        print("nothing to grade yet — run some slates first")
        return

    df = pd.concat(frames, ignore_index=True)
    payload = SC.write_json(df, args.output or "accuracy.json")
    if payload.get("score") is None:
        print(payload.get("note"))
        return
    c = payload["components"]
    print(f"\nMODEL SCORE: {payload['score']}/100  ({payload['grade']})")
    print(f"  skill vs baseline  {c['skill_vs_baseline']['points']:>5}/55   "
          f"(mean {c['skill_vs_baseline']['mean_gain_pct']}% better than a 3-game average)")
    print(f"  TD calibration     {c['td_calibration']['points']:>5}/30   "
          f"({c['td_calibration']['mean_abs_error_pts']} pts mean error)")
    print(f"  correlation        {c['correlation']['points']:>5}/15   "
          f"(mean r={c['correlation']['mean_corr']})")
    print(f"  graded on {payload['n']:,} player-games — {payload['population']}")


def cmd_audit(args):
    """Leakage / availability audit. Exits non-zero on failure so the pipeline
    stops rather than shipping a compromised model."""
    from . import leaktest
    print("leakage audit")
    ok = leaktest.run(seasons=tuple(args.seasons or (2024, 2025, 2026)))
    print("audit passed" if ok else "AUDIT FAILED")
    if not ok:
        raise SystemExit(2)


def cmd_serve(args):
    from . import serve as SV
    SV.serve(port=args.port, open_browser=not args.no_browser)


def main():
    ap = argparse.ArgumentParser(prog="nflproj")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("refresh").set_defaults(func=cmd_refresh)

    p = sub.add_parser("project")
    p.add_argument("--season", type=int)
    p.add_argument("--week", type=int)
    p.add_argument("--slate", choices=["tnf", "sat", "sun_early", "sun_late", "snf", "mnf", "all"],
                   help="restrict to one kickoff window")
    p.add_argument("--next", action="store_true",
                   help="auto-detect the next slate that has not kicked off")
    p.add_argument("--dashboard", help="path for the generated HTML dashboard")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--no-weather", action="store_true", help="skip the forecast fetch")
    p.add_argument("--accuracy", help="path to accuracy.json from the grade command")
    p.add_argument("--team", action="append", help="filter to team abbrev, repeatable")
    p.add_argument("--out", action="append", help="force player OUT, repeatable")
    p.add_argument("--doubtful", action="append", help="force player DOUBTFUL, repeatable")
    p.add_argument("--game", help="game_id for a weather override, e.g. 2026_03_NYG_TEN")
    p.add_argument("--wind", type=float)
    p.add_argument("--temp", type=float)
    p.add_argument("--output")
    p.set_defaults(func=cmd_project)

    b = sub.add_parser("backtest")
    b.add_argument("--start", type=int, default=2022)
    b.add_argument("--refit-every", type=int, default=6)
    b.add_argument("--output")
    b.set_defaults(func=cmd_backtest)

    gr = sub.add_parser("grade")
    gr.add_argument("--include-backtest", action="store_true", default=True,
                    help="blend walk-forward history in with the live record")
    gr.add_argument("--backtest",
                    help="evaluation artifact to grade; defaults to the live "
                         "model's (backtest_preds_v3.parquet)")
    gr.add_argument("--output")
    gr.set_defaults(func=cmd_grade)

    au = sub.add_parser("audit", help="run the leakage / feature-availability tests")
    au.add_argument("--seasons", type=int, nargs="*")
    au.set_defaults(func=cmd_audit)

    sv = sub.add_parser("serve", help="local dashboard with a working Refresh button")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--no-browser", action="store_true")
    sv.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
