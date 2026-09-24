#!/usr/bin/env python3
"""Weather diagnostic — prints exactly what each source returns and why.

  python3 diagnose_weather.py            # next outdoor game on the schedule
  python3 diagnose_weather.py GB 2026-09-24 20:15
"""
import sys, warnings
warnings.filterwarnings("ignore")
from datetime import datetime

from nflproj import weather as W, data as D, slates as S

if len(sys.argv) >= 4:
    team, day, t = sys.argv[1], sys.argv[2], sys.argv[3]
    kick = datetime.strptime(f"{day} {t}", "%Y-%m-%d %H:%M")
else:
    g = D.load_games()
    season, week, slate = S.next_slate(g)
    sched = S.slate_games(g, season, week, slate)
    row = sched.iloc[0]
    team, kick = row.home_team, S.kickoff(row)
    print(f"next slate: {season} wk{week} {slate} — {row.away_team} @ {team}")

lat, lon, venue, _ = W.STADIUMS[team]
print(f"\n{venue}  ({lat}, {lon})")
print(f"kickoff (ET)  {kick}")
print(f"kickoff (UTC) {W._as_utc(kick)}\n")

print("--- raw source results ---")
W.ERRORS.clear()
nws = W.from_nws(lat, lon, kick)
print(f"NWS        : {nws}")
for m in W.OM_MODELS:
    stamp = W._as_utc(kick).strftime("%Y-%m-%dT%H:00")
    print(f"{m:14}: {W._om_one(lat, lon, stamp, m)}")

print("\n--- errors captured ---")
for e in W.ERRORS or ["(none)"]:
    print(" ", e)

print("\n--- consensus ---")
W.CACHE = "/tmp/nflproj_diag_cache.json"   # bypass any cached failure
out = W.forecast(team, kick, "outdoors")
print(" summary   :", out.get("summary"))
print(" confidence:", out.get("confidence"))
print(" sources   :", out.get("sources"))
