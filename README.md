# nflproj — NFL player stat-line projection engine

Weekly QB/RB/WR/TE stat-line projections for every game, built on free nflverse
data, validated with a walk-forward backtest across 2022–2026.

No prop lines are ingested or scored. The system outputs stat projections; you
compare them to whatever prices you find yourself.

## Install

```bash
pip install pandas pyarrow scikit-learn
python -m nflproj.cli refresh          # downloads ~20MB to ~/nflproj_data
```

Set `NFLPROJ_DATA` to change the cache location.

## Weekly use

```bash
# whole slate
python -m nflproj.cli project --season 2026 --week 3

# one game, with late injury news applied by hand
python -m nflproj.cli project --season 2026 --week 3 \
    --team NYG --team TEN \
    --out "Malik Nabers" --doubtful "Theo Johnson"

# weather override when the forecast differs from the stored value
python -m nflproj.cli project --season 2026 --week 3 \
    --game 2026_03_NYG_TEN --wind 18 --temp 41
```

Output columns: `proj_attempts`, `proj_passing_yards`, `proj_passing_tds`,
`proj_passing_interceptions`, `proj_carries`, `proj_rushing_yards`,
`proj_targets`, `proj_receptions`, `proj_receiving_yards`,
`proj_anytime_td_prob`. A CSV is written alongside the printed table.

Run it Saturday for a first look, then rerun after final injury reports drop.
`--out` and `--doubtful` recompute vacated target and rush share for the whole
team, so ruling out a WR1 lifts his teammates automatically.

## Backtest

```bash
python -m nflproj.cli backtest --start 2022
```

Walk-forward: for each week, train on every game played before it, predict that
week cold, move on. Retrains every 6 weeks by default. See
`BACKTEST_REPORT.md` for the current numbers.

## How it works

**Data (all free, all nflverse)**
- `stats_player_week` — weekly box scores, 2021 onward
- `injuries` — official game-status reports, known before kickoff
- `snap_counts` — playing time, the best available route-participation proxy
- `games.csv` — schedule, closing spread and total, roof, temp, wind, rest days

**Features (79)**
- Player rolling usage over 3 games, 8 games, and career to date: targets,
  target share, air-yards share, carries, rush share, snap share, yards per
  target, yards per carry
- Opponent defense: rolling 8-game yards, targets, carries and TDs allowed to
  that specific position group
- Game context: implied team total derived from spread and total, home/away,
  rest days, divisional, dome
- Weather: temperature and wind, forced to neutral indoors
- Injury: own game status, practice participation, and **vacated target/rush
  share** — the sum of recent usage belonging to teammates ruled out or doubtful

**Models**
- One gradient-boosted regressor per stat, fit separately for QBs and for
  skill players
- Touchdowns as a Poisson rate, converted to P(anytime TD) = 1 − exp(−λ), then
  isotonically calibrated on a chronological holdout

**Leakage control.** Every player, team and defense feature is shifted so it
uses only prior games. The only same-week inputs are things genuinely knowable
before kickoff. When projecting a week that has already been played, the real
rows are dropped first so rolling windows cannot see the result.

## Limitations, stated plainly

- The edge over a naive trailing average is roughly 6–9% MAE on real
  contributors. That is a real but modest improvement, and it is what an honest
  walk-forward on this problem produces.
- The stored `temp`/`wind` in games.csv are observed values, not forecasts.
  Backtest weather is therefore slightly optimistic. Pass a forecast via
  `--wind`/`--temp` when projecting.
- Snap share is a proxy for routes run. True route participation is better and
  would be the highest-value data upgrade.
- High-probability TD buckets are still overstated (predicted 60%, actual 49%).
  Shade down anything above 50%.
- Nothing models in-game injuries, ejections or benchings. When a starter exits
  early, the projection is simply wrong and no amount of feature work fixes it.
- Roster assignment comes from recent appearances, so players who just changed
  teams can be misassigned for a week.

## Roadmap

1. True route participation and red-zone usage splits
2. Team-level pace and play-volume model feeding individual volume
3. Beat-reporter and practice-report sentiment layered on hard injury status
4. Postgres store and Next.js dashboard


## Slate-by-slate operation

The NFL week is five events, not one. Injury news lands hours before each
window, so the model is re-run per slate:

| Slate | Kickoff | Run at |
|---|---|---|
| `tnf` | Thu 8:15 PM ET | Thu 5:15 PM |
| `sun_early` | Sun 1:00 PM ET | Sun 11:00 AM |
| `sun_late` | Sun 4:05/4:25 PM ET | Sun 2:05 PM |
| `snf` | Sun 8:20 PM ET | Sun 5:20 PM |
| `mnf` | Mon 8:15 PM ET | Mon 5:15 PM |
| `sat` | Saturday, Dec/Jan | Sat 11:00 AM |

Sunday's 11:00 AM run matters most: inactives are official 90 minutes before
kickoff, so that run sees the real answer on every questionable player.

```bash
./run_slate.sh                                   # auto-detects the next slate
python3 -m nflproj.cli project --next             # same, without the wrapper
python3 -m nflproj.cli project --season 2026 --week 3 --slate snf
```

`--next` reads the schedule, finds the first game that has not kicked off, and
projects only that window.

### Recurring runs

`crontab.example` has the five entries. Install with `crontab -e`, paste, and
set `PROJ` to your install path. Each run refreshes data, projects the slate,
rewrites `dashboard.html`, and archives a dated copy under `runs/`.

On macOS, cron only fires if the machine is awake. Either use
`caffeinate`, or switch to launchd, which catches missed runs on wake:

```bash
# ~/Library/LaunchAgents/com.nflproj.slate.plist — see README for full example
launchctl load ~/Library/LaunchAgents/com.nflproj.slate.plist
```

## Dashboard

Every run regenerates `dashboard.html`, a single self-contained file. No server,
no build step — open it from disk or from your phone.

- One row per player, not per position. All 350 skill players on the slate.
- Game cards across the top show spread, both implied team totals, and
  indoors/wind. Click one to filter the table to that game.
- Sort any column. Filter by position, search by name.
- `Q` badge marks a questionable player; a green `+18%` badge marks a player
  inheriting vacated target share from an absent teammate.
- The anytime-TD column is a bar as well as a number, so the top of the slate
  is visible at a glance.

Dated copies land in `runs/`, so you can diff Sunday 11:00 AM against Saturday
and see exactly which players moved on the injury news.


## Matchup context

Click any game card to open its detail panel:

- **Game-time weather, from a consensus of free open sources.** Three
  independent forecasts are pulled and averaged: the US National Weather
  Service (api.weather.gov — official, no key, refined by the local forecast
  office) plus ECMWF IFS, NOAA GFS and DWD ICON as separate members via
  Open-Meteo. Each member is shown, along with the spread between them and a
  confidence label — when models disagree by 8+ mph on wind, the panel says
  "low — models disagree" rather than printing a false-precision average.
  Indoor venues short-circuit. Results cache for 90 minutes. Skip with
  `--no-weather`.

  Historical games missing weather in the schedule file are backfilled from
  ERA5 reanalysis (`features.fill_missing_weather`), the same measured-
  conditions dataset used in climate research — so the model trains on what
  actually happened rather than nulls.

- **Injury report by team,**- **Injury report by team,** ordered Out, Doubtful, Questionable, then by recent
  usage — so a 33%-target-share receiver sits above a backup safety. Each row
  shows the injury itself and the player's recent share. Reports publish
  Wednesday through Friday, so a Tuesday run correctly shows nothing yet.
- **Key players to watch,** three per team, scored on projected volume,
  touchdown equity, and inherited vacated share rather than reputation. Each
  carries a one-line reason.


## Model score

Every run grades itself. `nflproj grade` compares archived pre-kickoff
projections to what actually happened, blends in the walk-forward backtest, and
writes `accuracy.json`. The dashboard shows the result as a badge at the top,
with a 12-week trend and a tap-through breakdown.

```bash
python3 -m nflproj.cli grade        # writes accuracy.json
```

`run_slate.sh` runs this before every projection, so the score is always current.
It also archives each slate's CSV into `runs/`, which is what builds the live
track record over time.

**What the number is.** A skill score out of 100, not "percent correct" —
percent correct is meaningless for continuous projections.

| Component | Points | Full marks at |
|---|---|---|
| Skill vs naive baseline | 55 | 15% MAE reduction vs a 3-game average |
| Touchdown calibration | 30 | zero mean bucket error |
| Correlation with outcomes | 15 | mean r = 0.70 |

0 means no better than a trailing 3-game average. 55–70 is what an honest NFL
projection system looks like. **Above 85 means something is leaking** — treat a
sudden jump as a bug report, not a win.

Scored on players at 40%+ snap share. Including deep-bench zeros would inflate
every component.
