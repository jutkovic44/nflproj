# Quickstart

## One-time setup

You need the **nflproj.zip** download, not the individual files.

```bash
# 1. unpack it into your home folder (creates ~/nflproj)
cd ~/Downloads
unzip -o nflproj.zip -d ~

# 2. go there and confirm the layout
cd ~/nflproj
ls
# you should see: nflproj/  README.md  run_slate.sh  requirements.txt

# 3. dependencies
pip3 install -r requirements.txt

# 4. make the runners executable
chmod +x run_slate.sh run_week.sh

# 5. first data pull (~20MB, about a minute)
python3 -m nflproj.cli refresh

# 6. accuracy baseline (5-season backtest, ~2 minutes)
python3 -m nflproj.cli backtest --start 2022
python3 -m nflproj.cli grade
```

## If downloads fail with CERTIFICATE_VERIFY_FAILED

macOS python.org builds ship without the system root certificates. `pip3
install -r requirements.txt` now pulls in `certifi`, which the package uses
as a fallback trust store, so this should resolve itself. If it persists:

```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

## Pasting commands

Paste one command at a time, without the `#` comment lines — zsh tries to
interpret `~` and digits inside them and throws "unknown file attribute" or
"number expected". Those errors are noise, not failures.

**Every command must run from `~/nflproj`.** Python finds the package by
looking in the current directory, so running from anywhere else gives
`ModuleNotFoundError: No module named 'nflproj'`.

## Dashboard with a working Refresh button (recommended)

```bash
cd ~/nflproj
python3 -m nflproj.cli serve
```

Your browser opens http://localhost:8765. Press **Refresh** and it runs the
whole pipeline — nflverse data, injury reports, weather, regrade, reprojection
— streaming the log into the page, then reloads with fresh numbers. Takes
about 30 seconds.

Uses only Python's standard library. Nothing to install, nothing hosted, no
account. It binds to 127.0.0.1, so it is reachable from your machine only.
Leave the terminal open while you use it; ctrl-C stops it.

Opening `dashboard.html` directly from disk still works, but the button is
inert there — a file:// page has no way to run Python. It will tell you so.

## Every run, fresh data + dashboard

```bash
cd ~/nflproj && ./run_slate.sh
open dashboard.html
```

That single command refreshes nflverse data, regrades completed games,
auto-detects the next slate that has not kicked off, pulls live weather, and
rewrites `dashboard.html`.

## Thursday night, step by step

Injury reports are final Wednesday afternoon. Run any time Thursday:

```bash
cd ~/nflproj
./run_slate.sh          # picks up TNF automatically
open dashboard.html     # macOS. Linux: xdg-open. Windows: start
```

Click the game card to see the injury report, weather consensus, and key
players. Run it again at 5pm — data refreshes each time, so any Thursday
downgrade shows up.

## Targeting a specific slate

```bash
python3 -m nflproj.cli refresh                       # always refresh first
python3 -m nflproj.cli project --slate tnf --season 2026 --week 3
python3 -m nflproj.cli project --slate sun_early --season 2026 --week 3
python3 -m nflproj.cli project --slate sun_late --season 2026 --week 3
python3 -m nflproj.cli project --slate snf --season 2026 --week 3
python3 -m nflproj.cli project --slate mnf --season 2026 --week 3
python3 -m nflproj.cli project --season 2026 --week 3                 # whole week
```

## Reacting to news faster than the feed

nflverse updates injury data on a lag. If a player is ruled out on Twitter
before the feed catches up, force it:

```bash
python3 -m nflproj.cli project --slate tnf --season 2026 --week 3 \
  --out "Player Name" --doubtful "Other Player"
```

This recomputes vacated target and rush share for the whole team, so his
teammates' projections rise automatically.

## Automating it

```bash
crontab -e     # paste crontab.example, set PROJ=/Users/<you>/nflproj
crontab -l     # verify
```

Cron does not fire while a Mac is asleep. If yours sleeps, either run
`./run_slate.sh` by hand or switch to launchd.

## Files each run produces

| File | What |
|---|---|
| `dashboard.html` | the visualization — open this |
| `projections_<season>_wk<week>_<slate>.csv` | raw numbers |
| `accuracy.json` | current model score |
| `runs/` | dated archive of every run, builds the live track record |
