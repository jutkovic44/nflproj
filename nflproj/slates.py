"""Slate windows.

The NFL week is not one event, it is five. Injury news lands hours before each
window, so the model should be re-run per slate rather than once per week.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

SLATES = {
    "tnf":       {"label": "Thursday Night",  "days": ["Thursday", "Friday"], "start": "00:00", "end": "23:59"},
    "sat":       {"label": "Saturday",        "days": ["Saturday"],           "start": "00:00", "end": "23:59"},
    "sun_early": {"label": "Sunday 1:00 ET",  "days": ["Sunday"],             "start": "00:00", "end": "15:59"},
    "sun_late":  {"label": "Sunday 4:00 ET",  "days": ["Sunday"],             "start": "16:00", "end": "18:59"},
    "snf":       {"label": "Sunday Night",    "days": ["Sunday"],             "start": "19:00", "end": "23:59"},
    "mnf":       {"label": "Monday Night",    "days": ["Monday", "Tuesday"],  "start": "00:00", "end": "23:59"},
}
ORDER = ["tnf", "sat", "sun_early", "sun_late", "snf", "mnf"]

# how long before kickoff the run should happen, per slate
LEAD_HOURS = {"tnf": 3, "sat": 3, "sun_early": 2, "sun_late": 2, "snf": 3, "mnf": 3}


def classify(games: pd.DataFrame) -> pd.DataFrame:
    """Tag each scheduled game with its slate key."""
    g = games.copy()
    t = g["gametime"].fillna("13:00").astype(str).str.slice(0, 5)
    g["_t"] = t
    g["slate"] = None
    for key in ORDER:
        spec = SLATES[key]
        mask = (g["weekday"].isin(spec["days"]) & (g["_t"] >= spec["start"]) &
                (g["_t"] <= spec["end"]) & g["slate"].isna())
        g.loc[mask, "slate"] = key
    g["slate"] = g["slate"].fillna("sun_early")
    return g.drop(columns=["_t"])


def kickoff(row) -> datetime:
    t = str(row.get("gametime") or "")[:5]
    if len(t) != 5 or ":" not in t:
        t = "13:00"
    return datetime.strptime(f"{row['gameday']} {t}", "%Y-%m-%d %H:%M")


def slate_games(games: pd.DataFrame, season: int, week: int, slate: str) -> pd.DataFrame:
    g = classify(games)
    g = g[(g.season == season) & (g.week == week)]
    return g if slate in (None, "all") else g[g.slate == slate]


def next_slate(games: pd.DataFrame, now: datetime | None = None) -> tuple[int, int, str]:
    """The next slate that has not kicked off yet, as (season, week, slate)."""
    now = now or datetime.now()
    g = classify(games).dropna(subset=["gameday"])
    if "game_type" in g.columns:
        g = g[g.game_type == "REG"]
    g = g[g.gameday >= (now - timedelta(days=2)).strftime("%Y-%m-%d")].copy()
    g["kick"] = g.apply(kickoff, axis=1)
    upcoming = g[g.kick > now - timedelta(hours=4)].sort_values("kick")
    if not len(upcoming):
        raise ValueError("no upcoming games found in schedule")
    first = upcoming.iloc[0]
    return int(first.season), int(first.week), str(first.slate)


def slate_window(games: pd.DataFrame, season: int, week: int, slate: str) -> dict:
    sg = slate_games(games, season, week, slate)
    if not len(sg):
        return {}
    kicks = sg.apply(kickoff, axis=1)
    return {
        "slate": slate,
        "label": SLATES.get(slate, {}).get("label", "Full slate"),
        "games": int(len(sg)),
        "first_kickoff": kicks.min().strftime("%a %b %d, %-I:%M %p ET"),
        "run_by": (kicks.min() - timedelta(hours=LEAD_HOURS.get(slate, 2)))
                  .strftime("%a %b %d, %-I:%M %p ET"),
    }
