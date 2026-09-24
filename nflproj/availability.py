"""Feature availability contract.

Every feature must declare when its value becomes known. The pipeline refuses
to train or predict on any feature whose information was not available at the
prediction cutoff, so a future feature addition cannot silently reintroduce
leakage — an unregistered feature fails closed.

Availability classes
--------------------
STATIC      venue geometry, position, home/away — knowable indefinitely ahead
SCHEDULE    published when the schedule drops
PRIOR_GAME  derived only from games strictly before the one being predicted
GAMEDAY_AM  official injury report; final status ~90 min before kickoff
POSTGAME    known only after the game — never valid as a feature
MARKET_CLOSE closing lines; known at kickoff, NOT at an earlier cutoff
FORECAST    weather forecast issued before the cutoff
OBSERVED    measured game-day conditions — POSTGAME in disguise
"""
from __future__ import annotations

from dataclasses import dataclass, field

STATIC = "static"
SCHEDULE = "schedule"
PRIOR_GAME = "prior_game"
GAMEDAY_AM = "gameday_am"
FORECAST = "forecast"
MARKET_CLOSE = "market_close"
OBSERVED = "observed"
POSTGAME = "postgame"

# classes that are legitimately available before any pregame cutoff
ALLOWED = {STATIC, SCHEDULE, PRIOR_GAME, GAMEDAY_AM, FORECAST}
# classes that are only knowable at or after kickoff
FORBIDDEN = {MARKET_CLOSE, OBSERVED, POSTGAME}


@dataclass(frozen=True)
class Feature:
    name: str
    availability: str
    source: str
    note: str = ""
    group: str = "core"


def _rolling(name: str, source: str, note: str = "") -> Feature:
    return Feature(name, PRIOR_GAME, source, note or "shifted; excludes the current game")


REGISTRY: dict[str, Feature] = {}


def register(f: Feature) -> None:
    REGISTRY[f.name] = f


def register_many(features: list[Feature]) -> None:
    for f in features:
        register(f)


# --- static / schedule ------------------------------------------------------
register_many([
    Feature("is_home", SCHEDULE, "games.csv", "set when the schedule is published"),
    Feature("rest", SCHEDULE, "games.csv", "days since previous game"),
    Feature("div_game", SCHEDULE, "games.csv"),
    Feature("week", SCHEDULE, "games.csv"),
    Feature("pos_QB", STATIC, "roster"), Feature("pos_RB", STATIC, "roster"),
    Feature("pos_WR", STATIC, "roster"), Feature("pos_TE", STATIC, "roster"),
    Feature("games_played_prior", PRIOR_GAME, "player weeks"),
    # venue geometry replaces game-day roof state
    Feature("venue_fixed_dome", STATIC, "stadium metadata",
            "permanently enclosed venue; a property of the building", group="venue"),
    Feature("venue_retractable", STATIC, "stadium metadata",
            "retractable roof exists; its game-day position is NOT used", group="venue"),
])

# --- injury report ----------------------------------------------------------
register_many([
    Feature("status_doubtful", GAMEDAY_AM, "nflverse injuries"),
    Feature("status_questionable", GAMEDAY_AM, "nflverse injuries"),
    Feature("dnp", GAMEDAY_AM, "nflverse injuries", "practice participation"),
    Feature("vacated_target_share", GAMEDAY_AM, "injuries x prior usage"),
    Feature("vacated_rush_share", GAMEDAY_AM, "injuries x prior usage"),
])

# --- forbidden: retained for research, blocked as model inputs ---------------
register_many([
    Feature("spread_line", MARKET_CLOSE, "games.csv",
            "CLOSING spread; unknown at an earlier cutoff", group="market"),
    Feature("total_line", MARKET_CLOSE, "games.csv",
            "CLOSING total", group="market"),
    Feature("implied_team_total", MARKET_CLOSE, "derived from closing spread+total",
            group="market"),
    Feature("temp", OBSERVED, "games.csv",
            "measured game-day temperature, not a pregame forecast", group="weather"),
    Feature("wind", OBSERVED, "games.csv",
            "measured game-day wind, not a pregame forecast", group="weather"),
    Feature("is_dome", OBSERVED, "games.csv roof",
            "'closed' is a game-day decision at retractable venues", group="weather"),
])


WINDOW_SUFFIXES = ("_r3", "_r5", "_r8", "_exp")


def register_team_env(names: list[str]) -> None:
    """Team pace/tendency windows — prior games only, group 'team_env'.

    A team aggregate without a window suffix is a SAME-WEEK total (for a
    starting QB, team passing yards is essentially his own line), so it is
    registered as POSTGAME and blocked rather than silently admitted.
    """
    for n in names:
        if n in REGISTRY:
            continue
        if n.endswith(WINDOW_SUFFIXES):
            register(Feature(n, PRIOR_GAME, "team aggregates of prior games",
                             "shifted; excludes the current game", group="team_env"))
        else:
            register(Feature(n, POSTGAME, "same-week team aggregate",
                             "current-game total; not available pregame", group="team_env"))


def register_efficiency(names: list[str]) -> None:
    """Empirical-Bayes efficiency rates and prior-opportunity counters.

    Built from cumulative sums of games strictly before the one being
    predicted, so they carry no current-game information.
    """
    for n in names:
        if n not in REGISTRY:
            register(Feature(n, PRIOR_GAME, "cumulative prior-game production",
                             "EB-shrunk toward a positional prior", group="efficiency"))


def register_rollings(names: list[str]) -> None:
    """Rolling/expanding windows are generated programmatically in features.py."""
    for n in names:
        if n not in REGISTRY:
            register(_rolling(n, "player weeks / defense aggregates"))


class LeakageError(RuntimeError):
    pass


def audit(features: list[str], strict: bool = True) -> dict:
    """Check a feature list against the contract.

    Unregistered features fail closed: an unknown availability is treated as a
    violation rather than assumed safe.
    """
    unknown, violations, ok = [], [], []
    for name in features:
        f = REGISTRY.get(name)
        if f is None:
            unknown.append(name)
        elif f.availability in FORBIDDEN:
            violations.append((name, f.availability, f.note))
        else:
            ok.append(name)
    report = {"ok": ok, "violations": violations, "unregistered": unknown,
              "passed": not violations and not unknown}
    if strict and not report["passed"]:
        msg = []
        if violations:
            msg.append("features unavailable at prediction cutoff: "
                       + ", ".join(f"{n} ({a})" for n, a, _ in violations))
        if unknown:
            msg.append("unregistered features (fail closed): " + ", ".join(unknown))
        raise LeakageError("; ".join(msg))
    return report


def allowed(features: list[str], exclude_groups: set[str] | None = None) -> list[str]:
    """Filter a feature list down to contract-compliant entries."""
    exclude_groups = exclude_groups or set()
    out = []
    for name in features:
        f = REGISTRY.get(name)
        if f is None or f.availability in FORBIDDEN or f.group in exclude_groups:
            continue
        out.append(name)
    return out


def group_of(name: str) -> str:
    f = REGISTRY.get(name)
    return f.group if f else "unregistered"
