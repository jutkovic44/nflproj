"""Data acquisition from nflverse. All sources are free and public."""
from __future__ import annotations

import os
import ssl
import shutil
import urllib.request

import pandas as pd

DATA_DIR = os.environ.get("NFLPROJ_DATA", os.path.expanduser("~/nflproj_data"))
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

SKILL_POS = ["QB", "RB", "WR", "TE"]


def ssl_contexts() -> list[ssl.SSLContext]:
    """Trust stores to try, in order.

    The system store comes first because it is the one that carries corporate
    or proxy certificates. certifi is the fallback: macOS python.org builds
    ship without linking the system roots, so every HTTPS call there dies with
    CERTIFICATE_VERIFY_FAILED until either "Install Certificates.command" is
    run or certifi supplies its own trust store.
    """
    out = [ssl.create_default_context()]
    try:
        import certifi
        out.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    return out


def urlopen(url: str, timeout: float = 60.0):
    """Open a URL, trying each available trust store before giving up."""
    req = urllib.request.Request(url, headers={"User-Agent": "nflproj/0.2"})
    last = None
    for ctx in ssl_contexts():
        try:
            return urllib.request.urlopen(req, context=ctx, timeout=timeout)
        except ssl.SSLCertVerificationError as e:
            last = e
            continue
    raise last if last else RuntimeError("no SSL context available")


def _cache(url: str, fname: str, refresh: bool = False) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, fname)
    if refresh or not os.path.exists(path):
        with urlopen(url) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
    return path


def load_player_weeks(seasons) -> pd.DataFrame:
    """Weekly player box scores."""
    frames = []
    for yr in seasons:
        p = _cache(f"{BASE}/stats_player/stats_player_week_{yr}.parquet", f"stats_{yr}.parquet")
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df = df[df.season_type == "REG"]
    df = df[df.position.isin(SKILL_POS)]
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(0)
    return df.reset_index(drop=True)


def load_injuries(seasons) -> pd.DataFrame:
    """Official game-status injury reports (known before kickoff)."""
    frames = []
    for yr in seasons:
        try:
            p = _cache(f"{BASE}/injuries/injuries_{yr}.parquet", f"inj_{yr}.parquet")
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id", "report_status", "practice_status"])
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={"gsis_id": "player_id"})


def load_snaps(seasons) -> pd.DataFrame:
    frames = []
    for yr in seasons:
        try:
            p = _cache(f"{BASE}/snap_counts/snap_counts_{yr}.parquet", f"snaps_{yr}.parquet")
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_weekly_rosters(seasons) -> pd.DataFrame:
    """Per-week roster status: ACT, INA, RES, DEV, CUT.

    Week-level only — nflverse carries no transaction timestamp, so the moment a
    status changed is not recoverable. Inactive (INA) is a game-day decision
    published ~90 minutes before kickoff.
    """
    frames = []
    for yr in seasons:
        try:
            p = _cache(f"{BASE}/weekly_rosters/roster_weekly_{yr}.parquet", f"wroster_{yr}.parquet")
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={"gsis_id": "player_id"})


def load_depth_charts(seasons) -> pd.DataFrame:
    """Depth charts, normalised across two incompatible nflverse schemas.

    2025+  : daily snapshots carrying a real ISO timestamp (`dt`). This is the
             only genuinely timestamped role source in the project, and the
             only one whose information state can be reconstructed exactly at
             a historical cutoff.
    <=2024 : week-level rows with no timestamp. Availability cannot be
             verified, so these rows are marked accordingly and are not treated
             as timestamped evidence.
    """
    out = []
    for yr in seasons:
        try:
            p = _cache(f"{BASE}/depth_charts/depth_charts_{yr}.parquet", f"depth_{yr}.parquet")
            d = pd.read_parquet(p)
        except Exception:
            continue
        if "dt" in d.columns:                      # 2025+ timestamped schema
            d = d.rename(columns={"gsis_id": "player_id", "pos_abb": "position",
                                  "pos_rank": "depth_rank", "player_name": "full_name"})
            d["available_at"] = pd.to_datetime(d["dt"], utc=True)
            d["timestamped"] = True
            d["season"] = yr
            out.append(d[["season", "team", "player_id", "full_name", "position",
                          "depth_rank", "available_at", "timestamped"]])
        else:                                       # <=2024 week-level schema
            d = d.rename(columns={"gsis_id": "player_id", "club_code": "team",
                                  "depth_team": "depth_rank"})
            d["available_at"] = pd.NaT
            d["timestamped"] = False
            keep = ["season", "week", "team", "player_id", "full_name", "position",
                    "depth_rank", "available_at", "timestamped"]
            out.append(d[[c for c in keep if c in d.columns]])
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def load_games(refresh: bool = False) -> pd.DataFrame:
    """Schedule + closing spread/total + weather. Weather is observed after the
    fact in this file; at projection time supply a forecast instead (see README)."""
    p = _cache(GAMES_URL, "games.csv", refresh=refresh)
    return pd.read_csv(p, low_memory=False)


def refresh_all(seasons):
    for yr in seasons:
        for tag, fn in [
            (f"stats_player/stats_player_week_{yr}.parquet", f"stats_{yr}.parquet"),
            (f"injuries/injuries_{yr}.parquet", f"inj_{yr}.parquet"),
            (f"snap_counts/snap_counts_{yr}.parquet", f"snaps_{yr}.parquet"),
        ]:
            try:
                _cache(f"{BASE}/{tag}", fn, refresh=True)
            except Exception as e:
                msg = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in msg:
                    print(f"\n  SSL certificate verification failed ({fn}).\n"
                          "  Fix with either:\n"
                          "    pip3 install certifi\n"
                          "    '/Applications/Python 3.13/Install Certificates.command'\n")
                    raise SystemExit(1)
                print(f"  skip {fn}: {e}")
    _cache(GAMES_URL, "games.csv", refresh=True)
