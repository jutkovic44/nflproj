"""Game-time weather from free, open sources — consensus instead of one guess.

Three independent free sources, no API keys anywhere:

  NWS (api.weather.gov)     official US government forecast, refined by the
                            local forecast office, ~7 days out
  Open-Meteo multi-model    ECMWF IFS, NOAA GFS and DWD ICON pulled as separate
                            members, 16 days out
  Open-Meteo ERA5 archive   reanalysis of what actually happened, for
                            backfilling historical games

Averaging independent models beats any single one, and the spread between them
is itself information: when ECMWF and GFS disagree by 9 mph on wind, that is a
low-confidence forecast and the dashboard says so instead of printing false
precision.

Everything degrades gracefully. No network, no crash — just a summary saying
the forecast is unavailable.
"""
from __future__ import annotations

import json
import os
import statistics
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

UA = "nflproj/0.3 (personal projection tool)"

# populated during a forecast call so failures can be explained rather than
# silently collapsing to "no source reachable"
ERRORS: list[str] = []
CACHE = os.path.join(os.environ.get("NFLPROJ_DATA", os.path.expanduser("~/nflproj_data")),
                     "weather_cache.json")
CACHE_TTL_MIN = 90

# lat, lon, venue, fixed_roof
STADIUMS = {
    "ARI": (33.5277, -112.2626, "State Farm Stadium", True),
    "ATL": (33.7554, -84.4008, "Mercedes-Benz Stadium", True),
    "BAL": (39.2780, -76.6227, "M&T Bank Stadium", False),
    "BUF": (42.7738, -78.7870, "Highmark Stadium", False),
    "CAR": (35.2258, -80.8528, "Bank of America Stadium", False),
    "CHI": (41.8623, -87.6167, "Soldier Field", False),
    "CIN": (39.0955, -84.5161, "Paycor Stadium", False),
    "CLE": (41.5061, -81.6995, "Huntington Bank Field", False),
    "DAL": (32.7473, -97.0945, "AT&T Stadium", True),
    "DEN": (39.7439, -105.0201, "Empower Field", False),
    "DET": (42.3400, -83.0456, "Ford Field", True),
    "GB": (44.5013, -88.0622, "Lambeau Field", False),
    "HOU": (29.6847, -95.4107, "NRG Stadium", True),
    "IND": (39.7601, -86.1639, "Lucas Oil Stadium", True),
    "JAX": (30.3239, -81.6373, "EverBank Stadium", False),
    "KC": (39.0489, -94.4839, "Arrowhead Stadium", False),
    "LA": (33.9535, -118.3392, "SoFi Stadium", True),
    "LAC": (33.9535, -118.3392, "SoFi Stadium", True),
    "LV": (36.0909, -115.1833, "Allegiant Stadium", True),
    "MIA": (25.9580, -80.2389, "Hard Rock Stadium", False),
    "MIN": (44.9736, -93.2575, "U.S. Bank Stadium", True),
    "NE": (42.0909, -71.2643, "Gillette Stadium", False),
    "NO": (29.9511, -90.0812, "Caesars Superdome", True),
    "NYG": (40.8135, -74.0745, "MetLife Stadium", False),
    "NYJ": (40.8135, -74.0745, "MetLife Stadium", False),
    "PHI": (39.9008, -75.1675, "Lincoln Financial Field", False),
    "PIT": (40.4468, -80.0158, "Acrisure Stadium", False),
    "SEA": (47.5952, -122.3316, "Lumen Field", False),
    "SF": (37.4030, -121.9700, "Levi's Stadium", False),
    "TB": (27.9759, -82.5033, "Raymond James Stadium", False),
    "TEN": (36.1665, -86.7713, "Nissan Stadium", False),
    "WAS": (38.9077, -76.8645, "Northwest Stadium", False),
}

OM_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
OM_VARS = ("temperature_2m,wind_speed_10m,wind_gusts_10m,"
           "precipitation_probability,precipitation,weather_code")
OM_BASIC_VARS = "temperature_2m,wind_speed_10m,weather_code"
OM_FORECAST = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&hourly={vars}&models={model}"
               "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
               "&forecast_days=16&timezone=UTC")
OM_ARCHIVE = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
              "&start_date={d}&end_date={d}"
              "&hourly=temperature_2m,wind_speed_10m,precipitation"
              "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
              "&timezone=America%2FNew_York")

WMO = {0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
       45: "Fog", 48: "Freezing fog", 51: "Light drizzle", 53: "Drizzle", 55: "Drizzle",
       61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain",
       67: "Freezing rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow",
       77: "Snow grains", 80: "Rain showers", 81: "Rain showers", 82: "Heavy showers",
       85: "Snow showers", 86: "Snow showers", 95: "Thunderstorms",
       96: "Thunderstorms", 99: "Thunderstorms"}


def _get(url: str, timeout: float = 10.0) -> dict:
    import ssl as _ssl
    from .data import ssl_contexts
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    last = None
    for ctx in ssl_contexts():
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return json.load(r)
        except _ssl.SSLCertVerificationError as e:
            last = e
            continue
    raise last if last else RuntimeError("no SSL context available")


def _cache_read(key: str):
    try:
        with open(CACHE) as f:
            blob = json.load(f)
        row = blob.get(key)
        if not row:
            return None
        age = datetime.now() - datetime.fromisoformat(row["_t"])
        return row["v"] if age < timedelta(minutes=CACHE_TTL_MIN) else None
    except Exception:
        return None


def _cache_write(key: str, value) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        blob = {}
        if os.path.exists(CACHE):
            with open(CACHE) as f:
                blob = json.load(f)
        blob[key] = {"_t": datetime.now().isoformat(), "v": value}
        with open(CACHE, "w") as f:
            json.dump(blob, f)
    except Exception:
        pass


def _indoor(roof: str | None, home: str) -> bool:
    if roof in ("dome", "closed"):
        return True
    fixed = STADIUMS.get(home, (0, 0, "", False))[3]
    return bool(fixed) and roof != "open"


def _as_utc(kickoff: datetime) -> datetime:
    """Kickoff times come from the schedule as naive Eastern clock time.
    Stadium-local forecasts are in their own zone, so everything is compared
    in UTC. Previously this stripped tzinfo and compared clock faces, which
    silently missed every game outside the Eastern time zone."""
    aware = kickoff if kickoff.tzinfo else kickoff.replace(tzinfo=ET)
    return aware.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def from_nws(lat: float, lon: float, kickoff: datetime) -> dict | None:
    """National Weather Service hourly forecast. Official, free, no key."""
    try:
        pts = _get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")
        js = _get(pts["properties"]["forecastHourly"])
        target = _as_utc(kickoff)
        for p in js["properties"]["periods"]:
            start = datetime.fromisoformat(p["startTime"]).astimezone(timezone.utc)
            start = start.replace(minute=0, second=0, microsecond=0)
            if start == target:
                mph = [int(t) for t in str(p.get("windSpeed") or "").split() if t.isdigit()]
                pop = (p.get("probabilityOfPrecipitation") or {}).get("value")
                return {"source": "NWS", "temp": float(p["temperature"]),
                        "wind": float(max(mph)) if mph else None,
                        "precip": None if pop is None else float(pop),
                        "condition": p.get("shortForecast")}
        ERRORS.append("NWS: kickoff hour not in forecast periods")
        return None
    except Exception as e:
        ERRORS.append(f"NWS: {type(e).__name__} {e}")
        return None


def _om_one(lat: float, lon: float, stamp: str, model: str) -> dict | None:
    """One model, requested on its own.

    Asking for all three at once is fragile: not every model exposes every
    variable (precipitation_probability in particular), and a single
    unsupported variable makes Open-Meteo reject the whole request, which
    previously took all three members down together. Each model is now asked
    separately, with a reduced variable set as a fallback.
    """
    for var_set in (OM_VARS, OM_BASIC_VARS):
        try:
            js = _get(OM_FORECAST.format(lat=lat, lon=lon, vars=var_set, model=model))
        except Exception as e:
            ERRORS.append(f"{model}: {type(e).__name__} {e}")
            continue
        hourly = js.get("hourly", {})
        times = hourly.get("time", [])
        if stamp not in times:
            ERRORS.append(f"{model}: kickoff hour {stamp} not in forecast range")
            return None
        i = times.index(stamp)

        def pick(var):
            # single-model responses may or may not suffix the model name
            for key in (var, f"{var}_{model}"):
                series = hourly.get(key)
                if series and i < len(series) and series[i] is not None:
                    return series[i]
            return None

        temp, wind = pick("temperature_2m"), pick("wind_speed_10m")
        if temp is None and wind is None:
            continue
        code = pick("weather_code")
        return {"source": model.split("_")[0].upper(), "temp": temp, "wind": wind,
                "gust": pick("wind_gusts_10m"),
                "precip": pick("precipitation_probability"),
                "condition": WMO.get(int(code)) if code is not None else None}
    return None


def from_open_meteo(lat: float, lon: float, kickoff: datetime) -> list[dict]:
    """ECMWF, GFS and ICON as separate members so they can be compared."""
    stamp = _as_utc(kickoff).strftime("%Y-%m-%dT%H:00")
    out = []
    for model in OM_MODELS:
        m = _om_one(lat, lon, stamp, model)
        if m:
            out.append(m)
    return out


def historical(home_team: str, kickoff: datetime) -> dict | None:
    """ERA5 reanalysis — what conditions actually were. Backfills historical
    games whose weather is missing from the schedule file."""
    loc = STADIUMS.get(home_team)
    if loc is None:
        return None
    lat, lon, _, _ = loc
    key = f"era5:{home_team}:{kickoff:%Y%m%d%H}"
    hit = _cache_read(key)
    if hit is not None:
        return hit
    try:
        js = _get(OM_ARCHIVE.format(lat=lat, lon=lon, d=kickoff.strftime("%Y-%m-%d")))
        h = js["hourly"]
        i = h["time"].index(kickoff.strftime("%Y-%m-%dT%H:00"))  # archive uses ET
        val = {"temp": h["temperature_2m"][i], "wind": h["wind_speed_10m"][i],
               "precip_in": h["precipitation"][i]}
        _cache_write(key, val)
        return val
    except Exception:
        return None


def _blend(members: list[dict], field: str):
    vals = [m[field] for m in members if m.get(field) is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), (max(vals) - min(vals) if len(vals) > 1 else 0.0)


def forecast(home_team: str, kickoff: datetime, roof: str | None = None) -> dict:
    """Consensus conditions at kickoff. Always safe to render."""
    if _indoor(roof, home_team):
        return {"indoors": True,
                "venue": STADIUMS.get(home_team, (0, 0, "Indoors", True))[2],
                "summary": "Indoors — no weather factor", "sources": []}

    loc = STADIUMS.get(home_team)
    if loc is None:
        return {"indoors": False, "summary": "Venue unknown", "sources": []}
    lat, lon, venue, _ = loc

    key = f"fc:{home_team}:{kickoff:%Y%m%d%H}"
    hit = _cache_read(key)
    if hit is not None:
        return hit

    ERRORS.clear()
    members = []
    nws = from_nws(lat, lon, kickoff)
    if nws:
        members.append(nws)
    members += from_open_meteo(lat, lon, kickoff)

    if not members:
        return {"indoors": False, "venue": venue, "sources": [],
                "errors": list(ERRORS),
                "summary": "Forecast unavailable — " + (ERRORS[0] if ERRORS else "no source reachable")}

    temp, temp_spread = _blend(members, "temp")
    wind, wind_spread = _blend(members, "wind")
    precip, _ = _blend(members, "precip")
    gust, _ = _blend(members, "gust")
    conds = [m["condition"] for m in members if m.get("condition")]
    condition = max(set(conds), key=conds.count) if conds else None

    result = {
        "indoors": False, "venue": venue,
        "temp": None if temp is None else round(temp),
        "wind": None if wind is None else round(wind),
        "gust": None if gust is None else round(gust),
        "precip": None if precip is None else round(precip),
        "condition": condition,
        "wind_spread": None if wind_spread is None else round(wind_spread, 1),
        "temp_spread": None if temp_spread is None else round(temp_spread, 1),
        "confidence": _confidence(wind_spread, temp_spread, len(members)),
        "sources": [{"name": m["source"],
                     "temp": None if m.get("temp") is None else round(m["temp"]),
                     "wind": None if m.get("wind") is None else round(m["wind"])}
                    for m in members],
        "summary": _summary(temp, wind, gust, precip, condition),
        "impact": _impact(wind, precip, temp, gust),
        "errors": list(ERRORS),
    }
    _cache_write(key, result)
    return result


def _summary(temp, wind, gust, precip, condition) -> str:
    bits = []
    if temp is not None:
        bits.append(f"{round(temp)}\u00b0F")
    if wind is not None:
        w = f"{round(wind)} mph wind"
        if gust and gust - wind >= 7:
            w += f" (gusts {round(gust)})"
        bits.append(w)
    if condition:
        bits.append(condition)
    if precip and precip >= 20:
        bits.append(f"{round(precip)}% precip")
    return " \u00b7 ".join(bits) if bits else "Conditions unavailable"


def _confidence(wind_spread, temp_spread, n) -> str:
    if n < 2:
        return "single source"
    if (wind_spread or 0) >= 8 or (temp_spread or 0) >= 12:
        return "low \u2014 models disagree"
    if (wind_spread or 0) >= 4 or (temp_spread or 0) >= 6:
        return "moderate"
    return "high \u2014 models agree"


def _impact(wind, precip, temp, gust) -> str | None:
    notes = []
    if wind and wind >= 15:
        notes.append("wind at 15+ mph suppresses deep passing and field goals")
    elif gust and gust >= 25:
        notes.append("gusts above 25 mph make the deep ball unreliable")
    if precip and precip >= 50:
        notes.append("rain likely \u2014 favours volume rushing")
    if temp is not None and temp <= 25:
        notes.append("severe cold, expect a run-leaning script")
    return "; ".join(notes) if notes else None
