# odds_client_ncaaf.py
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone as tz
from typing import Any, Dict, List, Optional
import requests
from cache_ttl import get as cache_get, setex as cache_setex
from markets_ncaaf import NCAAF_SPORT_KEY
import perf

BASE = "https://api.the-odds-api.com"
API_KEY = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or ""
REGIONS = os.getenv("ODDS_REGIONS", "us")
ODDS_FORMAT = "american"
PREFERRED_BOOKMAKER_KEYS = [b for b in os.getenv("ODDS_PREFERRED_BOOKS","").lower().split(",") if b]

# --- BEGIN: resilient HTTP + backoff for NCAAF odds ---
import time, random

# Pooled session (reuse sockets)
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=0)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
_session.headers.update({"User-Agent":"MoraBets/1.0 (+NCAAF v4)"})

# Backoff / pacing knobs (env-tunable, safe defaults)
BACKOFF_BASE_MS  = int(os.getenv("ODDS_BACKOFF_BASE_MS", "250"))     # first 429 wait
BACKOFF_MAX_MS   = int(os.getenv("ODDS_BACKOFF_MAX_MS",  "4000"))    # cap
MAX_RETRIES      = int(os.getenv("ODDS_MAX_RETRIES",      "4"))      # attempts per call
EVENT_PAUSE_MS   = int(os.getenv("ODDS_EVENT_PAUSE_MS",   "120"))    # tiny pause after each event-odds fetch

# Cache TTLs, if not already defined in file
try:
    CACHE_SEC_EVENTS
except NameError:
    CACHE_SEC_EVENTS = int(os.getenv("CACHE_SEC_EVENTS_NCAAF", "45"))
try:
    CACHE_SEC_EVENT_ODDS
except NameError:
    CACHE_SEC_EVENT_ODDS = int(os.getenv("CACHE_SEC_EVENT_ODDS_NCAAF", "45"))

def _get_json(path: str, **params):
    # Keep your existing BASE, API_KEY, etc.
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY/THE_ODDS_API_KEY not set")

    p = {"apiKey": API_KEY}
    p.update(params)

    delay_ms = BACKOFF_BASE_MS
    last_err = None
    url = f"{BASE}/v4{path}" if "/v4/" not in BASE and not path.startswith("/v4/") else f"{BASE}{path}"

    for attempt in range(MAX_RETRIES):
        try:
            r = _session.get(url, params=p, timeout=20)
            if r.status_code == 429:
                # Respect Retry-After and add jitter
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(float(retry_after), delay_ms/1000.0)
                    except:
                        sleep_s = delay_ms/1000.0
                else:
                    sleep_s = delay_ms/1000.0
                time.sleep(sleep_s + random.uniform(0, 0.25))
                delay_ms = min(delay_ms * 2, BACKOFF_MAX_MS)
                last_err = f"429 backoff (attempt {attempt+1}/{MAX_RETRIES})"
                continue

            r.raise_for_status()

            # Optional: record remaining requests for telemetry
            rem = r.headers.get("X-Requests-Remaining") or r.headers.get("x-requests-remaining")
            if rem and rem.isdigit():
                try:
                    cache_setex("odds:ncaaf:remaining", 30, int(rem))
                except Exception:
                    pass

            return r.json()

        except requests.RequestException as e:
            last_err = str(e)
            # small progressive backoff even on transient errors
            time.sleep((delay_ms/1000.0) + random.uniform(0, 0.1))
            delay_ms = min(delay_ms * 2, BACKOFF_MAX_MS)

    raise RuntimeError(f"Odds API request failed after retries: {last_err}")
# --- END: resilient HTTP + backoff for NCAAF odds ---

def list_events_ncaaf(hours_ahead: int = 48, date: Optional[str] = None) -> List[Dict[str, Any]]:
    with perf.span("ncaaf:list_events", {"ha": hours_ahead, "date": date or ""}):
        if date:
            start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=tz.utc, hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            key = f"ncaaf:events:date:{date}"
        else:
            start = datetime.utcnow().replace(tzinfo=tz.utc, microsecond=0)
            end = start + timedelta(hours=hours_ahead)
            key = f"ncaaf:events:ha:{hours_ahead}"
        hit = cache_get(key)
        if hit is not None: return hit
        data = _get_json(
            f"/sports/{NCAAF_SPORT_KEY}/events",
            commenceTimeFrom=start.isoformat().replace("+00:00","Z"),
            commenceTimeTo=end.isoformat().replace("+00:00","Z"),
        )
        cache_setex(key, CACHE_SEC_EVENTS, data)
        try: perf.mark("ncaaf.events", len(data))
        except Exception: pass
        return data

def event_odds_ncaaf(event_id: str, markets: List[str]) -> Dict[str, Any]:
    with perf.span("ncaaf:event_odds", {"eid": event_id, "mk": len(markets)}):
        mk = ",".join(markets)
        key = f"ncaaf:event:{event_id}:mk:{mk}"
        hit = cache_get(key)
        if hit is not None: return hit
        base_params = {"regions": REGIONS, "oddsFormat": ODDS_FORMAT, "markets": mk}
        params = dict(base_params)
        if PREFERRED_BOOKMAKER_KEYS:
            params["bookmakers"] = ",".join(PREFERRED_BOOKMAKER_KEYS)
        data = _get_json(f"/sports/{NCAAF_SPORT_KEY}/events/{event_id}/odds", **params)
        if not (data.get("bookmakers") or []):
            data = _get_json(f"/sports/{NCAAF_SPORT_KEY}/events/{event_id}/odds", **base_params)
        cache_setex(key, CACHE_SEC_EVENT_ODDS, data)

        # Tiny pacing to avoid API rate spikes (env-tunable)
        if EVENT_PAUSE_MS > 0:
            time.sleep(EVENT_PAUSE_MS / 1000.0)

        return data
