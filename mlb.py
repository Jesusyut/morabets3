# contextual/mlb.py
# MLB-specific contextual enrichment for Mora Bets (safe, additive).
# Public API (just two functions):
#   - get_mlb_contextual_hit_rate(player_name, stat_type, threshold) -> dict
#   - get_mlb_contextual_hit_rate_cached(player_name, stat_type, threshold) -> dict
#
# Design goals:
# - No breaking changes to existing no-vig pipeline.
# - Short timeouts, retry, graceful fallbacks.
# - Optional Redis cache by day; fallback to in-process LRU memo.
# - Minimal stat mapping for batter props; extend as needed.
from __future__ import annotations
import os, math, time, json, hashlib
from datetime import date
from functools import lru_cache
from typing import Any, Dict, List
import requests

MLB_API = "https://statsapi.mlb.com/api/v1"
TIMEOUT = float(os.getenv("MLB_TIMEOUT","4"))

_session = requests.Session()
_session.headers.update({"User-Agent":"MoraBets/ctx-mlb/1.0"})

# Optional Redis cache
try:
    from redis import Redis
    REDIS = Redis.from_url(os.getenv("REDIS_URL","redis://localhost:6379/0"), decode_responses=True)
except Exception:
    REDIS = None

# Map common prop stats to MLB keys
STAT_KEY_MAP = {
    # hits & totals
    "batter_hits": "hits", "hits": "hits",
    "batter_total_bases": "totalBases", "total_bases": "totalBases", "tb":"totalBases",
    # HR / runs / RBI
    "batter_home_runs": "homeRuns", "home_runs":"homeRuns",
    "batter_runs": "runs", "runs":"runs",
    "batter_runs_batted_in": "rbi", "rbi":"rbi",
    # walks / steals
    "batter_walks": "baseOnBalls", "walks":"baseOnBalls",
    "batter_stolen_bases": "stolenBases", "stolen_bases":"stolenBases",
}

def _cache_key(player: str, stat: str, th: float) -> str:
    today = date.today().isoformat()
    raw = f"{today}|{player}|{stat}|{th}"
    return "ctxmlb:" + hashlib.md5(raw.encode()).hexdigest()

@lru_cache(maxsize=2048)
def _memo_get(key: str) -> str:
    # lightweight local memo store using key<->payload double insertion
    return key

def _get(url: str, params: dict | None = None) -> requests.Response:
    # tiny retry
    for i in range(3):
        try:
            r = _session.get(url, params=params, timeout=TIMEOUT)
            if r.ok:
                return r
        except Exception:
            pass
        time.sleep(0.15 * (i + 1))
    raise RuntimeError(f"MLB request failed: {url}")

def _resolve_player_id(name: str) -> int:
    r = _get(f"{MLB_API}/people/search", params={"names": name})
    js = r.json() or {}
    people = js.get("people") or []
    if not people:
        raise ValueError(f"player not found: {name}")
    return int(people[0]["id"])

def _game_logs(pid: int, season: int, group: str = "hitting") -> List[Dict[str, Any]]:
    r = _get(f"{MLB_API}/people/{pid}/stats", params={"stats": "gameLog", "season": season, "group": group})
    js = r.json() or {}
    stats = js.get("stats") or []
    if not stats: return []
    splits = stats[0].get("splits") or []
    return splits

def _confidence_label(rate: float, n: int) -> str:
    if n < 6:
        return "low"
    # simple z-score style separation from 0.5 using binomial SE
    se = math.sqrt(max(rate * (1 - rate), 1e-9) / max(n, 1))
    z = abs(rate - 0.5) / max(se, 1e-9)
    if n >= 8 and z >= 1.5:
        return "high"
    if z >= 0.8:
        return "medium"
    return "low"

def get_mlb_contextual_hit_rate(player_name: str, stat_type: str, threshold: float) -> Dict[str, Any]:
    """Compute last-10 game 'over threshold' rate for a batter stat.
    Returns a dict with hit_rate, sample_size, confidence, and threshold.
    Never raises to the caller; use cached wrapper to get safe fallbacks.
    """
    pid = _resolve_player_id(player_name)
    stat_key = STAT_KEY_MAP.get((stat_type or "").lower(), stat_type)

    logs = _game_logs(pid, date.today().year, "hitting")
    # If early season or injured, peek a bit into last season
    if len(logs) < 10:
        logs += _game_logs(pid, date.today().year - 1, "hitting")

    vals: List[float] = []
    for s in logs[:10]:
        st = s.get("stat") or {}
        vals.append(float(st.get(stat_key, 0) or 0))

    n = len(vals)
    if n == 0:
        return {"hit_rate": 0.0, "sample_size": 0, "confidence": "low", "threshold": float(threshold)}

    overs = sum(1 for v in vals if v >= float(threshold))
    rate = overs / n
    return {
        "hit_rate": round(rate, 4),
        "sample_size": n,
        "confidence": _confidence_label(rate, n),
        "threshold": float(threshold),
    }

def get_mlb_contextual_hit_rate_cached(player_name: str, stat_type: str, threshold: float) -> Dict[str, Any]:
    """Redis + memo cached accessor. Always returns a payload; never raises."""
    ck = _cache_key(player_name, stat_type, float(threshold))

    # Redis layer (preferred)
    if REDIS:
        cached = REDIS.get(ck)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass

    # In-process memo fallback (stores both key and payload as keys)
    try:
        payload = json.loads(_memo_get(ck))
        return payload
    except Exception:
        pass

    # Live compute with guard
    try:
        payload = get_mlb_contextual_hit_rate(player_name, stat_type, threshold)
    except Exception:
        payload = {"hit_rate": 0.0, "sample_size": 0, "confidence": "low", "threshold": float(threshold)}

    # Write back
    try:
        if REDIS:
            # 6 hours TTL; adjust if you run collectors
            REDIS.setex(ck, 6 * 3600, json.dumps(payload))
        _memo_get.cache_clear()
        _memo_get(json.dumps(payload)); _memo_get(ck)
    except Exception:
        pass

    return payload
