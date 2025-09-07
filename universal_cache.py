# universal_cache.py
import os, json, time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Dict, Callable
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

PHX_TZ = ZoneInfo("America/Phoenix") if ZoneInfo else None

# Optional Redis
REDIS_URL = os.getenv("REDIS_URL", "")
_redis = None
if REDIS_URL:
    try:
        import redis
        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        _redis = None

# In-proc fallback
_mem: Dict[str, Dict[str, Any]] = {}

CACHE_PREFIX = os.getenv("CACHE_PREFIX", "")
CACHE_VERSION = os.getenv("CACHE_VERSION", "v1")  # bump to invalidate

SLOTS = [8, 13, 18]  # local hours

def _now_local() -> datetime:
    utc = datetime.now(timezone.utc)
    return utc.astimezone(PHX_TZ) if PHX_TZ else utc

def current_slot(dt: Optional[datetime] = None) -> tuple[str, datetime]:
    dt = dt or _now_local()
    h = dt.hour
    if h < 8:
        slot = "night"
        next_b = dt.replace(hour=8, minute=0, second=0, microsecond=0)
    elif h < 13:
        slot = "morning"
        next_b = dt.replace(hour=13, minute=0, second=0, microsecond=0)
    elif h < 18:
        slot = "afternoon"
        next_b = dt.replace(hour=18, minute=0, second=0, microsecond=0)
    else:
        slot = "night"
        next_b = (dt + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    return slot, next_b

def _ttl_to_next_boundary(next_boundary: datetime) -> int:
    now = _now_local()
    ttl = int((next_boundary - now).total_seconds())
    return max(ttl, 60)

def slot_key(namespace: str, league: str, suffix: str = "") -> str:
    dt = _now_local()
    d = dt.date().isoformat()
    slot, _ = current_slot(dt)
    prefix = f"{CACHE_PREFIX}:" if CACHE_PREFIX else ""
    suf = f":{suffix}" if suffix else ""
    return f"{prefix}{CACHE_VERSION}:{namespace}:{league}:{d}:{slot}{suf}"

def get_json(key: str) -> Optional[Any]:
    if _redis:
        raw = _redis.get(key)
        return json.loads(raw) if raw else None
    rec = _mem.get(key)
    if rec and rec["exp"] > time.time():
        return json.loads(rec["val"])
    return None

def set_json(key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
    _, next_b = current_slot()
    ttl = ttl_seconds if ttl_seconds is not None else _ttl_to_next_boundary(next_b)
    raw = json.dumps(value)
    if _redis:
        _redis.setex(key, ttl, raw)
    else:
        _mem[key] = {"exp": time.time() + ttl, "val": raw}

def get_or_set_slot(namespace: str, league: str, fetcher: Callable[[], Any], suffix: str = "") -> Any:
    k = slot_key(namespace, league, suffix=suffix)
    cached = get_json(k)
    if cached is not None:
        return cached
    data = fetcher()
    set_json(k, data)
    return data
