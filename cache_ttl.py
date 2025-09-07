# cache_ttl.py
from __future__ import annotations
import os, time, json
from typing import Any, Optional

_REDIS_URL = os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_REST_URL")
_USE_REDIS = False
_r = None

if _REDIS_URL:
    try:
        import redis  # pip install redis
        _r = redis.from_url(_REDIS_URL, decode_responses=True)
        _USE_REDIS = True
    except Exception:
        _r = None
        _USE_REDIS = False

_mem: dict[str, tuple[float, str]] = {}

# counters
_hits = 0
_miss = 0
_sets = 0
def metrics():  # for /_perf/recent inspection
    return {"hits": _hits, "miss": _miss, "sets": _sets}

def setex(key: str, ttl_sec: int, value: Any) -> None:
    global _sets; _sets += 1
    s = json.dumps(value, separators=(",", ":"))
    if _USE_REDIS and _r:
        try:
            _r.setex(key, ttl_sec, s)
            return
        except Exception:
            pass
    _mem[key] = (time.time() + ttl_sec, s)

def get(key: str) -> Optional[Any]:
    global _hits, _miss
    if _USE_REDIS and _r:
        try:
            s = _r.get(key)
            if s is not None:
                _hits += 1
                return json.loads(s)
            else:
                _miss += 1
                return None
        except Exception:
            pass
    tup = _mem.get(key)
    if not tup:
        _miss += 1
        return None
    exp, s = tup
    if time.time() > exp:
        _mem.pop(key, None)
        _miss += 1
        return None
    try:
        _hits += 1
        return json.loads(s)
    except Exception:
        _miss += 1
        return None
