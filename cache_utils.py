# cache_utils.py — fast, non-blocking Redis with memory fallback
from __future__ import annotations
import os, json, time, logging
from typing import Any, Optional

LOG = logging.getLogger("cache")

try:
    from redis import Redis  # type: ignore
except Exception:
    Redis = None  # type: ignore

REDIS_URL = os.environ.get("REDIS_URL", "")

_redis = None          # type: ignore
redis_healthy = False
memory_cache: dict[str, Any] = {}
_last_health_check = 0.0

def _connect(url: str) -> bool:
    global _redis, redis_healthy
    if not Redis or not url:
        _redis = None
        redis_healthy = False
        return False
    try:
        _redis = Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2, health_check_interval=30)
        _redis.ping()
        redis_healthy = True
        LOG.info("✅ Redis connected: %s", url)
        return True
    except Exception as e:
        LOG.warning("❌ Redis connect failed for %s: %s", url, e)
        _redis = None
        redis_healthy = False
        return False

def init_cache() -> bool:
    """Attempt connection to REDIS_URL; otherwise leave memory fallback."""
    url = os.environ.get("REDIS_URL", "")
    ok = _connect(url) if url else False
    if not ok and os.environ.get("REDIS_FALLBACK_LOCAL","false").lower() in ("1","true","on"):
        # Optional: try localhost
        ok = _connect("redis://localhost:6379/0")
    return ok

def _health_tick(period: float = 30.0) -> None:
    """Ping Redis at most every `period` seconds; auto-retry if disconnected."""
    global _last_health_check, redis_healthy
    now = time.time()
    if now - _last_health_check < period:
        return
    _last_health_check = now
    if _redis is None:
        # Try to connect if env present
        init_cache()
        return
    try:
        _redis.ping()
        if not redis_healthy:
            LOG.info("✅ Redis reconnected")
        redis_healthy = True
    except Exception as e:
        if redis_healthy:
            LOG.warning("❌ Redis lost: %s", e)
        redis_healthy = False
        # Try reconnect once
        init_cache()

def cache_set(key: str, value: Any, ex: Optional[int] = None) -> bool:
    """Set key (Redis if healthy, else memory). `value` is JSON-encoded for Redis."""
    _health_tick()
    if _redis and redis_healthy:
        try:
            payload = json.dumps(value, separators=(",", ":"))
            if ex:
                _redis.setex(key, ex, payload)
            else:
                _redis.set(key, payload)
            memory_cache[key] = value
            return True
        except Exception as e:
            LOG.warning("redis set failed %s: %s", key, e)
    # fallback
    memory_cache[key] = value
    return False

def cache_get(key: str) -> Any:
    """Get key (Redis preferred, else memory). Returns parsed JSON or raw memory value."""
    _health_tick()
    if _redis and redis_healthy:
        try:
            v = _redis.get(key)
            if v is not None:
                try:
                    return json.loads(v if isinstance(v, (str, bytes)) else v)
                except Exception:
                    # value wasn't json — return as-is
                    return v.decode("utf-8") if isinstance(v, bytes) else v
        except Exception as e:
            LOG.warning("redis get failed %s: %s", key, e)
    return memory_cache.get(key)

def cache_incr(key: str) -> int:
    """Atomic-ish counter; mirrors to memory cache."""
    _health_tick()
    if _redis and redis_healthy:
        try:
            n = int(_redis.incr(key))
            memory_cache[key] = n
            return n
        except Exception as e:
            LOG.warning("redis incr failed %s: %s", key, e)
    memory_cache[key] = int(memory_cache.get(key, 0)) + 1
    return memory_cache[key]

def cache_exists(key: str) -> bool:
    _health_tick()
    if _redis and redis_healthy:
        try:
            return bool(_redis.exists(key)) or key in memory_cache
        except Exception as e:
            LOG.warning("redis exists failed %s: %s", key, e)
    return key in memory_cache

def cache_backend() -> str:
    return "redis" if redis_healthy else "memory"

# Initialize on import (non-blocking)
init_cache()
