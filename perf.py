# perf.py
from __future__ import annotations
import os, time, threading, json
from contextvars import ContextVar
from collections import deque
from typing import Any, Dict, List, Optional

# Enable globally via env or per-request via perf.enable()
PERF_DEFAULT = os.getenv("PERF_TRACE", "0") == "1"
PERF_WARN_MS = int(os.getenv("PERF_WARN_MS", "400"))  # warn if span > 400ms
_RING_SIZE = int(os.getenv("PERF_RING", "50"))

_ctx_trace: ContextVar[Optional[Dict[str, Any]]] = ContextVar("trace", default=None)
_ring = deque(maxlen=_RING_SIZE)
_lock = threading.Lock()

def enable(request_id: str | None = None):
    # start a new trace dict
    t = {
        "id": request_id or f"req-{int(time.time()*1000)}",
        "start": time.perf_counter(),
        "spans": [],            # list of {name, ms, extra}
        "marks": {},            # arbitrary counters/totals
        "kvs": {},              # small key->value map
    }
    _ctx_trace.set(t)

def disable():
    _ctx_trace.set(None)

def is_enabled() -> bool:
    t = _ctx_trace.get()
    return t is not None

def mark(key: str, inc: float = 1.0):
    t = _ctx_trace.get()
    if not t: return
    t["marks"][key] = t["marks"].get(key, 0.0) + inc

def kv(key: str, value: Any):
    t = _ctx_trace.get()
    if not t: return
    t["kvs"][key] = value

class span:
    """with perf.span('name', extra={...}): ..."""
    def __init__(self, name: str, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        self.extra = extra or {}
        self.t0 = 0.0
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        t = _ctx_trace.get()
        if not t: return False
        ms = (time.perf_counter() - self.t0) * 1000.0
        rec = {"name": self.name, "ms": round(ms, 1)}
        if self.extra:
            rec["extra"] = self.extra
        t["spans"].append(rec)
        # simple warning hook (stdout)
        if ms >= PERF_WARN_MS:
            print(f"[PERF] slow span '{self.name}' {ms:.1f}ms extra={self.extra}")
        return False

def snapshot() -> Optional[Dict[str, Any]]:
    t = _ctx_trace.get()
    if not t: return None
    out = dict(t)
    out["total_ms"] = round((time.perf_counter() - t["start"]) * 1000.0, 1)
    return out

def push_current():
    t = snapshot()
    if not t: return
    with _lock:
        _ring.append(t)

def recent() -> List[Dict[str, Any]]:
    with _lock:
        return list(_ring)

def to_header(obj: Dict[str, Any]) -> str:
    try:
        # Tiny summary for headers
        base = {
            "id": obj.get("id"),
            "total_ms": obj.get("total_ms"),
            "spans": obj.get("spans", [])[-6:],  # last few spans
            "marks": obj.get("marks", {}),
        }
        return json.dumps(base, separators=(",", ":"))
    except Exception:
        return "{}"
