# contextual/__init__.py
from __future__ import annotations
from typing import Any, Dict, Optional

# Try to import the real MLB function from contextual/mlb.py
try:
    from .mlb import get_mlb_contextual_hit_rate_cached as _mlb_ctx_cached  # new API
except Exception:  # if mlb.py is missing or WIP, keep app alive
    _mlb_ctx_cached = None  # type: ignore


def get_mlb_contextual_hit_rate_cached(player: str, stat: str, line: float) -> Optional[Dict[str, Any]]:
    """Public MLB function (new API)."""
    if _mlb_ctx_cached is None:
        return None
    try:
        return _mlb_ctx_cached(player, stat, line)
    except Exception:
        return None


# ------- Back-compat aliases (old imports still work) -------

def get_contextual_hit_rate(player: str, stat: str, line: float) -> Optional[Dict[str, Any]]:
    """
    Legacy name used by older modules (e.g., odds_api.py).
    Delegate to the MLB function so we don't break imports.
    """
    return get_mlb_contextual_hit_rate_cached(player, stat, line)


def get_contextual_hit_rate_cached(player: str, stat: str, line: float) -> Optional[Dict[str, Any]]:
    """If anyone imported the cached legacy name, keep it working."""
    return get_mlb_contextual_hit_rate_cached(player, stat, line)
