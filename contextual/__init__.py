# contextual/__init__.py
from __future__ import annotations
from typing import Any, Dict, Optional

# Try to import the real MLB function from contextual/mlb.py
try:
    from .mlb import get_mlb_contextual_hit_rate_cached as _mlb_ctx_cached  # new API
except Exception:  # if mlb.py is missing or WIP, keep app alive
    _mlb_ctx_cached = None  # type: ignore

try:
    from stats_providers.mlb_providers import provider_chain
except Exception:
    provider_chain = None

# Placeholder for LLM functions (if present)
try:
    from contextual_llm import get_mlb_contextual_hit_rate_llm, compute_context_from_logs
except Exception:
    get_mlb_contextual_hit_rate_llm = None
    compute_context_from_logs = None

def get_mlb_contextual_hit_rate_cached(player: str, stat: str, line: float) -> Optional[Dict[str, Any]]:
    """
    Same public API as before. Internally, try providers in STATS_PROVIDER_ORDER.
    If a provider returns logs, compute context (deterministic or LLM-adjusted).
    """
    if provider_chain is None:
        # Fallback to legacy function if providers not available
        try:
            return _mlb_ctx_cached(player, stat, line)
        except Exception as e:
            raise RuntimeError(f"legacy fallback failed: {e}")
    
    last_n = 10
    errors = []
    for name, client in provider_chain():
        try:
            logs = client.get_game_logs(player, last_n=last_n)
            if not logs:
                continue
            # deterministic baseline
            # Map logs -> numeric series happens inside LLM adapter; pass raw logs through:
            if get_mlb_contextual_hit_rate_llm:
                ctx = get_mlb_contextual_hit_rate_llm(player, stat, line, logs)
                if ctx:
                    ctx["source"] = f"{name}:{ctx.get('source','')}"
                    return ctx
            # fall back: derive hit_rate from logs deterministically
            # Build a tiny series for known keys:
            key = {"batter_hits":"h","hits":"h","batter_total_bases":"tb","total_bases":"tb","tb":"tb",
                   "batter_home_runs":"hr","home_runs":"hr","batter_walks":"bb","walks":"bb",
                   "batter_stolen_bases":"sb","stolen_bases":"sb","batter_runs_batted_in":"rbi","rbi":"rbi","batter_runs":"r","runs":"r"}.get(stat.lower())
            series = [float(g.get(key, 0) or 0) for g in logs] if key else []
            if series and compute_context_from_logs:
                base = compute_context_from_logs(series, line)
                if base:
                    return {"hit_rate": base.get("hit_rate_smooth", 0.5), **base, "source": f"{name}:deterministic"}
            continue
        except Exception as e:
            errors.append(f"{name}:{e}")
            continue
    # as a last resort, call your legacy cached function (keeps backward compat)
    try:
        return _mlb_ctx_cached(player, stat, line)
    except Exception as e2:
        # bubble up so /__diag/stats_health shows the failure
        raise RuntimeError("all providers failed: " + "; ".join(errors + [str(e2)]))

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

__all__ = ["get_mlb_contextual_hit_rate_cached"]
