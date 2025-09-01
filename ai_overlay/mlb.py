from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

# Optional: use contextual if available
try:
    from contextual import get_mlb_contextual_hit_rate_cached as ctx_cached
except Exception:
    ctx_cached = None  # type: ignore


def _get_baseline_over(prop: Dict[str, Any]) -> Optional[float]:
    """Find the market fair P(over) on the prop dict."""
    # Primary shape
    try:
        p = float(prop.get("fair", {}).get("prob", {}).get("over"))
        if 0.0 <= p <= 1.0:
            return p
    except Exception:
        pass
    # Common flat fallbacks
    for k in ("no_vig_prob_over", "fair_prob_over"):
        try:
            p = float(prop.get(k))
            if 0.0 <= p <= 1.0:
                return p
        except Exception:
            pass
    return None


def _clip01(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, x))


def _derive_ctx(prop: Dict[str, Any]) -> Dict[str, Any]:
    """Use attached enrichment if present; otherwise try cached function; else blank."""
    try:
        ctx = prop.get("enrichment", {}).get("mlb_context")
        if isinstance(ctx, dict) and "hit_rate" in ctx:
            return ctx
    except Exception:
        pass
    try:
        if ctx_cached and str(prop.get("league", "")).lower() == "mlb":
            return ctx_cached(
                str(prop.get("player", "")),
                str(prop.get("stat", "")),
                float(prop.get("line", 0) or 0.0),
            ) or {}
    except Exception:
        pass
    return {}


def compute_mlb_ai_overlay(prop: Dict[str, Any], min_edge: float = 0.06) -> Optional[Dict[str, Any]]:
    """Deterministic 'blender' model for MLB batter props."""
    if str(prop.get("league", "")).lower() != "mlb":
        return None
    stat = str(prop.get("stat", "")).lower()
    STAT_OK = {
        "batter_hits", "hits",
        "batter_total_bases", "total_bases", "tb",
        "batter_home_runs", "home_runs",
        "batter_runs", "runs",
        "batter_runs_batted_in", "rbi",
        "batter_walks", "walks",
        "batter_stolen_bases", "stolen_bases",
    }
    if ("batter" not in stat) and (stat not in STAT_OK):
        return None

    p_mkt = _get_baseline_over(prop)
    if p_mkt is None:
        return None

    ctx = _derive_ctx(prop)
    p_ctx = float(ctx.get("hit_rate", 0.0) or 0.0)
    n_ctx = int(ctx.get("sample_size", 0) or 0)
    conf  = str(ctx.get("confidence", "low"))

    p_mkt = _clip01(float(p_mkt))
    p_ctx = _clip01(float(p_ctx))

    # Shrinkage weight for context (bounded)
    w_ctx = (1.5 if conf == "high" else 1.0 if conf == "medium" else 0.5) * math.sqrt(max(n_ctx, 0) / 10.0)
    w_ctx = max(0.0, min(w_ctx, 1.5))
    w_mkt = 1.0

    p_model_over = (w_mkt * p_mkt + w_ctx * p_ctx) / (w_mkt + w_ctx if (w_mkt + w_ctx) > 0 else 1.0)
    p_model_over = _clip01(p_model_over)
    p_model_under = 1.0 - p_model_over

    edge_over  = round(p_model_over  - p_mkt, 6)
    edge_under = round(p_model_under - (1.0 - p_mkt), 6)

    return {
        "model_ver": "mlb-v0.1",
        "p_model_over":  round(p_model_over, 6),
        "p_model_under": round(p_model_under, 6),
        "edge_over":  edge_over,
        "edge_under": edge_under,
        "flag_over":  edge_over  >= float(min_edge),
        "flag_under": edge_under >= float(min_edge),
        "inputs": {
            "p_mkt_over": round(p_mkt, 6),
            "p_ctx_over": round(p_ctx, 6),
            "n_ctx": int(n_ctx),
            "confidence": conf,
            "min_edge": float(min_edge),
        },
    }


def attach_mlb_ai_overlay(props: List[Dict[str, Any]], min_edge: float = 0.06) -> None:
    """Attach the overlay to each MLB batter prop; never throw."""
    for p in props:
        try:
            o = compute_mlb_ai_overlay(p, min_edge=min_edge)
            if o:
                p.setdefault("ai", {}).update(o)
        except Exception:
            # swallow errors; never break the page
            pass
