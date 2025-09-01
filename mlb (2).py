# ai_overlay/mlb.py
from __future__ import annotations
import os, math
from typing import Any, Dict, List, Optional
try:
    from contextual import get_mlb_contextual_hit_rate_cached as ctx_cached
except Exception:
    ctx_cached = None  # type: ignore
AI_OVERLAY_ENABLED = os.getenv("AI_OVERLAY_ENABLED", "false").lower() == "true"
MODEL_VER = "mlb-v0.1"
def _get_baseline_over(prop: Dict[str, Any]) -> Optional[float]:
    try:
        fair = prop.get("fair") or {}
        prob = fair.get("prob") or {}
        p_over = float(prob.get("over"))
        if 0 <= p_over <= 1: return p_over
    except Exception:
        pass
    try:
        p_over = float(prop.get("no_vig_prob_over"))
        if 0 <= p_over <= 1: return p_over
    except Exception:
        pass
    return None
def _clip01(x: float, lo=0.01, hi=0.99) -> float:
    return max(lo, min(hi, x))
def _conf_weight(conf: str) -> float:
    if conf == "high": return 1.5
    if conf == "medium": return 1.0
    return 0.5
def _derive_ctx_from_prop(prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        en = prop.get("enrichment") or {}
        mlb_ctx = en.get("mlb_context")
        if isinstance(mlb_ctx, dict) and "hit_rate" in mlb_ctx:
            return mlb_ctx
    except Exception:
        pass
    try:
        if ctx_cached and str(prop.get("league","")).lower() == "mlb":
            return ctx_cached(
                str(prop.get("player","")),
                str(prop.get("stat","")),
                float(prop.get("line", 0) or 0.0)
            )
    except Exception:
        pass
    return None
def compute_mlb_ai_overlay(prop: Dict[str, Any], min_edge: float = 0.05) -> Optional[Dict[str, Any]]:
    try:
        if str(prop.get("league","")).lower() != "mlb": return None
        if "batter" not in str(prop.get("stat","")).lower(): return None
        p_mkt = _get_baseline_over(prop)
        if p_mkt is None:
            return None
        ctx = _derive_ctx_from_prop(prop) or {}
        p_ctx = float(ctx.get("hit_rate", 0.0) or 0.0)
        n_ctx = int(ctx.get("sample_size", 0) or 0)
        conf  = str(ctx.get("confidence", "low"))
        p_mkt = _clip01(float(p_mkt))
        p_ctx = _clip01(float(p_ctx))
        w_ctx = _conf_weight(conf) * math.sqrt(max(n_ctx, 0) / 10.0)
        w_ctx = max(0.0, min(w_ctx, 1.5))
        w_mkt = 1.0
        p_model_over = (w_mkt * p_mkt + w_ctx * p_ctx) / (w_mkt + w_ctx if (w_mkt + w_ctx) > 0 else 1.0)
        p_model_over = _clip01(p_model_over)
        p_model_under = 1.0 - p_model_over
        edge_over  = round(p_model_over  - p_mkt, 6)
        edge_under = round(p_model_under - (1.0 - p_mkt), 6)
        flagged_over  = edge_over  >= float(min_edge)
        flagged_under = edge_under >= float(min_edge)
        return {
            "model_ver": "mlb-v0.1",
            "p_model_over":  round(p_model_over, 6),
            "p_model_under": round(p_model_under, 6),
            "edge_over":  edge_over,
            "edge_under": edge_under,
            "flag_over": flagged_over,
            "flag_under": flagged_under,
            "inputs": {
                "p_mkt_over": round(p_mkt, 6),
                "p_ctx_over": round(p_ctx, 6),
                "n_ctx": int(n_ctx),
                "confidence": conf,
                "weights": {"market": w_mkt, "context": round(w_ctx,3)},
                "min_edge": float(min_edge),
            }
        }
    except Exception:
        return None
def attach_mlb_ai_overlay(props: List[Dict[str, Any]], min_edge: float = 0.05) -> None:
    if not AI_OVERLAY_ENABLED:
        return
    for p in props:
        try:
            overlay = compute_mlb_ai_overlay(p, min_edge=min_edge)
            if overlay:
                p.setdefault("ai", {}).update(overlay)
        except Exception:
            pass
