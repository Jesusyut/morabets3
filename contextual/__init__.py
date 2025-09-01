try:
    from .mlb import get_mlb_contextual_hit_rate_cached  # re-export
except Exception:
    # Allow import even if mlb.py has issues; callers should handle None
    def get_mlb_contextual_hit_rate_cached(*args, **kwargs):
        return None

__all__ = ["get_mlb_contextual_hit_rate_cached"]
