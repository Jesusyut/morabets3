from .mlb import get_mlb_contextual_hit_rate, get_mlb_contextual_hit_rate_cached

# Legacy aliases for backward compatibility
def get_contextual_hit_rate(*args, **kwargs):
    return get_mlb_contextual_hit_rate(*args, **kwargs)

def get_contextual_hit_rate_cached(*args, **kwargs):
    return get_mlb_contextual_hit_rate_cached(*args, **kwargs)

__all__ = [
    "get_mlb_contextual_hit_rate",
    "get_mlb_contextual_hit_rate_cached",
    "get_contextual_hit_rate",
    "get_contextual_hit_rate_cached",
]
