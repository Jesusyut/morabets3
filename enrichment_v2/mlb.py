"""
MLB Props Enrichment v2 - Free MLB Stats API with Top-K selection and Redis caching
"""
import os
import time
import logging
import requests
from datetime import date, datetime
from typing import List, Dict, Any, Optional
from functools import lru_cache

# Configure logging
LOG = logging.getLogger("enrichment_v2")

# MLB Stats API configuration
MLB_BASE = "https://statsapi.mlb.com/api/v1"
TIMEOUT = (3, 6)
UA = {"User-Agent": "MoraBets/2.0"}

# Redis configuration (optional)
try:
    from redis import Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    R = Redis.from_url(REDIS_URL, decode_responses=True)
    REDIS_TTL = int(os.getenv("REDIS_TTL", "3600"))  # 1 hour default
except Exception:
    R = None
    LOG.warning("Redis not available, using in-memory cache")

# Configuration
MAX_ENRICHMENT_TIME = float(os.getenv("ENRICHMENT_TIMEOUT", "2.5"))  # seconds
MAX_PROPS_TO_ENRICH = int(os.getenv("ENRICHMENT_MAX_PROPS", "100"))
TOP_K_PLAYERS = int(os.getenv("ENRICHMENT_TOP_K", "20"))

# Stat mapping from frontend to MLB API
STAT_MAP = {
    "batter_hits": "hits",
    "hits": "hits", 
    "batter_total_bases": "totalBases",
    "total_bases": "totalBases",
    "tb": "totalBases",
    "batter_home_runs": "homeRuns",
    "home_runs": "homeRuns",
    "batter_runs": "runs",
    "runs": "runs",
    "batter_runs_batted_in": "rbi",
    "rbi": "rbi",
    "batter_walks": "baseOnBalls",
    "walks": "baseOnBalls",
    "batter_stolen_bases": "stolenBases",
    "stolen_bases": "stolenBases",
}

def _get_cache_key(player: str, stat: str, line: float) -> str:
    """Generate cache key for enrichment data."""
    today = date.today().isoformat()
    return f"enrich_v2:{today}:{player}:{stat}:{line}"

def _get_from_cache(key: str) -> Optional[Dict[str, Any]]:
    """Get enrichment data from cache."""
    if R is None:
        return None
    try:
        data = R.get(key)
        return eval(data) if data else None
    except Exception as e:
        LOG.warning(f"Cache get failed: {e}")
        return None

def _set_in_cache(key: str, data: Dict[str, Any]) -> None:
    """Store enrichment data in cache."""
    if R is None:
        return
    try:
        R.setex(key, REDIS_TTL, str(data))
    except Exception as e:
        LOG.warning(f"Cache set failed: {e}")

def _fetch_player_id(player_name: str) -> Optional[int]:
    """Fetch MLB player ID by name."""
    try:
        url = f"{MLB_BASE}/people/search"
        params = {"names": player_name}
        r = requests.get(url, params=params, timeout=TIMEOUT, headers=UA)
        r.raise_for_status()
        
        data = r.json()
        people = data.get("people", [])
        if people:
            return int(people[0]["id"])
        return None
    except Exception as e:
        LOG.warning(f"Player ID fetch failed for {player_name}: {e}")
        return None

def _fetch_game_logs(player_id: int, season: int, group: str = "hitting") -> List[Dict[str, Any]]:
    """Fetch recent game logs for a player."""
    try:
        url = f"{MLB_BASE}/people/{player_id}/stats"
        params = {"stats": "gameLog", "season": season, "group": group}
        r = requests.get(url, params=params, timeout=TIMEOUT, headers=UA)
        r.raise_for_status()
        
        data = r.json()
        stats = data.get("stats", [])
        if stats:
            return stats[0].get("splits", [])
        return []
    except Exception as e:
        LOG.warning(f"Game logs fetch failed for player {player_id}: {e}")
        return []

def _calculate_hit_rate(logs: List[Dict[str, Any]], stat_key: str, threshold: float) -> Dict[str, Any]:
    """Calculate hit rate and confidence from game logs."""
    if not logs:
        return {"hit_rate": 0.0, "sample_size": 0, "confidence": "low"}
    
    # Extract values for the stat
    values = []
    for game in logs:
        stat_block = game.get("stat", {})
        value = float(stat_block.get(stat_key, 0) or 0)
        values.append(value)
    
    # Calculate hit rate (games above threshold)
    above_threshold = sum(1 for v in values if v > threshold)
    total_games = len(values)
    
    if total_games == 0:
        return {"hit_rate": 0.0, "sample_size": 0, "confidence": "low"}
    
    hit_rate = above_threshold / total_games
    
    # Calculate confidence based on sample size and variance
    if total_games < 5:
        confidence = "low"
    elif total_games < 10:
        confidence = "medium"
    else:
        # Higher confidence for larger samples
        confidence = "high"
    
    return {
        "hit_rate": round(hit_rate, 4),
        "sample_size": total_games,
        "confidence": confidence,
        "threshold": threshold,
        "games_above": above_threshold,
        "total_games": total_games
    }

def enrich_props_mlb_v2(props: List[Dict[str, Any]]) -> int:
    """
    Enrich MLB props with contextual data from MLB Stats API.
    
    Args:
        props: List of prop dictionaries
        
    Returns:
        int: Number of props successfully enriched
    """
    if not props:
        return 0
    
    start_time = time.time()
    enriched_count = 0
    
    # Filter to MLB batter props only
    mlb_batter_props = []
    for prop in props:
        league = str(prop.get("league", "")).lower()
        stat = str(prop.get("stat", "")).lower()
        
        if league == "mlb" and ("batter" in stat or stat in STAT_MAP):
            mlb_batter_props.append(prop)
    
    if not mlb_batter_props:
        LOG.info("No MLB batter props found for enrichment")
        return 0
    
    # Sort by line value to prioritize higher-value props
    mlb_batter_props.sort(key=lambda p: float(p.get("line", 0) or 0), reverse=True)
    
    # Take top K props for enrichment
    props_to_enrich = mlb_batter_props[:TOP_K_PLAYERS]
    
    LOG.info(f"Starting enrichment for {len(props_to_enrich)} MLB batter props")
    
    for prop in props_to_enrich:
        # Check timeout
        if time.time() - start_time > MAX_ENRICHMENT_TIME:
            LOG.warning(f"Enrichment timeout after {enriched_count} props")
            break
        
        # Check max props limit
        if enriched_count >= MAX_PROPS_TO_ENRICH:
            LOG.info(f"Reached max enrichment limit: {MAX_PROPS_TO_ENRICH}")
            break
        
        try:
            player = prop.get("player", "")
            stat = prop.get("stat", "")
            line = float(prop.get("line", 0) or 0)
            
            if not player or not stat or line <= 0:
                continue
            
            # Check cache first
            cache_key = _get_cache_key(player, stat, line)
            cached_data = _get_from_cache(cache_key)
            
            if cached_data:
                prop.setdefault("enrichment", {})["mlb_context"] = cached_data
                enriched_count += 1
                continue
            
            # Fetch from MLB API
            player_id = _fetch_player_id(player)
            if not player_id:
                continue
            
            # Map stat to MLB API field
            mlb_stat = STAT_MAP.get(stat.lower(), stat.lower())
            
            # Get current and previous season data
            current_year = datetime.now().year
            logs = _fetch_game_logs(player_id, current_year, "hitting")
            
            # If not enough current year data, add previous year
            if len(logs) < 10:
                prev_logs = _fetch_game_logs(player_id, current_year - 1, "hitting")
                logs.extend(prev_logs)
            
            # Calculate hit rate
            enrichment_data = _calculate_hit_rate(logs, mlb_stat, line)
            
            # Cache the result
            _set_in_cache(cache_key, enrichment_data)
            
            # Attach to prop
            prop.setdefault("enrichment", {})["mlb_context"] = enrichment_data
            enriched_count += 1
            
            LOG.debug(f"Enriched {player} {stat} {line}: {enrichment_data}")
            
        except Exception as e:
            LOG.warning(f"Enrichment failed for prop: {e}")
            continue
    
    elapsed = time.time() - start_time
    LOG.info(f"Enrichment completed: {enriched_count}/{len(props_to_enrich)} props enriched in {elapsed:.2f}s")
    
    return enriched_count
