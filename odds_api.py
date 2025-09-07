import requests
from datetime import datetime, timedelta
import os
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional

from novig import american_to_prob, novig_two_way as no_vig_two_way

logger = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com"
API_KEY = os.getenv("ODDS_API_KEY", "").strip()

# Use bookmaker *keys* (lowercase). Override via ODDS_BOOKMAKERS if needed.
PREFERRED_BOOKMAKER_KEYS: List[str] = [
    b.strip() for b in (os.getenv("ODDS_BOOKMAKERS") or
                        "fanduel,draftkings,betmgm,caesars,pointsbetus").split(",")
    if b.strip()
]

SELECTED_BOOKS = {"draftkings", "fanduel", "betmgm"}  # keep small to reduce noise

# ----- Optional / legacy modules made tolerant (no hard failures) -----
try:
    # legacy; not required for EV
    from contextual import get_contextual_hit_rate  # type: ignore
except Exception:
    def get_contextual_hit_rate(*args, **kwargs):
        return None

try:
    # legacy; not required for EV
    from fantasy import get_fantasy_hit_rate  # type: ignore
except Exception:
    def get_fantasy_hit_rate(*args, **kwargs):
        return None

try:
    from mlb_game_enrichment import classify_game_environment  # type: ignore
except Exception:
    def classify_game_environment(total_point, over_odds, under_odds):
        # neutral fallback if enrichment module is absent
        return "Neutral"

try:
    from team_abbreviations import TEAM_ABBREVIATIONS  # type: ignore
except Exception:
    TEAM_ABBREVIATIONS = {}

# ------------------------ Core helpers ------------------------

def fair_probs_from_two_sided(over_am, under_am):
    """Return (p_over, p_under) no-vig from two American prices."""
    return no_vig_two_way(int(over_am), int(under_am))

def fair_odds_from_prob(p: float) -> int:
    """Convert probability to American odds."""
    if p <= 0 or p >= 1:
        return 0
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))

def best_two_sided_prices(prices):
    """Find best home/away odds from list of price dicts"""
    # prices: list of dicts like {"book":"draftkings","home":-120,"away":+100}
    # returns (best_home_odds, best_away_odds)
    home_best = None
    away_best = None
    for p in prices:
        if p.get("book") not in SELECTED_BOOKS:
            continue
        h = p.get("home")
        a = p.get("away")
        if h is not None:
            if home_best is None or h > home_best:  # less negative / more positive is better
                home_best = h
        if a is not None:
            if away_best is None or a > away_best:
                away_best = a
    return home_best, away_best

def total_consensus(totals):
    """Calculate consensus total line and no-vig probabilities"""
    # totals: list of dicts {"book":"draftkings","line":9.5,"over":-110,"under":-105}
    lines = []
    over_under_pairs = []
    for t in totals:
        if t.get("book") not in SELECTED_BOOKS:
            continue
        line = t.get("line")
        o = t.get("over")
        u = t.get("under")
        if line is not None:
            lines.append(float(line))
        if o is not None and u is not None:
            over_under_pairs.append((o, u))
    line_avg = round(sum(lines) / len(lines), 1) if lines else None
    pair = None
    if over_under_pairs:
        # median pair — arbitrary but stable
        over_under_pairs.sort(key=lambda x: (x[0] + x[1]))
        pair = over_under_pairs[len(over_under_pairs) // 2]
    over_fair = under_fair = None
    if pair:
        over_fair, under_fair = no_vig_two_way(pair[0], pair[1])
    return line_avg, over_fair, under_fair

# ---------- pairing & normalization helpers ----------
def _norm_point(val) -> Optional[str]:
    """Normalize line so '0.5' pairs with '0.50' (3 dp string)."""
    if val is None:
        return None
    try:
        return f"{Decimal(str(val)).quantize(Decimal('0.001'))}"
    except (InvalidOperation, ValueError, TypeError):
        s = str(val).strip()
        return s if s else None

def _resolve_side_and_player(name: Optional[str], desc: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Books flip name/description; always detect the side and player."""
    n = (name or "").strip(); d = (desc or "").strip()
    ln, ld = n.lower(), d.lower()
    if ln in ("over","under"): return ln, d or n
    if ld in ("over","under"): return ld, n or d
    if "over" in ln:  return "over",  d or n
    if "under" in ln: return "under", d or n
    if "over" in ld:  return "over",  n or d
    if "under" in ld: return "under", n or d
    return None, None

def _pair_outcomes(bookmakers: List[Dict[str, Any]], stat_key: str):
    """
    Returns: (player, stat_key, point_key) -> {'over': {price, book}|None, 'under': {...}|None}
    Prefer FanDuel when duplicates occur.
    """
    sidebook = defaultdict(lambda: {"over": None, "under": None})
    for bk in bookmakers or []:
        book_name = (bk.get("key") or bk.get("title") or "").strip().lower()
        for m in bk.get("markets") or []:
            if m.get("key") != stat_key:
                continue
            for o in m.get("outcomes") or []:
                side, player = _resolve_side_and_player(o.get("name"), o.get("description"))
                if side not in ("over","under") or not player:
                    continue
                price = o.get("price")
                if price is None:
                    continue
                point = _norm_point(o.get("point"))
                key = (player, stat_key, point)
                keep = sidebook[key][side] is None or sidebook[key][side].get("book") != "fanduel"
                if keep:
                    sidebook[key][side] = {"price": int(price), "book": book_name}
    return sidebook

def _attach_fair_or_implied(row: Dict[str, Any]) -> None:
    """
    Compute final True Odds on the row:
      - both sides -> no-vig fair probs
      - one side   -> implied from that side (other = 1-p)
      - fallback   -> implied from generic 'odds'
    """
    shop = row.get("shop") or {}
    over_am  = (shop.get("over")  or {}).get("american")
    under_am = (shop.get("under") or {}).get("american")
    fallback = row.get("odds")

    row.setdefault("fair", {})
    row["fair"].setdefault("prob", {"over": 0.0, "under": 0.0})

    if over_am is not None and under_am is not None:
        p_over, p_under = fair_probs_from_two_sided(over_am, under_am)
        if p_over is not None and p_under is not None:
            row["fair"]["prob"]["over"]  = round(float(p_over), 4)
            row["fair"]["prob"]["under"] = round(float(p_under), 4)
            row["fair"]["american"] = {
                "over":  fair_odds_from_prob(p_over),
                "under": fair_odds_from_prob(p_under),
            }
            row["fair"]["book"] = (shop.get("over") or {}).get("book") or (shop.get("under") or {}).get("book") or (row.get("bookmaker") or "")
            return

    if over_am is not None and under_am is None:
        p = american_to_prob(over_am)
        row["fair"]["prob"]["over"]  = round(p, 4)
        row["fair"]["prob"]["under"] = round(1.0 - p, 4)
        row["fair"]["book"] = (shop.get("over") or {}).get("book") or (row.get("bookmaker") or "")
        return

    if under_am is not None and over_am is None:
        p = american_to_prob(under_am)
        row["fair"]["prob"]["under"] = round(p, 4)
        row["fair"]["prob"]["over"]  = round(1.0 - p, 4)
        row["fair"]["book"] = (shop.get("under") or {}).get("book") or (row.get("bookmaker") or "")
        return

    if fallback is not None:
        p = american_to_prob(fallback)
        row["fair"]["prob"]["over"]  = round(p, 4)
        row["fair"]["prob"]["under"] = round(1.0 - p, 4)
        row["fair"]["book"] = row.get("bookmaker") or ""
        return

def _is_zero_prob(row: Dict[str, Any]) -> bool:
    p = (row.get("fair") or {}).get("prob") or {}
    return (p.get("over", 0.0) == 0.0) and (p.get("under", 0.0) == 0.0)

def _ensure_shop_and_fallback(row: Dict[str, Any]) -> None:
    """Guarantee at least one price present so _attach_fair_or_implied can work."""
    if ("shop" not in row or not row["shop"]) and row.get("odds") is not None:
        try:
            row["shop"] = {"over": {"american": int(row["odds"]), "book": (row.get("bookmaker") or "")}}
        except Exception:
            pass

def _finalize_fair(rows: List[Dict[str, Any]]) -> None:
    """Final pass after enrichment: recompute any rows that ended up 0/0."""
    for row in rows:
        if _is_zero_prob(row):
            _ensure_shop_and_fallback(row)
            _attach_fair_or_implied(row)

def _event_odds(event_id: str, markets: List[str]) -> Dict[str, Any]:
    """Try with bookmaker KEYS first; if empty, retry without 'bookmakers'."""
    base_params = {
        "apiKey": API_KEY, "regions": "us", "oddsFormat": "american",
        "markets": ",".join(markets),
    }
    params = dict(base_params)
    if PREFERRED_BOOKMAKER_KEYS:
        params["bookmakers"] = ",".join(PREFERRED_BOOKMAKER_KEYS)
    r = requests.get(f"{BASE}/v4/sports/baseball_mlb/events/{event_id}/odds", params=params, timeout=20)
    r.raise_for_status()
    data = r.json() or {}
    if not (data.get("bookmakers") or []):
        r2 = requests.get(f"{BASE}/v4/sports/baseball_mlb/events/{event_id}/odds", params=base_params, timeout=20)
        r2.raise_for_status()
        data = r2.json() or {}
    return data

# ------------------------ Moneylines & environments ------------------------

def get_favored_team(game):
    """
    Determine the favored team based on moneyline odds
    Lower odds = favored team (e.g., -140 is favored over +120)
    """
    home_odds = game.get("home_odds")
    away_odds = game.get("away_odds")
    if home_odds is None or away_odds is None:
        return None
    home_team = game.get("home_team")
    away_team = game.get("away_team")
    return home_team if home_odds < away_odds else away_team

def parse_game_data():
    """Fetch moneylines with preferred sportsbooks first, fallback to all if needed"""
    now = datetime.utcnow()
    future = now + timedelta(hours=48)
    start_time = now.replace(microsecond=0).isoformat() + "Z"
    end_time = future.replace(microsecond=0).isoformat() + "Z"

    if not API_KEY:
        print("[ERROR] ODDS_API_KEY is not set")
        return []

    # Try preferred sportsbooks first
    try:
        print(f"[DEBUG] Fetching moneylines from preferred sportsbooks: {PREFERRED_BOOKMAKERS_KEYS}")
    except NameError:
        # Typo guard: ensure we reference the right var
        print(f"[DEBUG] Fetching moneylines from preferred sportsbooks: {PREFERRED_BOOKMAKER_KEYS}")

    try:
        response = requests.get(
            f"{BASE}/v4/sports/baseball_mlb/odds",
            params={
                "apiKey": API_KEY,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
                "commenceTimeFrom": start_time,
                "commenceTimeTo": end_time,
                "bookmakers": ",".join(PREFERRED_BOOKMAKER_KEYS)
            },
            timeout=20
        )
        response.raise_for_status()
        data = response.json()
        print(f"[INFO] Retrieved {len(data)} moneyline matchups from preferred sportsbooks")
        if data:
            return data
        else:
            print("[WARNING] No moneylines from preferred sportsbooks, falling back to all sportsbooks")
    except Exception as e:
        print(f"[ERROR] Failed to fetch odds from preferred sportsbooks: {e}, falling back to all sportsbooks")

    # Fallback to all sportsbooks
    try:
        print("[DEBUG] Fetching moneylines from all sportsbooks")
        response = requests.get(
            f"{BASE}/v4/sports/baseball_mlb/odds",
            params={
                "apiKey": API_KEY,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
                "commenceTimeFrom": start_time,
                "commenceTimeTo": end_time
            },
            timeout=20
        )
        response.raise_for_status()
        data = response.json()
        print(f"[INFO] Retrieved moneylines for {len(data)} MLB games")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to fetch odds from all sportsbooks: {e}")
        return []

def get_matchup_map():
    """Get today's games with accurate team matchups from Odds API"""
    now = datetime.utcnow()
    future = now + timedelta(hours=48)
    start_time = now.replace(microsecond=0).isoformat() + "Z"
    end_time = future.replace(microsecond=0).isoformat() + "Z"

    if not API_KEY:
        print("[ERROR] ODDS_API_KEY is not set")
        return {}

    try:
        response = requests.get(
            f"{BASE}/v4/sports/baseball_mlb/odds",
            params={
                "apiKey": API_KEY,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
                "commenceTimeFrom": start_time,
                "commenceTimeTo": end_time,
                "bookmakers": ",".join(PREFERRED_BOOKMAKER_KEYS)
            },
            timeout=20
        )
        response.raise_for_status()
        games = response.json()
        matchup_map = {}
        for game in games:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            game_id = game.get("id", "")

            home_abbr = TEAM_ABBREVIATIONS.get(home_team, home_team)
            away_abbr = TEAM_ABBREVIATIONS.get(away_team, away_team)

            matchup_str = f"{away_abbr} @ {home_abbr}"
            matchup_map[matchup_str] = {
                "teams": [home_abbr, away_abbr],
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team
            }
        print(f"[INFO] Built matchup map with {len(matchup_map)} games: {list(matchup_map.keys())}")
        return matchup_map
    except Exception as e:
        print(f"[ERROR] Failed to build matchup map: {e}")
        return {}

def get_mlb_totals_odds():
    """Fetch over/under totals odds for MLB games"""
    now = datetime.utcnow()
    future = now + timedelta(hours=48)
    start_time = now.replace(microsecond=0).isoformat() + "Z"
    end_time = future.replace(microsecond=0).isoformat() + "Z"

    if not API_KEY:
        print("[ERROR] ODDS_API_KEY is not set")
        return []

    try:
        print("[DEBUG] Fetching MLB totals odds")
        response = requests.get(
            f"{BASE}/v4/sports/baseball_mlb/odds",
            params={
                "apiKey": API_KEY,
                "regions": "us",
                "markets": "totals",
                "oddsFormat": "american",
                "commenceTimeFrom": start_time,
                "commenceTimeTo": end_time,
                "bookmakers": ",".join(PREFERRED_BOOKMAKER_KEYS)
            },
            timeout=20
        )
        response.raise_for_status()
        data = response.json()
        print(f"[INFO] Retrieved totals odds for {len(data)} MLB games")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to fetch totals odds: {e}")
        return []

def get_mlb_game_environment_map():
    """Get environment classification and favored team for each MLB game (tolerant)."""
    totals_data = get_mlb_totals_odds()
    moneyline_data = parse_game_data()
    env_map = {}

    # Create a lookup for moneyline odds by team matchup
    moneyline_lookup = {}
    for game in moneyline_data:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        if not (home_team and away_team):
            continue

        home_abbr = TEAM_ABBREVIATIONS.get(home_team, home_team)
        away_abbr = TEAM_ABBREVIATIONS.get(away_team, away_team)
        matchup_key = f"{away_abbr} @ {home_abbr}"

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    outcomes = market.get("outcomes", [])
                    home_odds = None
                    away_odds = None
                    for outcome in outcomes:
                        if outcome.get("name") == home_team:
                            home_odds = outcome.get("price")
                        elif outcome.get("name") == away_team:
                            away_odds = outcome.get("price")
                    if home_odds is not None and away_odds is not None:
                        favored_team = home_abbr if home_odds < away_odds else away_abbr
                        moneyline_lookup[matchup_key] = {
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                            "favored_team": favored_team
                        }
                        break
            if matchup_key in moneyline_lookup:
                break

    for game in totals_data:
        try:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            if not (home_team and away_team):
                continue

            home_abbr = TEAM_ABBREVIATIONS.get(home_team, home_team)
            away_abbr = TEAM_ABBREVIATIONS.get(away_team, away_team)
            matchup_key = f"{away_abbr} @ {home_abbr}"

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "totals":
                        continue

                    outcomes = market.get("outcomes", [])
                    total_point = None
                    over_odds = None
                    under_odds = None
                    for outcome in outcomes:
                        if outcome.get("name") == "Over":
                            total_point = outcome.get("point")
                            over_odds = outcome.get("price")
                        elif outcome.get("name") == "Under":
                            under_odds = outcome.get("price")

                    if total_point is not None and over_odds is not None and under_odds is not None:
                        label = classify_game_environment(total_point, over_odds, under_odds)

                        moneyline_info = moneyline_lookup.get(matchup_key, {})
                        home_best, away_best = best_two_sided_prices([{
                            "book": bookmaker.get("key", "").lower(),
                            "home": moneyline_info.get("home_odds"),
                            "away": moneyline_info.get("away_odds")
                        }])

                        home_fair = away_fair = None
                        favorite = None
                        if home_best is not None and away_best is not None:
                            home_fair, away_fair = no_vig_two_way(home_best, away_best)
                            if home_fair is not None and away_fair is not None:
                                favorite = "home" if home_fair > away_fair else "away"

                        total_line, over_fair, under_fair = total_consensus([{
                            "book": bookmaker.get("key", "").lower(),
                            "line": total_point,
                            "over": over_odds,
                            "under": under_odds
                        }])

                        high_scoring = False
                        if total_line is not None:
                            high_scoring = (total_line >= 9.0) or (over_fair is not None and over_fair > 0.55)

                        env_map[matchup_key] = {
                            "environment": label,
                            "total": total_point,
                            "over_odds": over_odds,
                            "under_odds": under_odds,
                            "favored_team": moneyline_info.get("favored_team"),
                            "home_team": home_abbr,
                            "away_team": away_abbr,
                            "no_vig": {
                                "moneyline": {
                                    "home_prob": round(home_fair, 3) if home_fair is not None else None,
                                    "away_prob": round(away_fair, 3) if away_fair is not None else None,
                                    "favorite": favorite
                                },
                                "totals": {
                                    "line": total_line,
                                    "over_prob": round(over_fair, 3) if over_fair is not None else None,
                                    "under_prob": round(under_fair, 3) if under_fair is not None else None,
                                    "high_scoring": high_scoring
                                }
                            }
                        }
                        print(f"[ENV] {matchup_key}: {label} (Total: {total_point})")
                        break
                if matchup_key in env_map:
                    break
        except Exception as e:
            logger.debug(f"Error processing game environment for {game}: {e}")
            continue

    print(f"[INFO] Classified {len(env_map)} game environments with favored teams")
    return env_map

# ------------------------ MLB player props ------------------------

def fetch_player_props():
    """Fetch player props with preferred sportsbooks first, fallback to all if needed"""
    now = datetime.utcnow()
    future = now + timedelta(hours=48)
    start_time = now.replace(microsecond=0).isoformat() + "Z"
    end_time = future.replace(microsecond=0).isoformat() + "Z"

    if not API_KEY:
        print("[ERROR] ODDS_API_KEY is not set")
        return []

    # 1) events list
    try:
        event_resp = requests.get(
            f"{BASE}/v4/sports/baseball_mlb/events",
            params={
                "apiKey": API_KEY,
                "commenceTimeFrom": start_time,
                "commenceTimeTo": end_time
            },
            timeout=20
        )
        event_resp.raise_for_status()
        events = event_resp.json()
        print(f"[INFO] Found {len(events)} events")
    except Exception as e:
        print(f"[ERROR] Failed to fetch MLB events: {e}")
        return []

    all_props = []
    print(f"[DEBUG] Starting prop collection for {len(events)} events")

    # Verified markets (Odds API naming)
    markets_batch_1 = ["batter_hits", "batter_home_runs", "batter_total_bases"]
    markets_batch_2 = ["pitcher_strikeouts", "pitcher_earned_runs", "pitcher_outs", "pitcher_hits_allowed"]
    print(f"[DEBUG] Using verified markets: {markets_batch_1 + markets_batch_2}")
    all_markets = [markets_batch_1, markets_batch_2]

    for event in events:
        eid = event.get("id")
        if not eid:
            continue

        home_team = event.get("home_team", "Unknown")
        away_team = event.get("away_team", "Unknown")
        matchup_key = f"{away_team} @ {home_team}"

        sidebook = defaultdict(lambda: {"over": None, "under": None})

        for batch_idx, markets in enumerate(all_markets):
            try:
                if batch_idx > 0:
                    time.sleep(1)  # small pause to respect rate limits

                data = _event_odds(eid, markets)

                if data.get("bookmakers"):
                    got = [m.get('key') for m in data.get('bookmakers', [])[0].get('markets', [])]
                    print(f"[DEBUG] Event {eid} batch {batch_idx} markets: {got}")

                for stat_key in markets:
                    batch_sidebook = _pair_outcomes(data.get("bookmakers", []), stat_key)
                    for key, sides in batch_sidebook.items():
                        if sides["over"] or sides["under"]:
                            sidebook[key] = sides
            except Exception as e:
                print(f"[ERROR] Failed to fetch props for event {eid} batch {batch_idx}: {e}")
                continue

        props_for_matchup = []
        for (player, stat_key, point), sides in sidebook.items():
            over  = sides.get('over')
            under = sides.get('under')

            row = {
                "player": player,
                "stat":   stat_key,
                "line":   point,
                "matchup": matchup_key,
            }

            # shop + basic fallback fields
            if over or under:
                row['shop'] = {}
                if over:  row['shop']['over']  = {'american': int(over['price']),  'book': over['book']}
                if under: row['shop']['under'] = {'american': int(under['price']), 'book': under['book']}
            row['bookmaker'] = (over or under or {}).get('book')
            row['odds']      = (over or under or {}).get('price')

            # side info (optional)
            if over and under:
                row['side'] = 'both'
            elif over:
                row['side'] = 'over'
            elif under:
                row['side'] = 'under'
            else:
                row['side'] = 'unknown'

            row['book'] = (over or under or {}).get('book', '')

            # compute True Odds
            _attach_fair_or_implied(row)

            props_for_matchup.append(row)

        _finalize_fair(props_for_matchup)
        all_props.extend(props_for_matchup)
        print(f"[DEBUG] Event {eid} ({matchup_key}): Collected {len(props_for_matchup)} props")

    print(f"[INFO] Final count of props: {len(all_props)}")
    # optional breakdown
    stat_counts: Dict[str, int] = {}
    for prop in all_props:
        stat = prop.get('stat', 'unknown')
        stat_counts[stat] = stat_counts.get(stat, 0) + 1
    print(f"[DEBUG] Props by stat type: {stat_counts}")
    return all_props

# ------------------------ Dedup & optional enrichment (no-op safe) ------------------------

def deduplicate_props(props):
    """Deduplicate props: keep one prop per unique player+stat+line combination"""
    unique_props = {}
    for prop in props:
        key = f"{prop.get('player')}_{prop.get('stat')}_{prop.get('line')}"
        if key not in unique_props:
            unique_props[key] = prop
        else:
            # Prefer better odds; for negative odds, closer to 0 is better. For positive, higher is better.
            current_odds = unique_props[key].get('odds')
            new_odds = prop.get('odds')
            try:
                if current_odds is None:
                    unique_props[key] = prop
                elif new_odds is None:
                    pass
                else:
                    # basic heuristic
                    if (current_odds < 0 and new_odds > current_odds) or (current_odds > 0 and new_odds > current_odds):
                        unique_props[key] = prop
            except Exception:
                pass
    deduplicated = list(unique_props.values())
    print(f"[INFO] Deduplication: {len(props)} props -> {len(deduplicated)} unique props")
    return deduplicated

def enrich_prop(prop):
    """
    Legacy enrichment wrapper — SAFE NO-OP if external modules are absent.
    Keeps structure but does not block deploy.
    """
    contextual = None
    try:
        contextual = get_contextual_hit_rate(
            prop.get("player"),  # name
            stat_type=prop.get("stat"),
            threshold=prop.get("line"),
        )
    except Exception as e:
        contextual = None

    fantasy = None
    try:
        fantasy = get_fantasy_hit_rate(prop.get("player"), threshold=prop.get("line"))
    except Exception as e:
        fantasy = None

    # Ensure we always return the prop with flags, but DO NOT rely on these for core flow
    out = dict(prop)
    out["contextual_hit_rate"] = contextual or {"hit_rate": None, "note": "not available"}
    out["fantasy_hit_rate"] = fantasy or {"hit_rate": None, "note": "not available"}
    out["enriched"] = bool(contextual or fantasy)
    return out

def enrich_player_props(props):
    """Parallel enrichment (optional). Safe if modules are missing."""
    if not props:
        return []
    print(f"[INFO] Starting enrichment for {len(props)} props (optional)")
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            enriched_props = list(executor.map(enrich_prop, props))
        ok = sum(1 for p in enriched_props if p.get("enriched"))
        print(f"[INFO] Enrichment complete: {ok}/{len(props)} props had extra signals")
        return enriched_props
    except Exception as e:
        print(f"[WARN] Enrichment failed globally: {e}")
        return props
