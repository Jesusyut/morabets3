# props_ufc.py - Odds-only UFC props
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime, timedelta, timezone

from odds_client_ufc import list_events_ufc, event_markets_ufc, event_odds_ufc
from markets_ufc import UFC_ML_MARKET, UFC_MOV_PATTERNS, UFC_TOTALS_MARKETS, MOV_CANON
from novig import novig_two_way, american_to_prob
import perf

MAX_WORKERS = int(os.getenv("ODDS_WORKERS", "4"))

def _any_matches(s: str, pats: List[str]) -> bool:
    t = s.lower()
    return any(p in t for p in pats)

def _canonical_bucket(outcome_name: str) -> str | None:
    t = (outcome_name or "").lower()
    for bucket, aliases in MOV_CANON.items():
        for a in aliases:
            if a in t: return bucket
    return None

def _collect_ml(bookmakers: List[Dict[str,Any]], fighters: Tuple[str,str]) -> List[Dict[str,Any]]:
    a, b = fighters
    best = {a: None, b: None}
    for bkr in bookmakers or []:
        bk = bkr.get("key","")
        for m in bkr.get("markets", []):
            if m.get("key") != UFC_ML_MARKET: continue
            for o in m.get("outcomes", []):
                name, price = o.get("name") or o.get("description"), o.get("price")
                if name in (a, b) and price is not None:
                    cur = best.get(name)
                    if (cur is None) or (abs(price) < abs(cur["price"])):
                        best[name] = {"price": int(price), "book": bk}
    rows = []
    if best[a] and best[b]:
        pa, pb = novig_two_way(best[a]["price"], best[b]["price"])
        rows.append({"type":"ml","fighter":a,"opponent":b,
                     "shop":{"ml":{"american":best[a]["price"],"book":best[a]["book"]}},
                     "fair":{"prob":{"ml":pa},"american":{"ml":best[a]["price"]}}})
        rows.append({"type":"ml","fighter":b,"opponent":a,
                     "shop":{"ml":{"american":best[b]["price"],"book":best[b]["book"]}},
                     "fair":{"prob":{"ml":pb},"american":{"ml":best[b]["price"]}}})
    return rows

def _collect_mov(bookmakers: List[Dict[str,Any]], fighter: str) -> Dict[str, Any]:
    best = {"ko": None, "sub": None, "dec": None}
    for bkr in bookmakers or []:
        bk = bkr.get("key","")
        for m in bkr.get("markets", []):
            k = (m.get("key") or "").lower()
            if not _any_matches(k, UFC_MOV_PATTERNS): continue
            for o in m.get("outcomes", []):
                name = (o.get("name") or o.get("description") or "")
                if fighter.lower() not in name.lower(): continue
                bucket = _canonical_bucket(name)
                if not bucket: continue
                price = o.get("price")
                if price is None: continue
                cur = best[bucket]
                if (cur is None) or (abs(price) < abs(cur["price"])):
                    best[bucket] = {"price": int(price), "book": bk}
    have = [b for b,v in best.items() if v]
    if len(have) < 2: return {}
    odds = [best[b]["price"] for b in ("ko","sub","dec") if best[b]]
    buckets = [b for b in ("ko","sub","dec") if best[b]]
    # Use american_to_prob for individual probabilities
    fair_prob = {}
    fair_amer = {}
    for bucket in buckets:
        if best[bucket]:
            prob = american_to_prob(best[bucket]["price"])
            fair_prob[bucket] = prob
            fair_amer[bucket] = best[bucket]["price"]
    return {"buckets": {b: {"american": best[b]["price"], "book": best[b]["book"]} for b in buckets},
            "fair": {"prob": fair_prob, "american": fair_amer}}

def fetch_ufc_totals_props(date_iso: Optional[str] = None, hours_ahead: int = 96) -> List[Dict[str, Any]]:
    """
    Fetch UFC totals O/U props for fight duration.
    Returns list of props in the required schema for /player_props endpoint.
    """
    with perf.span("ufc:fetch_totals_props", {"date": date_iso or "", "hours": hours_ahead}):
        # Get events within time window
        events = list_events_ufc(hours_ahead=hours_ahead, date=date_iso)
        perf.mark("ufc.events_seen", len(events))
        
        all_props = []
        
        def _process_event(ev):
            with perf.span("ufc:event_totals", {"eid": ev.get("id")}):
                eid = ev.get("id")
                if not eid:
                    return []
                
                away_team = ev.get("away_team", "")
                home_team = ev.get("home_team", "")
                if not away_team or not home_team:
                    return []
                
                # Build matchup as "Fighter A vs Fighter B"
                matchup = f"{away_team} vs {home_team}"
                
                # Get available markets for this event
                try:
                    mk = event_markets_ufc(eid)
                    seen_keys = {k for bk in mk.get("bookmakers", []) for k in (bk.get("markets") or [])}
                except Exception:
                    seen_keys = set()
                
                # Find totals markets
                totals_markets = [k for k in seen_keys if _any_matches(k, UFC_TOTALS_MARKETS)]
                if not totals_markets:
                    return []
                
                # Fetch odds for totals markets
                try:
                    data = event_odds_ufc(eid, totals_markets)
                except Exception:
                    return []
                
                bookmakers = data.get("bookmakers", [])
                if not bookmakers:
                    return []
                
                # Process each totals market
                event_props = []
                for market in totals_markets:
                    for bm in bookmakers:
                        book_key = bm.get("key", "")
                        
                        # Find the totals market
                        for mk in bm.get("markets", []):
                            if mk.get("key") != market:
                                continue
                            
                            # Look for over/under outcomes
                            over_outcome = None
                            under_outcome = None
                            line_value = None
                            
                            for outcome in mk.get("outcomes", []):
                                name = (outcome.get("name") or "").lower()
                                if "over" in name:
                                    over_outcome = outcome
                                    line_value = outcome.get("point")
                                elif "under" in name:
                                    under_outcome = outcome
                                    if line_value is None:
                                        line_value = outcome.get("point")
                            
                            # Only process if we have both sides and a line
                            if over_outcome and under_outcome and line_value is not None:
                                over_price = over_outcome.get("price")
                                under_price = under_outcome.get("price")
                                
                                if over_price is not None and under_price is not None:
                                    # Compute fair no-vig probabilities
                                    p_over, p_under = novig_two_way(over_price, under_price)
                                    
                                    # Build prop in required schema
                                    prop = {
                                        "league": "ufc",
                                        "matchup": matchup,
                                        "player": matchup,  # Same as matchup for UFC
                                        "stat": "fight_total_rounds",
                                        "line": float(line_value),
                                        "shop": {
                                            "over": int(over_price),
                                            "under": int(under_price),
                                            "book": book_key
                                        },
                                        "fair": {
                                            "prob": {
                                                "over": p_over,
                                                "under": p_under
                                            }
                                        }
                                    }
                                    event_props.append(prop)
                                    
                                    # Only process first bookmaker with valid data
                                    break
                
                return event_props
        
        # Process events concurrently
        with perf.span("ufc:concurrency", {"workers": MAX_WORKERS}):
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(_process_event, ev) for ev in events]
                for future in as_completed(futures):
                    try:
                        event_props = future.result()
                        all_props.extend(event_props)
                    except Exception as e:
                        perf.mark("ufc:event_error", 1)
                        continue
        
        perf.mark("ufc:total_props", len(all_props))
        return all_props

def fetch_ufc_props(date: str | None = None) -> List[Dict[str,Any]]:
    """Legacy function - now returns totals props in new format"""
    return fetch_ufc_totals_props(date_iso=date)

# Back-compat alias if earlier code called fetch_ufc_markets()
def fetch_ufc_markets(*args, **kwargs):
    return fetch_ufc_props(*args, **kwargs)
