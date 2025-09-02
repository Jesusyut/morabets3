from __future__ import annotations
import os, time, unicodedata, datetime as _dt
from typing import Dict, Any, List, Optional, Tuple

import requests

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode("ascii")
    s = s.replace(".", "").replace(",", "")
    for suf in (" Jr", " Jr.", " III", " II"): s = s.replace(suf, "")
    return " ".join(s.split())

def _http_json(url: str, params: dict=None) -> dict:
    r = requests.get(url, params=params or {}, headers={"User-Agent":"morabets/1.0"}, timeout=(2,5))
    r.raise_for_status()
    return r.json()

def _player_id(player_name: str) -> Optional[int]:
    j = _http_json(f"{MLB_STATS_API}/people/search", {"names": _norm(player_name).lower()})
    people = j.get("people") or j.get("searchPeople") or []
    if isinstance(people, dict): people = people.get("people", [])
    if people:
        pid = people[0].get("id") or people[0].get("personId")
        return int(pid) if isinstance(pid,int) else None
    return None

def _logs(pid: int, last_n: int=10) -> List[dict]:
    today = _dt.date.today()
    rows: List[dict] = []
    for season in (today.year, today.year-1):
        j = _http_json(f"{MLB_STATS_API}/people/{pid}/stats", {"stats":"gameLog","season":season,"group":"hitting"})
        splits = ((j.get("stats") or [{}])[0]).get("splits", [])
        for s in splits:
            st = s.get("stat") or {}
            rows.append({"date": s.get("date") or s.get("gameDate"),
                         "h": st.get("hits",0), "tb": st.get("totalBases",0),
                         "hr": st.get("homeRuns",0), "bb": st.get("baseOnBalls",0),
                         "sb": st.get("stolenBases",0), "rbi": st.get("rbi",0),
                         "r": st.get("runs",0)})
    rows = [r for r in rows if r.get("date")]
    rows.sort(key=lambda d: d["date"], reverse=True)
    return rows[:max(10,last_n)]

STAT = {
    "batter_hits":"h","hits":"h","h":"h",
    "batter_total_bases":"tb","total_bases":"tb","tb":"tb",
    "batter_home_runs":"hr","home_runs":"hr","hr":"hr",
    "batter_walks":"bb","walks":"bb","bb":"bb",
    "batter_stolen_bases":"sb","stolen_bases":"sb","sb":"sb",
    "batter_runs_batted_in":"rbi","rbi":"rbi",
    "batter_runs":"r","runs":"r",
}

def get_mlb_contextual_hit_rate_cached(player: str, stat: str, line: float, last_n: int=10) -> Optional[dict]:
    pid = _player_id(player)
    if not pid: return None
    logs = _logs(pid, last_n=last_n)
    key = STAT.get(stat.lower()) or STAT.get(stat.lower().replace("batter_",""))
    if not key: return None
    series = [float(g.get(key,0) or 0) for g in logs[:last_n]]
    n = len(series)
    suc = sum(1 for v in series if v >= float(line))
    alpha, league = 8.0, 0.50
    smooth = (suc + alpha*league) / (n + alpha) if n else league
    raw    = (suc / n) if n else 0.0
    conf   = "high" if n>=12 else "medium" if n>=6 else "low"
    return {"hit_rate": round(smooth,6), "hit_rate_raw": round(raw,6),
            "sample_size": n, "successes": suc, "confidence": conf, "threshold": float(line)}

def enrich_props_mlb_simple(props: List[Dict[str,Any]]) -> int:
    if not props: return 0
    topk   = int(os.getenv("CTX_SIMPLE_TOPK","200"))
    budget = float(os.getenv("CTX_SIMPLE_BUDGET_SEC","2.0"))
    t_end  = time.time() + budget
    # prefer fair.prob.over for ranking, if present
    def fair(p): 
        try: return float(p.get("fair",{}).get("prob",{}).get("over") or 0.0)
        except: return 0.0
    n=0
    for p in sorted(props, key=fair, reverse=True)[:topk]:
        if time.time() > t_end: break
        if str(p.get("league","")).lower() != "mlb": continue
        ctx = get_mlb_contextual_hit_rate_cached(p.get("player",""), p.get("stat",""), float(p.get("line",0) or 0), last_n=int(os.getenv("CTX_SIMPLE_LAST_N","10")))
        if ctx:
            p.setdefault("enrichment",{})["mlb_context"] = ctx
            n+=1
    return n
