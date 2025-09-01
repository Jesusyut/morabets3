# stats_providers/mlb_providers.py
import os, requests

TIMEOUT = (2, 4)
UA = {"User-Agent": "morabets/1.0"}

class MLBStatsAPI:
    """Free MLB StatsAPI based fetcher. Normalize to {date,h,tb,hr,bb,sb,rbi,r} items."""
    def _get(self, url):
        r = requests.get(url, timeout=TIMEOUT, headers=UA)
        r.raise_for_status()
        return r.json()

    def get_game_logs(self, player_name: str, last_n: int = 10) -> list[dict]:
        # Implement with your current StatsAPI flow.
        # Minimal example (replace with your existing logic):
        # 1) look up player id by name
        # 2) fetch recent games
        # 3) normalize per-game row fields to keys below
        # Return list of dicts like {"date":"YYYY-MM-DD","h":2,"tb":3,"hr":0,"bb":1,"sb":0,"rbi":1,"r":1}
        raise NotImplementedError  # fill with your current code

class SportsDataIO:
    """Paid provider example (replace endpoint/key as needed)."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.s = requests.Session()
        self.s.headers.update(UA)

    def get_game_logs(self, player_id_or_name: str, last_n: int = 10) -> list[dict]:
        # Example shape (you must map to your paid API):
        # url = f"https://api.sportsdata.io/v3/mlb/stats/json/PlayerGameStatsByPlayer/{season}/{player_id}?key={self.api_key}"
        # r = self.s.get(url, timeout=TIMEOUT); r.raise_for_status(); j = r.json()
        # normalize -> list[dict] with keys: date,h,tb,hr,bb,sb,rbi,r
        raise NotImplementedError  # fill with your provider mapping

def provider_chain():
    """
    Order from env: 'STATS_PROVIDER_ORDER=sportsdataio,mlb'
    Fallback to 'mlb' if unset.
    """
    order = os.getenv("STATS_PROVIDER_ORDER", "mlb")
    parts = [p.strip().lower() for p in order.split(",") if p.strip()]
    sd_key = os.getenv("SPORTSDATAIO_KEY")
    for name in parts:
        if name == "sportsdataio" and sd_key:
            yield ("sportsdataio", SportsDataIO(sd_key))
        elif name == "mlb":
            yield ("mlb", MLBStatsAPI())
