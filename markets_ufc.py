# markets_ufc.py
UFC_SPORT_KEY = "mma_mixed_martial_arts"
UFC_ML_MARKET = "h2h"
UFC_MOV_PATTERNS = ["method", "to_win_by", "win_by", "victory_method"]

# Totals markets for fight duration
UFC_TOTALS_MARKETS = ["totals", "total_rounds", "fight_totals", "rounds"]

MOV_CANON = {
    "ko":  ["ko", "tko", "ko/tko", "ko or tko", "technical knockout", "knockout"],
    "sub": ["submission", "wins by submission", "by submission"],
    "dec": ["decision", "points", "win on points", "by decision"],
}
