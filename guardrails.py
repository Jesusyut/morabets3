import os
def _f(name, default): return float(os.getenv(name, default))
def _i(name, default): return int(os.getenv(name, default))

MIN_AMERICAN_SINGLE   = _i("MIN_AMERICAN_SINGLE", -200)  # hide < -200 unless real edge
MAX_TRUE_PROB_SINGLE  = _f("MAX_TRUE_PROB_SINGLE", 0.75) # hide p*>0.75 unless edge
MIN_EDGE_SINGLE       = _f("MIN_EDGE_SINGLE", 0.02)      # need >= +2% edge to show
