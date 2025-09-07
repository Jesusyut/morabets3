import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, make_response
from flask_cors import CORS

# ----------------------------
# Optional matchup grouping (safe fallback)
# ----------------------------
try:
    from matchups import group_props_by_matchup
except Exception:
    def group_props_by_matchup(props, league):
        out = {}
        out.setdefault("Unknown @ Unknown", []).extend(props or [])
        return out

# ----------------------------
# Core odds adapters
# ----------------------------
from odds_api import fetch_player_props as fetch_mlb_player_props
from nfl_odds_api import fetch_nfl_player_props
from props_ncaaf import fetch_ncaaf_player_props
from props_ufc import fetch_ufc_totals_props

# If you added a simple MLB fallback in odds_api.py, we’ll try to import it:
try:
    from odds_api import fetch_player_props_simple as fetch_mlb_player_props_simple
except Exception:
    fetch_mlb_player_props_simple = None  # optional

# ----------------------------
# No-vig / EV helpers
# ----------------------------
from novig import american_to_prob, novig_two_way
try:
    from probability import book_breakeven_prob, ev_pct
except Exception:
    # Minimal internal versions in case probability.py lacks them
    def _american_to_decimal(odds: int) -> float:
        o = float(odds)
        return 1.0 + (o/100.0 if o > 0 else 100.0/abs(o))
    def book_breakeven_prob(american: int) -> float:
        return 1.0 / _american_to_decimal(int(american))
    def ev_value(p_true: float, american: int) -> float:
        d = _american_to_decimal(int(american))
        return p_true*(d - 1.0) - (1.0 - p_true)
    def ev_pct(p_true: float, american: int) -> float:
        return ev_value(p_true, american) * 100.0

# ----------------------------
# Guardrails (hide ultra-juice unless real edge)
# ----------------------------
try:
    from guardrails import MIN_AMERICAN_SINGLE, MAX_TRUE_PROB_SINGLE, MIN_EDGE_SINGLE
except Exception:
    MIN_AMERICAN_SINGLE = -200     # hide legs < -200 unless they clear edge
    MAX_TRUE_PROB_SINGLE = 0.75    # hide p* > 0.75 unless they clear edge
    MIN_EDGE_SINGLE      = 0.02    # require >= +2% edge vs breakeven

# ----------------------------
# Perf shim (safe if module missing)
# ----------------------------
try:
    import perf  # noqa
except Exception:
    class perf:  # type: ignore
        PERF_DEFAULT = False
        _enabled = False
        @classmethod
        def enable(cls, request_id=None): cls._enabled = True
        @classmethod
        def kv(cls, k, v): pass
        @classmethod
        def is_enabled(cls): return cls._enabled
        @classmethod
        def snapshot(cls): return {}
        @classmethod
        def to_header(cls, snap): return ""
        @classmethod
        def push_current(cls): pass
        @classmethod
        def disable(cls): cls._enabled = False
        @classmethod
        def recent(cls): return []

# ----------------------------
# Universal cache (safe shim if module missing)
# ----------------------------
try:
    from universal_cache import get_or_set_slot, slot_key, set_json, get_json, current_slot
except Exception:
    _SLOTS: Dict[str, Any] = {}
    def slot_key(kind, name): return f"{kind}:{name}"
    def set_json(k, v): _SLOTS[k] = v
    def get_json(k): return _SLOTS.get(k)
    def get_or_set_slot(kind, name, fn):
        k = slot_key(kind, name)
        if k not in _SLOTS:
            _SLOTS[k] = fn()
        return _SLOTS[k]
    def current_slot(): return {}

# -----------------------------------------------------------------------------
# App + config
# -----------------------------------------------------------------------------
log = logging.getLogger("app")
log.setLevel(logging.INFO)

def _norm_league(s: str = None) -> str:
    t = (s or "").strip().lower()
    aliases = {
        "ncaa": "ncaaf",
        "cfb": "ncaaf",
        "college_football": "ncaaf",
        "mma": "ufc",
        "udc": "ufc",
    }
    return aliases.get(t, t)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "mora-bets-secret-key-change-in-production")
CORS(app)

LICENSE_DB = os.environ.get("LICENSE_DB_PATH", "license_keys.json")

# -----------------------------------------------------------------------------
# Minimal public pages (no paywall)
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    return redirect(url_for("access"))

@app.route("/_debug/mlb")
def _debug_mlb():
    from odds_api import fetch_player_props as per_event
    out = {"ok": True}
    try:
        a = per_event()
        out["per_event_count"] = len(a or [])
        out["per_event_sample_market"] = (a[0].get("market") if a else None)
    except Exception as e:
        out["per_event_error"] = str(e)

    try:
        from odds_api import fetch_player_props_simple as simple
        b = simple()
        out["simple_count"] = len(b or [])
        out["simple_sample_market"] = (b[0].get("market") if b else None)
    except Exception as e:
        out["simple_error"] = str(e)

    out["env_has_api_key"] = bool(os.getenv("ODDS_API_KEY"))
    return jsonify(out)


@app.route("/access", methods=["GET"])
def access():
    """
    Simple access form: collect email → generate a gate key.
    If you don't have templates, we return a minimal HTML form.
    """
    try:
        return render_template("access.html")
    except Exception:
        return """
        <!doctype html><html><head><title>Mora Bets Access</title></head><body>
        <h1>Get Access to Mora Bets</h1>
        <form method="POST" action="/access">
          <label>Email</label><br/>
          <input type="email" name="email" placeholder="you@example.com" required/>
          <button type="submit">Get Gate Key</button>
        </form>
        <p style="margin-top:10px;">You'll receive a gate key on-screen. Use it to unlock the dashboard.</p>
        </body></html>
        """

@app.route("/access", methods=["POST"])
def access_submit():
    """
    Accept email, generate a key, store it in LICENSE_DB, show key and link to dashboard.
    """
    email = (request.form.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return redirect(url_for("access"))
    # Create human-ish key: first part of email + 4 random digits
    prefix = email.split("@")[0][:8].replace(".", "").replace("_", "")
    suffix = str(uuid.uuid4().int)[-4:]
    key = f"{prefix}{suffix}".lower()

    # Persist in file-store
    try:
        if os.path.exists(LICENSE_DB):
            with open(LICENSE_DB, "r") as f:
                keys = json.load(f)
        else:
            keys = {}
    except Exception:
        keys = {}
    keys[key] = {"email": email, "created": datetime.utcnow().isoformat() + "Z", "plan": "email_gate"}
    with open(LICENSE_DB, "w") as f:
        json.dump(keys, f)

    # Auto-login session
    session["licensed"] = True
    session["license_key"] = key
    session["access_level"] = "email"

    # Show key (or template if present)
    try:
        return render_template("verify.html", key=key, email=email)
    except Exception:
        return f"""
        <!doctype html><html><head><title>Gate Key</title></head><body>
        <h1>Your Gate Key</h1>
        <p>Email: <b>{email}</b></p>
        <p>Key: <code style="font-size:1.2em">{key}</code></p>
        <p><a href="/dashboard?key={key}">Open the Dashboard</a></p>
        </body></html>
        """

@app.route("/validate-key", methods=["POST"])
def validate_key():
    """
    Manual key validation (if someone pastes a key from the access page).
    """
    user_key = (request.form.get('key') or "").strip().lower()
    if not user_key:
        return jsonify({'valid': False})

    try:
        with open(LICENSE_DB, 'r') as f:
            keys = json.load(f)
    except Exception:
        keys = {}

    if user_key in keys:
        session["licensed"] = True
        session["license_key"] = user_key
        session["access_level"] = keys[user_key].get("plan", "email")
        return jsonify({'valid': True, 'redirect': url_for('dashboard')})

    return jsonify({'valid': False})

# -----------------------------------------------------------------------------
# License protection (dashboard only)
# -----------------------------------------------------------------------------
@app.before_request
def require_license():
    public_endpoints = {
        "home", "access", "access_submit",
        "healthz", "ping",
        "api_league_props", "player_props_legacy", "api_environment", "top_ev_props",
        "validate_key", "static"
    }
    if request.endpoint in public_endpoints or (request.path or "").startswith("/static"):
        return
    if request.endpoint == "dashboard" and session.get("licensed"):
        return
    if request.endpoint == "dashboard":
        return redirect(url_for("access"))

# -----------------------------------------------------------------------------
# Fetch helper (per league, with MLB fallback)
# -----------------------------------------------------------------------------
def get_player_props_for_league(league: str, *, date_str: str | None = None, nocache: bool = False):
    league = _norm_league(league or "mlb")

    # ---------- MLB ----------
    if league == "mlb":
        def _fetch():
            props = fetch_mlb_player_props()  # per-event fetcher
            if (not props) and fetch_mlb_player_props_simple:
                try:
                    props = fetch_mlb_player_props_simple()  # whole-board fallback
                except Exception:
                    pass
            return props or []

        if nocache:
            props = _fetch()
            set_json(slot_key("props", "mlb"), props)
            return props
        return get_or_set_slot("props", "mlb", _fetch)

    # ---------- NFL ----------
    elif league == "nfl":
        from nfl_odds_api import fetch_nfl_player_props
        if nocache:
            props = fetch_nfl_player_props(hours_ahead=96)
            set_json(slot_key("props", "nfl"), props)
            return props
        return get_or_set_slot("props", "nfl", lambda: fetch_nfl_player_props(hours_ahead=96))

    # ---------- NCAAF ----------
    elif league == "ncaaf":
        from props_ncaaf import fetch_ncaaf_player_props
        if nocache:
            props = fetch_ncaaf_player_props(date=date_str)
            set_json(slot_key("props", "ncaaf"), props)
            return props
        return get_or_set_slot("props", "ncaaf", lambda: fetch_ncaaf_player_props(date=date_str))

    # ---------- UFC ----------
    elif league == "ufc":
        from props_ufc import fetch_ufc_totals_props
        if nocache:
            props = fetch_ufc_totals_props(date_iso=date_str, hours_ahead=96)
            set_json(slot_key("props", "ufc"), props)
            return props
        return get_or_set_slot("props", "ufc", lambda: fetch_ufc_totals_props(date_iso=date_str, hours_ahead=96))

    # ---------- Unsupported ----------
    else:
        raise ValueError(f"Unsupported league: {league}")

# -----------------------------------------------------------------------------
# EV attach + guardrails
# -----------------------------------------------------------------------------
def _attach_ev_and_filter(props):
    """
    Expects each prop row to have:
      - row["fair"]["prob"]["over"/"under"] (no-vig p*)
      - row["shop"]["over"/"under"]["american"] (offered price)
    Attaches:
      - meta.over_breakeven / under_breakeven
      - meta.over_edge_pct / under_edge_pct
      - meta.over_ev_pct / under_ev_pct
    Filters heavy juice unless real edge.
    """
    out = []
    for row in props or []:
        fair = (row.get("fair") or {}).get("prob") or {}
        shop = row.get("shop") or {}
        over_am = (shop.get("over") or {}).get("american")
        under_am = (shop.get("under") or {}).get("american")
        p_over = fair.get("over")
        p_under = fair.get("under")

        row.setdefault("meta", {})
        m = row["meta"]

        def side_stats(p_true, american, side_key):
            if p_true is None or american is None: return
            be = book_breakeven_prob(int(american))
            edge = float(p_true) - be
            m[f"{side_key}_breakeven"] = round(be, 4)
            m[f"{side_key}_edge_pct"]  = round(edge*100.0, 2)
            m[f"{side_key}_ev_pct"]    = round(ev_pct(float(p_true), int(american)), 2)

        side_stats(p_over, over_am, "over")
        side_stats(p_under, under_am, "under")

        def hide(p_true, american):
            if p_true is None or american is None: return False
            be = book_breakeven_prob(int(american))
            # Hide heavy juice and high p* unless they CLEAR edge threshold
            if int(american) < MIN_AMERICAN_SINGLE or float(p_true) > MAX_TRUE_PROB_SINGLE:
                return (float(p_true) - be) < MIN_EDGE_SINGLE
            # Otherwise require minimal edge too
            return (float(p_true) - be) < MIN_EDGE_SINGLE

        drop_over  = hide(p_over, over_am)
        drop_under = hide(p_under, under_am)
        if not (drop_over and drop_under):
            out.append(row)
    return out

# -----------------------------------------------------------------------------
# Unified props API (EV only)
# -----------------------------------------------------------------------------
@app.route("/api/<league>/props")
def api_league_props(league):
    league = _norm_league(league or "mlb")
    date_str = request.args.get("date")
    nocache  = (request.args.get("nocache") == "1")

    # 1) fetch + EV attach
    rows  = get_player_props_for_league(league, date_str=date_str, nocache=nocache)
    props = _attach_ev_and_filter(rows)

    # 2) BACK-COMPAT for FE: mirror meta → ai.*
    for r in props:
        m = r.get("meta", {})
        r.setdefault("ai", {})
        # EV% (the thing you want to display)
        if "over_ev_pct" in m:   r["ai"]["ev_over_pct"]  = m["over_ev_pct"]
        if "under_ev_pct" in m:  r["ai"]["ev_under_pct"] = m["under_ev_pct"]
        # Edge as DECIMAL (0.03 = +3pp), not percent
        if "over_edge_pct" in m:  r["ai"]["edge_over"]  = round((m["over_edge_pct"]  or 0.0) / 100.0, 4)
        if "under_edge_pct" in m: r["ai"]["edge_under"] = round((m["under_edge_pct"] or 0.0) / 100.0, 4)
        # Clean source string (avoid [OBJECT OBJECT])
        src = (r.get("shop", {}).get("over", {}) or r.get("shop", {}).get("under", {}))
        r["source"] = (src.get("book") or r.get("book") or r.get("bookmaker") or "").upper()

    # 3) Stable ordering: sort by best EV side, desc
    def best_ev(row):
        m = row.get("meta", {})
        return max(m.get("over_ev_pct") or -1e9, m.get("under_ev_pct") or -1e9)
    props.sort(key=best_ev, reverse=True)

    # 4) Return FLAT list; empty matchups prevents FE regroup/re-sort
    return jsonify({
        "league": league,
        "date": date_str,
        "count": len(props),
        "props": props,
        "matchups": {},
        "enrichment_applied": True
    })

# Optional legacy alias
@app.route("/player_props")
def player_props_legacy():
    lg       = _norm_league(request.args.get("league", "mlb"))
    date_str = request.args.get("date")
    nocache  = (request.args.get("nocache") == "1")

    rows  = get_player_props_for_league(lg, date_str=date_str, nocache=nocache)
    props = _attach_ev_and_filter(rows)

    # FE back-compat (same as modern route)
    for r in props:
        m = r.get("meta", {})
        r.setdefault("ai", {})
        if "over_ev_pct" in m:   r["ai"]["ev_over_pct"]  = m["over_ev_pct"]
        if "under_ev_pct" in m:  r["ai"]["ev_under_pct"] = m["under_ev_pct"]
        if "over_edge_pct" in m:  r["ai"]["edge_over"]  = round((m["over_edge_pct"]  or 0.0) / 100.0, 4)
        if "under_edge_pct" in m: r["ai"]["edge_under"] = round((m["under_edge_pct"] or 0.0) / 100.0, 4)
        src = (r.get("shop", {}).get("over", {}) or r.get("shop", {}).get("under", {}))
        r["source"] = (src.get("book") or r.get("book") or r.get("bookmaker") or "").upper()

    def best_ev(row):
        m = row.get("meta", {})
        return max(m.get("over_ev_pct") or -1e9, m.get("under_ev_pct") or -1e9)
    props.sort(key=best_ev, reverse=True)

    return jsonify({
        "league": lg,
        "date": date_str,
        "count": len(props),
        "props": props,
        "matchups": {},
        "enrichment_applied": True
    })

# -----------------------------------------------------------------------------
# Top EV endpoint (great for landing/demo)
# -----------------------------------------------------------------------------
@app.route("/props/top")
def top_ev_props():
    league_in = request.args.get("league", "mlb")
    league = _norm_league(league_in)
    limit = int(request.args.get("limit", "50"))

    if league == "mlb":
        props = get_or_set_slot("props", "mlb", lambda: get_player_props_for_league("mlb"))
    elif league == "nfl":
        props = get_or_set_slot("props", "nfl", lambda: get_player_props_for_league("nfl"))
    elif league == "ncaaf":
        props = get_or_set_slot("props", "ncaaf", lambda: get_player_props_for_league("ncaaf"))
    elif league == "ufc":
        props = get_or_set_slot("props", "ufc", lambda: get_player_props_for_league("ufc"))
    else:
        return jsonify({"error": f"Unsupported league: {league_in}"}), 400

    props = _attach_ev_and_filter(props)

    # Flatten sides to rank by EV%
    legs = []
    for r in props:
        m = r.get("meta", {})
        if "over_ev_pct" in m:
            legs.append({"side":"over","ev_pct":m["over_ev_pct"],"edge_pct":m.get("over_edge_pct"),"row":r})
        if "under_ev_pct" in m:
            legs.append({"side":"under","ev_pct":m["under_ev_pct"],"edge_pct":m.get("under_edge_pct"),"row":r})
    legs.sort(key=lambda x: (x["ev_pct"] if x["ev_pct"] is not None else -9e9), reverse=True)
    return jsonify({"league": league, "count": min(limit, len(legs)), "legs": legs[:limit]})

# -----------------------------------------------------------------------------
# Dashboard (requires key; now unlocked via email gate)
# -----------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    if not session.get("licensed"):
        return redirect(url_for("access"))
    # Template version if present, else a simple table using /props/top
    try:
        return render_template("dashboard.html")
    except Exception:
        return """
        <!doctype html><html><head><title>Mora Bets Dashboard</title>
        <script>
        async function loadTop(){
          const res = await fetch('/props/top?league=mlb&limit=25');
          const j = await res.json();
          const rows = (j.legs||[]).map(x=>{
            const r=x.row||{}; const m=r.meta||{};
            return `<tr>
              <td>${r.player||'-'}</td>
              <td>${r.market||'-'}</td>
              <td>${x.side.toUpperCase()}</td>
              <td>${(m.over_ev_pct&&x.side==='over')?m.over_ev_pct:(m.under_ev_pct||'')}</td>
              <td>${(m.over_edge_pct&&x.side==='over')?m.over_edge_pct:(m.under_edge_pct||'')}</td>
            </tr>`;
          }).join('');
          document.getElementById('tb').innerHTML = rows;
        }
        window.onload = loadTop;
        </script></head>
        <body><h1>Mora Bets</h1>
        <p>Showing Top EV legs (MLB). Change league in the URL if needed.</p>
        <table border="1" cellpadding="6" cellspacing="0">
          <thead><tr><th>Player</th><th>Market</th><th>Side</th><th>EV% (per leg)</th><th>Edge vs BE (pp)</th></tr></thead>
          <tbody id="tb"></tbody>
        </table>
        </body></html>
        """

# -----------------------------------------------------------------------------
# Environments endpoint (placeholder so FE never 404s)
# -----------------------------------------------------------------------------
@app.route("/api/<league>/environment")
def api_environment(league):
    return jsonify({"environments": {}})

# -----------------------------------------------------------------------------
# Health & perf
# -----------------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})

@app.route("/ping")
def ping():
    return jsonify({"status": "running"})

@app.route("/_perf/recent", methods=["GET"])
def perf_recent():
    return jsonify({"recent": perf.recent()})

# -----------------------------------------------------------------------------
# Dev entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # For local testing
    app.run(debug=True, host="0.0.0.0", port=5001)
