import os
import json
import logging
import time
import requests
import stripe
import uuid
import hashlib
from datetime import datetime, timedelta, date
from typing import List, Dict, Any
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, make_response
from flask_cors import CORS

# ----------------------------
# Matchup grouping / environments
# ----------------------------
try:
    from matchups import group_props_by_matchup
except Exception:
    # minimal fallback: put every prop into "Unknown @ Unknown"
    def group_props_by_matchup(props, league):
        out = {}
        out.setdefault("Unknown @ Unknown", []).extend(props or [])
        return out

try:
    from environments import compute_environments_for_league
except Exception:
    def compute_environments_for_league(league: str) -> Dict[str, Dict]:
        return {}

# ----------------------------
# Core odds adapters
# ----------------------------
from odds_api import fetch_player_props as fetch_mlb_player_props
from nfl_odds_api import fetch_nfl_player_props
from props_ncaaf import fetch_ncaaf_player_props
from props_ufc import fetch_ufc_totals_props

# ----------------------------
# No-vig utils
# ----------------------------
from novig import american_to_prob, novig_two_way

# ----------------------------
# Cache metrics (safe shim)
# ----------------------------
try:
    from cache_ttl import metrics as cache_metrics
except Exception:
    def cache_metrics():
        return {}

# ----------------------------
# perf shim (safe if module missing)
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
# AI overlay (optional; kept)
# ----------------------------
try:
    from cache_ttl import cache_ttl  # if present
except Exception:
    def cache_ttl(seconds=60):
        def deco(fn): return fn
        return deco

try:
    from ai_scout import attach_ai_edges, get_ai_picks_cached  # your existing AI enrichment
except Exception:
    # Safe fallbacks if not present
    def attach_ai_edges(*args, **kwargs): return 0
    def get_ai_picks_cached(*args, **kwargs): return {"results": []}

# ----------------------------
# Universal cache helpers
# ----------------------------
from universal_cache import get_or_set_slot, slot_key, set_json, get_json, current_slot

# ----------------------------
# Optional tolerant import name for legacy overlay
# ----------------------------
try:
    from ai_overlay.mlb import attach_mlb_ai_overlay as _attach_edges
except Exception:
    try:
        from ai_scout import attach_ai_edges as _attach_edges
    except Exception:
        _attach_edges = None

# ----------------------------
# Bets5 contextual hit rate (tolerant import)
# ----------------------------
try:
    # Expose: get_contextual_hit_rate(player, stat, point, league) -> float 0..1 or 0..100
    from contextual import get_contextual_hit_rate
except Exception:
    def get_contextual_hit_rate(player, stat, point, league):
        return None  # safe fallback if module missing

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

# Stripe configuration
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')
LICENSE_DB = 'license_keys.json'
PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY")
PRICE_MONTHLY = os.environ.get("STRIPE_PRICE_ID_MONTHLY")
PRICE_YEARLY  = os.environ.get("STRIPE_PRICE_ID_YEARLY")
TRIAL_DAYS    = int(os.environ.get("TRIAL_DAYS", "3"))
APP_BASE_URL  = os.environ.get("APP_BASE_URL", "http://localhost:5000")

# AI Overlay configuration
AI_OVERLAY_ENABLED = os.getenv("AI_OVERLAY_ENABLED", "true").lower() in ("1", "true", "yes")
AI_MIN_EDGE   = float(os.getenv("AI_MIN_EDGE", "0.06"))   # NOTE: could be 0.06 (fraction) or 6 (percent)
AI_ATTACH_CAP = int(os.getenv("AI_ATTACH_CAP", "120"))

# Legacy price lookup
PRICE_LOOKUP = {
    'prod_SjjH7D6kkxRbJf': 'price_1RoFpPIzLEeC8QTz5kdeiLyf',  # Calculator Tool - $9.99/month
    'prod_Sjkk8GQGPBvuOP': 'price_1RoHFOIzLEeC8QTziT9k1t45'   # Mora Assist - $28.99
}

# -----------------------------------------------------------------------------
# Perf tracking
# -----------------------------------------------------------------------------
@app.before_request
def _perf_begin():
    want = getattr(perf, "PERF_DEFAULT", False) or (request.args.get("trace") == "1")
    if want:
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        perf.enable(request_id=f"{request.path}:{rid}")
        perf.kv("path", request.path)
        perf.kv("query", request.query_string.decode("utf-8"))

@app.after_request
def _perf_finish(resp):
    try:
        if perf.is_enabled():
            snap = perf.snapshot()
            if snap:
                resp.headers["X-Perf"] = perf.to_header(snap)
                perf.push_current()
            perf.disable()
    except Exception:
        pass
    return resp

# -----------------------------------------------------------------------------
# Public routes (landing/paywall/etc.)
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    return redirect(url_for("how_it_works"))

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/paywall")
def paywall():
    return render_template("index.html")

@app.route("/config", methods=["GET"])
def paywall_config():
    return jsonify({
        "publicKey": PUBLISHABLE_KEY,
        "priceMonthly": PRICE_MONTHLY,
        "priceYearly":  PRICE_YEARLY,
        "trialDays":    TRIAL_DAYS
    })

@app.route("/tool")
def tool():
    if session.get("licensed"):
        return redirect(url_for("dashboard"))
    else:
        return redirect(url_for("paywall") + "?message=You need a valid license key to access the tool.")

# -----------------------------------------------------------------------------
# Stripe checkout
# -----------------------------------------------------------------------------
@app.route("/create-checkout-session", methods=['POST'])
def create_checkout_session():
    try:
        product_id = request.form.get('product_id')
        data = None
        if not product_id:
            try:
                data = request.get_json(force=True)
                price_id = data.get("price_id") if data else None
            except:
                return jsonify({"error": "Missing product or price ID"}), 400
        else:
            price_id = PRICE_LOOKUP.get(product_id)

        if not price_id:
            return jsonify({"error": "Invalid product"}), 400

        if price_id not in [PRICE_MONTHLY, PRICE_YEARLY] + list(PRICE_LOOKUP.values()):
            return jsonify({"error": "Invalid price"}), 400

        subscription_data = {}
        if price_id == PRICE_MONTHLY and TRIAL_DAYS > 0:
            subscription_data["trial_period_days"] = TRIAL_DAYS
        if price_id == 'price_1RoFpPIzLEeC8QTz5kdeiLyf':
            subscription_data["trial_period_days"] = 3

        session_config = {
            'line_items': [{'price': price_id, 'quantity': 1}],
            'mode': 'subscription',
            'allow_promotion_codes': True,
            'success_url': f'{APP_BASE_URL}/dashboard?session_id={{CHECKOUT_SESSION_ID}}',
            'cancel_url': f'{APP_BASE_URL}/paywall?canceled=true',
        }
        if subscription_data:
            session_config['subscription_data'] = subscription_data

        if product_id == 'prod_Sjkk8GQGPBvuOP':
            session_config['phone_number_collection'] = {'enabled': True}
            session_config['custom_fields'] = [{
                'key': 'disclaimer',
                'label': {'type': 'custom', 'custom': 'Risk Acknowledgment (18+)'},
                'type': 'dropdown',
                'dropdown': {'options': [{'label': 'I agree (not financial advice)', 'value': 'agree'}]},
                'optional': False
            }]

        sess = stripe.checkout.Session.create(**session_config)
        if data:
            return jsonify({"id": sess.id, "url": sess.url})
        else:
            return redirect(sess.url or request.url_root, code=303)

    except Exception as e:
        log.error(f"Stripe checkout error: {e}")
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        return f"Checkout failed: {str(e)}", 500

# -----------------------------------------------------------------------------
# License verification
# -----------------------------------------------------------------------------
@app.route("/verify")
def verify():
    session_id = request.args.get('session_id')
    key = request.args.get('key')
    if key:
        return render_template('verify.html', key=key)
    if not session_id:
        return render_template('verify.html', error='Missing session ID.')

    try:
        sess = stripe.checkout.Session.retrieve(session_id, expand=['customer'])
        if not sess.customer_details:
            return render_template('verify.html', error="No customer details found")
        customer_email = sess.customer_details.email or "unknown@example.com"
        customer_name  = sess.customer_details.name or 'user'
        last = customer_name.split()[-1].lower()
        suffix = str(uuid.uuid4().int)[-4:]
        key = f'{last}{suffix}'

        try:
            with open(LICENSE_DB, 'r') as f:
                keys = json.load(f)
        except:
            keys = {}

        line_items = sess.get('line_items', {}).get('data', [])
        is_mora_assist = False
        if line_items:
            price_id = line_items[0].get('price', {}).get('id', '')
            is_mora_assist = price_id == 'price_1RoHFOIzLEeC8QTziT9k1t45'

        if is_mora_assist:
            phone_number = getattr(sess.customer_details, 'phone', 'Not provided')
            log.info(f"✅ Mora Assist purchase confirmed: {customer_email}, Phone: {phone_number}")
            return render_template('verify.html', mora_assist=True, email=customer_email, phone=phone_number)
        else:
            keys[key] = {'email': customer_email, 'plan': sess.mode}
            with open(LICENSE_DB, 'w') as f:
                json.dump(keys, f)
            log.info(f"✅ Generated license key for {customer_email}: {key}")
            return render_template('verify.html', key=key)

    except Exception as e:
        log.error(f"❌ Stripe verification error: {e}")
        return render_template('verify.html', error='Verification failed. Please contact support.')

@app.route("/verify-key")
def verify_key():
    user_key = request.args.get('key', '').strip()
    try:
        with open(LICENSE_DB, 'r') as f:
            keys = json.load(f)
    except Exception as e:
        log.error(f"Error loading license keys: {e}")
        return jsonify({'valid': False})

    is_valid = any(k.upper() == user_key.upper() and keys[k] for k in keys)
    log.info(f"Key verification for '{user_key}': {'Valid' if is_valid else 'Invalid'}")
    return jsonify({'valid': is_valid})

@app.route("/validate-key", methods=["POST"])
def validate_key():
    user_key = request.form.get('key', '').strip().lower()
    if user_key == 'mora-king':
        session["licensed"] = True
        session["license_key"] = user_key
        session["access_level"] = "creator"
        log.info("✅ Master key access granted")
        return jsonify({'valid': True, 'redirect': url_for('dashboard')})

    try:
        with open(LICENSE_DB, 'r') as f:
            keys = json.load(f)
    except:
        return jsonify({'valid': False})

    if user_key in keys:
        session["licensed"] = True
        session["license_key"] = user_key
        session["access_level"] = "premium"
        log.info(f"✅ License key validated: {user_key}")
        return jsonify({'valid': True, 'redirect': url_for('dashboard')})

    return jsonify({'valid': False})

# -----------------------------------------------------------------------------
# Dashboards (protected)
# -----------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    user_key = request.args.get('key', '').strip()
    if user_key:
        try:
            with open(LICENSE_DB, 'r') as f:
                keys = json.load(f)
        except Exception as e:
            log.error(f"Error loading license keys: {e}")
            return redirect(url_for('index') + '?message=System+error.+Please+try+again.')

        is_valid = any(k.upper() == user_key.upper() and keys[k] for k in keys)
        if not is_valid:
            log.info(f"Invalid key attempt: {user_key}")
            return redirect(url_for('index') + '?message=Invalid+key.+Please+try+again.')

        session["licensed"] = True
        session["license_key"] = user_key
        log.info(f"✅ Dashboard access granted for key: {user_key}")

    try:
        return render_template("dashboard.html", hits=0)
    except Exception as e:
        log.error(f"Error in dashboard route: {e}")
        return f"""<!DOCTYPE html><html><head><title>Mora Bets</title></head>
        <body><h1>Mora Bets - Sports Betting Analytics</h1>
        <p>System Status: Running</p><p>Error: {str(e)}</p><p><a href="/healthz">Health Check</a></p></body></html>"""

@app.route("/dashboard_legacy")
def dashboard_legacy():
    user_key = request.args.get('key', '').strip()
    if user_key:
        try:
            with open(LICENSE_DB, 'r') as f:
                keys = json.load(f)
        except Exception as e:
            log.error(f"Error loading license keys: {e}")
            return redirect(url_for('index') + '?message=System+error.+Please+try+again.')

        is_valid = any(k.upper() == user_key.upper() and keys[k] for k in keys)
        if not is_valid:
            log.info(f"Invalid key attempt: {user_key}")
            return redirect(url_for('index') + '?message=Invalid+key.+Please+try+again.')

        session["licensed"] = True
        session["license_key"] = user_key
        log.info(f"✅ Legacy dashboard access granted for key: {user_key}")

    try:
        return render_template("dashboard_legacy.html", hits=0)
    except Exception as e:
        log.error(f"Error in legacy dashboard route: {e}")
        return f"""<!DOCTYPE html><html><head><title>Mora Bets - Legacy</title></head>
        <body><h1>Mora Bets - Legacy Dashboard</h1>
        <p>System Status: Running</p><p>Error: {str(e)}</p><p><a href="/dashboard">New Dashboard</a></p></body></html>"""

# -----------------------------------------------------------------------------
# License protection
# -----------------------------------------------------------------------------
@app.before_request
def require_license():
    public_endpoints = [
        "home", "how_it_works", "paywall", "paywall_config", "tool",
        "verify", "verify_key", "validate_key", "create_checkout_session",
        "healthz", "ping", "static", "logout", "dashboard", "dashboard_legacy",
        "ai_edge_scout", "cron_prewarm", "api_league_props", "api_environment",
        "api_trends_l10", "api_ai_scout", "player_props_legacy"
    ]
    if request.endpoint in public_endpoints or request.path.startswith("/static") or request.path.startswith("/api/"):
        return
    if not session.get("licensed"):
        return redirect(url_for("paywall"))

# -----------------------------------------------------------------------------
# Fetch helper (DRY)
# -----------------------------------------------------------------------------
def get_player_props_for_league(league: str, *, date_str: str | None = None, nocache: bool = False):
    league = _norm_league(league or "mlb")

    if league == "mlb":
        if nocache:
            props = fetch_mlb_player_props()
            set_json(slot_key("props", "mlb"), props)
        else:
            props = get_or_set_slot("props", "mlb", fetch_mlb_player_props)
        return props

    if league == "nfl":
        if nocache:
            props = fetch_nfl_player_props(hours_ahead=96)
            set_json(slot_key("props", "nfl"), props)
        else:
            props = get_or_set_slot("props", "nfl", fetch_nfl_player_props)
        return props

    if league == "ncaaf":
        if nocache:
            props = fetch_ncaaf_player_props(date=date_str)
            set_json(slot_key("props", "ncaaf"), props)
        else:
            props = get_or_set_slot("props", "ncaaf", lambda: fetch_ncaaf_player_props(date=date_str))
        return props

    if league == "ufc":
        if nocache:
            props = fetch_ufc_totals_props(date_iso=date_str, hours_ahead=96)
            set_json(slot_key("props", "ufc"), props)
        else:
            props = get_or_set_slot("props", "ufc", lambda: fetch_ufc_totals_props(date_iso=date_str, hours_ahead=96))
        return props

    # TODO: add NBA/NHL fetchers when ready
    return []

# -----------------------------------------------------------------------------
# Contextual + no-vig EV enrichment (Bets5 style)
# -----------------------------------------------------------------------------
def _ctx_rate_fraction(player: str | None, stat: str | None, point: float | str | None, league: str) -> float | None:
    """
    Call Bets5 contextual model. It may return 0..1 or 0..100.
    Normalize to 0..1. Return None if unavailable.
    """
    try:
        r = get_contextual_hit_rate(player, stat, point, league)
        if r is None:
            return None
        r = float(r)
        return r/100.0 if r > 1.0 else r
    except Exception:
        return None

def _price_from(prop, key_name):
    """Accept numbers or dict like {'price': -120}."""
    v = prop.get(key_name)
    if isinstance(v, dict):
        return v.get("price") or v.get("odds")
    return v

def _compute_fair_probs(prop):
    """Build fair (no-vig) probabilities from over/under American odds."""
    over = _price_from(prop, "over")  or prop.get("over_odds")
    under = _price_from(prop, "under") or prop.get("under_odds")
    try:
        po = american_to_prob(float(over))   if over  is not None else None
        pu = american_to_prob(float(under))  if under is not None else None
        if po is None or pu is None:
            return None, None
        s = po + pu
        if s <= 0:
            return None, None
        return po/s, pu/s
    except Exception:
        return None, None

def _edge_threshold_pp() -> float:
    """
    Interpret AI_MIN_EDGE env:
      - if <= 1, treat as fraction (e.g., 0.06 -> 6.0 pp)
      - else already percent points (e.g., 6 -> 6.0 pp)
    """
    v = AI_MIN_EDGE
    return (v * 100.0) if v <= 1.0 else v

def enrich_with_context_and_edge(rows, league: str):
    """
    For each raw prop:
      - compute fair (no-vig) prob
      - fetch contextual hit rate (Bets5) → normalize
      - compute EV edge = contextual - fair
      - attach results under 'fair', 'contextual', 'ai'
    Returns a NEW list of props ready for FE.
    """
    thr = _edge_threshold_pp()
    out = []
    for r in rows:
        p = dict(r)  # shallow copy

        # Normalize common fields we might need
        p.setdefault("stat", p.get("stat_type"))
        p.setdefault("point", p.get("line", p.get("point")))
        p.setdefault("player", p.get("player_name") or p.get("fighter") or p.get("fighter_a"))

        # 1) fair probs (no-vig)
        fo, fu = _compute_fair_probs(p)
        if fo is not None and fu is not None:
            p.setdefault("fair", {}).setdefault("prob", {})
            p["fair"]["prob"]["over"] = fo
            p["fair"]["prob"]["under"] = fu

        # 2) contextual (Bets5)
        c_over = _ctx_rate_fraction(p.get("player"), p.get("stat"), p.get("point"), league)
        if c_over is not None:
            p.setdefault("contextual", {}).setdefault("prob", {})
            p["contextual"]["prob"]["over"]  = c_over
            p["contextual"]["prob"]["under"] = 1.0 - c_over

        # 3) edge (only if we have both)
        p.setdefault("ai", {})
        if fo is not None and c_over is not None:
            edge_over  = (c_over - fo) * 100.0
            edge_under = -edge_over  # since fu = 1-fo, cu = 1-c_over
            p["ai"]["edge_over"]  = round(edge_over, 1)
            p["ai"]["edge_under"] = round(edge_under, 1)
            # Pick side if sizable threshold met
            if edge_over >= thr:
                p["ai"]["pick"] = "OVER"
            elif edge_under >= thr:
                p["ai"]["pick"] = "UNDER"
            else:
                p["ai"]["pick"] = None
        else:
            p["ai"]["edge_over"] = p["ai"]["edge_under"] = None
            p["ai"]["pick"] = None

        out.append(p)
    return out

# -----------------------------------------------------------------------------
# Unified props API (enrich once, group once, add environments)
# -----------------------------------------------------------------------------
@app.route("/api/<league>/props")
def api_league_props(league):
    league = _norm_league(league or "mlb")
    date_str = request.args.get("date")
    nocache  = (request.args.get("nocache") == "1")

    # 1) Fetch
    rows = get_player_props_for_league(league, date_str=date_str, nocache=nocache)

    # 2) Enrich ONCE — Bets5 contextual + no-vig EV
    props = enrich_with_context_and_edge(rows, league)

    # (Optional) Add GPT blurbs AFTER edges (won’t overwrite edge numbers if your fn only adds reasons)
    # try:
    #     attach_ai_edges(props, min_edge=_edge_threshold_pp(), cap=AI_ATTACH_CAP)
    # except Exception:
    #     pass

    # 3) Group (Bets5-style)
    try:
        grouped = group_props_by_matchup(props, league)
    except Exception as e:
        log.exception("group_props_by_matchup failed: %s", e)
        grouped = {}

    # 4) Environments (safe if empty)
    try:
        env_map = compute_environments_for_league(league) or {}
    except Exception as e:
        log.exception("compute_environments_for_league failed: %s", e)
        env_map = {}

    return jsonify({
        "league": league,
        "date": date_str,
        "count": len(props),
        "props": props,          # flat list for list views
        "matchups": grouped,     # matchup -> [props] for Today’s Games
        "environments": env_map, # FE can ignore if {}
        "enrichment_applied": True
    })

# Optional: legacy alias for older FE calls (same shape)
@app.route("/player_props")
def player_props_legacy():
    lg       = _norm_league(request.args.get("league", "mlb"))
    date_str = request.args.get("date")
    nocache  = (request.args.get("nocache") == "1")

    rows = get_player_props_for_league(lg, date_str=date_str, nocache=nocache)
    props = enrich_with_context_and_edge(rows, lg)
    grouped = group_props_by_matchup(props, lg)
    try:
        env_map = compute_environments_for_league(lg) or {}
    except Exception:
        env_map = {}

    return jsonify({
        "league": lg, "date": date_str, "count": len(props),
        "props": props, "matchups": grouped, "environments": env_map,
        "enrichment_applied": True
    })

# Environments endpoint (so FE never 404s)
@app.route("/api/<league>/environment")
def api_environment(league):
    try:
        return jsonify({"environments": compute_environments_for_league(_norm_league(league)) or {}})
    except Exception:
        return jsonify({"environments": {}})

# -----------------------------------------------------------------------------
# AI endpoints (kept)
# -----------------------------------------------------------------------------
@app.route("/ai/edge_scout")
def ai_edge_scout():
    from openai import OpenAI
    from ai_scout import scout_cached_for_league

    league_in = request.args.get("league")
    league = _norm_league(league_in)
    if league != "mlb":
        return jsonify({"error": f"league '{league_in}' not supported in v1"}), 400

    rows = get_or_set_slot("props", "mlb", fetch_mlb_player_props)

    try:
        client = OpenAI()
    except Exception as e:
        return jsonify({"error": "OPENAI_API_KEY not configured", "detail": str(e)}), 503

    nocache = request.args.get("nocache") == "1"
    out = scout_cached_for_league(client, rows, league="mlb", top_k=30, force_refresh=nocache)
    return jsonify(out)

@app.route("/contextual/hit_rates", methods=["POST"])
def contextual_hit_rates():
    return jsonify({"results": []})

@app.route("/api/trends/l10", methods=["GET"])
def api_trends_l10():
    return jsonify({"results": []})

@app.route("/api/ai_scout")
def api_ai_scout():
    league = _norm_league(request.args.get("league", "mlb"))
    if league == "mlb":
        props = fetch_mlb_player_props()
    elif league == "nfl":
        props = fetch_nfl_player_props()
    elif league == "ncaaf":
        props = fetch_ncaaf_player_props()
    elif league == "ufc":
        props = fetch_ufc_totals_props()
    else:
        return jsonify({"error": f"Unsupported league: {league}"}), 400

    try:
        attach_ai_edges(props, min_edge=_edge_threshold_pp(), cap=AI_ATTACH_CAP)
    except Exception:
        pass

    data = get_ai_picks_cached(league.upper(), None, props)
    return jsonify(data)

# -----------------------------------------------------------------------------
# Health & utils
# -----------------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})

@app.route("/ping")
def ping():
    return jsonify({"status": "running"})

@app.route("/_cron/prewarm")
def cron_prewarm():
    token = request.args.get("key")
    if token != os.getenv("CRON_KEY"):
        return jsonify({"error":"unauthorized"}), 401

    leagues = [l.strip() for l in (request.args.get("leagues","mlb,nfl,ncaaf,ufc").split(",")) if l.strip()]
    out = {}
    for L in leagues:
        L = _norm_league(L)
        try:
            if L == "mlb":
                props = fetch_mlb_player_props()
                set_json(slot_key("props", "mlb"), props)
                try:
                    from openai import OpenAI
                    from ai_scout import scout_cached_for_league
                    client = OpenAI()
                    _ = scout_cached_for_league(client, props, league="mlb", top_k=30, force_refresh=True)
                except Exception as e:
                    out["mlb_ai"] = f"error: {e}"
            elif L == "nfl":
                props = fetch_nfl_player_props()
                set_json(slot_key("props", "nfl"), props)
            elif L == "ncaaf":
                props = fetch_ncaaf_player_props()
                set_json(slot_key("props", "ncaaf"), props)
            elif L == "ufc":
                props = fetch_ufc_totals_props(hours_ahead=96)
                set_json(slot_key("props", "ufc"), props)
            else:
                out[L] = "skipped: unsupported"
        except Exception as e:
            out[L] = f"error: {e}"

    out["status"] = "ok"
    return jsonify(out)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("how_it_works"))

# -----------------------------------------------------------------------------
# Perf debug
# -----------------------------------------------------------------------------
@app.route("/_perf/recent", methods=["GET"])
def perf_recent():
    return jsonify({"recent": perf.recent()})

@app.route("/_perf/cache", methods=["GET"])
def perf_cache():
    return jsonify({"cache": cache_metrics()})

# -----------------------------------------------------------------------------
# Dev entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
