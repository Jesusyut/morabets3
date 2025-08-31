import os
import json
import logging
import time
import requests
import stripe
import uuid
import hashlib
from datetime import datetime, timedelta, date
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, make_response
from flask_cors import CORS
from typing import List, Dict, Any

# Core odds modules
from odds_api import fetch_player_props as fetch_mlb_player_props
from nfl_odds_api import fetch_nfl_player_props
from props_ncaaf import fetch_ncaaf_player_props
from props_ufc import fetch_ufc_totals_props

# Helper modules
from novig import american_to_prob, novig_two_way
from cache_ttl import metrics as cache_metrics
import perf

log = logging.getLogger("app")
log.setLevel(logging.INFO)

def _norm_league(s: str | None) -> str:
    """Normalize league names with aliases"""
    t = (s or "").strip().lower()
    aliases = {
        "ncaa": "ncaaf",
        "cfb": "ncaaf", 
        "college_football": "ncaaf",
        "ncaaf": "ncaaf",
        "nfl": "nfl",
        "mlb": "mlb",
        "mma": "ufc",
        "udc": "ufc",
        "ufc": "ufc",
    }
    return aliases.get(t, t)

# Flask app setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "mora-bets-secret-key-change-in-production")
CORS(app)

# Stripe configuration
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')
LICENSE_DB = 'license_keys.json'
PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY")
PRICE_MONTHLY = os.environ.get("STRIPE_PRICE_ID_MONTHLY")
PRICE_YEARLY = os.environ.get("STRIPE_PRICE_ID_YEARLY")
TRIAL_DAYS = int(os.environ.get("TRIAL_DAYS", "3"))
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:5000")

# Legacy price lookup for backward compatibility
PRICE_LOOKUP = {
    'prod_SjjH7D6kkxRbJf': 'price_1RoFpPIzLEeC8QTz5kdeiLyf',  # Calculator Tool - $9.99/month
    'prod_Sjkk8GQGPBvuOP': 'price_1RoHFOIzLEeC8QTziT9k1t45'   # Mora Assist - $28.99
}

# Performance tracking
@app.before_request
def _perf_begin():
    want = perf.PERF_DEFAULT or (request.args.get("trace") == "1")
    if want:
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        perf.enable(request_id=f"{request.path}:{rid}")
        perf.kv("path", request.path)
        perf.kv("query", request.query_string.decode("utf-8"))

@app.after_request
def _perf_finish(resp):
    if perf.is_enabled():
        snap = perf.snapshot()
        if snap:
            resp.headers["X-Perf"] = perf.to_header(snap)
            perf.push_current()
        perf.disable()
    return resp

# Public routes
@app.route("/")
def home():
    """Redirect to how-it-works landing page"""
    return redirect(url_for("how_it_works"))

@app.route("/how-it-works")
def how_it_works():
    """Landing page explaining how Mora Bets works"""
    return render_template("how_it_works.html")

@app.route("/paywall")
def paywall():
    """Pricing page with Stripe checkout options"""
    return render_template("index.html")

@app.route("/config", methods=["GET"])
def paywall_config():
    """Return paywall configuration for frontend"""
    return jsonify({
        "publicKey": PUBLISHABLE_KEY,
        "priceMonthly": PRICE_MONTHLY,
        "priceYearly": PRICE_YEARLY,
        "trialDays": TRIAL_DAYS
    })

@app.route("/tool")
def tool():
    """Tool access page - requires valid license"""
    if session.get("licensed"):
        return redirect(url_for("dashboard"))
    else:
        return redirect(url_for("paywall") + "?message=You need a valid license key to access the tool.")

# Stripe checkout
@app.route("/create-checkout-session", methods=['POST'])
def create_checkout_session():
    """Create Stripe checkout session - supports both legacy and new pricing"""
    try:
        # Handle legacy form-based product ID
        product_id = request.form.get('product_id')
        
        # Handle new JSON-based price ID for monthly/yearly toggle
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
        
        # Validate that price_id is one of our accepted prices
        if price_id not in [PRICE_MONTHLY, PRICE_YEARLY] + list(PRICE_LOOKUP.values()):
            return jsonify({"error": "Invalid price"}), 400
            
        # Configure session
        subscription_data = {}
        
        # Add trial only for monthly subscription
        if price_id == PRICE_MONTHLY and TRIAL_DAYS > 0:
            subscription_data["trial_period_days"] = TRIAL_DAYS
        
        # Legacy trial for old price
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
        
        # Enable phone collection and disclaimer for legacy Mora Assist
        if product_id == 'prod_Sjkk8GQGPBvuOP':
            session_config['phone_number_collection'] = {'enabled': True}
            session_config['custom_fields'] = [
                {
                    'key': 'disclaimer',
                    'label': {
                        'type': 'custom',
                        'custom': 'Risk Acknowledgment (18+)'
                    },
                    'type': 'dropdown',
                    'dropdown': {
                        'options': [
                            {'label': 'I agree (not financial advice)', 'value': 'agree'}
                        ]
                    },
                    'optional': False
                }
            ]
        
        session = stripe.checkout.Session.create(**session_config)
        
        # Return JSON response for new API or redirect for legacy
        if data:
            return jsonify({"id": session.id, "url": session.url})
        else:
            return redirect(session.url or request.url_root, code=303)
            
    except Exception as e:
        log.error(f"Stripe checkout error: {e}")
        if data:
            return jsonify({"error": str(e)}), 400
        else:
            return f"Checkout failed: {str(e)}", 500

# License verification
@app.route("/verify")
def verify():
    """Handle Stripe success and generate license key"""
    session_id = request.args.get('session_id')
    key = request.args.get('key')  # For direct key display
    
    if key:
        return render_template('verify.html', key=key)
    
    if not session_id:
        return render_template('verify.html', error='Missing session ID.')

    try:
        session = stripe.checkout.Session.retrieve(session_id, expand=['customer'])
        if not session.customer_details:
            return render_template('verify.html', error="No customer details found")
        customer_email = session.customer_details.email or "unknown@example.com"
        customer_name = session.customer_details.name or 'user'
        last = customer_name.split()[-1].lower()
        suffix = str(uuid.uuid4().int)[-4:]
        key = f'{last}{suffix}'

        # Load existing keys
        try:
            with open(LICENSE_DB, 'r') as f:
                keys = json.load(f)
        except:
            keys = {}

        # Check if this is Mora Assist (no license key needed)
        line_items = session.get('line_items', {}).get('data', [])
        is_mora_assist = False
        if line_items:
            price_id = line_items[0].get('price', {}).get('id', '')
            is_mora_assist = price_id == 'price_1RoHFOIzLEeC8QTziT9k1t45'
        
        if is_mora_assist:
            # Mora Assist - no license key, just confirmation
            phone_number = getattr(session.customer_details, 'phone', 'Not provided')
            log.info(f"✅ Mora Assist purchase confirmed: {customer_email}, Phone: {phone_number}")
            return render_template('verify.html', mora_assist=True, email=customer_email, phone=phone_number)
        else:
            # Calculator Tool - generate license key
            keys[key] = {'email': customer_email, 'plan': session.mode}
            with open(LICENSE_DB, 'w') as f:
                json.dump(keys, f)

            log.info(f"✅ Generated license key for {customer_email}: {key}")
            return render_template('verify.html', key=key)
        
    except Exception as e:
        log.error(f"❌ Stripe verification error: {e}")
        return render_template('verify.html', error='Verification failed. Please contact support.')

@app.route("/verify-key")
def verify_key():
    """Verify license key for dashboard access"""
    user_key = request.args.get('key', '').strip()
    
    # Load keys from JSON file
    try:
        with open(LICENSE_DB, 'r') as f:
            keys = json.load(f)
    except Exception as e:
        log.error(f"Error loading license keys: {e}")
        return jsonify({'valid': False})
    
    # Check if key exists and is valid (case-insensitive)
    is_valid = False
    for key in keys:
        if key.upper() == user_key.upper() and keys[key]:
            is_valid = True
            break
    
    log.info(f"Key verification for '{user_key}': {'Valid' if is_valid else 'Invalid'}")
    
    return jsonify({'valid': is_valid})

@app.route("/validate-key", methods=["POST"])
def validate_key():
    """Validate license key and grant access"""
    user_key = request.form.get('key', '').strip().lower()
    
    # Check master key first
    if user_key == 'mora-king':
        session["licensed"] = True
        session["license_key"] = user_key
        session["access_level"] = "creator"
        log.info("✅ Master key access granted")
        return jsonify({'valid': True, 'redirect': url_for('dashboard')})
    
    # Check license database
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

# Protected dashboard routes
@app.route("/dashboard")
def dashboard():
    """Main Mora Bets dashboard - protected route"""
    # Check for key parameter
    user_key = request.args.get('key', '').strip()
    
    if user_key:
        # Validate key
        try:
            with open(LICENSE_DB, 'r') as f:
                keys = json.load(f)
        except Exception as e:
            log.error(f"Error loading license keys: {e}")
            return redirect(url_for('index') + '?message=System+error.+Please+try+again.')
        
        # Check if key exists and is valid (case-insensitive)
        is_valid = False
        for key in keys:
            if key.upper() == user_key.upper() and keys[key]:
                is_valid = True
                break
        
        if not is_valid:
            log.info(f"Invalid key attempt: {user_key}")
            return redirect(url_for('index') + '?message=Invalid+key.+Please+try+again.')
        
        # Key is valid, set session and render dashboard
        session["licensed"] = True
        session["license_key"] = user_key
        log.info(f"✅ Dashboard access granted for key: {user_key}")
    
    try:
        return render_template("dashboard.html", hits=0)
    except Exception as e:
        log.error(f"Error in dashboard route: {e}")
        return f'''
        <!DOCTYPE html>
        <html>
        <head><title>Mora Bets</title></head>
        <body>
        <h1>Mora Bets - Sports Betting Analytics</h1>
        <p>System Status: Running</p>
        <p>Error: {str(e)}</p>
        <p><a href="/healthz">Health Check</a></p>
        </body>
        </html>
        '''

@app.route("/dashboard_legacy")
def dashboard_legacy():
    """Legacy dashboard - preserved for backward compatibility"""
    # Check for key parameter
    user_key = request.args.get('key', '').strip()
    
    if user_key:
        # Validate key
        try:
            with open(LICENSE_DB, 'r') as f:
                keys = json.load(f)
        except Exception as e:
            log.error(f"Error loading license keys: {e}")
            return redirect(url_for('index') + '?message=System+error.+Please+try+again.')
        
        # Check if key exists and is valid (case-insensitive)
        is_valid = False
        for key in keys:
            if key.upper() == user_key.upper() and keys[key]:
                is_valid = True
                break
        
        if not is_valid:
            log.info(f"Invalid key attempt: {user_key}")
            return redirect(url_for('index') + '?message=Invalid+key.+Please+try+again.')
        
        # Key is valid, set session and render dashboard
        session["licensed"] = True
        session["license_key"] = user_key
        log.info(f"✅ Legacy dashboard access granted for key: {user_key}")
    
    try:
        return render_template("dashboard_legacy.html", hits=0)
    except Exception as e:
        log.error(f"Error in legacy dashboard route: {e}")
        return f'''
        <!DOCTYPE html>
        <html>
        <head><title>Mora Bets - Legacy</title></head>
        <body>
        <h1>Mora Bets - Legacy Dashboard</h1>
        <p>System Status: Running</p>
        <p>Error: {str(e)}</p>
        <p><a href="/dashboard">New Dashboard</a></p>
        </body>
        </html>
        '''

# License protection middleware
@app.before_request
def require_license():
    """Protect dashboard routes except public pages and API endpoints"""
    # Allow access to public pages, verification, health checks, API endpoints, and static files
    public_endpoints = [
        "home", "how_it_works", "paywall", "paywall_config", "tool", "verify", "verify_key", "validate_key", "create_checkout_session", 
        "healthz", "ping", "static", "logout", "dashboard", "dashboard_legacy"
    ]
    
    # Also allow access to any route starting with /api/
    if request.endpoint in public_endpoints or request.path.startswith("/static") or request.path.startswith("/api/"):
        return
    
    # Check if user has valid license in session for protected routes
    if not session.get("licensed"):
        return redirect(url_for("paywall"))

# Core player props endpoint
@app.route("/player_props")
def get_props():
    """Get player props grouped by matchup for MLB, NFL, NCAAF, and UFC"""
    try:
        league_in = request.args.get("league")
        league = _norm_league(league_in)
        date_str = request.args.get("date")  # YYYY-MM-DD optional
        log.info("props: league=%s (norm=%s) date=%s", league_in, league, date_str)

        if league == "mlb":
            props = fetch_mlb_player_props()
            # Group by matchup - need to create matchup from event data
            grouped = {}
            for prop in props:
                # Create matchup from player team info or use default
                matchup = prop.get("matchup", "Unknown")
                if matchup == "Unknown":
                    # Try to extract from other fields or use a default
                    matchup = "MLB Game"
                if matchup not in grouped:
                    grouped[matchup] = []
                grouped[matchup].append(prop)
            return jsonify(grouped)

        elif league == "nfl":
            props = fetch_nfl_player_props(hours_ahead=96)
            # Group by matchup
            grouped = {}
            for prop in props:
                matchup = prop.get("matchup", "Unknown")
                if matchup not in grouped:
                    grouped[matchup] = []
                grouped[matchup].append(prop)
            return jsonify(grouped)

        elif league == "ncaaf":
            props = fetch_ncaaf_player_props(date=date_str)
            # Group by matchup
            grouped = {}
            for prop in props:
                matchup = prop.get("matchup", "Unknown")
                if matchup not in grouped:
                    grouped[matchup] = []
                grouped[matchup].append(prop)
            return jsonify(grouped)

        elif league == "ufc":
            # Use the existing UFC totals function
            props = fetch_ufc_totals_props(date_iso=date_str, hours_ahead=96)
            # Group by matchup
            grouped = {}
            for prop in props:
                matchup = prop.get("matchup", "Unknown")
                if matchup not in grouped:
                    grouped[matchup] = []
                grouped[matchup].append(prop)
            return jsonify(grouped)

        else:
            raise ValueError(f"Unsupported league: {league_in}")

    except Exception as e:
        log.exception("props endpoint failure")
        return jsonify({"error": str(e)}), 503

# Stub endpoints to avoid UI errors
@app.route("/contextual/hit_rates", methods=["POST"])
def contextual_hit_rates():
    """Stub endpoint - returns empty results"""
    return jsonify({"results": []})

@app.route("/api/trends/l10", methods=["GET"])
def api_trends_l10():
    """Stub endpoint - returns empty results"""
    return jsonify({"results": []})

# Health and utility endpoints
@app.route("/healthz")
def healthz():
    """Health check endpoint"""
    return jsonify({"ok": True})

@app.route("/ping")
def ping():
    """Ping endpoint"""
    return jsonify({"status": "running"})

@app.route("/logout")
def logout():
    """Clear license session for testing"""
    session.clear()
    return redirect(url_for("how_it_works"))

# Performance endpoints
@app.route("/_perf/recent", methods=["GET"])
def perf_recent():
    return jsonify({"recent": perf.recent()})

@app.route("/_perf/cache", methods=["GET"])
def perf_cache():
    return jsonify({"cache": cache_metrics()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)