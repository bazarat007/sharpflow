"""
SharpFlow v1 — Polymarket Sharp Wallet Intelligence
=====================================================
Single-file deployment version. Drop this on Railway and it:
1. Fetches resolved markets from Polymarket
2. Pulls trade data for each market
3. Scores every wallet using the hybrid sharpness algorithm
4. Serves results via a web API + simple dashboard
5. Refreshes data twice daily

Environment variables needed:
  PORT              (Railway sets this automatically)
  TELEGRAM_BOT_TOKEN (optional — add later for alerts)
  TELEGRAM_CHAT_ID   (optional — add later for alerts)
"""

import os
import json
import time
import math
import logging
import threading
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

import requests
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# ================================================================
# LOGGING
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sharpflow")

# ================================================================
# CONFIG
# ================================================================
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Scoring weights (ROI-heavy — validated by 80K-config grid search, spread +0.30)
W_CLV = 0.20
W_TIMING = 0.20
W_CONSISTENCY = 0.20
W_ROI = 0.40

MIN_MARKETS = 8            # Validated: 8 markets with $100 notional filter
MARKETS_TO_FETCH = 750     # How many resolved markets to fetch from API
TRADES_PER_MARKET = 500    # Max trades per market
MIN_VOLUME = 2000          # Minimum market volume (USD)
API_DELAY = 0.3            # Slightly faster
SCORING_WINDOW_MONTHS = 2  # Validated: recent behavior predicts better than historical
MIN_TRADE_NOTIONAL = 100   # Filter dust trades below $100 notional

# Category keywords
SPORTS_KW = ["sports","nfl","nba","mlb","nhl","soccer","football","basketball",
             "baseball","ufc","mma","tennis","boxing","cricket","f1","golf",
             "rugby","hockey","super bowl","playoffs","championship","game","match",
             "world series","stanley cup","premier league","champions league",
             "la liga", "serie a", "bundesliga", "euros", "copa", "olympic",
             "ncaa", "march madness", "draft", "mvp", "heisman", "batting",
             "touchdown", "goal scored", "points scored", "win the",
             "win against", "beat the", "defeat", "series", "season"]
POLITICS_KW = ["politics","election","president","congress","senate","governor",
               "trump","biden","democrat","republican","vote","legislation",
               "cabinet","supreme court","government","policy","fed","tariff",
               "executive order","parliament","minister","geopolitics","primary",
               "inaugural","impeach","veto","gop","dnc","rnc","liberal",
               "conservative","poll","approval rating","midterm","swing state",
               "electoral","white house","pentagon","state department","cia",
               "fbi","doj","attorney general","speaker of","majority leader",
               "secretary of","ambassador","nato","un general","g7","g20",
               "sanctions","ceasefire","war","conflict","ukraine","russia",
               "china","iran","israel","gaza","palestine","north korea",
               "negotiate","treaty","border","immigration","refugee","asylum",
               "ban","mandate","regulation","deregulat","executive action",
               "pardon","indictment","arraign","conviction","verdict"]

# Sport-specific sub-category keywords
SPORT_SUBCATS = {
    "nba": ["nba", "basketball", "lakers", "celtics", "warriors", "bucks", "nuggets",
            "76ers", "knicks", "nets", "heat", "bulls", "suns", "mavs", "mavericks",
            "cavaliers", "cavs", "thunder", "grizzlies", "timberwolves", "clippers",
            "rockets", "spurs", "pelicans", "kings", "pacers", "hawks", "hornets",
            "pistons", "wizards", "blazers", "jazz", "raptors", "magic",
            "rebounds", "assists", "triple double", "all-star", "dunk contest"],
    "nfl": ["nfl", "super bowl", "touchdown", "quarterback", "chiefs", "eagles",
            "49ers", "cowboys", "ravens", "bills", "dolphins", "jets", "patriots",
            "steelers", "bengals", "browns", "texans", "colts", "jaguars", "titans",
            "broncos", "raiders", "chargers", "seahawks", "rams", "cardinals",
            "commanders", "giants", "bears", "packers", "lions", "vikings", "saints",
            "buccaneers", "bucs", "falcons", "panthers", "rushing yards",
            "passing yards", "field goal"],
    "mlb": ["mlb", "baseball", "world series", "home run", "batting", "pitcher",
            "yankees", "dodgers", "astros", "braves", "mets", "phillies", "padres",
            "cubs", "red sox", "mariners", "twins", "guardians", "orioles", "rangers",
            "blue jays", "brewers", "diamondbacks", "reds", "pirates", "royals",
            "tigers", "rockies", "marlins", "nationals", "white sox", "athletics",
            "strikeout", "rbi", "inning"],
    "nhl": ["nhl", "hockey", "stanley cup", "hat trick", "goalie",
            "bruins", "maple leafs", "canadiens", "oilers", "avalanche",
            "penguins", "lightning", "hurricanes", "stars", "wild", "jets",
            "flames", "canucks", "kraken", "predators", "islanders", "devils",
            "red wings", "blues", "sharks", "blackhawks", "capitals", "flyers"],
    "soccer": ["soccer", "premier league", "champions league", "la liga", "serie a",
               "bundesliga", "ligue 1", "mls", "world cup", "euros", "copa america",
               "europa league", "liverpool", "manchester united", "manchester city",
               "arsenal", "chelsea", "tottenham", "barcelona", "real madrid",
               "bayern", "psg", "juventus", "inter milan", "ac milan", "dortmund"],
    "ufc": ["ufc", "mma", "boxing", "fight", "knockout", "submission",
            "featherweight", "lightweight", "welterweight", "middleweight",
            "heavyweight", "bantamweight", "flyweight", "dana white", "octagon"],
}

# Telegram (optional)
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ================================================================
# DATABASE
# ================================================================

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor

def db_conn():
    """Get a database connection."""
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL)

def db_init():
    """Create tables if they don't exist."""
    if not DATABASE_URL:
        logger.warning("No DATABASE_URL set — running without persistence")
        return

    conn = db_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id BIGSERIAL PRIMARY KEY,
        wallet TEXT NOT NULL,
        condition_id TEXT NOT NULL,
        side TEXT NOT NULL,
        outcome TEXT,
        price DOUBLE PRECISION,
        size DOUBLE PRECISION,
        timestamp BIGINT,
        title TEXT,
        category TEXT,
        winning_outcome TEXT,
        ingested_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(wallet, condition_id, timestamp, side, price, size)
    );
    CREATE INDEX IF NOT EXISTS idx_trades_wallet ON trades(wallet);
    CREATE INDEX IF NOT EXISTS idx_trades_condition ON trades(condition_id);
    CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS wallet_scores (
        id BIGSERIAL PRIMARY KEY,
        wallet TEXT NOT NULL,
        display_name TEXT,
        sharpness_score DOUBLE PRECISION,
        tier TEXT,
        clv_score DOUBLE PRECISION,
        timing_score DOUBLE PRECISION,
        consistency_score DOUBLE PRECISION,
        roi_score DOUBLE PRECISION,
        avg_clv DOUBLE PRECISION,
        win_rate DOUBLE PRECISION,
        total_pnl DOUBLE PRECISION,
        total_trades INT,
        distinct_markets INT,
        early_entry_pct DOUBLE PRECISION,
        scored_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_ws_wallet ON wallet_scores(wallet);
    CREATE INDEX IF NOT EXISTS idx_ws_scored_at ON wallet_scores(scored_at);
    CREATE INDEX IF NOT EXISTS idx_ws_tier ON wallet_scores(tier);
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS market_intel (
        id BIGSERIAL PRIMARY KEY,
        condition_id TEXT NOT NULL,
        question TEXT,
        category TEXT,
        market_price DOUBLE PRECISION,
        smart_direction TEXT,
        smart_money_yes_pct DOUBLE PRECISION,
        price_gap DOUBLE PRECISION,
        divergence DOUBLE PRECISION,
        interest_score DOUBLE PRECISION,
        sharp_yes_count INT,
        sharp_no_count INT,
        sharp_yes_notional DOUBLE PRECISION,
        sharp_no_notional DOUBLE PRECISION,
        dull_yes_count INT,
        dull_no_count INT,
        recorded_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_mi_condition ON market_intel(condition_id);
    CREATE INDEX IF NOT EXISTS idx_mi_recorded ON market_intel(recorded_at);
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id BIGSERIAL PRIMARY KEY,
        started_at TIMESTAMPTZ DEFAULT NOW(),
        finished_at TIMESTAMPTZ,
        markets_analyzed INT,
        trades_ingested INT,
        wallets_scored INT,
        convergence_signals INT,
        markets_with_intel INT,
        status TEXT,
        notes TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS markets (
        id BIGSERIAL PRIMARY KEY,
        condition_id TEXT NOT NULL UNIQUE,
        question TEXT,
        category TEXT,
        slug TEXT,
        volume DOUBLE PRECISION,
        outcome_yes TEXT,
        outcome_no TEXT,
        resolved BOOLEAN DEFAULT FALSE,
        winning_outcome TEXT,
        resolved_at TIMESTAMPTZ,
        end_date TIMESTAMPTZ,
        first_seen_at TIMESTAMPTZ DEFAULT NOW(),
        last_updated_at TIMESTAMPTZ DEFAULT NOW(),
        final_price_yes DOUBLE PRECISION,
        final_price_no DOUBLE PRECISION,
        total_traders INT,
        open_interest DOUBLE PRECISION
    );
    CREATE INDEX IF NOT EXISTS idx_markets_cid ON markets(condition_id);
    CREATE INDEX IF NOT EXISTS idx_markets_resolved ON markets(resolved);
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS convergence_snapshots (
        id BIGSERIAL PRIMARY KEY,
        condition_id TEXT NOT NULL,
        market_title TEXT,
        side TEXT,
        signal_type TEXT,
        strength DOUBLE PRECISION,
        wallet_count INT,
        elite_count INT,
        sharp_count INT,
        total_notional DOUBLE PRECISION,
        market_price_at_signal DOUBLE PRECISION,
        wallets_json TEXT,
        detected_at TIMESTAMPTZ DEFAULT NOW(),
        -- outcome tracking (filled in later when market resolves)
        market_resolved BOOLEAN DEFAULT FALSE,
        winning_side TEXT,
        resolved_at TIMESTAMPTZ,
        signal_correct BOOLEAN,
        price_at_resolution DOUBLE PRECISION
    );
    CREATE INDEX IF NOT EXISTS idx_cs_condition ON convergence_snapshots(condition_id);
    CREATE INDEX IF NOT EXISTS idx_cs_detected ON convergence_snapshots(detected_at);
    CREATE INDEX IF NOT EXISTS idx_cs_resolved ON convergence_snapshots(market_resolved);
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS wallet_positions (
        id BIGSERIAL PRIMARY KEY,
        wallet TEXT NOT NULL,
        condition_id TEXT NOT NULL,
        outcome TEXT,
        size DOUBLE PRECISION,
        avg_price DOUBLE PRECISION,
        notional DOUBLE PRECISION,
        cur_price DOUBLE PRECISION,
        unrealized_pnl DOUBLE PRECISION,
        title TEXT,
        snapshot_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_wp_wallet ON wallet_positions(wallet);
    CREATE INDEX IF NOT EXISTS idx_wp_condition ON wallet_positions(condition_id);
    CREATE INDEX IF NOT EXISTS idx_wp_snapshot ON wallet_positions(snapshot_at);
    """)

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database initialized successfully")


def db_save_trades(trades_list):
    """Bulk insert trades, skipping duplicates."""
    if not DATABASE_URL or not trades_list:
        return 0
    conn = db_conn()
    cur = conn.cursor()

    rows = []
    for t in trades_list:
        rows.append((
            t.get("wallet", ""),
            t.get("condition_id", ""),
            t.get("side", "BUY"),
            t.get("outcome", ""),
            t.get("price", 0),
            t.get("size", 0),
            t.get("timestamp", 0),
            t.get("title", ""),
            t.get("category", "other"),
            t.get("winning_outcome", ""),
        ))

    if not rows:
        conn.close()
        return 0

    inserted = 0
    try:
        execute_values(cur, """
            INSERT INTO trades (wallet, condition_id, side, outcome, price, size, timestamp, title, category, winning_outcome)
            VALUES %s
            ON CONFLICT (wallet, condition_id, timestamp, side, price, size) DO NOTHING
        """, rows, page_size=1000)
        inserted = cur.rowcount
        conn.commit()
    except Exception as e:
        logger.error(f"DB trade insert error: {e}")
        conn.rollback()

    cur.close()
    conn.close()
    return inserted


def db_save_wallet_scores(scored_wallets):
    """Save current wallet scores snapshot."""
    if not DATABASE_URL or not scored_wallets:
        return
    conn = db_conn()
    cur = conn.cursor()

    rows = []
    for w in scored_wallets:
        rows.append((
            w["wallet"], w["display_name"], w["sharpness_score"], w["tier"],
            w.get("clv_score", 0), w.get("timing_score", 0),
            w.get("consistency_score", 0), w.get("roi_score", 0),
            w.get("avg_clv", 0), w.get("win_rate", 0),
            w.get("total_pnl", 0), w.get("total_trades", 0),
            w.get("distinct_markets", 0), w.get("early_entry_pct", 0),
        ))

    try:
        execute_values(cur, """
            INSERT INTO wallet_scores (wallet, display_name, sharpness_score, tier,
                clv_score, timing_score, consistency_score, roi_score,
                avg_clv, win_rate, total_pnl, total_trades, distinct_markets, early_entry_pct)
            VALUES %s
        """, rows, page_size=500)
        conn.commit()
    except Exception as e:
        logger.error(f"DB wallet score insert error: {e}")
        conn.rollback()

    cur.close()
    conn.close()


def db_save_market_intel(intel_list):
    """Save market intelligence snapshot."""
    if not DATABASE_URL or not intel_list:
        return
    conn = db_conn()
    cur = conn.cursor()

    rows = []
    for m in intel_list:
        rows.append((
            m["condition_id"], m.get("question", ""), m.get("category", ""),
            m.get("market_price", 0), m.get("smart_direction", ""),
            m.get("smart_money_yes_pct", 0), m.get("price_gap", 0),
            m.get("divergence", 0), m.get("interest_score", 0),
            m.get("sharp_yes_count", 0), m.get("sharp_no_count", 0),
            m.get("sharp_yes_notional", 0), m.get("sharp_no_notional", 0),
            m.get("dull_yes_count", 0), m.get("dull_no_count", 0),
        ))

    try:
        execute_values(cur, """
            INSERT INTO market_intel (condition_id, question, category, market_price,
                smart_direction, smart_money_yes_pct, price_gap, divergence, interest_score,
                sharp_yes_count, sharp_no_count, sharp_yes_notional, sharp_no_notional,
                dull_yes_count, dull_no_count)
            VALUES %s
        """, rows, page_size=500)
        conn.commit()
    except Exception as e:
        logger.error(f"DB market intel insert error: {e}")
        conn.rollback()

    cur.close()
    conn.close()


def db_save_pipeline_run(stats_dict):
    """Record a pipeline run."""
    if not DATABASE_URL:
        return
    conn = db_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO pipeline_runs (finished_at, markets_analyzed, trades_ingested,
                wallets_scored, convergence_signals, markets_with_intel, status, notes)
            VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
        """, (
            stats_dict.get("markets", 0), stats_dict.get("trades", 0),
            stats_dict.get("wallets", 0), stats_dict.get("convergence", 0),
            stats_dict.get("intel", 0), stats_dict.get("status", "ok"),
            stats_dict.get("notes", ""),
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"DB pipeline run insert error: {e}")
        conn.rollback()
    cur.close()
    conn.close()


def db_get_trade_count():
    """Get total trades in the database."""
    if not DATABASE_URL:
        return 0
    try:
        conn = db_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trades")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except:
        return 0


def db_get_all_trades(months_back=6):
    """Load trades from the database for scoring. Only loads last N months for efficiency."""
    if not DATABASE_URL:
        return []
    try:
        conn = db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Calculate cutoff timestamp (N months ago)
        cutoff = int((datetime.utcnow() - timedelta(days=months_back * 30)).timestamp())
        cur.execute("""
            SELECT wallet, condition_id, side, outcome, price, size,
                   timestamp, title, category, winning_outcome
            FROM trades
            WHERE timestamp > %s
            ORDER BY timestamp DESC
        """, (cutoff,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        trades = [dict(r) for r in rows]

        # Derive sub-category from title for sports trades
        for t in trades:
            if t.get("category") == "sports" and t.get("title"):
                title_lower = t["title"].lower()
                for sport, keywords in SPORT_SUBCATS.items():
                    if any(kw in title_lower for kw in keywords):
                        t["sub_category"] = sport
                        break

        logger.info(f"  DB query returned {len(trades):,} trades (last {months_back} months)")
        return trades
    except Exception as e:
        logger.error(f"DB load trades error: {e}")
        return []


def db_save_markets(markets_list):
    """Save/update market metadata. Upsert on condition_id."""
    if not DATABASE_URL or not markets_list:
        return
    conn = db_conn()
    cur = conn.cursor()

    rows = []
    for m in markets_list:
        cid = m.get("condition_id") or m.get("conditionId", "")
        if not cid:
            continue
        question = m.get("question", "") or m.get("title", "")
        resolved = m.get("resolved", False) or m.get("closed", False)
        winning = m.get("winning_outcome", "") or ""
        volume = float(m.get("volume", 0) or m.get("volumeNum", 0) or 0)
        slug = m.get("slug", "") or m.get("market_slug", "")
        end_date = m.get("endDate") or m.get("end_date_iso") or None

        rows.append((
            cid, question, categorize(question, "")[0] if question else "other",
            slug, volume, resolved, winning, end_date,
        ))

    if not rows:
        conn.close()
        return

    try:
        for row in rows:
            cur.execute("""
                INSERT INTO markets (condition_id, question, category, slug, volume, resolved, winning_outcome, end_date, last_updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (condition_id) DO UPDATE SET
                    volume = GREATEST(markets.volume, EXCLUDED.volume),
                    resolved = EXCLUDED.resolved,
                    winning_outcome = CASE WHEN EXCLUDED.winning_outcome != '' THEN EXCLUDED.winning_outcome ELSE markets.winning_outcome END,
                    last_updated_at = NOW()
            """, row)
        conn.commit()
    except Exception as e:
        logger.error(f"DB markets save error: {e}")
        conn.rollback()

    cur.close()
    conn.close()


def db_save_convergence_signals(signals_list):
    """Save convergence signal snapshots for backtesting."""
    if not DATABASE_URL or not signals_list:
        return
    conn = db_conn()
    cur = conn.cursor()

    for s in signals_list:
        # Check if we already logged this exact signal (same market + side + similar time)
        cid = s.get("condition_id", "")
        side = s.get("side", "")

        try:
            cur.execute("""
                SELECT id FROM convergence_snapshots
                WHERE condition_id = %s AND side = %s
                AND detected_at > NOW() - INTERVAL '2 hours'
            """, (cid, side))

            if cur.fetchone():
                continue  # Already recorded this signal recently

            wallets_json = json.dumps(s.get("wallets", []))

            cur.execute("""
                INSERT INTO convergence_snapshots
                    (condition_id, market_title, side, signal_type, strength,
                     wallet_count, elite_count, sharp_count, total_notional,
                     market_price_at_signal, wallets_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                cid, s.get("market_title", ""), side,
                s.get("signal_type", ""), s.get("strength", 0),
                s.get("wallet_count", 0), s.get("elite_count", 0),
                s.get("sharp_count", 0), s.get("total_notional", 0),
                s.get("market_price", 0), wallets_json,
            ))
        except Exception as e:
            logger.error(f"DB convergence save error: {e}")
            conn.rollback()
            continue

    try:
        conn.commit()
    except:
        pass
    cur.close()
    conn.close()


def db_save_wallet_positions(scored_wallets):
    """Snapshot open positions for sharp+ wallets."""
    if not DATABASE_URL:
        return
    sharp_wallets = [w for w in scored_wallets if w["tier"] in ("elite", "sharp")]
    if not sharp_wallets:
        return

    conn = db_conn()
    cur = conn.cursor()
    total_saved = 0

    for wdata in sharp_wallets:
        wallet_addr = wdata["wallet"]
        try:
            positions = api_get(f"{DATA_API}/positions", {"user": wallet_addr, "sizeThreshold": 1})
            if not positions or not isinstance(positions, list):
                continue

            for pos in positions:
                size = float(pos.get("size", 0) or 0)
                avg_price = float(pos.get("avgPrice", 0) or 0)
                if size < 1:
                    continue
                notional = size * avg_price
                cur_price = float(pos.get("curPrice", 0) or 0)
                pnl = (cur_price - avg_price) * size if cur_price > 0 else 0
                cid = pos.get("conditionId") or pos.get("market") or ""
                outcome = pos.get("outcome", "")
                title = pos.get("title", "") or pos.get("eventTitle", "")

                cur.execute("""
                    INSERT INTO wallet_positions
                        (wallet, condition_id, outcome, size, avg_price, notional, cur_price, unrealized_pnl, title)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (wallet_addr, cid, outcome, size, avg_price, notional, cur_price, pnl, title))
                total_saved += 1

        except Exception as e:
            logger.debug(f"Position snapshot error for {wallet_addr[:10]}: {e}")

    try:
        conn.commit()
    except Exception as e:
        logger.error(f"DB position snapshot commit error: {e}")
        conn.rollback()

    cur.close()
    conn.close()
    logger.info(f"  Position snapshots saved: {total_saved} across {len(sharp_wallets)} sharp wallets")


def db_update_convergence_outcomes():
    """Check if any unresolved convergence signals have resolved and grade them."""
    if not DATABASE_URL:
        return
    conn = db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # Find unresolved convergence signals
        cur.execute("""
            SELECT cs.id, cs.condition_id, cs.side
            FROM convergence_snapshots cs
            WHERE cs.market_resolved = FALSE
        """)
        unresolved = cur.fetchall()

        if not unresolved:
            cur.close()
            conn.close()
            return

        # Check each against the markets table
        for row in unresolved:
            cur.execute("""
                SELECT resolved, winning_outcome FROM markets
                WHERE condition_id = %s AND resolved = TRUE
            """, (row["condition_id"],))
            mkt = cur.fetchone()

            if mkt and mkt["resolved"]:
                winning = mkt["winning_outcome"] or ""
                signal_correct = None
                if winning and row["side"]:
                    signal_correct = (row["side"].upper() == winning.upper())

                cur.execute("""
                    UPDATE convergence_snapshots
                    SET market_resolved = TRUE, winning_side = %s,
                        resolved_at = NOW(), signal_correct = %s
                    WHERE id = %s
                """, (winning, signal_correct, row["id"]))

        conn.commit()
        resolved_count = sum(1 for r in unresolved if True)  # just for logging
        logger.info(f"  Checked {len(unresolved)} unresolved convergence signals")

    except Exception as e:
        logger.error(f"DB convergence outcome update error: {e}")
        conn.rollback()

    cur.close()
    conn.close()


# ================================================================
# STATE
# ================================================================
state = {
    "scored_wallets": [],
    "convergence_signals": [],
    "market_intelligence": [],
    "whale_positions": [],
    "markets_analyzed": 0,
    "trades_analyzed": 0,
    "wallets_scanned": 0,
    "last_refresh": None,
    "status": "initializing",
    "progress": "",
    "error": None,
}

# ================================================================
# DATA PIPELINE
# ================================================================

def api_get(url, params=None, retries=3):
    """Rate-limited GET request with retries."""
    for attempt in range(retries):
        time.sleep(API_DELAY)
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
            elif r.status_code >= 500:
                time.sleep(3)
            else:
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
    return None


def categorize(question, tags_str):
    """Categorize a market as sports/politics/other with optional sport sub-category."""
    q = question.lower()
    t = tags_str.lower()
    combined = q + " " + t

    if any(kw in combined for kw in SPORTS_KW):
        # Try to identify specific sport
        sub = None
        for sport, keywords in SPORT_SUBCATS.items():
            if any(kw in combined for kw in keywords):
                sub = sport
                break
        return ("sports", sub)

    if any(kw in combined for kw in POLITICS_KW):
        return ("politics", None)

    return ("other", None)


def fetch_resolved_markets(limit=MARKETS_TO_FETCH):
    """Fetch resolved markets from Gamma API, sorted by volume."""
    logger.info(f"Fetching up to {limit} resolved markets...")
    state["progress"] = "Fetching resolved markets..."
    all_markets = []
    offset = 0

    while len(all_markets) < limit:
        data = api_get(f"{GAMMA_API}/markets", {
            "closed": "true", "limit": 100, "offset": offset,
            "order": "volume", "ascending": "false",
        })
        if not data or not isinstance(data, list) or len(data) == 0:
            break

        for m in data:
            cid = m.get("conditionId") or m.get("condition_id")
            vol = float(m.get("volume", 0) or 0)
            if not cid or vol < MIN_VOLUME:
                continue

            tags = [t.get("label", "") if isinstance(t, dict) else str(t)
                    for t in (m.get("tags") or [])]

            # Determine winner from multiple sources
            winning = None
            tokens = m.get("tokens", [])
            if isinstance(tokens, list):
                for tok in tokens:
                    if isinstance(tok, dict) and tok.get("winner"):
                        winning = tok.get("outcome", "").lower()

            # Fallback: check outcomePrices (resolved markets show "1" and "0")
            if not winning:
                try:
                    op = m.get("outcomePrices", "")
                    outcomes = m.get("outcomes", "")
                    if op and outcomes:
                        # outcomePrices can be a JSON string like "[\"1\",\"0\"]" or a list
                        if isinstance(op, str):
                            op = json.loads(op)
                        if isinstance(outcomes, str):
                            outcomes = json.loads(outcomes)
                        if isinstance(op, list) and isinstance(outcomes, list) and len(op) == len(outcomes):
                            for price, outcome_name in zip(op, outcomes):
                                if float(price) >= 0.95:  # This outcome won
                                    winning = outcome_name.lower()
                                    break
                except Exception:
                    pass

            # Fallback 2: check if market question resolved YES/NO based on resolved field
            if not winning and m.get("resolved"):
                # Try to get resolution from the market data
                res = m.get("resolution", "")
                if res:
                    winning = str(res).lower()

            cat, sub_cat = categorize(m.get("question", ""), " ".join(tags))

            all_markets.append({
                "condition_id": cid,
                "question": m.get("question", ""),
                "slug": m.get("slug", ""),
                "volume": vol,
                "category": cat,
                "sub_category": sub_cat,
                "winning_outcome": winning,
                "tokens": tokens,
                "resolved": True,
                "closed": True,
            })

        logger.info(f"  Fetched batch at offset {offset}, total: {len(all_markets)}")
        state["progress"] = f"Fetched {len(all_markets)} markets..."
        offset += 100
        if len(data) < 100:
            break

    # Prioritize sports + politics
    sp = [m for m in all_markets if m["category"] in ("sports", "politics")]
    other = [m for m in all_markets if m["category"] == "other"]
    result = (sp + other)[:limit]

    # Diagnostic: how many have resolved outcomes?
    with_winner = sum(1 for m in result if m["winning_outcome"])
    without_winner = sum(1 for m in result if not m["winning_outcome"])
    logger.info(f"Collected {len(result)} markets ({len(sp)} sports/politics)")
    logger.info(f"  With resolved outcome: {with_winner} | Without: {without_winner}")

    return result


def fetch_trades_for_markets(markets):
    """Fetch trades for each market from Data API."""
    logger.info(f"Fetching trades for {len(markets)} markets...")
    all_trades = []

    for i, mkt in enumerate(markets):
        cid = mkt["condition_id"]
        data = api_get(f"{DATA_API}/trades", {
            "market": cid, "limit": TRADES_PER_MARKET,
        })

        if data and isinstance(data, list):
            # Log a sample trade from the first market to see field names
            if i == 0 and len(data) > 0:
                sample = data[0]
                logger.info(f"  Sample trade fields: {list(sample.keys())}")
                logger.info(f"  Sample trade: wallet={sample.get('proxyWallet', sample.get('maker', sample.get('taker', 'NONE')))}, "
                           f"outcome={sample.get('outcome','?')}, side={sample.get('side','?')}, "
                           f"price={sample.get('price','?')}, size={sample.get('size','?')}")

            for t in data:
                wallet = t.get("proxyWallet") or t.get("maker") or t.get("taker") or ""
                price = float(t.get("price", 0) or 0)
                size = float(t.get("size", 0) or 0)
                if not wallet or price <= 0 or size <= 0:
                    continue

                all_trades.append({
                    "wallet": wallet,
                    "side": t.get("side", "BUY"),
                    "price": price,
                    "size": size,
                    "timestamp": int(t.get("timestamp", 0) or 0),
                    "condition_id": cid,
                    "outcome": t.get("outcome", ""),
                    "title": t.get("title", mkt["question"]),
                    "name": t.get("name", ""),
                    "pseudonym": t.get("pseudonym", ""),
                    "category": mkt["category"],
                    "sub_category": mkt.get("sub_category"),
                    "winning_outcome": mkt["winning_outcome"],
                })

        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i+1}/{len(markets)} markets, {len(all_trades)} trades")
            state["progress"] = f"Fetching trades: {i+1}/{len(markets)} markets, {len(all_trades)} trades..."

    logger.info(f"Collected {len(all_trades)} total trades")
    return all_trades


# ================================================================
# SCORING ENGINE
# ================================================================

def score_all_wallets(trades, markets_lookup):
    """Compute hybrid sharpness scores for all wallets."""
    logger.info("Scoring wallets...")
    state["progress"] = "Scoring wallets..."

    # Group trades by wallet
    wallet_trades = defaultdict(list)
    for t in trades:
        wallet_trades[t["wallet"]].append(t)

    state["wallets_scanned"] = len(wallet_trades)
    results = []

    # Diagnostics
    skip_reasons = {"below_threshold": 0, "no_win_outcome": 0, "no_clvs": 0, "scored": 0}
    max_distinct = 0

    for wallet, wtrades in wallet_trades.items():
        # Group by market, filtering dust trades
        market_groups = defaultdict(list)
        for t in wtrades:
            notional = t["price"] * t["size"]
            if notional < MIN_TRADE_NOTIONAL:
                continue
            market_groups[t["condition_id"]].append(t)

        distinct = len(market_groups)
        max_distinct = max(max_distinct, distinct)

        if distinct < MIN_MARKETS:
            skip_reasons["below_threshold"] += 1
            continue

        clvs = []
        timing_scores = []
        market_pnls = {}
        total_inv = 0.0
        total_ret = 0.0
        categories = set()
        trade_count = 0

        for cid, mtrades in market_groups.items():
            mpnl = 0.0
            for t in mtrades:
                price = t["price"]
                side = t["side"]
                size = t["size"]
                outcome = t.get("outcome", "").lower()
                win_outcome = (t.get("winning_outcome") or "").lower()
                categories.add(t.get("category", "other"))

                if not win_outcome:
                    continue

                outcome_won = (outcome == win_outcome)
                res_price = 1.0 if outcome_won else 0.0
                trade_count += 1

                if side == "BUY":
                    # Skip "fake edge" trades — buying at 95c+ on near-certain outcomes
                    # These inflate win rate without indicating real predictive ability
                    if price >= 0.95 and outcome_won:
                        continue

                    clvs.append(res_price - price)
                    timing_scores.append((0.5 - price) if outcome_won else (price - 0.5))
                    pnl = (res_price - price) * size
                    total_inv += price * size
                    total_ret += res_price * size
                    mpnl += pnl
                elif side == "SELL":
                    clvs.append(price - res_price)
                    timing_scores.append((price - 0.5) if not outcome_won else (0.5 - price))
                    pnl = (price - res_price) * size
                    total_inv += (1.0 - price) * size
                    total_ret += (1.0 - res_price) * size
                    mpnl += pnl

            market_pnls[cid] = mpnl

        if not clvs or trade_count == 0:
            skip_reasons["no_clvs"] += 1
            continue

        skip_reasons["scored"] += 1

        # Component scores
        avg_clv = float(np.mean(clvs))
        # CLV score: -0.1 maps to 0, +0.42 maps to 1
        clv_score = max(0, min(1, (avg_clv + 0.1) / 0.52))

        avg_timing = float(np.mean(timing_scores)) if timing_scores else 0
        early_pct = sum(1 for t in timing_scores if t > 0) / len(timing_scores) if timing_scores else 0
        # FIXED: Lower threshold from 0.3 to 0.1, wider range
        # Now: 10% early entries = 0, 70% early entries = 1
        timing_score = max(0, min(1, (early_pct - 0.1) / 0.6))

        mkt_won = sum(1 for p in market_pnls.values() if p > 0)
        mkt_lost = sum(1 for p in market_pnls.values() if p <= 0)
        win_rate = mkt_won / len(market_pnls) if market_pnls else 0

        # FIXED: Use sqrt scaling instead of linear division by 100
        # 15 markets → 0.39, 30 → 0.55, 50 → 0.71, 100 → 1.0
        # This stops punishing wallets just for having fewer (but sufficient) markets
        market_count_factor = min(1.0, (distinct / 100) ** 0.5)
        consistency_score = win_rate * market_count_factor

        total_pnl = total_ret - total_inv
        roi = total_pnl / total_inv if total_inv > 0 else 0
        roi_score = max(0, min(1, (roi + 0.2) / 1.0))

        sharpness = W_CLV * clv_score + W_TIMING * timing_score + W_CONSISTENCY * consistency_score + W_ROI * roi_score

        # Tier assignment — grid search validated (80K configs, Feb 2026):
        # - Sharp: score >= 0.38, profitable, positive CLV, 8+ markets
        # - Average: moderate score
        # - Dull: everything else
        if sharpness >= 0.38 and total_pnl > 0 and avg_clv > 0:
            tier = "sharp"
        elif sharpness >= 0.25:
            tier = "average"
        else:
            tier = "dull"

        sample = wtrades[0]
        name = sample.get("pseudonym") or sample.get("name") or wallet[:12] + "..."

        # Category-specific scoring
        # Group trades by category AND sub-category, score each independently
        # Re-derive sub-category from title since DB trades may lack it
        cat_trades = defaultdict(list)
        subcat_trades = defaultdict(list)
        for t in wtrades:
            cat = t.get("category", "other")
            cat_trades[cat].append(t)

            # Derive sub-category from title if not already set
            sc = t.get("sub_category")
            if not sc and cat == "sports":
                title = (t.get("title") or "").lower()
                for sport, keywords in SPORT_SUBCATS.items():
                    if any(kw in title for kw in keywords):
                        sc = sport
                        break
            if sc:
                subcat_trades[sc].append(t)

        category_scores = {}
        CAT_MIN_MARKETS = 8  # Minimum markets per category/sub-category

        # Score both categories and sub-categories in one pass
        all_cat_groups = list(cat_trades.items()) + list(subcat_trades.items())

        for cat, ctrades in all_cat_groups:
            cat_market_groups = defaultdict(list)
            for t in ctrades:
                cat_market_groups[t["condition_id"]].append(t)

            cat_distinct = len(cat_market_groups)
            if cat_distinct < CAT_MIN_MARKETS:
                continue

            cat_clvs = []
            cat_timing = []
            cat_market_pnls = {}
            cat_inv = 0.0
            cat_ret = 0.0

            for cid, mtrades in cat_market_groups.items():
                mpnl = 0.0
                for t in mtrades:
                    p = t["price"]
                    s = t["side"]
                    sz = t["size"]
                    o = (t.get("outcome") or "").lower()
                    wo = (t.get("winning_outcome") or "").lower()
                    if not wo:
                        continue
                    ow = (o == wo)
                    rp = 1.0 if ow else 0.0

                    if s == "BUY":
                        if p >= 0.95 and ow:
                            continue
                        cat_clvs.append(rp - p)
                        cat_timing.append((0.5 - p) if ow else (p - 0.5))
                        pnl = (rp - p) * sz
                        cat_inv += p * sz
                        cat_ret += rp * sz
                        mpnl += pnl
                    elif s == "SELL":
                        cat_clvs.append(p - rp)
                        cat_timing.append((p - 0.5) if not ow else (0.5 - p))
                        pnl = (p - rp) * sz
                        cat_inv += (1.0 - p) * sz
                        cat_ret += (1.0 - rp) * sz
                        mpnl += pnl
                cat_market_pnls[cid] = mpnl

            if not cat_clvs:
                continue

            c_avg_clv = float(np.mean(cat_clvs))
            c_clv_score = max(0, min(1, (c_avg_clv + 0.1) / 0.52))
            c_early_pct = sum(1 for t in cat_timing if t > 0) / len(cat_timing) if cat_timing else 0
            c_timing_score = max(0, min(1, (c_early_pct - 0.1) / 0.6))
            c_mkt_won = sum(1 for p in cat_market_pnls.values() if p > 0)
            c_win_rate = c_mkt_won / len(cat_market_pnls) if cat_market_pnls else 0
            c_mkt_factor = min(1.0, (cat_distinct / 50) ** 0.5)
            c_consist = c_win_rate * c_mkt_factor
            c_pnl = cat_ret - cat_inv
            c_roi = c_pnl / cat_inv if cat_inv > 0 else 0
            c_roi_score = max(0, min(1, (c_roi + 0.2) / 1.0))

            c_sharpness = W_CLV * c_clv_score + W_TIMING * c_timing_score + W_CONSISTENCY * c_consist + W_ROI * c_roi_score

            if c_sharpness >= 0.38 and c_pnl > 0 and c_avg_clv > 0:
                c_tier = "sharp"
            elif c_sharpness >= 0.25:
                c_tier = "average"
            else:
                c_tier = "dull"

            category_scores[cat] = {
                "tier": c_tier,
                "sharpness": round(c_sharpness, 4),
                "markets": cat_distinct,
                "win_rate": round(c_win_rate, 4),
                "pnl": round(c_pnl, 2),
                "avg_clv": round(c_avg_clv, 4),
            }

        results.append({
            "wallet": wallet,
            "display_name": name,
            "sharpness_score": round(sharpness, 4),
            "tier": tier,
            "clv_score": round(clv_score, 4),
            "timing_score": round(timing_score, 4),
            "consistency_score": round(consistency_score, 4),
            "roi_score": round(roi_score, 4),
            "avg_clv": round(avg_clv, 4),
            "avg_timing_alpha": round(avg_timing, 4),
            "early_entry_pct": round(early_pct, 4),
            "win_rate": round(win_rate, 4),
            "roi": round(roi, 4),
            "total_pnl": round(total_pnl, 2),
            "total_invested": round(total_inv, 2),
            "total_trades": trade_count,
            "distinct_markets": distinct,
            "markets_won": mkt_won,
            "markets_lost": mkt_lost,
            "categories": sorted(categories),
            "category_scores": category_scores,
            "scored_at": datetime.utcnow().isoformat(),
        })

    results.sort(key=lambda x: x["sharpness_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    tiers = defaultdict(int)
    for r in results:
        tiers[r["tier"]] += 1

    # Count category-specific sharps
    cat_sharp_counts = defaultdict(int)
    for r in results:
        for cat, cs in r.get("category_scores", {}).items():
            if cs["tier"] == "sharp":
                cat_sharp_counts[cat] += 1

    logger.info(f"Scored {len(results)} wallets: {dict(tiers)}")
    logger.info(f"  Category-specific sharps: {dict(cat_sharp_counts)}")
    logger.info(f"  Diagnostics: {dict(skip_reasons)}")
    logger.info(f"  Max distinct markets any wallet had: {max_distinct}")
    logger.info(f"  Threshold: {MIN_MARKETS} markets required")
    return results


# ================================================================
# CONVERGENCE DETECTION
# ================================================================

def detect_convergence(scored_wallets, trades):
    """
    Detect markets where multiple sharp wallets align.
    
    ONLY looks at positions on OPEN/ACTIVE markets.
    Filters out:
      - Resolved/closed markets (the big one — no point signaling on finished events)
      - Dead markets (price > 90c or < 10c — outcome is already obvious)
      - Dust positions (< $50 notional)
      - Markets where the same small cluster appears everywhere
    """
    logger.info("Detecting convergence signals...")
    state["progress"] = "Detecting convergence..."

    sharp_wallets = {w["wallet"]: w for w in scored_wallets if w["sharpness_score"] >= 0.38}
    if not sharp_wallets:
        logger.info("No sharp wallets found for convergence detection")
        return []

    # ---- Step 0: Build set of OPEN market condition IDs ----
    state["progress"] = "Fetching open markets for convergence..."
    logger.info("Fetching open/active markets...")
    open_market_ids = set()
    open_market_meta = {}  # cid -> {question, slug, end_date, volume}
    offset = 0

    while True:
        data = api_get(f"{GAMMA_API}/markets", {
            "closed": "false",
            "limit": 100,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        })
        if not data or not isinstance(data, list) or len(data) == 0:
            break

        for m in data:
            cid = m.get("conditionId") or m.get("condition_id") or ""
            if cid:
                open_market_ids.add(cid)
                open_market_meta[cid] = {
                    "question": m.get("question", ""),
                    "slug": m.get("slug", ""),
                    "end_date": m.get("endDate", ""),
                    "volume": float(m.get("volume", 0) or 0),
                }

                # Also grab current prices from tokens if available
                tokens = m.get("tokens", [])
                if isinstance(tokens, list):
                    for tok in tokens:
                        if isinstance(tok, dict):
                            tp = float(tok.get("price", 0) or 0)
                            if tp > 0:
                                outcome = (tok.get("outcome", "") or "").lower()
                                if outcome in ("yes", "y"):
                                    open_market_meta[cid]["yes_price"] = tp

        offset += 100
        if len(data) < 100 or offset >= 2000:  # Cap at 2000 open markets
            break

    logger.info(f"Found {len(open_market_ids)} open markets")

    if not open_market_ids:
        logger.warning("No open markets found — cannot detect convergence")
        return []

    logger.info(f"Checking current positions for {len(sharp_wallets)} sharp wallets...")

    # ---- Fetch CURRENT positions for sharp wallets ----
    market_positions = defaultdict(lambda: {"yes": [], "no": []})
    market_titles = {}
    market_prices = {}  # cid -> current mid price (from position avg prices)
    wallets_checked = 0

    for wallet_addr, wdata in sharp_wallets.items():
        try:
            positions = api_get(f"{DATA_API}/positions", {"user": wallet_addr, "sizeThreshold": 10})
            if positions and isinstance(positions, list):
                for pos in positions:
                    cid = pos.get("conditionId") or pos.get("market") or ""
                    if not cid:
                        continue

                    # CRITICAL FILTER: Only count positions on OPEN markets
                    if cid not in open_market_ids:
                        continue

                    size = float(pos.get("size", 0) or 0)
                    avg_price = float(pos.get("avgPrice", 0) or pos.get("price", 0) or 0)
                    cur_price = float(pos.get("curPrice", 0) or pos.get("currentPrice", 0) or 0)

                    # Skip dust positions (less than $50 notional value)
                    notional = size * avg_price
                    if notional < 50:
                        continue

                    outcome = (pos.get("outcome", "") or "").lower()
                    title = pos.get("title", "") or pos.get("eventTitle", "") or ""

                    # Use open market metadata for better title and price
                    meta = open_market_meta.get(cid, {})
                    if meta.get("question"):
                        title = meta["question"]
                    if title:
                        market_titles[cid] = title

                    # Use market's current YES price if available
                    if meta.get("yes_price"):
                        market_prices[cid] = meta["yes_price"]
                    elif cur_price > 0:
                        market_prices[cid] = cur_price
                    elif avg_price > 0:
                        market_prices.setdefault(cid, avg_price)

                    entry = {
                        "wallet": wallet_addr,
                        "display_name": wdata["display_name"],
                        "sharpness_score": wdata["sharpness_score"],
                        "tier": wdata["tier"],
                        "size": size,
                        "price": avg_price,
                        "notional": round(notional, 2),
                    }

                    if outcome in ("yes", "y", "1", ""):
                        market_positions[cid]["yes"].append(entry)
                    else:
                        market_positions[cid]["no"].append(entry)

            wallets_checked += 1
            if wallets_checked % 5 == 0:
                state["progress"] = f"Scanning positions: {wallets_checked}/{len(sharp_wallets)} wallets..."

        except Exception as e:
            logger.debug(f"Failed to fetch positions for {wallet_addr[:10]}: {e}")

    logger.info(f"Scanned {wallets_checked} wallets, found positions in {len(market_positions)} markets")

    # ---- Verify markets are actually OPEN (not resolved) ----
    # Fetch market status from Gamma API for each candidate market
    logger.info(f"Verifying {len(market_positions)} markets are still active...")
    state["progress"] = f"Verifying market status for {len(market_positions)} markets..."

    resolved_markets = set()
    verified_count = 0

    # Build a set of all condition IDs we need to check
    cids_to_check = list(market_positions.keys())

    for cid in cids_to_check:
        # Check via Gamma API if market is still open
        try:
            data = api_get(f"{GAMMA_API}/markets", {"conditionId": cid, "limit": 1})
            if data and isinstance(data, list) and len(data) > 0:
                mkt = data[0]
                is_closed = mkt.get("closed", False)
                is_resolved = mkt.get("resolved", False)
                end_date = mkt.get("endDate", "") or mkt.get("end_date_iso", "")

                # Update title if we got a better one
                question = mkt.get("question", "")
                if question:
                    market_titles[cid] = question

                # Get current price from tokens
                tokens = mkt.get("tokens", [])
                if isinstance(tokens, list):
                    for tok in tokens:
                        if isinstance(tok, dict):
                            tp = float(tok.get("price", 0) or 0)
                            if tp > 0:
                                market_prices[cid] = tp
                                break

                # Also check outcomePrices for resolution
                try:
                    op = mkt.get("outcomePrices", "")
                    if op:
                        if isinstance(op, str):
                            op = json.loads(op)
                        if isinstance(op, list):
                            for p in op:
                                if float(p) >= 0.95:
                                    is_resolved = True
                                    break
                except Exception:
                    pass

                if is_closed or is_resolved:
                    resolved_markets.add(cid)

            verified_count += 1
            if verified_count % 10 == 0:
                state["progress"] = f"Verifying markets: {verified_count}/{len(cids_to_check)}..."

        except Exception as e:
            logger.debug(f"Failed to verify market {cid[:10]}: {e}")

    # Remove resolved markets
    for cid in resolved_markets:
        del market_positions[cid]

    logger.info(f"Market verification: {len(resolved_markets)} resolved/closed removed, {len(market_positions)} active remain")

    # ---- Score convergence with quality filters ----
    signals = []
    skipped_dead = 0
    skipped_small = 0

    for cid, pos in market_positions.items():
        # FILTER: Skip dead markets (price implies outcome is already known)
        mid_price = market_prices.get(cid, 0.5)
        if mid_price > 0.90 or mid_price < 0.10:
            skipped_dead += 1
            continue

        for direction, wallets in [("YES", pos["yes"]), ("NO", pos["no"])]:
            # Deduplicate by wallet
            seen = {}
            for w in wallets:
                addr = w["wallet"]
                if addr not in seen or w["size"] > seen[addr]["size"]:
                    seen[addr] = w
            unique_wallets = list(seen.values())

            # FILTER: Need at least 3 distinct wallets for a meaningful signal
            if len(unique_wallets) < 3:
                skipped_small += 1
                continue

            combined_score = sum(w["sharpness_score"] for w in unique_wallets)
            if combined_score < 1.0:
                continue

            elite_count = sum(1 for w in unique_wallets if w["tier"] == "elite")
            sharp_count = sum(1 for w in unique_wallets if w["tier"] == "sharp")
            total_size = sum(w["size"] for w in unique_wallets)
            total_notional = sum(w.get("notional", 0) for w in unique_wallets)

            # FILTER: Minimum $200 combined notional
            if total_notional < 200:
                continue

            opp = pos["no"] if direction == "YES" else pos["yes"]
            opp_seen = {}
            for w in opp:
                if w["wallet"] not in opp_seen:
                    opp_seen[w["wallet"]] = w
            opp_score = sum(w["sharpness_score"] for w in opp_seen.values())
            agreement = combined_score / (combined_score + opp_score) if (combined_score + opp_score) > 0 else 1.0

            # Strength formula: wallet diversity + combined sharpness + agreement + money committed
            # Batch signals don't have exact entry timestamps, so freshness defaults to 0.5
            freshness = 0.5  # unknown entry time from batch positions

            strength = (
                0.20 * min(1, len(unique_wallets) / 5) +    # More independent wallets = stronger
                0.20 * min(1, combined_score / 2.5) +        # Higher combined sharpness = stronger
                0.20 * agreement +                            # Less opposition = stronger
                0.20 * min(1, total_notional / 5000) +        # More money committed = stronger
                0.20 * freshness                              # Freshness (unknown for batch)
            )

            if sharp_count >= 5 and len(unique_wallets) >= 6:
                sig_type = "STRONG_CONVERGENCE"
            elif sharp_count >= 3 and len(unique_wallets) >= 4:
                sig_type = "CONVERGENCE"
            else:
                sig_type = "MILD_CONVERGENCE"

            title = market_titles.get(cid, cid[:20])

            signals.append({
                "market_id": cid,
                "market_title": title,
                "side": direction,
                "signal_type": sig_type,
                "strength": round(strength, 4),
                "wallet_count": len(unique_wallets),
                "elite_count": elite_count,
                "sharp_count": sharp_count,
                "combined_sharpness": round(combined_score, 4),
                "opposition_score": round(opp_score, 4),
                "agreement_ratio": round(agreement, 4),
                "total_size": round(total_size, 2),
                "total_notional": round(total_notional, 2),
                "market_price": round(mid_price, 3),
                "freshness": freshness,
                "avg_age_hours": None,
                "earliest_entry": None,
                "latest_entry": None,
                "wallets": sorted(unique_wallets, key=lambda w: w["sharpness_score"], reverse=True)[:10],
                "source": "batch",
            })

    # Deduplicate signals from related markets (same wallets, similar titles)
    # Sort by strength, then remove signals where >75% of wallets overlap with a stronger signal
    signals.sort(key=lambda s: s["strength"], reverse=True)
    filtered_signals = []
    seen_wallet_sets = []

    for sig in signals:
        sig_wallets = set(w["wallet"] for w in sig["wallets"])

        is_duplicate = False
        for prev_wallets in seen_wallet_sets:
            overlap = len(sig_wallets & prev_wallets)
            if overlap >= len(sig_wallets) * 0.75:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_signals.append(sig)
            seen_wallet_sets.append(sig_wallets)

    logger.info(f"Convergence: {len(signals)} raw → {len(filtered_signals)} after dedup "
                f"(skipped {skipped_dead} dead markets, {skipped_small} too few wallets)")
    return filtered_signals


# ================================================================
# MARKET INTELLIGENCE
# ================================================================

def compute_market_intelligence(scored_wallets):
    """
    For every open market with sharp wallet activity, compute:
      - Smart money direction (weighted YES vs NO)
      - Dull money direction
      - Sharp/Dull divergence score
      - Price gap (market price vs smart money implied price)
    
    This is the core "should I bet on this" view.
    """
    logger.info("Computing market intelligence...")
    state["progress"] = "Computing market intelligence..."

    # Build wallet lookup by tier — including category-specific tiers
    wallet_lookup = {w["wallet"]: w for w in scored_wallets}

    # For category-specific sharp detection
    def is_sharp_for_category(wallet_data, category, sub_category=None):
        """Check if wallet is sharp overall, for the category, or for the sub-category."""
        if wallet_data["tier"] == "sharp":
            return True
        cat_scores = wallet_data.get("category_scores", {})
        if category in cat_scores and cat_scores[category]["tier"] == "sharp":
            return True
        if sub_category and sub_category in cat_scores and cat_scores[sub_category]["tier"] == "sharp":
            return True
        return False

    sharp_addrs = {w["wallet"] for w in scored_wallets if w["tier"] == "sharp"}
    # Also include wallets that are sharp in ANY category
    for w in scored_wallets:
        for cat, cs in w.get("category_scores", {}).items():
            if cs["tier"] == "sharp":
                sharp_addrs.add(w["wallet"])

    dull_addrs = {w["wallet"] for w in scored_wallets if w["tier"] == "dull"}
    avg_addrs = {w["wallet"] for w in scored_wallets if w["tier"] == "average"}

    # Fetch open markets
    state["progress"] = "Fetching open markets..."
    open_markets = {}
    offset = 0
    while True:
        data = api_get(f"{GAMMA_API}/markets", {
            "closed": "false", "limit": 100, "offset": offset,
            "order": "volume", "ascending": "false",
        })
        if not data or not isinstance(data, list) or len(data) == 0:
            break
        for m in data:
            cid = m.get("conditionId") or m.get("condition_id") or ""
            if not cid:
                continue
            vol = float(m.get("volume", 0) or 0)
            if vol < 1000:
                continue

            # Get current price
            yes_price = 0.5
            tokens = m.get("tokens", [])
            if isinstance(tokens, list):
                for tok in tokens:
                    if isinstance(tok, dict):
                        tp = float(tok.get("price", 0) or 0)
                        outcome = (tok.get("outcome", "") or "").lower()
                        if tp > 0 and outcome in ("yes", "y"):
                            yes_price = tp

            q = m.get("question", "")
            tags_str = " ".join(
                t.get("label", "") if isinstance(t, dict) else str(t)
                for t in (m.get("tags") or [])
            )
            mkt_cat, mkt_sub = categorize(q, tags_str)
            open_markets[cid] = {
                "condition_id": cid,
                "question": q,
                "slug": m.get("slug", ""),
                "volume": vol,
                "yes_price": yes_price,
                "category": mkt_cat,
                "sub_category": mkt_sub,
                "end_date": m.get("endDate", ""),
            }
        offset += 100
        if len(data) < 100 or offset >= 1500:
            break

    logger.info(f"Found {len(open_markets)} open markets")

    # Skip dead-priced markets
    active_markets = {cid: m for cid, m in open_markets.items()
                      if 0.05 < m["yes_price"] < 0.95}
    logger.info(f"Active (price 5-95%): {len(active_markets)} markets")

    # Scan positions for ALL scored wallets (not just sharp) to get dull money too
    state["progress"] = "Scanning wallet positions for market intelligence..."
    market_positions = defaultdict(lambda: {
        "sharp_yes": [], "sharp_no": [],
        "dull_yes": [], "dull_no": [],
        "avg_yes": [], "avg_no": [],
    })

    wallets_to_scan = [w for w in scored_wallets if w["tier"] == "sharp"]
    # Also include wallets that are sharp in any category but not overall sharp
    cat_sharp_wallets = [w for w in scored_wallets if w["tier"] != "sharp"
                         and any(cs["tier"] == "sharp" for cs in w.get("category_scores", {}).values())]
    # Limit dull wallet scanning to top 100 by trade count (most active)
    dull_wallets = sorted([w for w in scored_wallets if w["tier"] == "dull"],
                          key=lambda w: w["total_trades"], reverse=True)[:100]
    sharp_wallets_list = wallets_to_scan + cat_sharp_wallets
    scan_list = sharp_wallets_list + dull_wallets

    # Also collect whale positions during this scan
    whale_positions = []
    WHALE_MIN_NOTIONAL = 25000

    scanned = 0
    for wdata in scan_list:
        wallet_addr = wdata["wallet"]
        try:
            positions = api_get(f"{DATA_API}/positions", {"user": wallet_addr, "sizeThreshold": 10})
            if not positions or not isinstance(positions, list):
                continue

            for pos in positions:
                cid = pos.get("conditionId") or pos.get("market") or ""
                if cid not in active_markets:
                    continue

                size = float(pos.get("size", 0) or 0)
                avg_price = float(pos.get("avgPrice", 0) or 0)
                notional = size * avg_price
                if notional < 20:
                    continue

                outcome = (pos.get("outcome", "") or "").lower()
                tier = wdata["tier"]
                # Check category-specific sharpness for this market's category and sub-category
                mkt_category = active_markets[cid].get("category", "other")
                mkt_sub = active_markets[cid].get("sub_category")
                is_cat_sharp = is_sharp_for_category(wdata, mkt_category, mkt_sub)

                entry = {
                    "wallet": wallet_addr,
                    "display_name": wdata["display_name"],
                    "tier": tier,
                    "is_category_sharp": is_cat_sharp,
                    "sharpness_score": wdata["sharpness_score"],
                    "category_sharpness": wdata.get("category_scores", {}).get(mkt_category, {}).get("sharpness", 0),
                    "size": size,
                    "avg_price": avg_price,
                    "notional": round(notional, 2),
                }

                is_yes = outcome in ("yes", "y", "1", "")

                if is_cat_sharp:
                    if is_yes:
                        market_positions[cid]["sharp_yes"].append(entry)
                    else:
                        market_positions[cid]["sharp_no"].append(entry)
                elif tier == "dull":
                    if is_yes:
                        market_positions[cid]["dull_yes"].append(entry)
                    else:
                        market_positions[cid]["dull_no"].append(entry)

                # Collect whale positions
                if notional >= WHALE_MIN_NOTIONAL:
                    mkt = active_markets[cid]
                    whale_positions.append({
                        "wallet": wallet_addr,
                        "display_name": wdata["display_name"],
                        "tier": tier,
                        "sharpness_score": wdata["sharpness_score"],
                        "is_sharp": is_cat_sharp,
                        "is_category_sharp": is_cat_sharp and tier != "sharp",
                        "category_detail": mkt_sub or mkt_category,
                        "outcome": outcome.upper() if outcome in ("yes", "y", "1") else "NO",
                        "title": mkt["question"][:80],
                        "condition_id": cid,
                        "notional": round(notional, 2),
                        "size": round(size, 2),
                        "avg_price": round(avg_price, 4),
                    })

            scanned += 1
            if scanned % 10 == 0:
                state["progress"] = f"Market intel: scanned {scanned}/{len(scan_list)} wallets..."

        except Exception as e:
            logger.debug(f"Failed scanning {wallet_addr[:10]}: {e}")

    logger.info(f"Scanned {scanned} wallets, found activity in {len(market_positions)} open markets")

    # Compute intelligence for each market
    intel_results = []
    for cid, pos in market_positions.items():
        mkt = active_markets.get(cid)
        if not mkt:
            continue

        # Deduplicate within each group
        def dedup(entries):
            seen = {}
            for e in entries:
                addr = e["wallet"]
                if addr not in seen or e["notional"] > seen[addr]["notional"]:
                    seen[addr] = e
            return list(seen.values())

        sy = dedup(pos["sharp_yes"])
        sn = dedup(pos["sharp_no"])
        dy = dedup(pos["dull_yes"])
        dn = dedup(pos["dull_no"])

        # Need at least 2 sharp wallets (overall or category-specific) to be actionable
        total_sharp = len(sy) + len(sn)
        if total_sharp < 2:
            continue

        # Compute weighted scores (sharpness-weighted capital)
        sharp_yes_score = sum(e["sharpness_score"] * e["notional"] for e in sy)
        sharp_no_score = sum(e["sharpness_score"] * e["notional"] for e in sn)
        sharp_total_score = sharp_yes_score + sharp_no_score

        dull_yes_notional = sum(e["notional"] for e in dy)
        dull_no_notional = sum(e["notional"] for e in dn)
        dull_total = dull_yes_notional + dull_no_notional

        # Smart money direction: what % of sharp-weighted capital is on YES
        if sharp_total_score > 0:
            smart_money_yes_pct = sharp_yes_score / sharp_total_score
        else:
            smart_money_yes_pct = 0.5  # No data = neutral

        # Dull money direction
        if dull_total > 0:
            dull_money_yes_pct = dull_yes_notional / dull_total
        else:
            dull_money_yes_pct = 0.5

        # Smart money implied price (what sharps think this market is worth)
        smart_implied = smart_money_yes_pct

        # Price gap: difference between market price and smart money implied
        market_price = mkt["yes_price"]
        price_gap = smart_implied - market_price  # Positive = sharps think YES is underpriced

        # Divergence: how much do sharp and dull disagree?
        # 0 = same direction, 1 = completely opposite
        divergence = abs(smart_money_yes_pct - dull_money_yes_pct)

        # Interest score: combination of sharp conviction, divergence, and price gap
        sharp_conviction = abs(smart_money_yes_pct - 0.5) * 2  # 0 = split, 1 = unanimous
        interest_score = (
            0.35 * sharp_conviction +            # How strongly sharps agree
            0.25 * divergence +                    # How much sharp/dull disagree
            0.25 * min(1, abs(price_gap) / 0.20) + # How big the price gap is
            0.15 * min(1, total_sharp / 5)         # How many sharps are involved
        )

        # Determine signal direction
        if smart_money_yes_pct > 0.65:
            smart_direction = "YES"
            direction_strength = smart_money_yes_pct
        elif smart_money_yes_pct < 0.35:
            smart_direction = "NO"
            direction_strength = 1 - smart_money_yes_pct
        else:
            smart_direction = "SPLIT"
            direction_strength = 0.5

        # Count category-specific sharps
        cat_sharp_yes = [e for e in sy if e.get("is_category_sharp")]
        cat_sharp_no = [e for e in sn if e.get("is_category_sharp")]

        intel_results.append({
            "condition_id": cid,
            "question": mkt["question"],
            "slug": mkt["slug"],
            "category": mkt["category"],
            "sub_category": mkt.get("sub_category"),
            "volume": mkt["volume"],
            "market_price": round(market_price, 3),

            # Smart money
            "smart_direction": smart_direction,
            "smart_money_yes_pct": round(smart_money_yes_pct, 3),
            "smart_implied_price": round(smart_implied, 3),
            "sharp_yes_count": len(sy),
            "sharp_no_count": len(sn),
            "sharp_yes_notional": round(sum(e["notional"] for e in sy), 2),
            "sharp_no_notional": round(sum(e["notional"] for e in sn), 2),
            "cat_sharp_yes": len(cat_sharp_yes),
            "cat_sharp_no": len(cat_sharp_no),
            "sharp_wallets_yes": sorted(sy, key=lambda e: e.get("category_sharpness", e["sharpness_score"]), reverse=True)[:5],
            "sharp_wallets_no": sorted(sn, key=lambda e: e.get("category_sharpness", e["sharpness_score"]), reverse=True)[:5],

            # Dull money
            "dull_money_yes_pct": round(dull_money_yes_pct, 3),
            "dull_yes_count": len(dy),
            "dull_no_count": len(dn),
            "dull_yes_notional": round(dull_yes_notional, 2),
            "dull_no_notional": round(dull_no_notional, 2),

            # Signals
            "price_gap": round(price_gap, 3),
            "divergence": round(divergence, 3),
            "sharp_conviction": round(sharp_conviction, 3),
            "interest_score": round(interest_score, 4),
        })

    # Sort by interest score (most interesting markets first)
    intel_results.sort(key=lambda m: m["interest_score"], reverse=True)

    # Sort whale positions by notional
    whale_positions.sort(key=lambda w: w["notional"], reverse=True)

    logger.info(f"Market intelligence: {len(intel_results)} markets with sharp wallet activity")
    logger.info(f"Whale positions: {len(whale_positions)} positions >= ${WHALE_MIN_NOTIONAL}, {sum(1 for w in whale_positions if w['is_sharp'])} from sharp wallets")
    return intel_results, whale_positions


# ================================================================
# LEADERBOARD SEEDING
# ================================================================

def fetch_leaderboard_wallets(limit=200):
    """
    Pull top traders from Polymarket's leaderboard.
    These are the biggest whales — some sharp, some lucky.
    We force-scan their full history so the scoring engine can decide.
    """
    logger.info(f"Fetching top {limit} leaderboard wallets...")
    state["progress"] = "Seeding from Polymarket leaderboard..."

    wallets = set()

    # Try multiple leaderboard periods to get a diverse set
    for period in ["all", "daily", "weekly", "monthly"]:
        data = api_get(f"{DATA_API}/leaderboard", {"limit": 100, "period": period})
        if data and isinstance(data, list):
            for entry in data:
                addr = entry.get("proxyWallet") or entry.get("userAddress") or entry.get("address") or ""
                if addr:
                    wallets.add(addr)
            logger.info(f"  Leaderboard [{period}]: found {len(data)} entries")
        elif isinstance(data, dict):
            # Some endpoints return {"leaderboard": [...]}
            entries = data.get("leaderboard", data.get("data", data.get("results", [])))
            if isinstance(entries, list):
                for entry in entries:
                    addr = entry.get("proxyWallet") or entry.get("userAddress") or entry.get("address") or ""
                    if addr:
                        wallets.add(addr)
                logger.info(f"  Leaderboard [{period}]: found {len(entries)} entries")

    logger.info(f"Found {len(wallets)} unique leaderboard wallets")
    return list(wallets)[:limit]


def fetch_wallet_trade_history(wallet_addr, markets_lookup):
    """
    Fetch the full trade history for a specific wallet.
    Returns normalized trades enriched with market resolution data.
    """
    all_trades = []
    offset = 0
    max_trades = 2000

    while offset < max_trades:
        data = api_get(f"{DATA_API}/trades", {
            "user": wallet_addr,
            "limit": min(1000, max_trades - offset),
            "offset": offset,
        })
        if not data or not isinstance(data, list) or len(data) == 0:
            break

        for t in data:
            price = float(t.get("price", 0) or 0)
            size = float(t.get("size", 0) or 0)
            if price <= 0 or size <= 0:
                continue

            cid = t.get("conditionId") or t.get("market") or ""
            mkt = markets_lookup.get(cid, {})

            all_trades.append({
                "wallet": wallet_addr,
                "side": t.get("side", "BUY"),
                "price": price,
                "size": size,
                "timestamp": int(t.get("timestamp", 0) or 0),
                "condition_id": cid,
                "outcome": t.get("outcome", ""),
                "title": t.get("title", "") or mkt.get("question", ""),
                "name": t.get("name", ""),
                "pseudonym": t.get("pseudonym", ""),
                "category": mkt.get("category", "other"),
                "winning_outcome": mkt.get("winning_outcome", ""),
            })

        offset += len(data)
        if len(data) < 1000:
            break

    return all_trades


def seed_from_leaderboard(existing_trades, markets_lookup):
    """
    Fetch leaderboard wallets' full histories and merge with existing trades.
    Only fetches history for wallets we don't already have good data on.
    """
    logger.info("=" * 50)
    logger.info("LEADERBOARD SEEDING")
    logger.info("=" * 50)

    lb_wallets = fetch_leaderboard_wallets(200)
    if not lb_wallets:
        logger.info("No leaderboard data returned. Skipping seeding.")
        return existing_trades

    # Find which wallets we already have enough data for
    existing_wallet_trades = defaultdict(int)
    for t in existing_trades:
        existing_wallet_trades[t["wallet"]] += 1

    new_trades = list(existing_trades)  # Start with existing
    wallets_seeded = 0
    wallets_skipped = 0

    for i, wallet in enumerate(lb_wallets):
        # Skip if we already have 20+ trades for this wallet
        if existing_wallet_trades.get(wallet, 0) >= 20:
            wallets_skipped += 1
            continue

        trades = fetch_wallet_trade_history(wallet, markets_lookup)
        if trades:
            new_trades.extend(trades)
            wallets_seeded += 1
            logger.debug(f"  Seeded {wallet[:10]}... with {len(trades)} trades")

        if (i + 1) % 20 == 0:
            state["progress"] = f"Seeding leaderboard: {i+1}/{len(lb_wallets)} wallets ({wallets_seeded} new)..."

    logger.info(f"Seeding complete: {wallets_seeded} new wallets added, {wallets_skipped} already had data")
    logger.info(f"Total trades now: {len(new_trades)} (was {len(existing_trades)})")
    return new_trades


# ================================================================
# TELEGRAM ALERTS
# ================================================================

def tg_send(text):
    """Send a Telegram message if configured."""
    if not TG_TOKEN or not TG_CHAT:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False


def alert_convergence(signals):
    """Alert 1: New convergence signal (3+ sharps align on open market)."""
    if not signals:
        return
    for s in signals[:5]:  # Max 5 alerts per refresh
        wallets = ", ".join(w["display_name"] for w in s["wallets"][:4])
        emoji = {"STRONG_CONVERGENCE": "🔴", "CONVERGENCE": "🟡", "MILD_CONVERGENCE": "🟢"}.get(s["signal_type"], "⚪")
        tg_send(
            f"{emoji} <b>CONVERGENCE SIGNAL</b>\n\n"
            f"📊 <b>{s['market_title'][:70]}</b>\n"
            f"📍 Direction: <b>{s['side']}</b>\n"
            f"\U0001f45b {s['wallet_count']} sharp wallets\n"
            f"💪 Strength: {s['strength']:.2f}\n"
            f"💰 Combined: ${s.get('total_notional', 0):,.0f}\n"
            f"📈 Market price: {s.get('market_price', 0):.0%}\n\n"
            f"🔍 Wallets: {wallets}"
        )


def alert_high_divergence(intel_markets):
    """Alert 2: High divergence market (sharp vs dull disagree with big price gap)."""
    if not intel_markets:
        return

    # Get previously alerted markets to avoid repeats
    prev_alerted = state.get("_alerted_divergence", set())
    new_alerted = set()

    for m in intel_markets:
        cid = m["condition_id"]
        # Only alert if: divergence >= 30%, price gap >= 8%, and at least 2 sharp wallets
        if (m["divergence"] >= 0.30 and abs(m["price_gap"]) >= 0.08
                and (m["sharp_yes_count"] + m["sharp_no_count"]) >= 2):

            if cid in prev_alerted:
                new_alerted.add(cid)
                continue  # Already alerted this market

            new_alerted.add(cid)
            gap = m["price_gap"]
            direction = "underpriced" if gap > 0 else "overpriced"

            sharp_side = "YES" if m["smart_money_yes_pct"] > 0.5 else "NO"
            dull_side = "YES" if m["dull_money_yes_pct"] > 0.5 else "NO"

            tg_send(
                f"⚡ <b>SHARP/DULL DIVERGENCE</b>\n\n"
                f"📊 <b>{m['question'][:70]}</b>\n\n"
                f"🧠 Sharp money: <b>{sharp_side}</b> ({m['smart_money_yes_pct']:.0%} YES)\n"
                f"🐑 Dull money: <b>{dull_side}</b> ({m['dull_money_yes_pct']:.0%} YES)\n"
                f"📍 Divergence: <b>{m['divergence']:.0%}</b>\n\n"
                f"💹 Market price: {m['market_price']:.0%}\n"
                f"📐 Price gap: <b>{gap:+.0%}</b> (YES looks {direction})\n"
                f"👛 {m['sharp_yes_count']} sharp YES · {m['sharp_no_count']} sharp NO\n"
                f"💰 Sharp capital: ${m['sharp_yes_notional'] + m['sharp_no_notional']:,.0f}"
            )

    state["_alerted_divergence"] = new_alerted


def alert_sharp_moves(scored_wallets):
    """Alert 3: Top sharp wallet takes a large new position (>$500 notional)."""
    # Track top 20 sharp wallets by sharpness score
    sharp_wallets = sorted(
        [w for w in scored_wallets if w["tier"] == "sharp"],
        key=lambda w: w["sharpness_score"], reverse=True
    )[:20]
    if not sharp_wallets:
        return

    # Track previously seen positions to detect new ones
    prev_positions = state.get("_sharp_positions", {})
    current_positions = {}

    for wdata in sharp_wallets:
        wallet_addr = wdata["wallet"]
        try:
            positions = api_get(f"{DATA_API}/positions", {"user": wallet_addr, "sizeThreshold": 10})
            if not positions or not isinstance(positions, list):
                continue

            for pos in positions:
                cid = pos.get("conditionId") or pos.get("market") or ""
                size = float(pos.get("size", 0) or 0)
                avg_price = float(pos.get("avgPrice", 0) or 0)
                notional = size * avg_price
                if notional < 500:
                    continue

                pos_key = f"{wallet_addr}:{cid}"
                current_positions[pos_key] = notional

                # Alert if this is a NEW position we haven't seen before
                if pos_key not in prev_positions:
                    outcome = pos.get("outcome", "?")
                    title = pos.get("title", "") or pos.get("eventTitle", "Unknown market")

                    tg_send(
                        f"\U0001f40b <b>SHARP WALLET MOVE</b>\n\n"
                        f"\U0001f45b <b>{wdata['display_name']}</b> (Score: {wdata['sharpness_score']*100:.0f})\n"
                        f"\U0001f4ca {title[:70]}\n"
                        f"\U0001f4cd Side: <b>{outcome}</b>\n"
                        f"\U0001f4b0 Position: <b>${notional:,.0f}</b>\n"
                        f"\U0001f4c8 Avg entry: ${avg_price:.2f}"
                    )
                # Alert if position grew significantly (>50% increase)
                elif notional > prev_positions[pos_key] * 1.5:
                    outcome = pos.get("outcome", "?")
                    title = pos.get("title", "") or pos.get("eventTitle", "Unknown market")
                    prev_val = prev_positions[pos_key]

                    tg_send(
                        f"\U0001f40b <b>SHARP ADDING TO POSITION</b>\n\n"
                        f"\U0001f45b <b>{wdata['display_name']}</b> (Score: {wdata['sharpness_score']*100:.0f})\n"
                        f"\U0001f4ca {title[:70]}\n"
                        f"\U0001f4cd Side: <b>{outcome}</b>\n"
                        f"\U0001f4b0 ${prev_val:,.0f} \u2192 <b>${notional:,.0f}</b> (+{(notional/prev_val-1)*100:.0f}%)"
                    )

        except Exception as e:
            logger.debug(f"Failed scanning sharp {wallet_addr[:10]}: {e}")

    state["_sharp_positions"] = current_positions


def send_daily_digest(scored_wallets, intel_markets, signals):
    """Alert 4: Daily summary digest — sent once per day."""
    last_digest = state.get("_last_digest_date", "")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if last_digest == today:
        return  # Already sent today

    tiers = defaultdict(int)
    for w in scored_wallets:
        tiers[w["tier"]] += 1

    # Top 3 markets by interest
    top_markets = ""
    if intel_markets:
        for i, m in enumerate(intel_markets[:3], 1):
            gap = m["price_gap"]
            top_markets += (
                f"\n{i}. <b>{m['question'][:50]}</b>"
                f"\n   Smart $: {'YES' if m['smart_money_yes_pct'] > 0.5 else 'NO'} "
                f"({m['smart_money_yes_pct']:.0%}) · Gap: {gap:+.0%} · "
                f"Div: {m['divergence']:.0%}"
            )
    else:
        top_markets = "\n   No market intel data yet"

    # New elite/sharp wallets (scored recently)
    recent_sharps = [w for w in scored_wallets
                     if w["tier"] in ("elite", "sharp")
                     and w.get("scored_at", "").startswith(today)]

    tg_send(
        f"📋 <b>SHARPFLOW DAILY DIGEST</b>\n"
        f"📅 {today}\n\n"
        f"<b>System Status</b>\n"
        f"👛 {len(scored_wallets)} wallets scored\n"
        f"   ⚡ {tiers['elite']} Elite · 📊 {tiers['sharp']} Sharp\n"
        f"   ⚖️ {tiers['average']} Average · ❌ {tiers['dull']} Dull\n"
        f"🎯 {len(signals)} convergence signals\n"
        f"📊 {len(intel_markets)} markets tracked\n\n"
        f"<b>Top Markets by Smart Money Interest</b>"
        f"{top_markets}\n\n"
        f"{'🆕 ' + str(len(recent_sharps)) + ' new sharp+ wallets today' if recent_sharps else ''}"
        f"🔗 Check dashboard for full details"
    )

    state["_last_digest_date"] = today


# ================================================================
# FULL PIPELINE
# ================================================================

def run_pipeline():
    """Execute the full data pipeline."""
    try:
        state["status"] = "running"
        state["error"] = None
        start = time.time()

        # Step 1: Fetch markets
        markets = fetch_resolved_markets(MARKETS_TO_FETCH)
        state["markets_analyzed"] = len(markets)
        if not markets:
            state["status"] = "error"
            state["error"] = "No markets fetched. API may be down."
            return

        markets_lookup = {m["condition_id"]: m for m in markets}

        # Step 2: Fetch trades
        trades = fetch_trades_for_markets(markets)
        state["trades_analyzed"] = len(trades)
        if not trades:
            state["status"] = "error"
            state["error"] = "No trades fetched."
            return

        # Step 2.5: Seed from Polymarket leaderboard
        trades = seed_from_leaderboard(trades, markets_lookup)
        state["trades_analyzed"] = len(trades)

        # Step 2.6: Merge with historical trades from database
        if DATABASE_URL:
            state["progress"] = "Loading historical trades from database..."
            logger.info("Loading historical trades from database...")
            try:
                db_trades = db_get_all_trades(SCORING_WINDOW_MONTHS)
                if db_trades:
                    # Build a set of existing trade signatures to avoid duplicates
                    existing_sigs = set()
                    for t in trades:
                        sig = (t["wallet"], t["condition_id"], t.get("timestamp", 0), t["side"], t["price"])
                        existing_sigs.add(sig)

                    # Add DB trades that aren't already in memory
                    added = 0
                    for t in db_trades:
                        sig = (t["wallet"], t["condition_id"], t.get("timestamp", 0), t["side"], t["price"])
                        if sig not in existing_sigs:
                            trades.append(t)
                            existing_sigs.add(sig)
                            added += 1

                    # Also expand the markets lookup with any markets from DB trades
                    for t in db_trades:
                        cid = t.get("condition_id", "")
                        if cid and cid not in markets_lookup:
                            markets_lookup[cid] = {
                                "condition_id": cid,
                                "question": t.get("title", ""),
                                "category": t.get("category", "other"),
                                "winning_outcome": t.get("winning_outcome", ""),
                            }

                    logger.info(f"  Loaded {len(db_trades):,} DB trades, added {added:,} new. Total trades for scoring: {len(trades):,}")
                    state["trades_analyzed"] = len(trades)
                    state["progress"] = f"Scoring with {len(trades):,} trades ({added:,} from history)..."
            except Exception as e:
                logger.error(f"Error loading DB trades: {e}")

        # Step 3: Score wallets
        scored = score_all_wallets(trades, markets_lookup)

        # Seed the live scoring cache with batch results
        # This is the source of truth for "who is sharp"
        with live_wallet_scores_lock:
            for w in scored:
                w["source"] = "batch"
                live_wallet_scores[w["wallet"]] = w
            logger.info(f"Seeded live scoring cache with {len(scored)} batch-scored wallets")

        state["scored_wallets"] = scored

        # Step 4: Detect convergence
        signals = detect_convergence(scored, trades)

        # Merge batch convergence with any live convergence signals
        live_signals = [s for s in live_convergence_signals if s.get("source") == "live"]
        all_signals = signals + live_signals
        state["convergence_signals"] = all_signals

        # Step 5: Compute market intelligence (also collects whale positions)
        intel, whale_positions = compute_market_intelligence(scored)
        state["market_intelligence"] = intel
        state["whale_positions"] = whale_positions

        elapsed = time.time() - start
        state["status"] = "ready"
        state["last_refresh"] = datetime.utcnow().isoformat()
        state["progress"] = f"Complete! {len(scored)} wallets scored, {len(intel)} markets analyzed in {elapsed/60:.1f} min"

        logger.info(f"Pipeline complete in {elapsed/60:.1f} minutes")
        logger.info(f"  Markets: {len(markets)} | Trades: {len(trades)} | Wallets scored: {len(scored)} | Convergence: {len(signals)} | Market intel: {len(intel)}")

        # ---- PERSIST TO DATABASE ----
        if DATABASE_URL:
            state["progress"] = "Saving to database..."
            logger.info("Persisting data to database...")
            try:
                inserted = db_save_trades(trades)
                logger.info(f"  Trades saved: {inserted} new (of {len(trades)} total)")

                db_save_wallet_scores(scored)
                logger.info(f"  Wallet scores snapshot saved: {len(scored)}")

                db_save_market_intel(intel)
                logger.info(f"  Market intel snapshot saved: {len(intel)}")

                db_save_markets(markets)
                logger.info(f"  Market metadata saved: {len(markets)}")

                db_save_convergence_signals(signals)
                logger.info(f"  Convergence signals saved: {len(signals)}")

                state["progress"] = "Snapshotting wallet positions..."
                db_save_wallet_positions(scored)

                db_update_convergence_outcomes()

                db_save_pipeline_run({
                    "markets": len(markets), "trades": len(trades),
                    "wallets": len(scored), "convergence": len(signals),
                    "intel": len(intel), "status": "ok",
                    "notes": f"Elapsed: {elapsed/60:.1f} min",
                })
                total_db_trades = db_get_trade_count()
                logger.info(f"  Total trades in database: {total_db_trades:,}")
            except Exception as e:
                logger.error(f"Database persistence error: {e}")

        state["progress"] = f"Complete! {len(scored)} wallets scored, {len(intel)} markets analyzed in {elapsed/60:.1f} min"

        # ---- TELEGRAM ALERTS ----
        logger.info("Running alert checks...")

        # Alert 1: Convergence signals
        alert_convergence(signals)

        # Alert 2: High divergence markets
        alert_high_divergence(intel)

        # Alert 3: Elite wallet moves
        alert_sharp_moves(scored)

        # Alert 4: Daily digest (once per day)
        send_daily_digest(scored, intel, signals)

        logger.info("Alert checks complete")

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        state["status"] = "error"
        state["error"] = str(e)


# ================================================================
# FASTAPI APP
# ================================================================

app = FastAPI(title="SharpFlow", version="1.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/test-telegram")
def test_telegram():
    """Send a test message to verify Telegram is connected."""
    if not TG_TOKEN or not TG_CHAT:
        return {"status": "error", "message": "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID not set in environment variables"}
    success = tg_send(
        f"✅ <b>SharpFlow Telegram Connected!</b>\n\n"
        f"You'll receive alerts for:\n"
        f"🎯 Convergence signals (3+ sharps align)\n"
        f"⚡ High divergence markets (sharp vs dull disagree)\n"
        f"🐋 Elite wallet moves (>$500 positions)\n"
        f"📋 Daily digest summary"
    )
    if success:
        return {"status": "ok", "message": "Test message sent! Check your Telegram."}
    return {"status": "error", "message": "Failed to send. Check your bot token and chat ID."}


@app.get("/api/debug-env")
def debug_env():
    """Check which env vars are set (values hidden)."""
    db_url = os.getenv("DATABASE_URL", "")
    return {
        "DATABASE_URL_set": bool(db_url),
        "DATABASE_URL_length": len(db_url),
        "DATABASE_URL_starts_with": db_url[:15] + "..." if db_url else "(empty)",
        "DATABASE_URL_from_global": bool(DATABASE_URL),
        "all_db_vars": [k for k in os.environ if "DATABASE" in k.upper() or "POSTGRES" in k.upper() or "PG" in k.upper()],
    }


@app.get("/api/live")
def get_live_status():
    """Get live trade listener status."""
    try:
        from live_listener import get_listener_status
        status = get_listener_status()
        # Also get DB stats if listener has a db
        if ALCHEMY_URL and DATABASE_URL:
            from live_listener import _listener_instance
            if _listener_instance:
                status["db_stats"] = _listener_instance.db.get_stats()
        return status
    except ImportError:
        return {"running": False, "error": "live_listener module not found"}
    except Exception as e:
        return {"running": False, "error": str(e)}


@app.get("/api/wallets")
def get_wallets(
    sort: str = Query("sharpness"),
    tier: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    cat_sharp: Optional[str] = Query(None, description="Filter to wallets sharp in this category"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    wallets = state["scored_wallets"]
    if tier:
        wallets = [w for w in wallets if w["tier"] == tier]
    if category:
        wallets = [w for w in wallets if category in w.get("categories", [])]
    if cat_sharp:
        wallets = [w for w in wallets if w.get("category_scores", {}).get(cat_sharp, {}).get("tier") == "sharp"]

    sort_keys = {
        "sharpness": "sharpness_score", "pnl": "total_pnl",
        "clv": "clv_score", "winrate": "win_rate", "markets": "distinct_markets",
    }
    key = sort_keys.get(sort, "sharpness_score")
    wallets = sorted(wallets, key=lambda w: w.get(key, 0), reverse=True)

    total = len(wallets)
    wallets = wallets[offset:offset + limit]
    for i, w in enumerate(wallets):
        w["rank"] = offset + i + 1

    return {"wallets": wallets, "total": total, "offset": offset, "limit": limit}


@app.get("/api/wallets/{address}")
def get_wallet(address: str):
    """Get wallet score data."""
    for w in state["scored_wallets"]:
        if w["wallet"].lower() == address.lower():
            return w
    return {"error": "Wallet not found"}


@app.get("/api/wallets/{address}/live")
def get_wallet_live(address: str):
    """
    Get a wallet's CURRENT positions and recent trades.
    This hits the Polymarket API in real-time for fresh data.
    """
    # Get score data
    score_data = None
    for w in state["scored_wallets"]:
        if w["wallet"].lower() == address.lower():
            score_data = w
            break

    # Fetch current positions
    positions_raw = api_get(f"{DATA_API}/positions", {"user": address, "sizeThreshold": 1})
    positions = []
    if positions_raw and isinstance(positions_raw, list):
        for pos in positions_raw:
            size = float(pos.get("size", 0) or 0)
            avg_price = float(pos.get("avgPrice", 0) or 0)
            if size < 1:
                continue
            positions.append({
                "condition_id": pos.get("conditionId", ""),
                "title": pos.get("title", "") or pos.get("eventTitle", "Unknown"),
                "outcome": pos.get("outcome", ""),
                "size": round(size, 2),
                "avg_price": round(avg_price, 4),
                "notional": round(size * avg_price, 2),
                "current_price": float(pos.get("curPrice", 0) or 0),
            })

    # Sort positions by notional value
    positions.sort(key=lambda p: p["notional"], reverse=True)

    # Fetch recent trades
    trades_raw = api_get(f"{DATA_API}/trades", {"user": address, "limit": 50})
    recent_trades = []
    if trades_raw and isinstance(trades_raw, list):
        for t in trades_raw:
            price = float(t.get("price", 0) or 0)
            size = float(t.get("size", 0) or 0)
            if price <= 0 or size <= 0:
                continue
            recent_trades.append({
                "side": t.get("side", "BUY"),
                "outcome": t.get("outcome", ""),
                "price": round(price, 4),
                "size": round(size, 2),
                "notional": round(price * size, 2),
                "title": t.get("title", "") or t.get("eventTitle", "Unknown"),
                "timestamp": int(t.get("timestamp", 0) or 0),
                "condition_id": t.get("conditionId", ""),
            })

    return {
        "wallet": address,
        "score": score_data,
        "positions": positions,
        "position_count": len(positions),
        "total_position_value": round(sum(p["notional"] for p in positions), 2),
        "recent_trades": recent_trades,
        "trade_count": len(recent_trades),
    }


@app.get("/api/convergence")
def get_convergence(min_strength: float = Query(0.0)):
    signals = state["convergence_signals"]
    if min_strength > 0:
        signals = [s for s in signals if s["strength"] >= min_strength]
    return {"signals": signals, "total": len(signals)}


@app.get("/api/whales")
def get_whale_positions(min_notional: float = Query(25000)):
    """
    Surface the largest positions on active markets.
    Data is cached during pipeline run — returns instantly.
    """
    whales = state.get("whale_positions", [])
    filtered = [w for w in whales if w["notional"] >= min_notional]
    return {
        "whales": filtered[:50],
        "total": len(filtered),
        "min_notional": min_notional,
        "sharp_whales": sum(1 for w in filtered if w["is_sharp"]),
    }


@app.get("/api/markets")
def get_market_intelligence(limit: int = Query(50, ge=1, le=200)):
    """
    Market Intelligence endpoint.
    Returns open markets ranked by how interesting the smart money positioning is.
    For each market: sharp money direction, dull money direction, divergence score.
    """
    return {"markets": state.get("market_intelligence", [])[:limit],
            "total": len(state.get("market_intelligence", []))}


@app.get("/api/markets/{condition_id}")
def get_single_market_intel(condition_id: str):
    """Get smart money breakdown for a specific market."""
    for m in state.get("market_intelligence", []):
        if m["condition_id"] == condition_id:
            return m
    return {"error": "Market not found in intelligence data"}


@app.get("/api/stats")
def get_stats():
    tiers = defaultdict(int)
    for w in state["scored_wallets"]:
        tiers[w["tier"]] += 1
    return {
        "status": state["status"],
        "progress": state["progress"],
        "markets_analyzed": state["markets_analyzed"],
        "trades_analyzed": state["trades_analyzed"],
        "wallets_scanned": state["wallets_scanned"],
        "wallets_scored": len(state["scored_wallets"]),
        "tiers": dict(tiers),
        "convergence_signals": len(state["convergence_signals"]),
        "markets_with_intel": len(state.get("market_intelligence", [])),
        "last_refresh": state["last_refresh"],
        "error": state["error"],
        "db_connected": bool(DATABASE_URL),
        "db_total_trades": db_get_trade_count() if DATABASE_URL else None,
    }


@app.get("/api/db")
def get_db_stats():
    """Database statistics."""
    if not DATABASE_URL:
        return {"connected": False, "message": "No DATABASE_URL configured"}
    try:
        conn = db_conn()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM trades")
        trade_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT wallet) FROM trades")
        wallet_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT condition_id) FROM trades")
        market_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM wallet_scores")
        score_snapshots = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM pipeline_runs")
        run_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM markets")
        markets_stored = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM markets WHERE resolved = TRUE")
        markets_resolved = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM convergence_snapshots")
        convergence_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM convergence_snapshots WHERE signal_correct = TRUE")
        convergence_correct = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM convergence_snapshots WHERE market_resolved = TRUE")
        convergence_resolved = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM wallet_positions")
        position_snapshots = cur.fetchone()[0]

        cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM trades WHERE timestamp > 0")
        ts_row = cur.fetchone()
        min_ts = ts_row[0] if ts_row[0] else 0
        max_ts = ts_row[1] if ts_row[1] else 0

        cur.close()
        conn.close()

        return {
            "connected": True,
            "total_trades": trade_count,
            "unique_wallets": wallet_count,
            "unique_markets": market_count,
            "score_snapshots": score_snapshots,
            "pipeline_runs": run_count,
            "markets_stored": markets_stored,
            "markets_resolved": markets_resolved,
            "convergence_signals_stored": convergence_count,
            "convergence_resolved": convergence_resolved,
            "convergence_correct": convergence_correct,
            "convergence_accuracy": f"{convergence_correct/convergence_resolved*100:.0f}%" if convergence_resolved > 0 else "N/A",
            "position_snapshots": position_snapshots,
            "earliest_trade": datetime.utcfromtimestamp(min_ts).isoformat() if min_ts > 0 else None,
            "latest_trade": datetime.utcfromtimestamp(max_ts).isoformat() if max_ts > 0 else None,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}



# ================================================================
# HISTORICAL BACKFILL
# ================================================================

backfill_state = {
    "running": False,
    "progress": "",
    "markets_found": 0,
    "markets_processed": 0,
    "markets_skipped": 0,
    "trades_ingested": 0,
    "errors": 0,
    "started_at": None,
    "finished_at": None,
    "last_error": None,
}


def get_db_market_condition_ids():
    """Get all condition_ids we already have trades for in the database."""
    if not DATABASE_URL:
        return set()
    try:
        conn = db_conn()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT condition_id FROM trades")
        result = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error getting existing condition IDs: {e}")
        return set()


def run_backfill(max_markets=10000, min_volume=500):
    """
    Backfill historical data from Polymarket.
    Pages through ALL resolved markets and fetches trades for any we don't have yet.
    Saves to Postgres in batches — fully resumable.
    """
    if backfill_state["running"]:
        logger.warning("Backfill already running, skipping")
        return

    backfill_state["running"] = True
    backfill_state["started_at"] = datetime.utcnow().isoformat()
    backfill_state["finished_at"] = None
    backfill_state["errors"] = 0
    backfill_state["trades_ingested"] = 0
    backfill_state["markets_processed"] = 0
    backfill_state["markets_skipped"] = 0
    backfill_state["last_error"] = None

    logger.info(f"Starting historical backfill (max {max_markets} markets, min vol ${min_volume})")

    try:
        # Step 1: Get condition IDs we already have
        existing_cids = get_db_market_condition_ids()
        logger.info(f"Already have trades for {len(existing_cids)} markets in DB")
        backfill_state["progress"] = f"Found {len(existing_cids)} existing markets. Fetching market list..."

        # Step 2: Page through ALL resolved markets from Gamma API
        all_markets = []
        offset = 0
        while len(all_markets) < max_markets:
            try:
                data = api_get(f"{GAMMA_API}/markets", {
                    "closed": "true", "limit": 100, "offset": offset,
                    "order": "volume", "ascending": "false",
                })
                if not data or not isinstance(data, list) or len(data) == 0:
                    break

                for m in data:
                    cid = m.get("conditionId") or m.get("condition_id")
                    vol = float(m.get("volume", 0) or 0)
                    if not cid or vol < min_volume:
                        continue

                    tags = [t.get("label", "") if isinstance(t, dict) else str(t)
                            for t in (m.get("tags") or [])]

                    # Determine winner
                    winning = None
                    tokens = m.get("tokens", [])
                    if isinstance(tokens, list):
                        for tok in tokens:
                            if isinstance(tok, dict) and tok.get("winner"):
                                winning = tok.get("outcome", "").lower()

                    if not winning:
                        try:
                            op = m.get("outcomePrices", "")
                            outcomes = m.get("outcomes", "")
                            if op and outcomes:
                                if isinstance(op, str):
                                    op = json.loads(op)
                                if isinstance(outcomes, str):
                                    outcomes = json.loads(outcomes)
                                if isinstance(op, list) and isinstance(outcomes, list) and len(op) == len(outcomes):
                                    for price, outcome_name in zip(op, outcomes):
                                        if float(price) >= 0.95:
                                            winning = outcome_name.lower()
                                            break
                        except Exception:
                            pass

                    if not winning and m.get("resolved"):
                        res = m.get("resolution", "")
                        if res:
                            winning = str(res).lower()

                    cat, sub_cat = categorize(m.get("question", ""), " ".join(tags))

                    all_markets.append({
                        "condition_id": cid,
                        "question": m.get("question", ""),
                        "slug": m.get("slug", ""),
                        "volume": vol,
                        "category": cat,
                        "winning_outcome": winning,
                        "tokens": tokens,
                        "resolved": bool(m.get("resolved") or m.get("closed")),
                    })

                offset += 100
                backfill_state["progress"] = f"Scanning markets: {len(all_markets)} found (offset {offset})..."
                logger.info(f"  Backfill: fetched {len(all_markets)} markets at offset {offset}")

                if len(data) < 100:
                    break

            except Exception as e:
                logger.error(f"Backfill market fetch error at offset {offset}: {e}")
                backfill_state["errors"] += 1
                backfill_state["last_error"] = str(e)
                offset += 100
                time.sleep(2)

        backfill_state["markets_found"] = len(all_markets)
        logger.info(f"Backfill: found {len(all_markets)} total resolved markets")

        # Step 3: Filter to markets we don't have yet
        new_markets = [m for m in all_markets if m["condition_id"] not in existing_cids]
        logger.info(f"Backfill: {len(new_markets)} new markets to process ({len(all_markets) - len(new_markets)} already in DB)")
        backfill_state["markets_skipped"] = len(all_markets) - len(new_markets)

        # Step 4: Save market metadata for ALL markets
        if DATABASE_URL:
            db_save_markets(all_markets)
            logger.info(f"  Market metadata saved for {len(all_markets)} markets")

        # Step 5: Fetch trades for new markets and save in batches
        batch_trades = []
        batch_size = 20  # Save to DB every 20 markets
        empty_markets = 0

        for i, mkt in enumerate(new_markets):
            cid = mkt["condition_id"]
            try:
                # Try both parameter names
                data = api_get(f"{DATA_API}/trades", {
                    "market": cid, "limit": TRADES_PER_MARKET,
                })

                # If empty, try with conditionId parameter
                if not data or (isinstance(data, list) and len(data) == 0):
                    data = api_get(f"{DATA_API}/trades", {
                        "conditionId": cid, "limit": TRADES_PER_MARKET,
                    })

                # Log first few results for diagnostics
                if i < 3:
                    logger.info(f"  Backfill diag: market {cid[:16]}... returned {len(data) if isinstance(data, list) else type(data)}")

                if data and isinstance(data, list) and len(data) > 0:
                    for t in data:
                        wallet = t.get("proxyWallet") or t.get("maker") or t.get("taker") or ""
                        price = float(t.get("price", 0) or 0)
                        size = float(t.get("size", 0) or 0)
                        if not wallet or price <= 0 or size <= 0:
                            continue

                        batch_trades.append({
                            "wallet": wallet,
                            "side": t.get("side", "BUY"),
                            "price": price,
                            "size": size,
                            "timestamp": int(t.get("timestamp", 0) or 0),
                            "condition_id": cid,
                            "outcome": t.get("outcome", ""),
                            "title": t.get("title", mkt["question"]),
                            "category": mkt["category"],
                            "winning_outcome": mkt["winning_outcome"],
                        })
                else:
                    empty_markets += 1

            except Exception as e:
                logger.debug(f"Backfill trade fetch error for {cid[:12]}: {e}")
                backfill_state["errors"] += 1
                backfill_state["last_error"] = str(e)

            backfill_state["markets_processed"] = i + 1

            # Save batch to DB periodically
            if (i + 1) % batch_size == 0 or i == len(new_markets) - 1:
                if batch_trades and DATABASE_URL:
                    inserted = db_save_trades(batch_trades)
                    backfill_state["trades_ingested"] += inserted
                    logger.info(f"  Backfill batch: {i+1}/{len(new_markets)} markets, +{inserted} trades (total: {backfill_state['trades_ingested']})")
                    batch_trades = []

                backfill_state["progress"] = (
                    f"Processing: {i+1}/{len(new_markets)} new markets "
                    f"({backfill_state['trades_ingested']:,} trades ingested, "
                    f"{empty_markets} empty, {backfill_state['errors']} errors)"
                )

            # Brief pause to avoid rate limiting
            if (i + 1) % 50 == 0:
                time.sleep(1)

        backfill_state["finished_at"] = datetime.utcnow().isoformat()
        backfill_state["running"] = False
        backfill_state["progress"] = (
            f"Complete! {backfill_state['markets_processed']} markets processed, "
            f"{backfill_state['trades_ingested']:,} new trades ingested, "
            f"{backfill_state['markets_skipped']} skipped (already had data)"
        )

        # Save a pipeline run record for the backfill
        if DATABASE_URL:
            db_save_pipeline_run({
                "markets": backfill_state["markets_found"],
                "trades": backfill_state["trades_ingested"],
                "wallets": 0, "convergence": 0, "intel": 0,
                "status": "backfill_complete",
                "notes": backfill_state["progress"],
            })

        logger.info(f"Backfill complete: {backfill_state['progress']}")

        # Send Telegram notification
        tg_send(
            f"📦 <b>BACKFILL COMPLETE</b>\n\n"
            f"📊 Markets found: {backfill_state['markets_found']:,}\n"
            f"🆕 New markets: {backfill_state['markets_processed']:,}\n"
            f"⏭ Skipped (existing): {backfill_state['markets_skipped']:,}\n"
            f"📈 Trades ingested: {backfill_state['trades_ingested']:,}\n"
            f"❌ Errors: {backfill_state['errors']}\n\n"
            f"💾 Total trades in DB: {db_get_trade_count():,}"
        )

    except Exception as e:
        logger.error(f"Backfill fatal error: {e}", exc_info=True)
        backfill_state["running"] = False
        backfill_state["last_error"] = str(e)
        backfill_state["progress"] = f"Error: {e}"


@app.get("/api/backfill/start")
def start_backfill(max_markets: int = Query(10000), min_volume: int = Query(500)):
    """Start a historical backfill job."""
    if not DATABASE_URL:
        return {"status": "error", "message": "No database configured"}
    if backfill_state["running"]:
        return {"status": "already_running", "progress": backfill_state["progress"]}

    thread = threading.Thread(target=run_backfill, args=(max_markets, min_volume), daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": f"Backfill started. Fetching up to {max_markets} markets with min volume ${min_volume}. Check /api/backfill/status for progress.",
    }


@app.get("/api/backfill/status")
def backfill_status():
    """Check backfill progress."""
    return {
        **backfill_state,
        "db_total_trades": db_get_trade_count() if DATABASE_URL else 0,
    }


@app.get("/api/live_trades/stats")
def live_trades_stats():
    """Get live_trades table stats."""
    if not DATABASE_URL:
        return {"error": "No database configured"}
    try:
        conn = db_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM live_trades")
        count = cur.fetchone()[0]
        cur.execute("SELECT pg_size_pretty(pg_total_relation_size('live_trades'))")
        size = cur.fetchone()[0]
        cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM live_trades")
        min_ts, max_ts = cur.fetchone()
        conn.close()
        return {
            "row_count": count,
            "table_size": size,
            "oldest_timestamp": min_ts,
            "newest_timestamp": max_ts,
            "retention_days": LIVE_TRADES_RETENTION_DAYS,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/live_trades/cleanup")
def trigger_retention_cleanup():
    """Manually trigger retention cleanup."""
    thread = threading.Thread(target=run_retention_cleanup, daemon=True)
    thread.start()
    return {"status": "started", "message": f"Retention cleanup running (deleting rows older than {LIVE_TRADES_RETENTION_DAYS} days)"}


@app.get("/api/live_scoring")
def get_live_scoring_status():
    """Get live event-driven scoring stats."""
    with live_wallet_scores_lock:
        sharp_count = sum(1 for s in live_wallet_scores.values() if s.get("sharpness_score", 0) >= 0.38)
        total_scored = len(live_wallet_scores)
        convergence_count = len(live_convergence_signals)
        batch_count = sum(1 for s in live_wallet_scores.values() if s.get("source") == "batch")
    return {
        **live_scoring_stats,
        "wallets_tracked": total_scored,
        "batch_scored_wallets": batch_count,
        "sharp_wallets": sharp_count,
        "live_convergence_signals": convergence_count,
        "convergence_details": live_convergence_signals[:10],
    }


# ================================================================
# BACKTESTING ENGINE
# ================================================================

def backtest_score_wallets(trades, weights, min_markets=8, elite_score=0.65, elite_markets=25, sharp_score=0.45):
    """
    Score wallets using specified weights. Returns dict of wallet -> {tier, score, ...}.
    Standalone scoring function for backtesting — doesn't touch global state.
    """
    w_clv, w_timing, w_consist, w_roi = weights

    wallet_trades = defaultdict(list)
    for t in trades:
        wallet_trades[t["wallet"]].append(t)

    scores = {}
    for wallet, wtrades in wallet_trades.items():
        market_groups = defaultdict(list)
        for t in wtrades:
            market_groups[t["condition_id"]].append(t)

        distinct = len(market_groups)
        if distinct < min_markets:
            continue

        clvs = []
        timing_scores_list = []
        market_pnls = {}
        total_inv = 0.0
        total_ret = 0.0
        trade_count = 0

        for cid, mtrades in market_groups.items():
            mpnl = 0.0
            for t in mtrades:
                price = t["price"]
                side = t["side"]
                size = t["size"]
                outcome = (t.get("outcome") or "").lower()
                win_outcome = (t.get("winning_outcome") or "").lower()
                if not win_outcome:
                    continue

                outcome_won = (outcome == win_outcome)
                res_price = 1.0 if outcome_won else 0.0
                trade_count += 1

                if side == "BUY":
                    if price >= 0.95 and outcome_won:
                        continue
                    clvs.append(res_price - price)
                    timing_scores_list.append((0.5 - price) if outcome_won else (price - 0.5))
                    pnl = (res_price - price) * size
                    total_inv += price * size
                    total_ret += res_price * size
                    mpnl += pnl
                elif side == "SELL":
                    clvs.append(price - res_price)
                    timing_scores_list.append((price - 0.5) if not outcome_won else (0.5 - price))
                    pnl = (price - res_price) * size
                    total_inv += (1.0 - price) * size
                    total_ret += (1.0 - res_price) * size
                    mpnl += pnl

            market_pnls[cid] = mpnl

        if not clvs or trade_count == 0:
            continue

        avg_clv = float(np.mean(clvs))
        clv_score = max(0, min(1, (avg_clv + 0.1) / 0.52))

        early_pct = sum(1 for t in timing_scores_list if t > 0) / len(timing_scores_list) if timing_scores_list else 0
        timing_score = max(0, min(1, (early_pct - 0.1) / 0.6))

        mkt_won = sum(1 for p in market_pnls.values() if p > 0)
        win_rate = mkt_won / len(market_pnls) if market_pnls else 0
        market_count_factor = min(1.0, (distinct / 100) ** 0.5)
        consistency_score = win_rate * market_count_factor

        total_pnl = total_ret - total_inv
        roi = total_pnl / total_inv if total_inv > 0 else 0
        roi_score = max(0, min(1, (roi + 0.2) / 1.0))

        sharpness = w_clv * clv_score + w_timing * timing_score + w_consist * consistency_score + w_roi * roi_score

        if sharpness >= elite_score and total_pnl > 0 and avg_clv > 0.02 and distinct >= elite_markets:
            tier = "elite"
        elif sharpness >= sharp_score and (total_pnl > 0 or (avg_clv > 0.05 and distinct >= 20)):
            tier = "sharp"
        elif sharpness >= 0.25:
            tier = "average"
        else:
            tier = "dull"

        scores[wallet] = {
            "tier": tier,
            "sharpness": sharpness,
            "avg_clv": avg_clv,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "distinct_markets": distinct,
        }

    return scores


def backtest_score_wallets_v2(trades, weights, min_markets=8):
    """
    Alternative scoring: 3-factor model with informational edge (CLV + time-weighted price alpha),
    consistency, and ROI.
    """
    w_edge, w_consist, w_roi = weights

    wallet_trades = defaultdict(list)
    for t in trades:
        wallet_trades[t["wallet"]].append(t)

    # Get market time ranges for time-lead calculation
    market_times = defaultdict(lambda: {"min_ts": float("inf"), "max_ts": 0})
    for t in trades:
        cid = t["condition_id"]
        ts = t.get("timestamp", 0) or 0
        if ts > 0:
            market_times[cid]["min_ts"] = min(market_times[cid]["min_ts"], ts)
            market_times[cid]["max_ts"] = max(market_times[cid]["max_ts"], ts)

    scores = {}
    for wallet, wtrades in wallet_trades.items():
        market_groups = defaultdict(list)
        for t in wtrades:
            market_groups[t["condition_id"]].append(t)

        distinct = len(market_groups)
        if distinct < min_markets:
            continue

        edge_scores = []
        market_pnls = {}
        total_inv = 0.0
        total_ret = 0.0
        trade_count = 0

        for cid, mtrades in market_groups.items():
            mpnl = 0.0
            mt = market_times.get(cid, {})
            market_duration = mt.get("max_ts", 0) - mt.get("min_ts", 0)
            if market_duration <= 0:
                market_duration = 1

            for t in mtrades:
                price = t["price"]
                side = t["side"]
                size = t["size"]
                outcome = (t.get("outcome") or "").lower()
                win_outcome = (t.get("winning_outcome") or "").lower()
                ts = t.get("timestamp", 0) or 0
                if not win_outcome:
                    continue

                outcome_won = (outcome == win_outcome)
                res_price = 1.0 if outcome_won else 0.0
                trade_count += 1

                # Price alpha
                if side == "BUY":
                    if price >= 0.95 and outcome_won:
                        continue
                    price_alpha = res_price - price
                    pnl = (res_price - price) * size
                    total_inv += price * size
                    total_ret += res_price * size
                elif side == "SELL":
                    price_alpha = price - res_price
                    pnl = (price - res_price) * size
                    total_inv += (1.0 - price) * size
                    total_ret += (1.0 - res_price) * size
                else:
                    continue

                mpnl += pnl

                # Time lead factor: how early in the market's life did they enter?
                # 1.0 = very early, 0.1 = very late
                if ts > 0 and mt.get("min_ts", 0) > 0:
                    time_position = (ts - mt["min_ts"]) / market_duration  # 0=earliest, 1=latest
                    time_lead = max(0.1, 1.0 - time_position * 0.9)
                else:
                    time_lead = 0.5

                # Informational edge = price_alpha weighted by time lead
                edge_scores.append(price_alpha * time_lead)

            market_pnls[cid] = mpnl

        if not edge_scores or trade_count == 0:
            continue

        avg_edge = float(np.mean(edge_scores))
        # Normalize: -0.1 = 0, +0.4 = 1
        edge_score = max(0, min(1, (avg_edge + 0.1) / 0.5))

        mkt_won = sum(1 for p in market_pnls.values() if p > 0)
        win_rate = mkt_won / len(market_pnls) if market_pnls else 0
        market_count_factor = min(1.0, (distinct / 100) ** 0.5)
        consistency_score = win_rate * market_count_factor

        total_pnl = total_ret - total_inv
        roi = total_pnl / total_inv if total_inv > 0 else 0
        roi_score = max(0, min(1, (roi + 0.2) / 1.0))

        sharpness = w_edge * edge_score + w_consist * consistency_score + w_roi * roi_score

        if sharpness >= 0.65 and total_pnl > 0 and avg_edge > 0.02 and distinct >= 25:
            tier = "elite"
        elif sharpness >= 0.45 and (total_pnl > 0 or (avg_edge > 0.05 and distinct >= 20)):
            tier = "sharp"
        elif sharpness >= 0.25:
            tier = "average"
        else:
            tier = "dull"

        scores[wallet] = {
            "tier": tier,
            "sharpness": sharpness,
            "avg_edge": avg_edge,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "distinct_markets": distinct,
        }

    return scores


def evaluate_tiers(wallet_scores, test_trades):
    """
    Given wallet tier assignments and test-set trades, calculate performance metrics per tier.
    Returns EV per dollar, win rate, trade count, and profit per tier.
    """
    tier_stats = {}
    for tier in ["elite", "sharp", "average", "dull", "unscored"]:
        tier_stats[tier] = {
            "trades": 0,
            "wins": 0,
            "total_invested": 0.0,
            "total_returned": 0.0,
            "wallets": 0,
        }

    # Count wallets per tier
    tier_wallet_counts = defaultdict(int)
    for w, data in wallet_scores.items():
        tier_wallet_counts[data["tier"]] += 1
    for tier, count in tier_wallet_counts.items():
        if tier in tier_stats:
            tier_stats[tier]["wallets"] = count

    # Evaluate each test trade
    for t in test_trades:
        wallet = t["wallet"]
        price = t["price"]
        side = t["side"]
        size = t["size"]
        outcome = (t.get("outcome") or "").lower()
        win_outcome = (t.get("winning_outcome") or "").lower()

        if not win_outcome or price <= 0 or price >= 1.0:
            continue

        tier = wallet_scores.get(wallet, {}).get("tier", "unscored")
        if tier not in tier_stats:
            tier = "unscored"

        outcome_won = (outcome == win_outcome)
        res_price = 1.0 if outcome_won else 0.0

        if side == "BUY":
            if price >= 0.95 and outcome_won:
                continue
            invested = price * size
            returned = res_price * size
        elif side == "SELL":
            invested = (1.0 - price) * size
            returned = (1.0 - res_price) * size
        else:
            continue

        tier_stats[tier]["trades"] += 1
        tier_stats[tier]["total_invested"] += invested
        tier_stats[tier]["total_returned"] += returned
        if returned > invested:
            tier_stats[tier]["wins"] += 1

    # Calculate final metrics
    results = {}
    for tier, s in tier_stats.items():
        if s["trades"] == 0:
            results[tier] = {
                "wallets": s["wallets"],
                "trades": 0,
                "win_rate": None,
                "ev_per_dollar": None,
                "total_profit": 0,
            }
        else:
            results[tier] = {
                "wallets": s["wallets"],
                "trades": s["trades"],
                "win_rate": round(s["wins"] / s["trades"], 4),
                "ev_per_dollar": round(s["total_returned"] / s["total_invested"], 4) if s["total_invested"] > 0 else None,
                "total_profit": round(s["total_returned"] - s["total_invested"], 2),
            }

    return results


def evaluate_divergence(wallet_scores, test_trades):
    """
    For each resolved market in the test set, check if sharp and dull money disagreed.
    When they did, who won?
    """
    sharp_tiers = {"elite", "sharp"}
    dull_tiers = {"dull"}

    # Group test trades by market
    market_trades = defaultdict(list)
    for t in test_trades:
        if (t.get("winning_outcome") or ""):
            market_trades[t["condition_id"]].append(t)

    results = {"total_markets": 0, "divergent_markets": 0, "sharp_won": 0, "dull_won": 0, "tie": 0}

    for cid, trades in market_trades.items():
        sharp_yes = 0.0
        sharp_no = 0.0
        dull_yes = 0.0
        dull_no = 0.0

        win_outcome = ""
        for t in trades:
            wallet = t["wallet"]
            tier = wallet_scores.get(wallet, {}).get("tier", "unscored")
            outcome = (t.get("outcome") or "").lower()
            notional = t["price"] * t["size"]
            win_outcome = (t.get("winning_outcome") or "").lower()

            if tier in sharp_tiers:
                if outcome == "yes":
                    sharp_yes += notional
                else:
                    sharp_no += notional
            elif tier in dull_tiers:
                if outcome == "yes":
                    dull_yes += notional
                else:
                    dull_no += notional

        if not win_outcome:
            continue

        sharp_total = sharp_yes + sharp_no
        dull_total = dull_yes + dull_no
        if sharp_total < 10 or dull_total < 10:
            continue

        results["total_markets"] += 1
        sharp_pct_yes = sharp_yes / sharp_total
        dull_pct_yes = dull_yes / dull_total

        # Divergence = they disagree by 30%+
        if abs(sharp_pct_yes - dull_pct_yes) < 0.30:
            continue

        results["divergent_markets"] += 1
        sharp_direction = "yes" if sharp_pct_yes > 0.5 else "no"
        dull_direction = "yes" if dull_pct_yes > 0.5 else "no"

        if sharp_direction == win_outcome:
            results["sharp_won"] += 1
        elif dull_direction == win_outcome:
            results["dull_won"] += 1
        else:
            results["tie"] += 1

    if results["divergent_markets"] > 0:
        results["sharp_win_rate"] = round(results["sharp_won"] / results["divergent_markets"], 4)
    else:
        results["sharp_win_rate"] = None

    return results


@app.get("/api/backtest")
def run_backtest(
    cutoff_months: int = Query(3, description="Months ago for train/test split"),
    min_markets: int = Query(8, description="Minimum markets for a wallet to qualify"),
    elite_score: float = Query(0.55, description="Minimum sharpness for elite tier"),
    elite_markets: int = Query(12, description="Minimum markets for elite tier"),
    sharp_score: float = Query(0.38, description="Minimum sharpness for sharp tier"),
):
    """
    Run a time-split backtest:
    1. Score wallets on training data (before cutoff)
    2. Evaluate their test-set performance (after cutoff)
    3. Compare V1 (4-factor) vs V2 (3-factor) scoring
    4. Test divergence signal accuracy
    """
    if not DATABASE_URL:
        return {"error": "No database configured"}

    logger.info(f"Running backtest: cutoff={cutoff_months}mo, min_markets={min_markets}")

    # Load all trades from DB (no time limit — we'll split ourselves)
    conn = db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT wallet, condition_id, side, outcome, price, size,
               timestamp, title, category, winning_outcome
        FROM trades WHERE winning_outcome IS NOT NULL AND winning_outcome != ''
        AND price > 0 AND size > 0
    """)
    all_trades = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()

    if not all_trades:
        return {"error": "No resolved trades in database"}

    # Split by time
    cutoff_ts = int((datetime.utcnow() - timedelta(days=cutoff_months * 30)).timestamp())
    train_trades = [t for t in all_trades if (t.get("timestamp") or 0) < cutoff_ts and (t.get("timestamp") or 0) > 0]
    test_trades = [t for t in all_trades if (t.get("timestamp") or 0) >= cutoff_ts]

    if len(train_trades) < 1000:
        return {"error": f"Not enough training data: {len(train_trades)} trades before cutoff"}
    if len(test_trades) < 500:
        return {"error": f"Not enough test data: {len(test_trades)} trades after cutoff"}

    logger.info(f"  Train: {len(train_trades):,} trades | Test: {len(test_trades):,} trades")

    # --- V1: Current 4-factor scoring ---
    v1_weights = (0.40, 0.25, 0.20, 0.15)
    v1_scores = backtest_score_wallets(train_trades, v1_weights, min_markets, elite_score, elite_markets, sharp_score)
    v1_tiers = evaluate_tiers(v1_scores, test_trades)
    v1_divergence = evaluate_divergence(v1_scores, test_trades)

    # --- V2: 3-factor informational edge scoring ---
    v2_weights = (0.50, 0.30, 0.20)
    v2_scores = backtest_score_wallets_v2(train_trades, v2_weights, min_markets)
    v2_tiers = evaluate_tiers(v2_scores, test_trades)
    v2_divergence = evaluate_divergence(v2_scores, test_trades)

    # --- V1 alternative weights ---
    alt_configs = [
        {"label": "CLV Heavy (60/15/15/10)", "weights": (0.60, 0.15, 0.15, 0.10)},
        {"label": "Balanced (25/25/25/25)", "weights": (0.25, 0.25, 0.25, 0.25)},
        {"label": "CLV+Timing (45/35/10/10)", "weights": (0.45, 0.35, 0.10, 0.10)},
    ]
    alt_results = []
    for cfg in alt_configs:
        alt_scores = backtest_score_wallets(train_trades, cfg["weights"], min_markets, elite_score, elite_markets, sharp_score)
        alt_tiers = evaluate_tiers(alt_scores, test_trades)
        # Key metric: elite EV per dollar
        elite_ev = alt_tiers.get("elite", {}).get("ev_per_dollar")
        sharp_ev = alt_tiers.get("sharp", {}).get("ev_per_dollar")
        dull_ev = alt_tiers.get("dull", {}).get("ev_per_dollar")
        alt_results.append({
            "label": cfg["label"],
            "weights": cfg["weights"],
            "elite_ev": elite_ev,
            "sharp_ev": sharp_ev,
            "dull_ev": dull_ev,
            "elite_trades": alt_tiers.get("elite", {}).get("trades", 0),
            "sharp_trades": alt_tiers.get("sharp", {}).get("trades", 0),
            "tier_counts": {t: alt_tiers[t]["wallets"] for t in alt_tiers},
        })

    # --- Baseline: what if you just followed market consensus? ---
    baseline_invested = 0.0
    baseline_returned = 0.0
    for t in test_trades:
        price = t["price"]
        outcome = (t.get("outcome") or "").lower()
        win_outcome = (t.get("winning_outcome") or "").lower()
        size = t["size"]
        if not win_outcome or price <= 0 or price >= 1.0:
            continue
        # Market consensus = buying at current price
        outcome_won = (outcome == win_outcome)
        res_price = 1.0 if outcome_won else 0.0
        baseline_invested += price * size
        baseline_returned += res_price * size

    baseline_ev = round(baseline_returned / baseline_invested, 4) if baseline_invested > 0 else None

    cutoff_date = (datetime.utcnow() - timedelta(days=cutoff_months * 30)).strftime("%Y-%m-%d")

    result = {
        "config": {
            "cutoff_date": cutoff_date,
            "cutoff_months": cutoff_months,
            "min_markets": min_markets,
            "elite_score": elite_score,
            "elite_markets": elite_markets,
            "sharp_score": sharp_score,
            "train_trades": len(train_trades),
            "test_trades": len(test_trades),
        },
        "baseline": {
            "ev_per_dollar": baseline_ev,
            "description": "Average EV if you copied every trade at its entry price",
        },
        "v1_current_model": {
            "name": "4-Factor (CLV 40%, Timing 25%, Consistency 20%, ROI 15%)",
            "tiers": v1_tiers,
            "divergence": v1_divergence,
        },
        "v2_edge_model": {
            "name": "3-Factor (Informational Edge 50%, Consistency 30%, ROI 20%)",
            "tiers": v2_tiers,
            "divergence": v2_divergence,
        },
        "weight_experiments": alt_results,
        "interpretation": {
            "ev_per_dollar": "Above 1.0 = profitable. 1.10 means 10% return per dollar invested.",
            "divergence_sharp_win_rate": "When sharp & dull disagree by 30%+, how often does the sharp side win?",
            "what_to_look_for": "Elite EV should be highest, Dull lowest. Bigger gap = better scoring system. Sharp win rate in divergence > 55% = real signal.",
        },
    }

    logger.info(f"Backtest complete. V1 elite EV: {v1_tiers.get('elite', {}).get('ev_per_dollar')} | V2 elite EV: {v2_tiers.get('elite', {}).get('ev_per_dollar')}")

    return result


# ================================================================
# REACT SPA DASHBOARD
# ================================================================

# Read the SPA HTML from file at startup, or use inline fallback
import pathlib as _pathlib
_spa_path = _pathlib.Path(__file__).parent / "spa.html"
if _spa_path.exists():
    REACT_SPA_HTML = _spa_path.read_text()
else:
    REACT_SPA_HTML = "<html><body style='background:#060d14;color:#c5d0de;padding:40px'>SharpFlow - spa.html not found</body></html>"


@app.get("/wallet/{address}", response_class=HTMLResponse)
def wallet_detail_page(address: str):
    return HTMLResponse(REACT_SPA_HTML)


@app.get("/markets", response_class=HTMLResponse)
def markets_page():
    return HTMLResponse(REACT_SPA_HTML)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(REACT_SPA_HTML)


# ================================================================
# EVENT-DRIVEN LIVE SCORING
# ================================================================

# In-memory cache for live wallet scores (updated on each trade)
live_wallet_scores = {}  # wallet -> score dict
live_wallet_scores_lock = threading.Lock()
live_convergence_signals = []  # latest live convergence signals
live_scoring_stats = {
    "trades_processed": 0,
    "wallets_rescored": 0,
    "convergence_checks": 0,
    "last_trade_time": None,
    "errors": 0,
}


def db_get_live_trades_for_wallet(wallet, months_back=2):
    """Pull all trades for a specific wallet from live_trades within the scoring window."""
    if not DATABASE_URL:
        return []
    try:
        conn = db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cutoff = int((datetime.utcnow() - timedelta(days=months_back * 30)).timestamp())
        cur.execute("""
            SELECT wallet, condition_id, side, outcome, price, size,
                   timestamp, title, category
            FROM live_trades
            WHERE wallet = %s AND timestamp > %s
            ORDER BY timestamp DESC
        """, (wallet, cutoff))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Error loading live trades for {wallet[:10]}: {e}")
        return []


def db_get_live_trades_for_market(condition_id, months_back=2):
    """Pull all trades for a specific market from live_trades."""
    if not DATABASE_URL:
        return []
    try:
        conn = db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cutoff = int((datetime.utcnow() - timedelta(days=months_back * 30)).timestamp())
        cur.execute("""
            SELECT wallet, condition_id, side, outcome, price, size,
                   timestamp, title, category
            FROM live_trades
            WHERE condition_id = %s AND timestamp > %s
            ORDER BY timestamp DESC
        """, (condition_id, cutoff))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Error loading live trades for market {condition_id[:10]}: {e}")
        return []


def score_single_wallet(wallet, trades):
    """Score a single wallet using the same algorithm as score_all_wallets.
    Returns a score dict or None if wallet doesn't meet thresholds."""

    # Group by market, filtering dust trades
    market_groups = defaultdict(list)
    for t in trades:
        notional = t["price"] * t["size"]
        if notional < MIN_TRADE_NOTIONAL:
            continue
        cid = t.get("condition_id", "")
        if not cid:
            continue
        market_groups[cid].append(t)

    distinct = len(market_groups)
    if distinct < MIN_MARKETS:
        return None

    clvs = []
    timing_scores = []
    market_pnls = {}
    total_inv = 0.0
    total_ret = 0.0
    categories = set()
    trade_count = 0

    for cid, mtrades in market_groups.items():
        mpnl = 0.0
        for t in mtrades:
            price = t["price"]
            side = t["side"]
            size = t["size"]
            outcome = (t.get("outcome") or "").lower()
            win_outcome = (t.get("winning_outcome") or "").lower()
            categories.add(t.get("category", "other"))

            # For live trades, we don't have winning_outcome yet (market not resolved)
            # Use market resolution from the markets table if available
            if not win_outcome:
                continue

            outcome_won = (outcome == win_outcome)
            res_price = 1.0 if outcome_won else 0.0
            trade_count += 1

            if side == "BUY":
                if price >= 0.95 and outcome_won:
                    continue
                clvs.append(res_price - price)
                timing_scores.append((0.5 - price) if outcome_won else (price - 0.5))
                pnl = (res_price - price) * size
                total_inv += price * size
                total_ret += res_price * size
                mpnl += pnl
            elif side == "SELL":
                clvs.append(price - res_price)
                timing_scores.append((price - 0.5) if not outcome_won else (0.5 - price))
                pnl = (price - res_price) * size
                total_inv += (1.0 - price) * size
                total_ret += (1.0 - res_price) * size
                mpnl += pnl

        market_pnls[cid] = mpnl

    if not clvs or trade_count == 0:
        return None

    avg_clv = float(np.mean(clvs))
    clv_score = max(0, min(1, (avg_clv + 0.1) / 0.52))

    avg_timing = float(np.mean(timing_scores)) if timing_scores else 0
    early_pct = sum(1 for t in timing_scores if t > 0) / len(timing_scores) if timing_scores else 0
    timing_score = max(0, min(1, (early_pct - 0.1) / 0.6))

    mkt_won = sum(1 for p in market_pnls.values() if p > 0)
    mkt_lost = sum(1 for p in market_pnls.values() if p <= 0)
    win_rate = mkt_won / len(market_pnls) if market_pnls else 0
    market_count_factor = min(1.0, (distinct / 100) ** 0.5)
    consistency_score = win_rate * market_count_factor

    total_pnl = total_ret - total_inv
    roi = total_pnl / total_inv if total_inv > 0 else 0
    roi_score = max(0, min(1, (roi + 0.2) / 1.0))

    sharpness = W_CLV * clv_score + W_TIMING * timing_score + W_CONSISTENCY * consistency_score + W_ROI * roi_score

    if sharpness >= 0.38 and total_pnl > 0 and avg_clv > 0:
        tier = "sharp"
    elif sharpness >= 0.25:
        tier = "average"
    else:
        tier = "dull"

    return {
        "wallet": wallet,
        "display_name": wallet[:12] + "...",
        "sharpness_score": round(sharpness, 4),
        "tier": tier,
        "clv_score": round(clv_score, 4),
        "timing_score": round(timing_score, 4),
        "consistency_score": round(consistency_score, 4),
        "roi_score": round(roi_score, 4),
        "avg_clv": round(avg_clv, 4),
        "avg_timing_alpha": round(avg_timing, 4),
        "early_entry_pct": round(early_pct, 4),
        "win_rate": round(win_rate, 4),
        "roi": round(roi, 4),
        "total_pnl": round(total_pnl, 2),
        "total_invested": round(total_inv, 2),
        "total_trades": trade_count,
        "distinct_markets": distinct,
        "markets_won": mkt_won,
        "markets_lost": mkt_lost,
        "categories": sorted(categories),
        "scored_at": datetime.utcnow().isoformat(),
        "source": "live",
    }


def check_live_convergence(condition_id):
    """Check if a market has convergence from live-scored sharp wallets.
    Lightweight version — only checks the specific market."""

    with live_wallet_scores_lock:
        sharp_wallets = {w: s for w, s in live_wallet_scores.items() if s["sharpness_score"] >= 0.38}

    if len(sharp_wallets) < 3:
        return None

    # Get all live trades for this market
    market_trades = db_get_live_trades_for_market(condition_id)
    if not market_trades:
        return None

    # Group by wallet and direction
    yes_wallets = []
    no_wallets = []

    for t in market_trades:
        wallet = t["wallet"]
        if wallet not in sharp_wallets:
            continue

        notional = t["price"] * t["size"]
        if notional < 50:
            continue

        wdata = sharp_wallets[wallet]
        entry = {
            "wallet": wallet,
            "display_name": wdata["display_name"],
            "sharpness_score": wdata["sharpness_score"],
            "tier": wdata["tier"],
            "size": t["size"],
            "price": t["price"],
            "notional": round(notional, 2),
            "entry_time": t.get("timestamp", 0),
        }

        side = t["side"].upper()
        if side == "BUY":
            yes_wallets.append(entry)
        else:
            no_wallets.append(entry)

    # Check both directions
    signals = []
    title = market_trades[0].get("title", condition_id[:20]) if market_trades else condition_id[:20]

    for direction, wallets, opposition in [("YES", yes_wallets, no_wallets), ("NO", no_wallets, yes_wallets)]:
        # Deduplicate by wallet (keep largest position)
        seen = {}
        for w in wallets:
            addr = w["wallet"]
            if addr not in seen or w["size"] > seen[addr]["size"]:
                seen[addr] = w
        unique = list(seen.values())

        if len(unique) < 3:
            continue

        combined_score = sum(w["sharpness_score"] for w in unique)
        if combined_score < 1.0:
            continue

        total_notional = sum(w["notional"] for w in unique)
        if total_notional < 200:
            continue

        opp_seen = {}
        for w in opposition:
            if w["wallet"] not in opp_seen:
                opp_seen[w["wallet"]] = w
        opp_score = sum(w["sharpness_score"] for w in opp_seen.values())
        agreement = combined_score / (combined_score + opp_score) if (combined_score + opp_score) > 0 else 1.0

        sharp_count = sum(1 for w in unique if w["tier"] == "sharp")
        total_size = sum(w["size"] for w in unique)

        # Signal decay: compute freshness from entry timestamps
        now_epoch = int(time.time())
        entry_times = [w.get("entry_time", 0) for w in unique if w.get("entry_time", 0) > 0]
        if entry_times:
            earliest_entry = min(entry_times)
            latest_entry = max(entry_times)
            avg_age_seconds = now_epoch - (sum(entry_times) / len(entry_times))
            avg_age_hours = avg_age_seconds / 3600

            # Freshness: 1.0 = just entered, decays to 0 over 72 hours (half-life ~24h)
            # Using exponential decay: freshness = e^(-age / half_life)
            half_life_hours = 24
            freshness = math.exp(-avg_age_hours * math.log(2) / half_life_hours)
            freshness = round(max(0, min(1, freshness)), 4)
        else:
            earliest_entry = 0
            latest_entry = 0
            avg_age_hours = 0
            freshness = 0.5  # unknown age, assume mid

        strength = (
            0.20 * min(1, len(unique) / 5) +
            0.20 * min(1, combined_score / 2.5) +
            0.20 * agreement +
            0.20 * min(1, total_notional / 5000) +
            0.20 * freshness  # fresh signals are stronger
        )

        if sharp_count >= 5 and len(unique) >= 6:
            sig_type = "STRONG_CONVERGENCE"
        elif sharp_count >= 3 and len(unique) >= 4:
            sig_type = "CONVERGENCE"
        else:
            sig_type = "MILD_CONVERGENCE"

        signals.append({
            "market_id": condition_id,
            "market_title": title,
            "side": direction,
            "signal_type": sig_type,
            "strength": round(strength, 4),
            "wallet_count": len(unique),
            "sharp_count": sharp_count,
            "combined_sharpness": round(combined_score, 4),
            "opposition_score": round(opp_score, 4),
            "agreement_ratio": round(agreement, 4),
            "total_size": round(total_size, 2),
            "total_notional": round(total_notional, 2),
            "freshness": freshness,
            "avg_age_hours": round(avg_age_hours, 1),
            "earliest_entry": earliest_entry,
            "latest_entry": latest_entry,
            "wallets": sorted(unique, key=lambda w: w["sharpness_score"], reverse=True)[:10],
            "source": "live",
            "detected_at": datetime.utcnow().isoformat(),
        })

    return signals if signals else None


def on_live_trades(trades_batch):
    """Callback from live listener. Uses batch-scored wallet data to detect
    real-time convergence when sharp wallets trade on open markets.
    
    Flow:
    1. Trade comes in → check if wallet is already scored (from batch pipeline)
    2. If wallet is sharp → check if this market now has convergence
    3. Update dashboard state with any new signals
    
    The batch pipeline provides the "who is sharp" baseline.
    The live pipeline provides the "what are they doing right now" signals.
    """

    if not trades_batch:
        return

    live_scoring_stats["trades_processed"] += len(trades_batch)
    live_scoring_stats["last_trade_time"] = datetime.utcnow().isoformat()

    # Collect unique wallets and markets from this batch
    affected_markets = set()
    sharp_trades_in_batch = 0

    with live_wallet_scores_lock:
        for t in trades_batch:
            wallet = t["wallet"]
            cid = t.get("condition_id", "")
            if not cid:
                continue

            # Check if this wallet is known sharp (from batch scoring)
            wallet_score = live_wallet_scores.get(wallet)
            if wallet_score and wallet_score.get("sharpness_score", 0) >= 0.38:
                affected_markets.add(cid)
                sharp_trades_in_batch += 1

    if sharp_trades_in_batch > 0:
        live_scoring_stats["wallets_rescored"] += sharp_trades_in_batch

    # Check convergence for markets where sharp wallets just traded
    for cid in affected_markets:
        try:
            signals = check_live_convergence(cid)
            if signals:
                live_scoring_stats["convergence_checks"] += 1
                with live_wallet_scores_lock:
                    # Remove old signals for this market, add new ones
                    live_convergence_signals[:] = [
                        s for s in live_convergence_signals if s["market_id"] != cid
                    ] + signals

                # Also update main state convergence signals
                if state.get("convergence_signals") is not None:
                    state["convergence_signals"] = [
                        s for s in state["convergence_signals"]
                        if not (s.get("market_id") == cid and s.get("source") == "live")
                    ] + signals

                logger.info(f"Live convergence detected on {signals[0]['market_title'][:40]}: "
                           f"{signals[0]['signal_type']} {signals[0]['side']} "
                           f"({signals[0]['wallet_count']} wallets, strength {signals[0]['strength']})")

        except Exception as e:
            live_scoring_stats["errors"] += 1
            logger.error(f"Live convergence check error for {cid[:10]}: {e}")


# ================================================================
# RETENTION POLICY
# ================================================================

LIVE_TRADES_RETENTION_DAYS = 180  # 6 months

def run_retention_cleanup():
    """Delete live_trades older than retention window and log results."""
    if not DATABASE_URL:
        return
    try:
        conn = db_conn()
        conn.autocommit = True
        cur = conn.cursor()

        # Get count before cleanup
        cur.execute("SELECT COUNT(*) FROM live_trades")
        before_count = cur.fetchone()[0]

        if before_count == 0:
            logger.info("Retention cleanup: live_trades is empty, nothing to do")
            conn.close()
            return

        # Delete rows older than retention window (timestamp is epoch bigint)
        cutoff_epoch = int(time.time()) - (LIVE_TRADES_RETENTION_DAYS * 86400)
        cur.execute("DELETE FROM live_trades WHERE timestamp < %s", (cutoff_epoch,))
        deleted = cur.rowcount

        # Get count after cleanup
        cur.execute("SELECT COUNT(*) FROM live_trades")
        after_count = cur.fetchone()[0]

        # Reclaim disk space if we deleted a significant amount
        if deleted > 10000:
            logger.info(f"Retention: running VACUUM on live_trades after deleting {deleted} rows...")
            cur.execute("VACUUM live_trades")

        # Log table size
        cur.execute("SELECT pg_size_pretty(pg_total_relation_size('live_trades'))")
        table_size = cur.fetchone()[0]

        conn.close()
        logger.info(
            f"Retention cleanup complete: {before_count} → {after_count} rows "
            f"({deleted} deleted), table size: {table_size}"
        )
    except Exception as e:
        logger.error(f"Retention cleanup failed: {e}")


# ================================================================
# STARTUP
# ================================================================

def background_pipeline():
    """Run pipeline on startup and then on schedule."""
    # Initialize database
    try:
        db_init()
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    # Run retention cleanup on startup
    run_retention_cleanup()

    # Initial run
    run_pipeline()

    # Schedule refreshes every 1 hour, retention cleanup every 24 hours
    cycles = 0
    while True:
        time.sleep(1 * 3600)  # 1 hour
        cycles += 1
        logger.info("Running scheduled refresh...")
        run_pipeline()

        # Run retention cleanup once per day (every 24 cycles)
        if cycles % 24 == 0:
            logger.info("Running nightly retention cleanup...")
            run_retention_cleanup()


# Start pipeline in background thread
pipeline_thread = threading.Thread(target=background_pipeline, daemon=True)
pipeline_thread.start()

# Start live trade listener if Alchemy URL is configured
ALCHEMY_URL = os.getenv("ALCHEMY_URL", "")
if ALCHEMY_URL and DATABASE_URL:
    try:
        from live_listener import start_listener, get_listener_status, _listener_instance
        listener = start_listener(DATABASE_URL, ALCHEMY_URL, poll_interval=5)
        listener.set_on_trades_callback(on_live_trades)
        logger.info("Live trade listener started with event-driven scoring callback")
    except Exception as e:
        logger.error(f"Failed to start live listener: {e}")
else:
    logger.info("Live listener disabled (set ALCHEMY_URL to enable)")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
