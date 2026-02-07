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

# Scoring weights
W_CLV = 0.40
W_TIMING = 0.25
W_CONSISTENCY = 0.20
W_ROI = 0.15

MIN_MARKETS = 15          # Minimum distinct resolved markets to score a wallet
MARKETS_TO_FETCH = 300    # How many resolved markets to analyze
TRADES_PER_MARKET = 500   # Max trades per market
MIN_VOLUME = 5000         # Minimum market volume (USD)
API_DELAY = 0.35          # Seconds between API calls

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

# Telegram (optional)
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

# ================================================================
# STATE
# ================================================================
state = {
    "scored_wallets": [],
    "convergence_signals": [],
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
    """Categorize a market as sports, politics, or other."""
    q = question.lower()
    t = tags_str.lower()
    if any(kw in q or kw in t for kw in SPORTS_KW):
        return "sports"
    if any(kw in q or kw in t for kw in POLITICS_KW):
        return "politics"
    return "other"


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

            cat = categorize(m.get("question", ""), " ".join(tags))

            all_markets.append({
                "condition_id": cid,
                "question": m.get("question", ""),
                "slug": m.get("slug", ""),
                "volume": vol,
                "category": cat,
                "winning_outcome": winning,
                "tokens": tokens,
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
        # Group by market
        market_groups = defaultdict(list)
        for t in wtrades:
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

        # Tier thresholds — tuned for real Polymarket data distributions
        # Real-world scores cluster lower than synthetic tests
        tier = "elite" if sharpness >= 0.55 else "sharp" if sharpness >= 0.38 else "average" if sharpness >= 0.22 else "dull"

        sample = wtrades[0]
        name = sample.get("pseudonym") or sample.get("name") or wallet[:12] + "..."

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
            "scored_at": datetime.utcnow().isoformat(),
        })

    results.sort(key=lambda x: x["sharpness_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    tiers = defaultdict(int)
    for r in results:
        tiers[r["tier"]] += 1
    logger.info(f"Scored {len(results)} wallets: {dict(tiers)}")
    logger.info(f"  Diagnostics: {dict(skip_reasons)}")
    logger.info(f"  Max distinct markets any wallet had: {max_distinct}")
    logger.info(f"  Threshold: {MIN_MARKETS} markets required")
    return results


# ================================================================
# CONVERGENCE DETECTION
# ================================================================

def detect_convergence(scored_wallets, trades):
    """Detect markets where multiple sharp wallets align."""
    logger.info("Detecting convergence signals...")
    state["progress"] = "Detecting convergence..."

    sharp_wallets = {w["wallet"]: w for w in scored_wallets if w["sharpness_score"] >= 0.38}
    if not sharp_wallets:
        return []

    # Group recent trades by market, only from sharp wallets
    market_positions = defaultdict(lambda: {"yes": [], "no": []})

    for t in trades:
        wallet = t["wallet"]
        if wallet not in sharp_wallets:
            continue

        cid = t["condition_id"]
        wdata = sharp_wallets[wallet]
        outcome = t.get("outcome", "").lower()
        side = t.get("side", "BUY")

        entry = {
            "wallet": wallet,
            "display_name": wdata["display_name"],
            "sharpness_score": wdata["sharpness_score"],
            "tier": wdata["tier"],
            "size": t["size"],
            "price": t["price"],
        }

        # Determine direction
        if side == "BUY":
            if outcome in ("yes", ""):
                market_positions[cid]["yes"].append(entry)
            else:
                market_positions[cid]["no"].append(entry)
        else:
            if outcome in ("yes", ""):
                market_positions[cid]["no"].append(entry)
            else:
                market_positions[cid]["yes"].append(entry)

    # Deduplicate (same wallet multiple trades = one entry, keep highest size)
    signals = []
    for cid, pos in market_positions.items():
        for direction, wallets in [("YES", pos["yes"]), ("NO", pos["no"])]:
            # Deduplicate by wallet
            seen = {}
            for w in wallets:
                addr = w["wallet"]
                if addr not in seen or w["size"] > seen[addr]["size"]:
                    seen[addr] = w
            unique_wallets = list(seen.values())

            if len(unique_wallets) < 3:
                continue

            combined_score = sum(w["sharpness_score"] for w in unique_wallets)
            if combined_score < 1.5:
                continue

            elite_count = sum(1 for w in unique_wallets if w["tier"] == "elite")
            total_size = sum(w["size"] for w in unique_wallets)

            # Get opposite side
            opp = pos["no"] if direction == "YES" else pos["yes"]
            opp_seen = {}
            for w in opp:
                addr = w["wallet"]
                if addr not in opp_seen:
                    opp_seen[addr] = w
            opp_score = sum(w["sharpness_score"] for w in opp_seen.values())
            agreement = combined_score / (combined_score + opp_score) if (combined_score + opp_score) > 0 else 1.0

            strength = 0.4 * min(1, len(unique_wallets) / 5) + 0.3 * min(1, combined_score / 3) + 0.3 * agreement

            if elite_count >= 2 and len(unique_wallets) >= 4:
                sig_type = "STRONG_CONVERGENCE"
            elif elite_count >= 1 and len(unique_wallets) >= 3:
                sig_type = "CONVERGENCE"
            else:
                sig_type = "MILD_CONVERGENCE"

            # Find market title from trades
            title = next((t["title"] for t in trades if t["condition_id"] == cid and t["title"]), cid[:16])

            signals.append({
                "market_id": cid,
                "market_title": title,
                "side": direction,
                "signal_type": sig_type,
                "strength": round(strength, 4),
                "wallet_count": len(unique_wallets),
                "elite_count": elite_count,
                "combined_sharpness": round(combined_score, 4),
                "opposition_score": round(opp_score, 4),
                "agreement_ratio": round(agreement, 4),
                "total_size": round(total_size, 2),
                "wallets": sorted(unique_wallets, key=lambda w: w["sharpness_score"], reverse=True)[:10],
            })

    signals.sort(key=lambda s: s["strength"], reverse=True)
    logger.info(f"Found {len(signals)} convergence signals")
    return signals


# ================================================================
# TELEGRAM (OPTIONAL)
# ================================================================

def tg_send(text):
    """Send a Telegram message if configured."""
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Telegram error: {e}")


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

        # Step 3: Score wallets
        scored = score_all_wallets(trades, markets_lookup)
        state["scored_wallets"] = scored

        # Step 4: Detect convergence
        signals = detect_convergence(scored, trades)
        state["convergence_signals"] = signals

        elapsed = time.time() - start
        state["status"] = "ready"
        state["last_refresh"] = datetime.utcnow().isoformat()
        state["progress"] = f"Complete! {len(scored)} wallets scored in {elapsed/60:.1f} min"

        logger.info(f"Pipeline complete in {elapsed/60:.1f} minutes")
        logger.info(f"  Markets: {len(markets)} | Trades: {len(trades)} | Wallets scored: {len(scored)} | Convergence signals: {len(signals)}")

        # Send Telegram notification
        tiers = defaultdict(int)
        for w in scored:
            tiers[w["tier"]] += 1

        tg_send(
            f"✅ <b>SharpFlow Data Refresh Complete</b>\n\n"
            f"📊 Markets analyzed: {len(markets)}\n"
            f"📈 Trades processed: {len(trades):,}\n"
            f"👛 Wallets scored: {len(scored)}\n"
            f"  ⚡ Elite: {tiers['elite']}\n"
            f"  📊 Sharp: {tiers['sharp']}\n"
            f"  ⚖️ Average: {tiers['average']}\n"
            f"  ❌ Dull: {tiers['dull']}\n"
            f"🎯 Convergence signals: {len(signals)}\n"
            f"⏱ Time: {elapsed/60:.1f} min"
        )

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        state["status"] = "error"
        state["error"] = str(e)


# ================================================================
# FASTAPI APP
# ================================================================

app = FastAPI(title="SharpFlow", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/wallets")
def get_wallets(
    sort: str = Query("sharpness"),
    tier: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    wallets = state["scored_wallets"]
    if tier:
        wallets = [w for w in wallets if w["tier"] == tier]
    if category:
        wallets = [w for w in wallets if category in w.get("categories", [])]

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
    for w in state["scored_wallets"]:
        if w["wallet"].lower() == address.lower():
            return w
    return {"error": "Wallet not found"}


@app.get("/api/convergence")
def get_convergence(min_strength: float = Query(0.0)):
    signals = state["convergence_signals"]
    if min_strength > 0:
        signals = [s for s in signals if s["strength"] >= min_strength]
    return {"signals": signals, "total": len(signals)}


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
        "last_refresh": state["last_refresh"],
        "error": state["error"],
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve a built-in dashboard — no separate frontend needed."""
    wallets = state["scored_wallets"]
    signals = state["convergence_signals"]
    status = state["status"]
    progress = state["progress"]

    # Build wallet rows
    wallet_rows = ""
    for w in wallets[:100]:
        tier_color = {"elite": "#00f5a0", "sharp": "#f5c842", "average": "#8a9bb5", "dull": "#f54242"}.get(w["tier"], "#666")
        tier_badge = {"elite": "⚡ ELITE", "sharp": "📊 SHARP", "average": "⚖️ AVG", "dull": "❌ DULL"}.get(w["tier"], "")
        pnl_color = "#00f5a0" if w["total_pnl"] >= 0 else "#f54242"
        pnl_sign = "+" if w["total_pnl"] >= 0 else ""
        pnl_fmt = f"${abs(w['total_pnl']):,.0f}" if abs(w["total_pnl"]) < 1e6 else f"${w['total_pnl']/1e6:,.1f}M"

        cats = ", ".join(w.get("categories", []))
        wallet_short = w["wallet"][:8] + "..." + w["wallet"][-4:]

        wallet_rows += f"""
        <tr onclick="this.querySelector('.detail-row')?.classList.toggle('hidden')" style="cursor:pointer;border-bottom:1px solid #1a2a3a">
          <td style="padding:10px 8px;color:#5a7a9a;font-size:13px">#{w['rank']}</td>
          <td style="padding:10px 8px">
            <div style="font-weight:600;color:#e8edf2;font-size:14px">{w['display_name']}</div>
            <div style="font-size:11px;color:#3a4a5a;font-family:monospace">{wallet_short}</div>
          </td>
          <td style="padding:10px 8px;text-align:center">
            <span style="background:{tier_color}18;color:{tier_color};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700">{tier_badge}</span>
          </td>
          <td style="padding:10px 8px;text-align:right;font-family:monospace;font-size:16px;font-weight:700;color:{tier_color}">{w['sharpness_score']*100:.0f}</td>
          <td style="padding:10px 8px;text-align:right;font-family:monospace;font-size:13px;color:#00d4ff">{w['avg_clv']:+.3f}</td>
          <td style="padding:10px 8px;text-align:right;font-family:monospace;font-size:13px;color:#a855f7">{w['early_entry_pct']:.0%}</td>
          <td style="padding:10px 8px;text-align:right;font-family:monospace;font-size:13px">{w['win_rate']:.0%}</td>
          <td style="padding:10px 8px;text-align:right;font-family:monospace;font-size:13px;color:{pnl_color}">{pnl_sign}{pnl_fmt}</td>
          <td style="padding:10px 8px;text-align:right;font-size:12px;color:#5a7a9a">{w['distinct_markets']}</td>
          <td style="padding:10px 8px;text-align:right;font-size:12px;color:#3a4a5a">{cats}</td>
        </tr>"""

    # Build convergence rows
    signal_rows = ""
    if signals:
        for s in signals[:20]:
            emoji = {"STRONG_CONVERGENCE": "🔴", "CONVERGENCE": "🟡", "MILD_CONVERGENCE": "🟢"}.get(s["signal_type"], "⚪")
            wallet_names = ", ".join(w["display_name"] for w in s["wallets"][:4])
            if len(s["wallets"]) > 4:
                wallet_names += f" +{len(s['wallets'])-4} more"
            signal_rows += f"""
            <tr style="border-bottom:1px solid #1a2a3a">
              <td style="padding:10px 8px;font-size:14px">{emoji}</td>
              <td style="padding:10px 8px;font-size:13px;color:#e8edf2">{s['market_title'][:60]}</td>
              <td style="padding:10px 8px;text-align:center;font-weight:700;color:{'#00f5a0' if s['side']=='YES' else '#f54242'}">{s['side']}</td>
              <td style="padding:10px 8px;text-align:center;font-family:monospace">{s['wallet_count']} ({s['elite_count']}⚡)</td>
              <td style="padding:10px 8px;text-align:center;font-family:monospace">{s['strength']:.2f}</td>
              <td style="padding:10px 8px;text-align:center;font-family:monospace">{s['agreement_ratio']:.0%}</td>
              <td style="padding:10px 8px;font-size:11px;color:#5a7a9a">{wallet_names}</td>
            </tr>"""
    else:
        signal_rows = '<tr><td colspan="7" style="padding:20px;text-align:center;color:#3a4a5a">No convergence signals detected yet</td></tr>'

    # Status banner
    if status == "initializing" or status == "running":
        status_html = f'<div style="background:#f5c84215;border:1px solid #f5c84230;color:#f5c842;padding:14px 20px;border-radius:8px;margin-bottom:20px;font-size:14px">⏳ <b>Pipeline running...</b> {progress}</div>'
    elif status == "error":
        status_html = f'<div style="background:#f5424215;border:1px solid #f5424230;color:#f54242;padding:14px 20px;border-radius:8px;margin-bottom:20px;font-size:14px">❌ <b>Error:</b> {state["error"]}</div>'
    else:
        status_html = f'<div style="background:#00f5a015;border:1px solid #00f5a030;color:#00f5a0;padding:14px 20px;border-radius:8px;margin-bottom:20px;font-size:14px">✅ <b>Live</b> — Last refresh: {state["last_refresh"] or "never"} — {len(wallets)} wallets scored</div>'

    tiers = defaultdict(int)
    for w in wallets:
        tiers[w["tier"]] += 1

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SharpFlow — Polymarket Intelligence</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#060d14; color:#c5d0de; font-family:-apple-system,system-ui,sans-serif; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ text-align:left; padding:8px; font-size:10px; font-weight:700; color:#3a4a5a; letter-spacing:1.5px; text-transform:uppercase; border-bottom:2px solid #1a2a3a; }}
  .card {{ background:#0a1018; border:1px solid #1a2a3a; border-radius:8px; padding:16px; }}
  .stat {{ text-align:center; }}
  .stat .val {{ font-size:28px; font-weight:800; font-family:monospace; }}
  .stat .lbl {{ font-size:10px; color:#3a4a5a; letter-spacing:1px; margin-top:4px; }}
  a {{ color:#00d4ff; text-decoration:none; }}
  @media(max-width:768px) {{
    .desktop-only {{ display:none; }}
    .stat .val {{ font-size:20px; }}
  }}
</style>
</head><body>
<div style="max-width:1400px;margin:0 auto;padding:20px">

  <!-- Header -->
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:10px">
    <div>
      <h1 style="font-size:22px;font-weight:800;color:#e8edf2;letter-spacing:-0.5px">SHARPFLOW</h1>
      <div style="font-size:12px;color:#4a6070">Polymarket Sharp Wallet Intelligence</div>
    </div>
    <div style="font-size:11px;color:#3a4a5a">
      API: <a href="/api/wallets">/api/wallets</a> ·
      <a href="/api/convergence">/api/convergence</a> ·
      <a href="/api/stats">/api/stats</a>
    </div>
  </div>

  {status_html}

  <!-- Stats -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:24px">
    <div class="card stat"><div class="val" style="color:#5a7a9a">{state['markets_analyzed']}</div><div class="lbl">MARKETS</div></div>
    <div class="card stat"><div class="val" style="color:#5a7a9a">{state['trades_analyzed']:,}</div><div class="lbl">TRADES</div></div>
    <div class="card stat"><div class="val" style="color:#00d4ff">{len(wallets)}</div><div class="lbl">WALLETS SCORED</div></div>
    <div class="card stat"><div class="val" style="color:#00f5a0">{tiers.get('elite',0)}</div><div class="lbl">ELITE</div></div>
    <div class="card stat"><div class="val" style="color:#f5c842">{tiers.get('sharp',0)}</div><div class="lbl">SHARP</div></div>
    <div class="card stat"><div class="val" style="color:#a855f7">{len(signals)}</div><div class="lbl">CONVERGENCE</div></div>
  </div>

  <!-- Convergence Signals -->
  <h2 style="font-size:14px;font-weight:700;color:#a855f7;margin-bottom:10px;letter-spacing:1px">🎯 CONVERGENCE SIGNALS</h2>
  <div class="card" style="margin-bottom:24px;overflow-x:auto">
    <table>
      <tr>
        <th style="width:30px"></th><th>Market</th><th style="text-align:center">Side</th>
        <th style="text-align:center">Wallets</th><th style="text-align:center">Strength</th>
        <th style="text-align:center">Agreement</th><th class="desktop-only">Sharp Wallets</th>
      </tr>
      {signal_rows}
    </table>
  </div>

  <!-- Wallet Leaderboard -->
  <h2 style="font-size:14px;font-weight:700;color:#00d4ff;margin-bottom:10px;letter-spacing:1px">👛 WALLET LEADERBOARD</h2>
  <div class="card" style="overflow-x:auto">
    <table>
      <tr>
        <th style="width:40px">#</th><th>Wallet</th><th style="text-align:center">Tier</th>
        <th style="text-align:right">Score</th><th style="text-align:right">CLV</th>
        <th style="text-align:right" class="desktop-only">Timing</th>
        <th style="text-align:right">Win%</th><th style="text-align:right">PnL</th>
        <th style="text-align:right" class="desktop-only">Markets</th>
        <th class="desktop-only">Category</th>
      </tr>
      {wallet_rows}
    </table>
  </div>

  <div style="margin-top:30px;padding:16px;text-align:center;font-size:11px;color:#2a3a4a;border-top:1px solid #111b26">
    SharpFlow v1.0 · Scoring: 40% CLV · 25% Timing · 20% Consistency · 15% ROI · Not financial advice
  </div>
</div>
</body></html>"""


# ================================================================
# STARTUP
# ================================================================

def background_pipeline():
    """Run pipeline on startup and then on schedule."""
    # Initial run
    run_pipeline()

    # Schedule refreshes every 12 hours
    while True:
        time.sleep(12 * 3600)  # 12 hours
        logger.info("Running scheduled refresh...")
        run_pipeline()


# Start pipeline in background thread
pipeline_thread = threading.Thread(target=background_pipeline, daemon=True)
pipeline_thread.start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
