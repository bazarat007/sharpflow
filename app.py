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

MIN_MARKETS = 8            # Lowered: minimum distinct resolved markets to score a wallet
MARKETS_TO_FETCH = 750     # Increased: how many resolved markets to analyze
TRADES_PER_MARKET = 500    # Max trades per market
MIN_VOLUME = 2000          # Lowered: minimum market volume (USD)
API_DELAY = 0.3            # Slightly faster

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
    "market_intelligence": [],
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

        # Tier assignment with sanity checks:
        # - Elite requires positive PnL AND positive CLV AND 12+ markets
        # - Sharp requires positive PnL OR (positive CLV with 15+ markets)
        # - This prevents wallets with high timing/consistency but no actual edge from ranking up
        if sharpness >= 0.55 and total_pnl > 0 and avg_clv > 0 and distinct >= 12:
            tier = "elite"
        elif sharpness >= 0.38 and (total_pnl > 0 or (avg_clv > 0.05 and distinct >= 15)):
            tier = "sharp"
        elif sharpness >= 0.22:
            tier = "average"
        else:
            tier = "dull"

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
            strength = (
                0.25 * min(1, len(unique_wallets) / 5) +    # More independent wallets = stronger
                0.25 * min(1, combined_score / 2.5) +        # Higher combined sharpness = stronger
                0.25 * agreement +                            # Less opposition = stronger
                0.25 * min(1, total_notional / 5000)          # More money committed = stronger
            )

            if elite_count >= 2 and len(unique_wallets) >= 4:
                sig_type = "STRONG_CONVERGENCE"
            elif elite_count >= 1 or (sharp_count >= 2 and len(unique_wallets) >= 4):
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
                "wallets": sorted(unique_wallets, key=lambda w: w["sharpness_score"], reverse=True)[:10],
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

    # Build wallet lookup by tier
    wallet_lookup = {w["wallet"]: w for w in scored_wallets}
    sharp_addrs = {w["wallet"] for w in scored_wallets if w["tier"] in ("elite", "sharp")}
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
            open_markets[cid] = {
                "condition_id": cid,
                "question": q,
                "slug": m.get("slug", ""),
                "volume": vol,
                "yes_price": yes_price,
                "category": categorize(q, " ".join(
                    t.get("label", "") if isinstance(t, dict) else str(t)
                    for t in (m.get("tags") or [])
                )),
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

    wallets_to_scan = [w for w in scored_wallets if w["tier"] in ("elite", "sharp", "dull")]
    # Limit dull wallet scanning to top 100 by trade count (most active)
    dull_wallets = sorted([w for w in scored_wallets if w["tier"] == "dull"],
                          key=lambda w: w["total_trades"], reverse=True)[:100]
    sharp_wallets_list = [w for w in scored_wallets if w["tier"] in ("elite", "sharp")]
    scan_list = sharp_wallets_list + dull_wallets

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

                entry = {
                    "wallet": wallet_addr,
                    "display_name": wdata["display_name"],
                    "tier": tier,
                    "sharpness_score": wdata["sharpness_score"],
                    "size": size,
                    "avg_price": avg_price,
                    "notional": round(notional, 2),
                }

                is_yes = outcome in ("yes", "y", "1", "")

                if tier in ("elite", "sharp"):
                    if is_yes:
                        market_positions[cid]["sharp_yes"].append(entry)
                    else:
                        market_positions[cid]["sharp_no"].append(entry)
                elif tier == "dull":
                    if is_yes:
                        market_positions[cid]["dull_yes"].append(entry)
                    else:
                        market_positions[cid]["dull_no"].append(entry)

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

        # Need at least 1 sharp wallet to be interesting
        total_sharp = len(sy) + len(sn)
        if total_sharp == 0:
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

        # Count elites
        elite_yes = [e for e in sy if e["tier"] == "elite"]
        elite_no = [e for e in sn if e["tier"] == "elite"]

        intel_results.append({
            "condition_id": cid,
            "question": mkt["question"],
            "slug": mkt["slug"],
            "category": mkt["category"],
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
            "elite_yes": len(elite_yes),
            "elite_no": len(elite_no),
            "sharp_wallets_yes": sorted(sy, key=lambda e: e["sharpness_score"], reverse=True)[:5],
            "sharp_wallets_no": sorted(sn, key=lambda e: e["sharpness_score"], reverse=True)[:5],

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

    logger.info(f"Market intelligence: {len(intel_results)} markets with sharp wallet activity")
    return intel_results


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

        # Step 2.5: Seed from Polymarket leaderboard
        trades = seed_from_leaderboard(trades, markets_lookup)
        state["trades_analyzed"] = len(trades)

        # Step 3: Score wallets
        scored = score_all_wallets(trades, markets_lookup)
        state["scored_wallets"] = scored

        # Step 4: Detect convergence
        signals = detect_convergence(scored, trades)
        state["convergence_signals"] = signals

        # Step 5: Compute market intelligence
        intel = compute_market_intelligence(scored)
        state["market_intelligence"] = intel

        elapsed = time.time() - start
        state["status"] = "ready"
        state["last_refresh"] = datetime.utcnow().isoformat()
        state["progress"] = f"Complete! {len(scored)} wallets scored, {len(intel)} markets analyzed in {elapsed/60:.1f} min"

        logger.info(f"Pipeline complete in {elapsed/60:.1f} minutes")
        logger.info(f"  Markets: {len(markets)} | Trades: {len(trades)} | Wallets scored: {len(scored)} | Convergence: {len(signals)} | Market intel: {len(intel)}")

        # Send Telegram notification
        tiers = defaultdict(int)
        for w in scored:
            tiers[w["tier"]] += 1

        tg_send(
            f"✅ <b>SharpFlow Data Refresh Complete</b>\n\n"
            f"📊 Markets analyzed: {len(markets)}\n"
            f"📈 Trades processed: {len(trades):,}\n"
            f"🐋 Leaderboard wallets seeded\n"
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
    }


@app.get("/wallet/{address}", response_class=HTMLResponse)
def wallet_detail_page(address: str):
    """Wallet detail page with live positions and recent trades."""
    # Get score data
    score = None
    for w in state["scored_wallets"]:
        if w["wallet"].lower() == address.lower():
            score = w
            break

    if not score:
        return HTMLResponse(f"""<html><body style="background:#060d14;color:#c5d0de;font-family:sans-serif;padding:40px">
        <h1>Wallet not found</h1><p>{address}</p><a href="/" style="color:#00d4ff">← Back to dashboard</a></body></html>""")

    tier_color = {"elite": "#00f5a0", "sharp": "#f5c842", "average": "#8a9bb5", "dull": "#f54242"}.get(score["tier"], "#666")
    tier_badge = {"elite": "⚡ ELITE", "sharp": "📊 SHARP", "average": "⚖️ AVG", "dull": "❌ DULL"}.get(score["tier"], "")
    pnl_color = "#00f5a0" if score["total_pnl"] >= 0 else "#f54242"
    pnl_sign = "+" if score["total_pnl"] >= 0 else ""

    # Fetch live data
    positions_raw = api_get(f"{DATA_API}/positions", {"user": address, "sizeThreshold": 1})
    position_rows = ""
    total_pos_value = 0
    if positions_raw and isinstance(positions_raw, list):
        positions = []
        for pos in positions_raw:
            size = float(pos.get("size", 0) or 0)
            avg_price = float(pos.get("avgPrice", 0) or 0)
            if size < 1:
                continue
            notional = size * avg_price
            total_pos_value += notional
            positions.append((notional, pos))

        positions.sort(key=lambda x: x[0], reverse=True)
        for notional, pos in positions[:30]:
            size = float(pos.get("size", 0) or 0)
            avg_price = float(pos.get("avgPrice", 0) or 0)
            outcome = pos.get("outcome", "?")
            title = pos.get("title", "") or pos.get("eventTitle", "Unknown")
            cur_price = float(pos.get("curPrice", 0) or 0)
            pnl_est = (cur_price - avg_price) * size if cur_price > 0 else 0
            pnl_c = "#00f5a0" if pnl_est >= 0 else "#f54242"

            position_rows += f"""
            <tr style="border-bottom:1px solid #1a2a3a">
              <td style="padding:8px;font-size:13px;color:#e8edf2;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{title[:55]}</td>
              <td style="padding:8px;text-align:center;font-weight:700;color:{'#00f5a0' if outcome.lower() in ('yes','y') else '#f54242'}">{outcome}</td>
              <td style="padding:8px;text-align:right;font-family:monospace;font-size:12px">{size:,.0f}</td>
              <td style="padding:8px;text-align:right;font-family:monospace;font-size:12px">${avg_price:.2f}</td>
              <td style="padding:8px;text-align:right;font-family:monospace;font-size:12px;color:#00d4ff">{f'${cur_price:.2f}' if cur_price > 0 else '—'}</td>
              <td style="padding:8px;text-align:right;font-family:monospace;font-size:12px;color:{pnl_c}">{f'${pnl_est:+,.0f}' if cur_price > 0 else '—'}</td>
              <td style="padding:8px;text-align:right;font-family:monospace;font-size:12px;color:#f5c842">${notional:,.0f}</td>
            </tr>"""

    if not position_rows:
        position_rows = '<tr><td colspan="7" style="padding:20px;text-align:center;color:#3a4a5a">No open positions</td></tr>'

    # Fetch recent trades
    trades_raw = api_get(f"{DATA_API}/trades", {"user": address, "limit": 30})
    trade_rows = ""
    if trades_raw and isinstance(trades_raw, list):
        for t in trades_raw:
            price = float(t.get("price", 0) or 0)
            size = float(t.get("size", 0) or 0)
            if price <= 0 or size <= 0:
                continue
            side = t.get("side", "BUY")
            outcome = t.get("outcome", "?")
            title = t.get("title", "Unknown")
            ts = int(t.get("timestamp", 0) or 0)
            time_str = datetime.utcfromtimestamp(ts).strftime("%b %d %H:%M") if ts > 0 else "—"
            side_color = "#00f5a0" if side == "BUY" else "#f54242"

            trade_rows += f"""
            <tr style="border-bottom:1px solid #1a2a3a">
              <td style="padding:8px;font-size:11px;color:#3a4a5a">{time_str}</td>
              <td style="padding:8px;font-weight:700;font-size:12px;color:{side_color}">{side}</td>
              <td style="padding:8px;font-size:13px;color:#e8edf2;max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{title[:50]}</td>
              <td style="padding:8px;text-align:center;font-weight:600;color:{'#00f5a0' if outcome.lower() in ('yes','y') else '#f54242'}">{outcome}</td>
              <td style="padding:8px;text-align:right;font-family:monospace;font-size:12px">${price:.2f}</td>
              <td style="padding:8px;text-align:right;font-family:monospace;font-size:12px">{size:,.0f}</td>
              <td style="padding:8px;text-align:right;font-family:monospace;font-size:12px;color:#f5c842">${price*size:,.0f}</td>
            </tr>"""

    if not trade_rows:
        trade_rows = '<tr><td colspan="7" style="padding:20px;text-align:center;color:#3a4a5a">No recent trades</td></tr>'

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{score['display_name']} — SharpFlow</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#060d14; color:#c5d0de; font-family:-apple-system,system-ui,sans-serif; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ text-align:left; padding:8px; font-size:10px; font-weight:700; color:#3a4a5a; letter-spacing:1.5px; text-transform:uppercase; border-bottom:2px solid #1a2a3a; }}
  .card {{ background:#0a1018; border:1px solid #1a2a3a; border-radius:8px; padding:16px; }}
  .stat {{ text-align:center; }}
  .stat .val {{ font-size:24px; font-weight:800; font-family:monospace; }}
  .stat .lbl {{ font-size:10px; color:#3a4a5a; letter-spacing:1px; margin-top:4px; }}
  a {{ color:#00d4ff; text-decoration:none; }}
</style>
</head><body>
<div style="max-width:1200px;margin:0 auto;padding:20px">

  <a href="/" style="font-size:12px;color:#3a4a5a">← Back to leaderboard</a>

  <div style="display:flex;align-items:center;gap:16px;margin:16px 0 20px 0;flex-wrap:wrap">
    <div>
      <h1 style="font-size:24px;font-weight:800;color:#e8edf2">{score['display_name']}</h1>
      <div style="font-size:12px;color:#3a4a5a;font-family:monospace">{address}</div>
    </div>
    <span style="background:{tier_color}18;color:{tier_color};padding:4px 12px;border-radius:6px;font-size:13px;font-weight:700">{tier_badge}</span>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:10px;margin-bottom:24px">
    <div class="card stat"><div class="val" style="color:{tier_color}">{score['sharpness_score']*100:.0f}</div><div class="lbl">SCORE</div></div>
    <div class="card stat"><div class="val" style="color:#00d4ff">{score['avg_clv']:+.3f}</div><div class="lbl">AVG CLV</div></div>
    <div class="card stat"><div class="val" style="color:#a855f7">{score['early_entry_pct']:.0%}</div><div class="lbl">EARLY ENTRY</div></div>
    <div class="card stat"><div class="val">{score['win_rate']:.0%}</div><div class="lbl">WIN RATE</div></div>
    <div class="card stat"><div class="val" style="color:{pnl_color}">{pnl_sign}${abs(score['total_pnl']):,.0f}</div><div class="lbl">PNL</div></div>
    <div class="card stat"><div class="val" style="color:#5a7a9a">{score['distinct_markets']}</div><div class="lbl">MARKETS</div></div>
    <div class="card stat"><div class="val" style="color:#5a7a9a">{score['total_trades']}</div><div class="lbl">TRADES</div></div>
    <div class="card stat"><div class="val" style="color:#f5c842">${total_pos_value:,.0f}</div><div class="lbl">OPEN VALUE</div></div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px">
    <div class="card stat"><div class="val" style="font-size:16px;color:#00d4ff">{score['clv_score']:.2f}</div><div class="lbl">CLV (40%)</div></div>
    <div class="card stat"><div class="val" style="font-size:16px;color:#a855f7">{score['timing_score']:.2f}</div><div class="lbl">TIMING (25%)</div></div>
    <div class="card stat"><div class="val" style="font-size:16px;color:#f5c842">{score['consistency_score']:.2f}</div><div class="lbl">CONSIST (20%)</div></div>
    <div class="card stat"><div class="val" style="font-size:16px;color:#00f5a0">{score['roi_score']:.2f}</div><div class="lbl">ROI (15%)</div></div>
  </div>

  <h2 style="font-size:14px;font-weight:700;color:#f5c842;margin-bottom:10px;letter-spacing:1px">📂 CURRENT POSITIONS</h2>
  <div class="card" style="margin-bottom:24px;overflow-x:auto">
    <table>
      <tr><th>Market</th><th style="text-align:center">Side</th><th style="text-align:right">Shares</th><th style="text-align:right">Entry</th><th style="text-align:right">Current</th><th style="text-align:right">Est P&L</th><th style="text-align:right">Value</th></tr>
      {position_rows}
    </table>
  </div>

  <h2 style="font-size:14px;font-weight:700;color:#00d4ff;margin-bottom:10px;letter-spacing:1px">📋 RECENT TRADES</h2>
  <div class="card" style="overflow-x:auto">
    <table>
      <tr><th>Time</th><th>Side</th><th>Market</th><th style="text-align:center">Outcome</th><th style="text-align:right">Price</th><th style="text-align:right">Shares</th><th style="text-align:right">Value</th></tr>
      {trade_rows}
    </table>
  </div>

  <div style="margin-top:30px;padding:16px;text-align:center;font-size:11px;color:#2a3a4a;border-top:1px solid #111b26">
    SharpFlow v1.0 · Data fetched live from Polymarket
  </div>
</div>
</body></html>"""


@app.get("/markets", response_class=HTMLResponse)
def markets_page():
    """Market Intelligence page — smart money positioning across open markets."""
    intel = state.get("market_intelligence", [])
    status = state["status"]

    if status != "ready" or not intel:
        return HTMLResponse(f"""<!DOCTYPE html><html><head>
        <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Markets — SharpFlow</title>
        <style>body {{ background:#060d14;color:#c5d0de;font-family:sans-serif;padding:40px; }}</style>
        </head><body>
        <a href="/" style="color:#00d4ff;text-decoration:none;font-size:12px">← Dashboard</a>
        <h1 style="margin-top:16px;color:#e8edf2">Market Intelligence</h1>
        <p style="margin-top:20px;color:#3a4a5a">{'Pipeline is still running...' if status != 'ready' else 'No market intelligence data yet. Wait for the next pipeline refresh.'}</p>
        </body></html>""")

    market_cards = ""
    for m in intel[:40]:
        # Smart money direction indicator
        sy_pct = m["smart_money_yes_pct"]
        gap = m["price_gap"]
        div = m["divergence"]
        mkt_price = m["market_price"]

        # Color and arrow for smart direction
        if m["smart_direction"] == "YES":
            dir_color = "#00f5a0"
            dir_arrow = "▲"
            dir_label = "SMART MONEY: YES"
        elif m["smart_direction"] == "NO":
            dir_color = "#f54242"
            dir_arrow = "▼"
            dir_label = "SMART MONEY: NO"
        else:
            dir_color = "#5a7a9a"
            dir_arrow = "◆"
            dir_label = "SMART MONEY: SPLIT"

        # Price gap indicator
        if abs(gap) >= 0.10:
            gap_color = "#f5c842"
            gap_label = "LARGE GAP"
        elif abs(gap) >= 0.05:
            gap_color = "#a855f7"
            gap_label = "MODERATE GAP"
        else:
            gap_color = "#3a4a5a"
            gap_label = "SMALL GAP"

        gap_direction = "underpriced" if gap > 0 else "overpriced"

        # Divergence badge
        if div >= 0.30:
            div_badge = f'<span style="background:#f5424220;color:#f54242;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700">⚡ HIGH DIVERGENCE</span>'
        elif div >= 0.15:
            div_badge = f'<span style="background:#f5c84220;color:#f5c842;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:700">DIVERGENT</span>'
        else:
            div_badge = ""

        # Smart money bar (visual gauge)
        yes_width = max(2, int(sy_pct * 100))
        no_width = 100 - yes_width

        # Dull money bar
        dy_pct = m["dull_money_yes_pct"]
        dull_yes_w = max(2, int(dy_pct * 100))
        dull_no_w = 100 - dull_yes_w

        # Wallet names
        sharp_yes_names = ", ".join(w["display_name"] for w in m.get("sharp_wallets_yes", [])[:3])
        sharp_no_names = ", ".join(w["display_name"] for w in m.get("sharp_wallets_no", [])[:3])

        elite_indicator = ""
        if m["elite_yes"] > 0 or m["elite_no"] > 0:
            elite_parts = []
            if m["elite_yes"] > 0:
                elite_parts.append(f'{m["elite_yes"]}⚡ YES')
            if m["elite_no"] > 0:
                elite_parts.append(f'{m["elite_no"]}⚡ NO')
            elite_indicator = f'<span style="color:#00f5a0;font-size:10px;font-weight:700">{" · ".join(elite_parts)}</span>'

        cat_badge = {"sports": "🏀", "politics": "🏛️", "other": "📊"}.get(m["category"], "📊")

        market_cards += f"""
        <div class="card" style="margin-bottom:12px;padding:16px">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;gap:8px">
            <div style="flex:1">
              <div style="font-size:14px;font-weight:600;color:#e8edf2;line-height:1.3">{cat_badge} {m['question'][:80]}</div>
              <div style="font-size:11px;color:#3a4a5a;margin-top:4px">
                Vol: ${m['volume']:,.0f} · Market: {mkt_price:.0%} · {m['sharp_yes_count']+m['sharp_no_count']} sharp wallets
                {' · ' + elite_indicator if elite_indicator else ''}
              </div>
            </div>
            <div style="text-align:right;min-width:90px">
              <div style="font-size:20px;font-weight:800;color:{dir_color}">{dir_arrow} {sy_pct:.0%}</div>
              <div style="font-size:9px;color:{dir_color};letter-spacing:0.5px">{dir_label}</div>
            </div>
          </div>

          <!-- Smart Money Bar -->
          <div style="margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;font-size:10px;color:#5a7a9a;margin-bottom:3px">
              <span>SHARP MONEY</span>
              <span>YES ${m['sharp_yes_notional']:,.0f} ({m['sharp_yes_count']}) vs NO ${m['sharp_no_notional']:,.0f} ({m['sharp_no_count']})</span>
            </div>
            <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;background:#111b26">
              <div style="width:{yes_width}%;background:#00f5a0;transition:width 0.3s"></div>
              <div style="width:{no_width}%;background:#f54242;transition:width 0.3s"></div>
            </div>
            <div style="font-size:10px;color:#3a4a5a;margin-top:2px">{sharp_yes_names}{' · ' + sharp_no_names if sharp_no_names else ''}</div>
          </div>

          <!-- Dull Money Bar -->
          <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;font-size:10px;color:#5a7a9a;margin-bottom:3px">
              <span>DULL MONEY</span>
              <span>YES ${m['dull_yes_notional']:,.0f} ({m['dull_yes_count']}) vs NO ${m['dull_no_notional']:,.0f} ({m['dull_no_count']})</span>
            </div>
            <div style="display:flex;height:6px;border-radius:3px;overflow:hidden;background:#111b26">
              <div style="width:{dull_yes_w}%;background:#00f5a050"></div>
              <div style="width:{dull_no_w}%;background:#f5424250"></div>
            </div>
          </div>

          <!-- Signals Row -->
          <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
            <div style="font-size:11px">
              <span style="color:#5a7a9a">Price Gap:</span>
              <span style="color:{gap_color};font-weight:700;font-family:monospace">{gap:+.0%}</span>
              <span style="color:#3a4a5a;font-size:10px">({gap_label})</span>
            </div>
            <div style="font-size:11px">
              <span style="color:#5a7a9a">Divergence:</span>
              <span style="font-family:monospace;color:{'#f54242' if div>=0.3 else '#f5c842' if div>=0.15 else '#3a4a5a'}">{div:.0%}</span>
            </div>
            <div style="font-size:11px">
              <span style="color:#5a7a9a">Interest:</span>
              <span style="font-family:monospace;color:#00d4ff">{m['interest_score']:.2f}</span>
            </div>
            {div_badge}
          </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Intelligence — SharpFlow</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#060d14; color:#c5d0de; font-family:-apple-system,system-ui,sans-serif; }}
  .card {{ background:#0a1018; border:1px solid #1a2a3a; border-radius:8px; }}
  a {{ color:#00d4ff; text-decoration:none; }}
</style>
</head><body>
<div style="max-width:900px;margin:0 auto;padding:20px">

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:10px">
    <div>
      <a href="/" style="font-size:12px;color:#3a4a5a">← Dashboard</a>
      <h1 style="font-size:22px;font-weight:800;color:#e8edf2;margin-top:8px">MARKET INTELLIGENCE</h1>
      <div style="font-size:12px;color:#4a6070">Smart money positioning across {len(intel)} open markets</div>
    </div>
    <div style="text-align:right;font-size:11px;color:#3a4a5a">
      Last refresh: {state.get('last_refresh', 'never')}<br>
      <a href="/api/markets">/api/markets</a>
    </div>
  </div>

  <div class="card" style="padding:14px;margin-bottom:20px">
    <div style="font-size:12px;color:#5a7a9a;line-height:1.6">
      <b style="color:#e8edf2">How to read this:</b>
      The <span style="color:#00f5a0">green bar</span> shows how much sharp money is on YES vs <span style="color:#f54242">NO</span>.
      The <b>Price Gap</b> is the difference between the market price and where smart money is positioned — a large gap means sharps see an opportunity the market hasn't priced in yet.
      <b style="color:#f54242">High Divergence</b> means sharp and dull money disagree — historically, sharp money wins that fight.
    </div>
  </div>

  {market_cards}

  <div style="margin-top:30px;padding:16px;text-align:center;font-size:11px;color:#2a3a4a;border-top:1px solid #111b26">
    SharpFlow v1.5 · Market Intelligence · Not financial advice
  </div>
</div>
</body></html>"""


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
        <tr onclick="window.location='/wallet/{w['wallet']}'" style="cursor:pointer;border-bottom:1px solid #1a2a3a" onmouseover="this.style.background='#0d1520'" onmouseout="this.style.background='none'">
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
            mkt_price = s.get('market_price', 0)
            notional = s.get('total_notional', 0)
            notional_fmt = f"${notional:,.0f}" if notional < 1e6 else f"${notional/1e6:,.1f}M"
            signal_rows += f"""
            <tr style="border-bottom:1px solid #1a2a3a">
              <td style="padding:10px 8px;font-size:14px">{emoji}</td>
              <td style="padding:10px 8px;font-size:13px;color:#e8edf2">{s['market_title'][:60]}</td>
              <td style="padding:10px 8px;text-align:center;font-weight:700;color:{'#00f5a0' if s['side']=='YES' else '#f54242'}">{s['side']}</td>
              <td style="padding:10px 8px;text-align:center;font-family:monospace;font-size:12px;color:#00d4ff">{mkt_price:.0%}</td>
              <td style="padding:10px 8px;text-align:center;font-family:monospace">{s['wallet_count']} ({s['elite_count']}⚡ {s.get('sharp_count',0)}📊)</td>
              <td style="padding:10px 8px;text-align:center;font-family:monospace">{s['strength']:.2f}</td>
              <td style="padding:10px 8px;text-align:center;font-family:monospace;font-size:12px;color:#f5c842">{notional_fmt}</td>
              <td style="padding:10px 8px;font-size:11px;color:#5a7a9a" class="desktop-only">{wallet_names}</td>
            </tr>"""
    else:
        signal_rows = '<tr><td colspan="8" style="padding:20px;text-align:center;color:#3a4a5a">No convergence signals detected yet</td></tr>'

    # Build hot markets section (top 8 from market intelligence)
    hot_markets_html = ""
    intel = state.get("market_intelligence", [])
    if intel:
        for m in intel[:8]:
            sy_pct = m["smart_money_yes_pct"]
            gap = m["price_gap"]
            div = m["divergence"]

            if m["smart_direction"] == "YES":
                dir_color = "#00f5a0"
                dir_arrow = "▲"
            elif m["smart_direction"] == "NO":
                dir_color = "#f54242"
                dir_arrow = "▼"
            else:
                dir_color = "#5a7a9a"
                dir_arrow = "◆"

            yes_w = max(2, int(sy_pct * 100))
            no_w = 100 - yes_w

            div_tag = ""
            if div >= 0.30:
                div_tag = '<span style="background:#f5424220;color:#f54242;padding:1px 4px;border-radius:2px;font-size:9px;font-weight:700;margin-left:4px">DIVERGENT</span>'

            cat_badge = {"sports": "🏀", "politics": "🏛️", "other": "📊"}.get(m["category"], "📊")

            hot_markets_html += f"""
            <div style="background:#0a1018;border:1px solid #1a2a3a;border-radius:6px;padding:12px;display:flex;justify-content:space-between;align-items:center;gap:10px">
              <div style="flex:1;min-width:0">
                <div style="font-size:12px;font-weight:600;color:#e8edf2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{cat_badge} {m['question'][:55]}</div>
                <div style="display:flex;height:4px;border-radius:2px;overflow:hidden;background:#111b26;margin-top:6px">
                  <div style="width:{yes_w}%;background:#00f5a0"></div>
                  <div style="width:{no_w}%;background:#f54242"></div>
                </div>
                <div style="font-size:10px;color:#3a4a5a;margin-top:3px">
                  Mkt: {m['market_price']:.0%} · Gap: <span style="color:{'#f5c842' if abs(gap)>=0.10 else '#5a7a9a'}">{gap:+.0%}</span> · {m['sharp_yes_count']+m['sharp_no_count']} sharps{div_tag}
                </div>
              </div>
              <div style="text-align:right;min-width:60px">
                <div style="font-size:18px;font-weight:800;color:{dir_color}">{dir_arrow}{sy_pct:.0%}</div>
                <div style="font-size:9px;color:#3a4a5a">smart $</div>
              </div>
            </div>"""
        hot_markets_html = f"""
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:8px">
          {hot_markets_html}
        </div>
        <div style="text-align:right;margin-top:8px"><a href="/markets" style="font-size:12px;color:#a855f7">View all {len(intel)} markets →</a></div>"""
    else:
        hot_markets_html = '<div style="padding:20px;text-align:center;color:#3a4a5a;background:#0a1018;border:1px solid #1a2a3a;border-radius:8px">Market intelligence data will appear after the next pipeline run</div>'

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
      <a href="/markets" style="font-size:13px;font-weight:700;color:#a855f7">📊 Market Intelligence</a> ·
      <a href="/api/wallets">/api/wallets</a> ·
      <a href="/api/markets">/api/markets</a> ·
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
    <div class="card stat"><div class="val" style="color:#f5c842">{len(intel)}</div><div class="lbl">MKTS TRACKED</div></div>
  </div>

  <!-- Convergence Signals -->
  <h2 style="font-size:14px;font-weight:700;color:#a855f7;margin-bottom:10px;letter-spacing:1px">🎯 CONVERGENCE SIGNALS</h2>
  <div class="card" style="margin-bottom:24px;overflow-x:auto">
    <table>
      <tr>
        <th style="width:30px"></th><th>Market</th><th style="text-align:center">Side</th>
        <th style="text-align:center">Price</th><th style="text-align:center">Wallets</th>
        <th style="text-align:center">Strength</th><th style="text-align:center">$ In</th>
        <th class="desktop-only">Sharp Wallets</th>
      </tr>
      {signal_rows}
    </table>
  </div>

  <!-- Hot Markets -->
  <h2 style="font-size:14px;font-weight:700;color:#a855f7;margin-bottom:10px;letter-spacing:1px">📊 HOT MARKETS — SMART MONEY VIEW</h2>
  <div style="margin-bottom:24px">
    {hot_markets_html}
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
        time.sleep(3 * 3600)  # 3 hours
        logger.info("Running scheduled refresh...")
        run_pipeline()


# Start pipeline in background thread
pipeline_thread = threading.Thread(target=background_pipeline, daemon=True)
pipeline_thread.start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
