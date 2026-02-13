"""
SharpFlow Live Trade Listener
Monitors Polymarket OrderFilled events on Polygon PoS in real-time.
Writes trades to Postgres as they happen.

Usage:
    # Standalone test
    python live_listener.py

    # Imported by app.py as a background thread
    from live_listener import start_listener
    start_listener(db_url, alchemy_url)

Requirements:
    pip install web3 psycopg2-binary requests
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timezone
from collections import defaultdict

import requests
import psycopg2
from web3 import Web3

logger = logging.getLogger("live_listener")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [LIVE] %(message)s"))
    logger.addHandler(handler)

# ================================================================
# POLYMARKET CONTRACTS
# ================================================================

# CTF Exchange (binary markets)
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
# NegRisk CTF Exchange (multi-outcome markets)
NEGRISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# Both contracts emit the same OrderFilled event
ORDER_FILLED_TOPIC = Web3.keccak(text="OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)")

# Platform addresses (taker in duplicate events — skip these)
PLATFORM_ADDRESSES = {
    CTF_EXCHANGE.lower(),
    NEGRISK_EXCHANGE.lower(),
}

# OrderFilled ABI for decoding
ORDER_FILLED_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "name": "orderHash", "type": "bytes32"},
        {"indexed": True, "name": "maker", "type": "address"},
        {"indexed": True, "name": "taker", "type": "address"},
        {"indexed": False, "name": "makerAssetId", "type": "uint256"},
        {"indexed": False, "name": "takerAssetId", "type": "uint256"},
        {"indexed": False, "name": "makerAmountFilled", "type": "uint256"},
        {"indexed": False, "name": "takerAmountFilled", "type": "uint256"},
        {"indexed": False, "name": "fee", "type": "uint256"},
    ],
    "name": "OrderFilled",
    "type": "event",
}


# ================================================================
# ASSET ID -> MARKET MAPPING CACHE
# ================================================================

class MarketMapper:
    """Maps token asset IDs to condition IDs (markets) and metadata."""

    def __init__(self):
        self.asset_to_market = {}  # asset_id -> {condition_id, question, outcome, category, ...}
        self.condition_cache = {}  # condition_id -> market metadata
        self._failed_cache = {}   # asset_id -> (fail_count, last_attempt_time)
        self._lock = threading.Lock()
        self._api_last_call = 0   # timestamp of last Gamma API call
        self._api_min_interval = 0.15  # minimum seconds between Gamma calls
        self._max_retries = 3     # max retries for failed lookups
        self._retry_backoff = 300 # seconds before retrying a failed lookup

    def lookup(self, asset_id):
        """Look up market info for an asset ID. Returns dict or None."""
        asset_str = str(asset_id)
        with self._lock:
            if asset_str in self.asset_to_market:
                return self.asset_to_market[asset_str]

            # Check if we recently failed on this asset_id (avoid hammering)
            if asset_str in self._failed_cache:
                fail_count, last_attempt = self._failed_cache[asset_str]
                if fail_count >= self._max_retries:
                    # Permanent failure — don't retry until backoff expires
                    if time.time() - last_attempt < self._retry_backoff:
                        return None
                    else:
                        # Backoff expired, reset and retry
                        del self._failed_cache[asset_str]

        # Rate limit: wait if we called too recently
        now = time.time()
        elapsed = now - self._api_last_call
        if elapsed < self._api_min_interval:
            time.sleep(self._api_min_interval - elapsed)
        self._api_last_call = time.time()

        # Try Gamma API
        info = self._fetch_from_gamma(asset_str)
        if info:
            with self._lock:
                self.asset_to_market[asset_str] = info
            return info

        # Track failure
        with self._lock:
            if asset_str in self._failed_cache:
                fail_count, _ = self._failed_cache[asset_str]
                self._failed_cache[asset_str] = (fail_count + 1, time.time())
            else:
                self._failed_cache[asset_str] = (1, time.time())

        return None

    def _fetch_from_gamma(self, asset_id):
        """Fetch market info from Polymarket Gamma API."""
        try:
            # Search by token ID (try open markets first, then closed)
            for closed_param in ["false", "true"]:
                url = f"https://gamma-api.polymarket.com/markets?clob_token_ids={asset_id}&closed={closed_param}"
                resp = requests.get(url, timeout=10)

                if resp.status_code == 429:
                    logger.warning(f"Gamma API rate limited (429), backing off")
                    time.sleep(5)
                    return None

                if resp.status_code != 200:
                    logger.warning(f"Gamma API returned {resp.status_code} for asset {asset_id[:20]}...")
                    continue

                markets = resp.json()
                if markets and len(markets) > 0:
                    m = markets[0]
                    # Debug: log ALL keys when condition_id is missing
                    if not m.get("condition_id"):
                        logger.info(
                            f"Gamma ALL KEYS for asset {asset_id[:20]}...: "
                            f"{sorted(m.keys())}"
                        )
                    parsed = self._parse_market(m, asset_id)
                    if parsed and parsed.get("condition_id"):
                        logger.debug(f"Mapped asset {asset_id[:20]}... → {parsed['question'][:60]}")
                        return parsed
                    else:
                        logger.warning(f"Gamma returned market but no condition_id for asset {asset_id[:20]}...")

        except requests.exceptions.Timeout:
            logger.warning(f"Gamma API timeout for asset {asset_id[:20]}...")
        except Exception as e:
            logger.warning(f"Gamma API lookup failed for {asset_id[:20]}...: {e}")

        return None

    def _parse_market(self, m, asset_id):
        """Parse Gamma API market response."""
        # condition_id can be empty for NegRisk (multi-outcome) markets
        # Fall back to question_id or neg_risk_market_id
        condition_id = m.get("condition_id", "") or m.get("question_id", "") or m.get("neg_risk_market_id", "")
        question = m.get("question", "")

        # Determine which outcome this token represents
        tokens = m.get("tokens", [])
        outcome = "unknown"
        for tok in tokens:
            if str(tok.get("token_id", "")) == str(asset_id):
                outcome = tok.get("outcome", "unknown")
                break

        # Categorize
        category = self._categorize(question)

        return {
            "condition_id": condition_id,
            "question": question,
            "outcome": outcome,
            "category": category,
            "end_date": m.get("end_date_iso", ""),
            "active": m.get("active", True),
            "closed": m.get("closed", False),
            "winning_outcome": m.get("winning_outcome", ""),
        }

    def _categorize(self, title):
        """Simple keyword categorization matching app.py logic."""
        title_lower = title.lower()
        sports_kw = ["sports", "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
                      "baseball", "ufc", "mma", "tennis", "boxing", "cricket", "f1", "golf",
                      "super bowl", "playoffs", "championship", "world series", "stanley cup",
                      "premier league", "champions league", "la liga", "serie a", "bundesliga"]
        politics_kw = ["politics", "election", "president", "congress", "senate", "governor",
                        "trump", "biden", "democrat", "republican", "vote", "legislation",
                        "cabinet", "supreme court", "government", "policy", "fed", "tariff",
                        "executive order", "parliament", "minister", "primary"]

        for kw in sports_kw:
            if kw in title_lower:
                return "sports"
        for kw in politics_kw:
            if kw in title_lower:
                return "politics"
        return "other"


# ================================================================
# TRADE PROCESSOR
# ================================================================

def decode_order_filled(log, w3):
    """Decode an OrderFilled event log into a trade dict."""
    # Indexed params are in topics
    # topics[0] = event signature
    # topics[1] = orderHash
    # topics[2] = maker (address, padded to 32 bytes)
    # topics[3] = taker (address, padded to 32 bytes)
    topics = log["topics"]
    if len(topics) < 4:
        return None

    order_hash = topics[1].hex() if isinstance(topics[1], bytes) else topics[1]
    maker = Web3.to_checksum_address("0x" + (topics[2].hex() if isinstance(topics[2], bytes) else topics[2])[-40:])
    taker = Web3.to_checksum_address("0x" + (topics[3].hex() if isinstance(topics[3], bytes) else topics[3])[-40:])

    # Non-indexed params are in data
    data = log["data"]
    if isinstance(data, str):
        data = bytes.fromhex(data[2:])

    # Decode: makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled, fee
    # Each is uint256 = 32 bytes
    if len(data) < 160:
        return None

    maker_asset_id = int.from_bytes(data[0:32], "big")
    taker_asset_id = int.from_bytes(data[32:64], "big")
    maker_amount = int.from_bytes(data[64:96], "big")
    taker_amount = int.from_bytes(data[96:128], "big")
    fee = int.from_bytes(data[128:160], "big")

    # Determine side: if makerAssetId == 0, maker is BUYING (paying USDC for tokens)
    if maker_asset_id == 0:
        side = "BUY"
        asset_id = taker_asset_id  # the token being bought
        # Price = USDC paid / tokens received
        # USDC has 6 decimals, tokens have 6 decimals on Polymarket
        usdc_amount = maker_amount / 1e6
        token_amount = taker_amount / 1e6
        price = usdc_amount / token_amount if token_amount > 0 else 0
    else:
        side = "SELL"
        asset_id = maker_asset_id  # the token being sold
        usdc_amount = taker_amount / 1e6
        token_amount = maker_amount / 1e6
        price = usdc_amount / token_amount if token_amount > 0 else 0

    return {
        "order_hash": order_hash,
        "maker": maker.lower(),
        "taker": taker.lower(),
        "side": side,
        "asset_id": str(asset_id),
        "price": round(price, 4),
        "size": round(token_amount, 2),
        "usdc_amount": round(usdc_amount, 2),
        "fee": fee / 1e6,
        "block_number": log["blockNumber"],
        "tx_hash": log["transactionHash"].hex() if isinstance(log["transactionHash"], bytes) else log["transactionHash"],
        "log_index": log["logIndex"],
        "contract": log["address"].lower(),
    }


def is_duplicate_event(trade):
    """
    Filter duplicate OrderFilled events per Paradigm's analysis.
    Keep only maker-focused events where maker is NOT an exchange contract.
    """
    return trade["maker"] in PLATFORM_ADDRESSES


# ================================================================
# DATABASE WRITER
# ================================================================

class DBWriter:
    """Writes live trades to the SharpFlow Postgres database."""

    def __init__(self, db_url):
        self.db_url = db_url
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = True
        return self._conn

    def ensure_live_table(self):
        """Create live_trades table if it doesn't exist."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS live_trades (
                    id SERIAL PRIMARY KEY,
                    tx_hash TEXT NOT NULL,
                    log_index INTEGER NOT NULL,
                    block_number BIGINT NOT NULL,
                    timestamp BIGINT,
                    wallet TEXT NOT NULL,
                    condition_id TEXT,
                    asset_id TEXT,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    usdc_amount REAL,
                    outcome TEXT,
                    title TEXT,
                    category TEXT DEFAULT 'other',
                    fee REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(tx_hash, log_index)
                );
                CREATE INDEX IF NOT EXISTS idx_live_wallet ON live_trades(wallet);
                CREATE INDEX IF NOT EXISTS idx_live_condition ON live_trades(condition_id);
                CREATE INDEX IF NOT EXISTS idx_live_block ON live_trades(block_number);
                CREATE INDEX IF NOT EXISTS idx_live_timestamp ON live_trades(timestamp);
            """)
        logger.info("live_trades table ready")

    def insert_trade(self, trade, market_info, block_timestamp):
        """Insert a single trade. Returns True if inserted, False if duplicate."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO live_trades (tx_hash, log_index, block_number, timestamp,
                        wallet, condition_id, asset_id, side, price, size, usdc_amount,
                        outcome, title, category, fee)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tx_hash, log_index) DO NOTHING
                """, (
                    trade["tx_hash"],
                    trade["log_index"],
                    trade["block_number"],
                    block_timestamp,
                    trade["maker"],
                    market_info.get("condition_id", "") if market_info else "",
                    trade["asset_id"],
                    trade["side"],
                    trade["price"],
                    trade["size"],
                    trade["usdc_amount"],
                    market_info.get("outcome", "") if market_info else "",
                    market_info.get("question", "")[:500] if market_info else "",
                    market_info.get("category", "other") if market_info else "other",
                    trade["fee"],
                ))
                return cur.rowcount > 0
        except psycopg2.Error as e:
            logger.error(f"DB insert error: {e}")
            self._conn = None
            return False

    def get_latest_block(self):
        """Get the most recent block number in live_trades."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(block_number) FROM live_trades")
                result = cur.fetchone()
                return result[0] if result and result[0] else None
        except Exception:
            return None

    def get_stats(self):
        """Get live trade stats."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*), COUNT(DISTINCT wallet), COUNT(DISTINCT condition_id),
                           MIN(timestamp), MAX(timestamp)
                    FROM live_trades
                """)
                row = cur.fetchone()
                return {
                    "total_trades": row[0],
                    "unique_wallets": row[1],
                    "unique_markets": row[2],
                    "earliest": row[3],
                    "latest": row[4],
                }
        except Exception:
            return {}


# ================================================================
# MAIN LISTENER
# ================================================================

class LiveTradeListener:
    """
    Polls Polygon for OrderFilled events and writes to DB.
    Designed to run as a background thread.
    """

    def __init__(self, alchemy_url, db_url, poll_interval=5, block_chunk=9):
        self.w3 = Web3(Web3.HTTPProvider(alchemy_url))
        self.alchemy_url = alchemy_url
        self.db = DBWriter(db_url)
        self.mapper = MarketMapper()
        self.poll_interval = poll_interval
        self.block_chunk = block_chunk
        self.running = False
        self._thread = None

        # Stats
        self.trades_ingested = 0
        self.trades_skipped = 0
        self.errors = 0
        self.last_block = 0
        self.last_poll_time = None
        self.started_at = None

    def start(self):
        """Start the listener in a background thread."""
        if self.running:
            logger.warning("Listener already running")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Live trade listener started")

    def stop(self):
        """Stop the listener."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=30)
        logger.info("Live trade listener stopped")

    def get_status(self):
        """Return current listener status."""
        mapper_stats = {
            "cached_assets": len(self.mapper.asset_to_market),
            "failed_lookups": len(self.mapper._failed_cache),
        }
        return {
            "running": self.running,
            "trades_ingested": self.trades_ingested,
            "trades_skipped": self.trades_skipped,
            "errors": self.errors,
            "last_block": self.last_block,
            "last_poll": self.last_poll_time,
            "started_at": self.started_at,
            "chain_connected": self.w3.is_connected() if self.w3 else False,
            "mapper": mapper_stats,
        }

    def _run_loop(self):
        """Main polling loop."""
        self.started_at = datetime.now(timezone.utc).isoformat()

        # Ensure table exists
        try:
            self.db.ensure_live_table()
        except Exception as e:
            logger.error(f"Failed to create live_trades table: {e}")
            self.running = False
            return

        # Check chain connection with retry
        connected = False
        for attempt in range(5):
            try:
                chain_id = self.w3.eth.chain_id
                logger.info(f"Connected to chain {chain_id} (Polygon={'yes' if chain_id == 137 else 'NO'})")
                connected = True
                break
            except Exception as e:
                logger.warning(f"RPC connection attempt {attempt+1}/5 failed: {e}")
                time.sleep(5)

        if not connected:
            # Try a raw request as fallback diagnostic
            try:
                import requests as req
                resp = req.post(self.w3.provider.endpoint_uri, json={
                    "jsonrpc": "2.0", "method": "eth_chainId", "params": [], "id": 1
                }, timeout=10)
                logger.error(f"Raw RPC response: {resp.status_code} {resp.text[:200]}")
            except Exception as e2:
                logger.error(f"Raw RPC also failed: {e2}")
            logger.error("Cannot connect to Polygon RPC after 5 attempts")
            self.running = False
            return

        # Determine starting block
        db_latest = self.db.get_latest_block()
        if db_latest:
            start_block = db_latest + 1
            logger.info(f"Resuming from block {start_block} (last in DB: {db_latest})")
        else:
            # Start from ~2 minutes ago (just catch real-time, don't backfill)
            current = self.w3.eth.block_number
            start_block = current - 60  # ~60 blocks = ~2 minutes on Polygon
            logger.info(f"No history in DB. Starting from block {start_block} (current: {current})")

        self.last_block = start_block

        while self.running:
            try:
                self._poll_once()
            except Exception as e:
                self.errors += 1
                logger.error(f"Poll error: {e}")
                time.sleep(min(self.poll_interval * 2, 60))

            time.sleep(self.poll_interval)

    def _poll_once(self):
        """Poll for new events since last block."""
        current_block = self.w3.eth.block_number
        from_block = self.last_block
        to_block = min(from_block + self.block_chunk, current_block)

        if from_block > current_block:
            return  # caught up

        # Use raw JSON-RPC for get_logs (more control over format)
        contracts = [CTF_EXCHANGE.lower(), NEGRISK_EXCHANGE.lower()]
        topic_hex = "0x" + ORDER_FILLED_TOPIC.hex() if isinstance(ORDER_FILLED_TOPIC, bytes) else ORDER_FILLED_TOPIC

        all_logs = []
        for contract_addr in contracts:
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "eth_getLogs",
                    "params": [{
                        "fromBlock": hex(from_block),
                        "toBlock": hex(to_block),
                        "address": contract_addr,
                        "topics": [topic_hex],
                    }],
                    "id": 1,
                }
                resp = requests.post(self.alchemy_url, json=payload, timeout=30)
                result = resp.json()

                if "error" in result:
                    logger.warning(f"RPC error for {contract_addr[:10]}: {result['error']}")
                    if self.block_chunk > 10:
                        self.block_chunk = max(10, self.block_chunk // 2)
                        logger.info(f"Reduced block chunk to {self.block_chunk}")
                    raise Exception(f"RPC error: {result['error']}")

                logs = result.get("result", [])
                # Convert hex fields to int for processing
                for log in logs:
                    log["blockNumber"] = int(log["blockNumber"], 16)
                    log["logIndex"] = int(log["logIndex"], 16)
                    log["transactionHash"] = log["transactionHash"]
                    log["address"] = log["address"]
                    # topics stay as hex strings
                all_logs.extend(logs)

            except requests.RequestException as e:
                logger.warning(f"HTTP error fetching logs: {e}")
                raise

        # Get block timestamps for this range (batch)
        block_timestamps = {}

        # Process events
        inserted = 0
        skipped = 0

        for log in all_logs:
            trade = self._decode_raw_log(log)
            if not trade:
                continue

            # Skip duplicate taker-focused events
            if is_duplicate_event(trade):
                skipped += 1
                continue

            # Get block timestamp (cache per block)
            bn = trade["block_number"]
            if bn not in block_timestamps:
                try:
                    block = self.w3.eth.get_block(bn)
                    block_timestamps[bn] = block["timestamp"]
                except Exception:
                    block_timestamps[bn] = int(time.time())

            # Look up market info
            market_info = self.mapper.lookup(trade["asset_id"])

            # Write to DB
            if self.db.insert_trade(trade, market_info, block_timestamps[bn]):
                inserted += 1

        self.trades_ingested += inserted
        self.trades_skipped += skipped
        self.last_block = to_block + 1
        self.last_poll_time = datetime.now(timezone.utc).isoformat()

        blocks_behind = current_block - to_block
        if inserted > 0 or blocks_behind > 100:
            logger.info(
                f"Blocks {from_block}-{to_block}: {inserted} trades inserted, "
                f"{skipped} dupes skipped, {len(all_logs)} events. "
                f"{'CAUGHT UP' if blocks_behind < 50 else f'{blocks_behind} blocks behind'}"
            )

        # Don't increase chunk beyond Alchemy free tier limit (10 blocks)
        # block_chunk stays at 9

    def _decode_raw_log(self, log):
        """Decode a raw JSON-RPC log dict into a trade dict."""
        topics = log.get("topics", [])
        if len(topics) < 4:
            return None

        order_hash = topics[1]
        maker = Web3.to_checksum_address("0x" + topics[2][-40:]).lower()
        taker = Web3.to_checksum_address("0x" + topics[3][-40:]).lower()

        data = log.get("data", "0x")
        if isinstance(data, str):
            data = bytes.fromhex(data[2:])

        if len(data) < 160:
            return None

        maker_asset_id = int.from_bytes(data[0:32], "big")
        taker_asset_id = int.from_bytes(data[32:64], "big")
        maker_amount = int.from_bytes(data[64:96], "big")
        taker_amount = int.from_bytes(data[96:128], "big")
        fee = int.from_bytes(data[128:160], "big")

        if maker_asset_id == 0:
            side = "BUY"
            asset_id = taker_asset_id
            usdc_amount = maker_amount / 1e6
            token_amount = taker_amount / 1e6
            price = usdc_amount / token_amount if token_amount > 0 else 0
        else:
            side = "SELL"
            asset_id = maker_asset_id
            usdc_amount = taker_amount / 1e6
            token_amount = maker_amount / 1e6
            price = usdc_amount / token_amount if token_amount > 0 else 0

        return {
            "order_hash": order_hash,
            "maker": maker,
            "taker": taker,
            "side": side,
            "asset_id": str(asset_id),
            "price": round(price, 4),
            "size": round(token_amount, 2),
            "usdc_amount": round(usdc_amount, 2),
            "fee": fee / 1e6,
            "block_number": log["blockNumber"],
            "tx_hash": log["transactionHash"],
            "log_index": log["logIndex"],
            "contract": log["address"].lower(),
        }


# ================================================================
# INTEGRATION HELPERS
# ================================================================

_listener_instance = None

def start_listener(db_url, alchemy_url, poll_interval=15):
    """Start the global listener instance. Called by app.py."""
    global _listener_instance

    if not alchemy_url:
        logger.warning("No ALCHEMY_URL configured, live listener disabled")
        return None

    if _listener_instance and _listener_instance.running:
        logger.info("Listener already running")
        return _listener_instance

    _listener_instance = LiveTradeListener(alchemy_url, db_url, poll_interval)
    _listener_instance.start()
    return _listener_instance


def get_listener_status():
    """Get status of the global listener."""
    if _listener_instance:
        return _listener_instance.get_status()
    return {"running": False, "error": "Not initialized"}


# ================================================================
# STANDALONE MODE
# ================================================================

if __name__ == "__main__":
    db_url = os.environ.get("DATABASE_URL")
    alchemy_url = os.environ.get("ALCHEMY_URL")

    if not db_url:
        print("ERROR: Set DATABASE_URL environment variable")
        sys.exit(1)
    if not alchemy_url:
        print("ERROR: Set ALCHEMY_URL environment variable")
        sys.exit(1)

    print(f"Starting SharpFlow Live Trade Listener")
    print(f"  Polygon RPC: {alchemy_url[:50]}...")
    print(f"  Database: {db_url[:30]}...")
    print(f"  Contracts: CTF Exchange + NegRisk Exchange")
    print(f"  Poll interval: 15s")
    print()

    listener = LiveTradeListener(alchemy_url, db_url)
    listener.start()

    try:
        while True:
            time.sleep(30)
            status = listener.get_status()
            stats = listener.db.get_stats()
            print(
                f"[STATUS] Trades: {status['trades_ingested']} ingested, "
                f"{status['trades_skipped']} dupes | "
                f"Block: {status['last_block']} | "
                f"DB: {stats.get('total_trades', 0)} total, "
                f"{stats.get('unique_wallets', 0)} wallets, "
                f"{stats.get('unique_markets', 0)} markets | "
                f"Errors: {status['errors']}"
            )
    except KeyboardInterrupt:
        print("\nShutting down...")
        listener.stop()
