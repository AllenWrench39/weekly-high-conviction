"""
Stock universe builder for downstream swing and weekly screeners.

Goals:
- fetch broad US stock symbol lists from NASDAQ + NYSE sources
- validate symbols against yfinance metadata
- keep only real EQUITY tickers with usable price info
- exclude ETFs and symbols that Yahoo cannot resolve cleanly
- save a clean universe.json so downstream scripts do not need to
  repeat ETF checks or basic ticker-info checks
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone

import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OUTPUT_FILE = "universe.json"
INVALID_FILE = "universe_invalid.json"

REQUEST_TIMEOUT_SECONDS = 15
YF_TIMEOUT_PER_SYMBOL = 5
MAX_WORKERS = 8
PROGRESS_INTERVAL = 100

SOURCE_URLS = {
    "nasdaq": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.json",
    "nyse": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.json",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SYMBOL FETCH
# ---------------------------------------------------------------------------

def fetch_symbols_from_url(url: str, retries: int = 3) -> list[str]:
    """Fetch raw exchange symbols from GitHub-backed JSON."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return payload
            log.warning("Unexpected payload type from %s", url)
            return []
        except Exception as exc:
            log.warning("Attempt %s failed for %s: %s", attempt, url, exc)
            time.sleep(3)
    log.error("Failed all retries for %s", url)
    return []


def clean_symbols(symbols: list[str]) -> list[str]:
    """
    Keep standard common-stock-style symbols only.

    This strips a lot of obvious junk before we hit yfinance:
    - max 5 chars
    - letters only
    - no dots, dashes, warrants, units, rights
    """
    cleaned = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not re.fullmatch(r"[A-Z]+", symbol):
            continue
        if len(symbol) > 5:
            continue
        cleaned.add(symbol)
    return sorted(cleaned)


def fetch_tickers() -> list[str]:
    """Fetch and clean the combined exchange universe."""
    all_symbols: list[str] = []
    for exchange_name, url in SOURCE_URLS.items():
        log.info("Fetching %s symbols from %s", exchange_name.upper(), url)
        symbols = fetch_symbols_from_url(url)
        log.info("Loaded %s raw %s symbols", len(symbols), exchange_name.upper())
        all_symbols.extend(symbols)

    cleaned = clean_symbols(all_symbols)
    log.info("Total unique cleaned symbols to validate: %s", len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
# YFINANCE VALIDATION
# ---------------------------------------------------------------------------

def _validate_symbol(symbol: str) -> dict | None:
    """
    Return a normalized universe entry for valid equities, otherwise None.

    A valid symbol must:
    - resolve in yfinance
    - have quoteType EQUITY
    - have at least one usable price field
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not isinstance(info, dict) or len(info) < 3:
            return None

        quote_type = str(info.get("quoteType", "")).upper()
        if quote_type != "EQUITY":
            return None

        regular_market_price = info.get("regularMarketPrice")
        previous_close = info.get("previousClose")
        market_cap = info.get("marketCap")
        avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day")
        short_name = info.get("shortName") or info.get("longName") or ""

        if regular_market_price is None and previous_close is None:
            return None

        return {
            "symbol": symbol,
            "type": "EQUITY",
            "name": short_name,
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": market_cap,
            "avg_volume": avg_volume,
        }
    except Exception:
        return None


def validate_symbol(symbol: str, timeout_seconds: int = YF_TIMEOUT_PER_SYMBOL) -> dict | str:
    """
    Validate a symbol with a hard timeout.

    Returns either a normalized dict or one of:
    - INVALID
    - TIMEOUT
    - ERROR
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_validate_symbol, symbol)
            result = future.result(timeout=timeout_seconds)
            return result if result else "INVALID"
    except FuturesTimeoutError:
        return "TIMEOUT"
    except Exception:
        return "ERROR"


def validate_universe(tickers: list[str]) -> tuple[list[dict], list[dict], dict]:
    """Validate all symbols and keep only usable equities."""
    valid_symbols: list[dict] = []
    invalid_symbols: list[dict] = []
    stats = {
        "valid_equity": 0,
        "invalid": 0,
        "timeout": 0,
        "error": 0,
    }

    start_time = time.time()
    total = len(tickers)
    log.info("Validating %s symbols against yfinance ticker info...", total)

    for index, symbol in enumerate(tickers, start=1):
        result = validate_symbol(symbol)

        if isinstance(result, dict):
            valid_symbols.append(result)
            stats["valid_equity"] += 1
        else:
            invalid_symbols.append({"symbol": symbol, "reason": result})
            stats[result.lower()] = stats.get(result.lower(), 0) + 1

        if index % PROGRESS_INTERVAL == 0 or index == total:
            elapsed = time.time() - start_time
            rate = index / elapsed if elapsed > 0 else 0
            remaining = total - index
            eta_minutes = (remaining / rate / 60) if rate > 0 else 0
            log.info(
                "Progress %s/%s (%.1f%%) | valid=%s invalid=%s timeout=%s error=%s | %.1f/s | ETA %.1fm",
                index,
                total,
                index / total * 100,
                stats["valid_equity"],
                stats["invalid"],
                stats["timeout"],
                stats["error"],
                rate,
                eta_minutes,
            )

    return valid_symbols, invalid_symbols, stats


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def save_universe(valid_symbols: list[dict], invalid_symbols: list[dict], stats: dict) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()

    universe_payload = {
        "generated_at": generated_at,
        "total_symbols": len(valid_symbols),
        "equities_count": len(valid_symbols),
        "symbols": valid_symbols,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file_handle:
        json.dump(universe_payload, file_handle, indent=2)

    invalid_payload = {
        "generated_at": generated_at,
        "total_invalid": len(invalid_symbols),
        "stats": stats,
        "invalid_symbols": invalid_symbols,
    }

    with open(INVALID_FILE, "w", encoding="utf-8") as file_handle:
        json.dump(invalid_payload, file_handle, indent=2)

    log.info("Saved %s clean equities to %s", len(valid_symbols), OUTPUT_FILE)
    log.info("Saved %s rejected symbols to %s", len(invalid_symbols), INVALID_FILE)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("STOCK UNIVERSE BUILDER")
    log.info("=" * 60)

    start_time = time.time()
    tickers = fetch_tickers()
    if not tickers:
        log.error("No symbols fetched. Aborting.")
        raise SystemExit(1)

    valid_symbols, invalid_symbols, stats = validate_universe(tickers)
    if not valid_symbols:
        log.error("No valid equities found. Universe not saved.")
        raise SystemExit(1)

    save_universe(valid_symbols, invalid_symbols, stats)

    elapsed_minutes = (time.time() - start_time) / 60
    log.info("=" * 60)
    log.info(
        "Done in %.1f minutes | kept=%s invalid=%s timeout=%s error=%s",
        elapsed_minutes,
        stats["valid_equity"],
        stats["invalid"],
        stats["timeout"],
        stats["error"],
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
