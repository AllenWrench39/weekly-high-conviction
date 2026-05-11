"""
Lightweight test harness for the weekly high conviction scanner.

What it does:
- checks that universe.json loads
- checks email env presence without sending
- fetches a small market regime basket from yfinance
- optionally imports weekly_high_conviction.py helpers when available
- runs a small sample-bar fetch / setup smoke test

Usage:
    python test_weekly_scanner.py

Optional env vars:
    TEST_TICKERS=AAPL,MSFT,NVDA
    TEST_SAMPLE_SIZE=3
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yfinance as yf


ROOT = Path(__file__).resolve().parent
UNIVERSE_FILE = ROOT / "universe.json"

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]
MARKET_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA", "RSP",
    "SMH", "XLK", "XLF", "XLI", "XLY",
    "XLV", "XLP", "XLU",
    "TLT", "HYG", "^VIX",
]


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def env_ok(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def test_universe_file() -> list[str]:
    print_header("Universe File")
    if not UNIVERSE_FILE.exists():
        print(f"FAIL: missing {UNIVERSE_FILE.name}")
        return []

    with UNIVERSE_FILE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        print("FAIL: universe.json has no symbols")
        return []

    sample = []
    for item in symbols:
        if isinstance(item, dict) and item.get("symbol"):
            sample.append(str(item["symbol"]).upper())
        elif isinstance(item, str):
            sample.append(item.upper())
        if len(sample) >= 5:
            break

    print(f"PASS: loaded {len(symbols)} raw universe entries")
    print(f"Sample: {sample}")
    return sample


def test_email_env() -> None:
    print_header("Email Environment")
    checks = {
        "EMAIL_ADDRESS": env_ok("EMAIL_ADDRESS"),
        "EMAIL_PASSWORD": env_ok("EMAIL_PASSWORD"),
        "RECIPIENT_EMAIL": env_ok("RECIPIENT_EMAIL"),
        "SENDER_EMAIL": env_ok("SENDER_EMAIL"),
        "SENDER_PASSWORD": env_ok("SENDER_PASSWORD"),
    }
    for key, ok in checks.items():
        print(f"{key}: {'SET' if ok else 'MISSING'}")


def test_market_data() -> None:
    print_header("Market Regime Basket")
    try:
        data = yf.download(
            tickers=MARKET_SYMBOLS,
            period="6mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        print(f"FAIL: market basket download error: {exc}")
        return

    available = []
    missing = []
    for symbol in MARKET_SYMBOLS:
        try:
            if symbol in data.columns.get_level_values(0) and not data[symbol].dropna().empty:
                available.append(symbol)
            else:
                missing.append(symbol)
        except Exception:
            missing.append(symbol)

    print(f"Available: {len(available)}/{len(MARKET_SYMBOLS)}")
    print(f"Good: {available}")
    if missing:
        print(f"Missing: {missing}")


def import_weekly_module():
    print_header("Module Import")
    try:
        import weekly_high_conviction as whc
        print("PASS: imported weekly_high_conviction.py")
        return whc
    except Exception as exc:
        print(f"FAIL: import error: {exc}")
        return None


def test_weekly_helpers(whc, sample_symbols: list[str]) -> None:
    print_header("Weekly Helper Smoke Test")
    if whc is None:
        print("SKIP: module did not import")
        return

    test_tickers = [
        x.strip().upper()
        for x in os.getenv("TEST_TICKERS", ",".join(DEFAULT_TICKERS)).split(",")
        if x.strip()
    ]

    if sample_symbols:
        merged = []
        for symbol in test_tickers + sample_symbols[:2]:
            if symbol not in merged:
                merged.append(symbol)
        test_tickers = merged

    sample_size = int(os.getenv("TEST_SAMPLE_SIZE", "3"))
    test_tickers = test_tickers[:sample_size]
    print(f"Testing tickers: {test_tickers}")

    if hasattr(whc, "build_macro_context"):
        try:
            macro = whc.build_macro_context()
            print("PASS: build_macro_context()")
            print({
                "spy_bullish": macro.get("spy_bullish"),
                "spy_weekly_perf": macro.get("spy_weekly_perf"),
                "tnx_current": macro.get("tnx_current"),
                "oil_price": macro.get("oil_price"),
            })
        except Exception as exc:
            print(f"FAIL: build_macro_context(): {exc}")

    if hasattr(whc, "get_all_bars"):
        try:
            bars = whc.get_all_bars(test_tickers)
            print(f"PASS: get_all_bars() returned {len(bars)} symbols")
            for symbol, df in list(bars.items())[:3]:
                print(f"  {symbol}: {len(df)} bars")
        except Exception as exc:
            print(f"FAIL: get_all_bars(): {exc}")


def main() -> int:
    print_header("Weekly Scanner Test Harness")
    sample_symbols = test_universe_file()
    test_email_env()
    test_market_data()
    whc = import_weekly_module()
    test_weekly_helpers(whc, sample_symbols)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
