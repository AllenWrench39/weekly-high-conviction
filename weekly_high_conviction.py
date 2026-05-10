"""
Weekly High-Conviction Trading System
- Primary data source: yfinance
- Backup data source: Alpaca (optional)
- Universe source: universe.json
- Outputs: email + CSV log + markdown report
"""

import csv
import json
import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Alpaca is optional backup only
ALPACA_AVAILABLE = False
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


# ============================================================================
# CONFIG
# ============================================================================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "")

UNIVERSE_JSON_PATH = os.getenv("UNIVERSE_JSON_PATH", "universe.json")
DAILY_SCREENER_CSV = os.getenv("DAILY_SCREENER_CSV", "")

MIN_CONVICTION_SCORE = float(os.getenv("MIN_CONVICTION_SCORE", "75"))
MAX_DRAWDOWN_FILTER = float(os.getenv("MAX_DRAWDOWN_FILTER", "0.08"))
MIN_MARKET_CAP_BILLIONS = float(os.getenv("MIN_MARKET_CAP_BILLIONS", "5"))
MIN_AVG_VOLUME = int(os.getenv("MIN_AVG_VOLUME", "500000"))
LOOKBACK_TECHNICAL_DAYS = int(os.getenv("LOOKBACK_TECHNICAL_DAYS", "180"))
EXCLUDE_EARNINGS_DAYS = int(os.getenv("EXCLUDE_EARNINGS_DAYS", "7"))
ENABLE_SPY_TREND_FILTER = os.getenv("ENABLE_SPY_TREND_FILTER", "true").lower() == "true"
SPY_BEARISH_CONVICTION_BOOST = float(os.getenv("SPY_BEARISH_CONVICTION_BOOST", "10"))

WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.40
WEIGHT_MACRO = 0.15
WEIGHT_RISK_ADJUSTED = 0.15

YF_BATCH_SIZE = int(os.getenv("YF_BATCH_SIZE", "75"))
ALPACA_BATCH_SIZE = int(os.getenv("ALPACA_BATCH_SIZE", "100"))


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("weekly_high_conviction.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

alpaca_data_client = None


# ============================================================================
# ALPACA BACKUP
# ============================================================================

def init_alpaca_backup() -> bool:
    global alpaca_data_client

    if not ALPACA_AVAILABLE:
        logger.info("Alpaca package not available - yfinance only mode")
        return False

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.info("Alpaca credentials not set - yfinance only mode")
        return False

    try:
        alpaca_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        logger.info("Alpaca backup initialized")
        return True
    except Exception as exc:
        logger.warning(f"Alpaca backup init failed: {exc}")
        return False


# ============================================================================
# UNIVERSE
# ============================================================================

def load_universe() -> List[Dict]:
    if not os.path.exists(UNIVERSE_JSON_PATH):
        logger.error(f"Universe file not found: {UNIVERSE_JSON_PATH}")
        return []

    try:
        with open(UNIVERSE_JSON_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        raw_symbols = payload.get("symbols", [])
        universe = []

        for item in raw_symbols:
            if not isinstance(item, dict):
                continue

            symbol = str(item.get("symbol", "")).strip().upper()
            symbol_type = str(item.get("type", "")).strip().upper()

            if not symbol or symbol_type != "EQUITY":
                continue

            if not symbol.replace(".", "").replace("-", "").isalnum():
                continue

            universe.append(
                {
                    "symbol": symbol,
                    "name": item.get("name", ""),
                    "sector": item.get("sector", ""),
                    "industry": item.get("industry", ""),
                    "market_cap": item.get("market_cap"),
                    "avg_volume": item.get("avg_volume"),
                }
            )

        deduped = {item["symbol"]: item for item in universe}
        result = list(deduped.values())
        logger.info(f"Loaded {len(result)} equities from {UNIVERSE_JSON_PATH}")
        return result
    except Exception as exc:
        logger.error(f"Failed to load universe file: {exc}")
        return []


def load_daily_screener_tickers() -> List[str]:
    if not DAILY_SCREENER_CSV:
        return []

    if not os.path.exists(DAILY_SCREENER_CSV):
        logger.info(f"Daily screener file not found: {DAILY_SCREENER_CSV}")
        return []

    try:
        df = pd.read_csv(DAILY_SCREENER_CSV)
        ticker_col = None
        for col in ["ticker", "Ticker", "symbol", "Symbol", "TICKER"]:
            if col in df.columns:
                ticker_col = col
                break

        if not ticker_col:
            return []

        tickers = [
            str(x).strip().upper()
            for x in df[ticker_col].dropna().unique().tolist()
            if str(x).strip()
        ]
        logger.info(f"Loaded {len(tickers)} tickers from daily screener")
        return tickers[:50]
    except Exception as exc:
        logger.warning(f"Daily screener load error: {exc}")
        return []


def build_universe() -> List[Dict]:
    universe = load_universe()
    if not universe:
        return []

    by_symbol = {item["symbol"]: item for item in universe}

    for symbol in load_daily_screener_tickers():
        if symbol not in by_symbol:
            by_symbol[symbol] = {
                "symbol": symbol,
                "name": "",
                "sector": "",
                "industry": "",
                "market_cap": None,
                "avg_volume": None,
            }

    result = list(by_symbol.values())
    logger.info(f"Final universe size: {len(result)}")
    return result


# ============================================================================
# DATA FETCHING
# ============================================================================

def _normalize_download_frame(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    try:
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            if ticker not in df.columns.get_level_values(0):
                return None
            sub = df[ticker].copy()
        else:
            sub = df.copy()

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        sub = sub.rename(columns=rename_map)

        needed = ["open", "high", "low", "close", "volume"]
        if not all(col in sub.columns for col in needed):
            return None

        sub = sub[needed].dropna()
        if len(sub) < 50:
            return None

        return sub.sort_index()
    except Exception:
        return None


def get_bars_yfinance_batch(tickers: List[str], num_days: int = LOOKBACK_TECHNICAL_DAYS) -> Dict[str, pd.DataFrame]:
    result = {}
    period_days = max(num_days + 40, 260)

    for i in range(0, len(tickers), YF_BATCH_SIZE):
        batch = tickers[i:i + YF_BATCH_SIZE]
        try:
            data = yf.download(
                tickers=batch,
                period=f"{period_days}d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            for ticker in batch:
                normalized = _normalize_download_frame(data, ticker)
                if normalized is not None:
                    result[ticker] = normalized
        except Exception as exc:
            logger.warning(f"yfinance batch failed for batch starting {batch[0]}: {exc}")

    return result


def get_bars_alpaca_batch(tickers: List[str], num_days: int = LOOKBACK_TECHNICAL_DAYS) -> Dict[str, pd.DataFrame]:
    if not alpaca_data_client or not tickers:
        return {}

    result = {}
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days + 40)

        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
        )
        bars = alpaca_data_client.get_stock_bars(request)

        for ticker in tickers:
            if ticker in bars.data and bars.data[ticker]:
                rows = bars.data[ticker]
                df = pd.DataFrame(
                    [
                        {
                            "open": float(x.open),
                            "high": float(x.high),
                            "low": float(x.low),
                            "close": float(x.close),
                            "volume": float(x.volume),
                            "timestamp": x.timestamp,
                        }
                        for x in rows
                    ]
                )
                if len(df) >= 50:
                    result[ticker] = df.set_index("timestamp").sort_index()
    except Exception as exc:
        logger.warning(f"Alpaca backup bars failed: {exc}")

    return result


def get_all_bars(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    logger.info("Fetching historical bars from yfinance...")
    bars = get_bars_yfinance_batch(tickers, LOOKBACK_TECHNICAL_DAYS)

    missing = [ticker for ticker in tickers if ticker not in bars]
    if missing:
        logger.info(f"yfinance missing bars for {len(missing)} symbols")
        logger.info("Trying Alpaca backup for missing symbols...")
        for i in range(0, len(missing), ALPACA_BATCH_SIZE):
            batch = missing[i:i + ALPACA_BATCH_SIZE]
            bars.update(get_bars_alpaca_batch(batch, LOOKBACK_TECHNICAL_DAYS))

    logger.info(f"Got bars for {len(bars)} symbols")
    return bars


def get_latest_price(ticker: str, bars: Optional[pd.DataFrame] = None) -> Optional[float]:
    if bars is not None and not bars.empty:
        try:
            price = float(bars["close"].iloc[-1])
            if price > 0:
                return price
        except Exception:
            pass

    try:
        info = yf.Ticker(ticker).fast_info
        if info:
            for key in ["lastPrice", "regularMarketPrice", "previousClose"]:
                value = info.get(key)
                if value and value > 0:
                    return float(value)
    except Exception:
        pass

    if alpaca_data_client:
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quotes = alpaca_data_client.get_stock_latest_quote(request)
            if ticker in quotes:
                ask = quotes[ticker].ask_price
                if ask and ask > 0:
                    return float(ask)
        except Exception:
            pass

    return None


# ============================================================================
# FUNDAMENTALS / EARNINGS / MACRO
# ============================================================================

def get_yf_info(ticker: str) -> Dict:
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def get_fundamentals(ticker: str, universe_meta: Dict) -> Dict:
    info = get_yf_info(ticker)

    return {
        "roe": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "earnings_growth": info.get("earningsGrowth"),
        "market_cap": universe_meta.get("market_cap") or info.get("marketCap"),
        "avg_volume": universe_meta.get("avg_volume") or info.get("averageVolume") or info.get("averageDailyVolume10Day"),
        "sector": universe_meta.get("sector") or info.get("sector", ""),
        "industry": universe_meta.get("industry") or info.get("industry", ""),
    }


def check_earnings_soon(ticker: str) -> bool:
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None or (isinstance(cal, dict) and not cal):
            return False

        earnings_date = None
        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")
            if isinstance(earnings_date, list) and earnings_date:
                earnings_date = earnings_date[0]

        if earnings_date:
            if hasattr(earnings_date, "date"):
                days_until = (earnings_date.date() - datetime.now().date()).days
            else:
                days_until = (earnings_date - datetime.now().date()).days
            return 0 <= days_until <= EXCLUDE_EARNINGS_DAYS
    except Exception:
        return False

    return False


def build_macro_context(lookback_days: int = 60) -> Dict:
    context = {
        "tnx_current": None,
        "tnx_trend": 0.0,
        "oil_price": None,
        "spy_weekly_perf": 0.0,
        "spy_bullish": True,
        "spy_pct_above_ma200": 0.0,
    }

    try:
        tnx = yf.download("^TNX", period=f"{lookback_days}d", progress=False)
        if not tnx.empty:
            if isinstance(tnx.columns, pd.MultiIndex):
                tnx.columns = tnx.columns.get_level_values(0)
            context["tnx_current"] = float(tnx["Close"].iloc[-1]) / 100
            if len(tnx) >= 20:
                context["tnx_trend"] = float(tnx["Close"].iloc[-1] - tnx["Close"].iloc[-20]) / 100
    except Exception:
        pass

    try:
        oil = yf.download("CL=F", period="30d", progress=False)
        if not oil.empty:
            if isinstance(oil.columns, pd.MultiIndex):
                oil.columns = oil.columns.get_level_values(0)
            context["oil_price"] = float(oil["Close"].iloc[-1])
    except Exception:
        pass

    try:
        spy = yf.download("SPY", period="1y", progress=False)
        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)

            if len(spy) >= 5:
                week_ago = float(spy["Close"].iloc[-5])
                current = float(spy["Close"].iloc[-1])
                context["spy_weekly_perf"] = (current - week_ago) / week_ago * 100

            if len(spy) >= 200:
                current = float(spy["Close"].iloc[-1])
                ma200 = float(spy["Close"].rolling(200).mean().iloc[-1])
                context["spy_bullish"] = current > ma200
                context["spy_pct_above_ma200"] = (current - ma200) / ma200 * 100
    except Exception:
        pass

    return context


# ============================================================================
# INDICATORS
# ============================================================================

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).rolling(period).mean()


def calculate_max_drawdown(prices: pd.Series) -> float:
    cummax = prices.expanding().max()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 5.0

    downside_dev = np.sqrt(np.mean(downside_returns ** 2))
    if downside_dev == 0:
        return 5.0

    return max(np.mean(excess_returns) / downside_dev * np.sqrt(252), 0.0)


# ============================================================================
# SCORING
# ============================================================================

def score_fundamental(fundamentals: Dict) -> Tuple[float, str]:
    score = 0.0
    details = []

    roe = fundamentals.get("roe")
    if roe is not None:
        if roe > 0.18:
            score += 35
            details.append(f"ROE {roe:.1%}")
        elif roe > 0.12:
            score += 20
        elif roe > 0.05:
            score += 10

    debt_to_equity = fundamentals.get("debt_to_equity")
    if debt_to_equity is not None:
        de_norm = debt_to_equity / 100 if debt_to_equity > 5 else debt_to_equity
        if de_norm < 0.8:
            score += 30
            details.append(f"D/E {de_norm:.2f}")
        elif de_norm < 1.5:
            score += 15

    earnings_growth = fundamentals.get("earnings_growth")
    if earnings_growth is not None:
        if earnings_growth > 0.10:
            score += 35
            details.append(f"EG {earnings_growth:.1%}")
        elif earnings_growth > 0.05:
            score += 20
        elif earnings_growth > 0:
            score += 10

    return min(score, 100), ", ".join(details) if details else "N/A"


def score_technical(df: pd.DataFrame) -> Tuple[float, str]:
    score = 0.0
    details = []

    try:
        rsi = calculate_rsi(df)
        atr = calculate_atr(df)
        ma_50 = df["close"].rolling(50).mean()
        ma_200 = df["close"].rolling(min(200, len(df) - 1)).mean()

        current_close = df["close"].iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_ma50 = ma_50.iloc[-1]
        current_ma200 = ma_200.iloc[-1] if not pd.isna(ma_200.iloc[-1]) else current_ma50

        if 40 <= current_rsi <= 70:
            score += 35
            details.append(f"RSI {current_rsi:.0f}")
        elif 35 < current_rsi < 75:
            score += 20

        if current_close > current_ma50:
            score += 20
            if current_ma50 > current_ma200:
                score += 20
                details.append("Above 50/200 MA")

        current_atr = atr.iloc[-1]
        if current_atr > 0:
            atr_prev = atr.iloc[-10] if len(atr) >= 10 else atr.iloc[0]
            if atr_prev > 0:
                atr_ratio = current_atr / atr_prev
                if 1.0 <= atr_ratio < 1.4:
                    score += 25
                    details.append("ATR rising")
                elif 0.8 < atr_ratio < 1.0:
                    score += 15
    except Exception as exc:
        logger.debug(f"Technical scoring error: {exc}")

    return min(score, 100), ", ".join(details) if details else "N/A"


def score_macro(sector: str, macro: Dict) -> Tuple[float, str]:
    score = 50.0
    details = []

    current_yield = macro.get("tnx_current")
    yield_trend = macro.get("tnx_trend", 0.0)
    oil_price = macro.get("oil_price")

    if current_yield is None:
        return score, "Neutral"

    if "Technology" in sector:
        if current_yield < 0.04 and yield_trend < 0.002:
            score = 80
            details.append(f"Tech: rates favorable ({current_yield:.2%})")
        elif current_yield < 0.045:
            score = 65
        else:
            score = 40
            details.append(f"Tech: rates headwind ({current_yield:.2%})")
    elif "Energy" in sector:
        if oil_price is not None and oil_price > 75:
            score = 80
            details.append(f"Energy: oil ${oil_price:.0f}")
        elif oil_price is not None and oil_price > 65:
            score = 65
        else:
            score = 40
    elif "Financial" in sector:
        if current_yield > 0.04:
            score = 75
            details.append("Financials: rates favorable")
        else:
            score = 55
    elif "Healthcare" in sector or "Consumer Staples" in sector:
        score = 65
        details.append("Defensive sector")

    return min(score, 100), ", ".join(details) if details else "Neutral"


def score_risk_adjusted(df: pd.DataFrame) -> Tuple[float, str]:
    score = 0.0
    details = []

    try:
        returns = df["close"].pct_change().dropna()
        sortino = calculate_sortino_ratio(returns)

        if sortino > 1.0:
            score += 50
            details.append(f"Sortino {sortino:.2f}")
        elif sortino > 0.5:
            score += 30
        elif sortino > 0:
            score += 15

        recent_prices = df["close"].iloc[max(0, len(df) - 63):]
        max_dd = calculate_max_drawdown(recent_prices)

        if max_dd > -0.12:
            score += 50
            details.append(f"DD {max_dd:.1%}")
        elif max_dd > -0.18:
            score += 30
        elif max_dd > -0.25:
            score += 15
    except Exception as exc:
        logger.debug(f"Risk scoring error: {exc}")

    return min(score, 100), ", ".join(details) if details else "N/A"


def calculate_conviction(df: pd.DataFrame, fundamentals: Dict, macro: Dict) -> Tuple[float, Dict]:
    fund_score, fund_detail = score_fundamental(fundamentals)
    tech_score, tech_detail = score_technical(df)
    macro_score, macro_detail = score_macro(fundamentals.get("sector", ""), macro)
    risk_score, risk_detail = score_risk_adjusted(df)

    conviction = (
        fund_score * WEIGHT_FUNDAMENTAL
        + tech_score * WEIGHT_TECHNICAL
        + macro_score * WEIGHT_MACRO
        + risk_score * WEIGHT_RISK_ADJUSTED
    )

    return conviction, {
        "fundamental": fund_score,
        "technical": tech_score,
        "macro": macro_score,
        "risk_adjusted": risk_score,
        "overall": conviction,
        "fundamental_detail": fund_detail,
        "technical_detail": tech_detail,
        "macro_detail": macro_detail,
        "risk_detail": risk_detail,
    }


# ============================================================================
# TRADE LEVELS
# ============================================================================

def calculate_rtr(current_price: float, df: pd.DataFrame, atr_14: float) -> Dict:
    result = {
        "entry": None,
        "stop": None,
        "tp1": None,
        "tp2": None,
        "rr_ratio": None,
        "risk_distance": None,
    }

    try:
        entry = current_price * 1.001
        result["entry"] = entry

        lowest_10d = df["close"].tail(10).min()
        atr_stop = entry - (1.5 * atr_14) if atr_14 and atr_14 > 0 else entry * 0.95
        stop = max(min(lowest_10d, atr_stop), entry * 0.85)
        result["stop"] = stop

        risk_distance = entry - stop
        if risk_distance > 0 and risk_distance < entry * 0.15:
            result["tp1"] = entry + (1.5 * risk_distance)
            result["tp2"] = entry + (2.5 * risk_distance)
            result["risk_distance"] = risk_distance
            result["rr_ratio"] = (result["tp1"] - entry) / risk_distance
        else:
            result["stop"] = entry * 0.95
            result["risk_distance"] = entry * 0.05
            result["tp1"] = entry * 1.075
            result["tp2"] = entry * 1.125
            result["rr_ratio"] = 1.5
    except Exception as exc:
        logger.debug(f"RTR calc error: {exc}")

    return result


# ============================================================================
# PERSPECTIVES
# ============================================================================

def generate_perspectives(ticker: str, breakdown: Dict, df: pd.DataFrame) -> Dict[str, str]:
    perspectives = {}

    try:
        rsi = calculate_rsi(df).iloc[-1]
        atr = calculate_atr(df).iloc[-1]
        atr_pct = (atr / df["close"].iloc[-1]) * 100

        if breakdown["technical"] > 70:
            perspectives["Momentum Quant"] = (
                f"{ticker} shows strong momentum with RSI at {rsi:.0f} in an attractive range. "
                f"Trend alignment supports continuation."
            )
        elif breakdown["technical"] > 50:
            perspectives["Momentum Quant"] = (
                f"{ticker} has constructive momentum with RSI at {rsi:.0f}. "
                f"Watch for follow-through confirmation."
            )
        else:
            perspectives["Momentum Quant"] = (
                f"{ticker} has weaker momentum with RSI at {rsi:.0f}. "
                f"Trend confirmation is still needed."
            )

        if rsi > 65:
            perspectives["Mean Reversion Quant"] = (
                f"{ticker} looks extended at RSI {rsi:.0f}. "
                f"A pullback toward moving averages would improve the setup."
            )
        elif rsi < 45:
            perspectives["Mean Reversion Quant"] = (
                f"{ticker} is nearer a mean-reversion buy zone at RSI {rsi:.0f}. "
                f"Bounce potential is improving."
            )
        else:
            perspectives["Mean Reversion Quant"] = (
                f"{ticker} is in neutral RSI territory at {rsi:.0f}. "
                f"No strong mean-reversion edge here."
            )

        if atr_pct > 3.5:
            perspectives["Volatility Quant"] = (
                f"{ticker} has elevated volatility at {atr_pct:.1f}% ATR. "
                f"Smaller size is appropriate."
            )
        elif atr_pct < 1.5:
            perspectives["Volatility Quant"] = (
                f"{ticker} has compressed volatility at {atr_pct:.1f}% ATR. "
                f"Vol expansion may follow."
            )
        else:
            perspectives["Volatility Quant"] = (
                f"{ticker} has normal volatility at {atr_pct:.1f}% ATR. "
                f"Standard swing sizing fits."
            )

        if breakdown["fundamental"] > 70:
            perspectives["Fundamental Quant"] = (
                f"{ticker} screens as high quality on profitability, leverage, and growth."
            )
        elif breakdown["fundamental"] > 40:
            perspectives["Fundamental Quant"] = (
                f"{ticker} has mixed but workable fundamentals."
            )
        else:
            perspectives["Fundamental Quant"] = (
                f"{ticker} has weaker fundamentals, so risk control matters more."
            )
    except Exception as exc:
        logger.debug(f"Perspectives error: {exc}")
        for label in ["Momentum Quant", "Mean Reversion Quant", "Volatility Quant", "Fundamental Quant"]:
            perspectives[label] = f"Analysis incomplete for {ticker}."

    return perspectives


# ============================================================================
# SCREENING ENGINE
# ============================================================================

def screen_stocks(universe: List[Dict], macro: Dict) -> List[Dict]:
    qualified = []
    tickers = [item["symbol"] for item in universe]
    meta_by_symbol = {item["symbol"]: item for item in universe}

    logger.info(f"Starting weekly screen on {len(tickers)} symbols")
    all_bars = get_all_bars(tickers)

    processed = 0
    for ticker, bars in all_bars.items():
        processed += 1
        if processed % 100 == 0:
            logger.info(f"Processed {processed}/{len(all_bars)} | qualified={len(qualified)}")

        try:
            if check_earnings_soon(ticker):
                continue

            fundamentals = get_fundamentals(ticker, meta_by_symbol.get(ticker, {}))

            market_cap = fundamentals.get("market_cap")
            if not market_cap or market_cap / 1e9 < MIN_MARKET_CAP_BILLIONS:
                continue

            avg_volume = fundamentals.get("avg_volume")
            if not avg_volume or avg_volume < MIN_AVG_VOLUME:
                continue

            conviction, breakdown = calculate_conviction(bars, fundamentals, macro)
            if conviction < MIN_CONVICTION_SCORE:
                continue

            recent_prices = bars["close"].iloc[max(0, len(bars) - 63):]
            max_dd = calculate_max_drawdown(recent_prices)
            if max_dd < -MAX_DRAWDOWN_FILTER:
                continue

            current_price = get_latest_price(ticker, bars)
            if not current_price or current_price <= 0:
                continue

            atr_14 = calculate_atr(bars).iloc[-1]
            rtr = calculate_rtr(current_price, bars, atr_14)
            if not rtr["rr_ratio"] or rtr["rr_ratio"] < 1.0:
                continue

            perspectives = generate_perspectives(ticker, breakdown, bars)

            qualified.append(
                {
                    "ticker": ticker,
                    "conviction_score": conviction,
                    "breakdown": breakdown,
                    "perspectives": perspectives,
                    "market_cap_b": market_cap / 1e9,
                    "avg_volume": avg_volume,
                    "current_price": current_price,
                    "entry": rtr["entry"],
                    "stop": rtr["stop"],
                    "tp1": rtr["tp1"],
                    "tp2": rtr["tp2"],
                    "rr_ratio": rtr["rr_ratio"],
                    "risk_distance": rtr["risk_distance"],
                    "max_drawdown": max_dd,
                    "sector": fundamentals.get("sector", "N/A"),
                    "industry": fundamentals.get("industry", "N/A"),
                    "screening_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        except Exception as exc:
            logger.debug(f"Error processing {ticker}: {exc}")

    qualified.sort(key=lambda x: (x["rr_ratio"], x["conviction_score"]), reverse=True)
    logger.info(f"Screening complete: {len(qualified)} qualified trades")
    return qualified


# ============================================================================
# OUTPUTS
# ============================================================================

def log_to_csv(trades: List[Dict]) -> None:
    try:
        file_exists = os.path.exists("weekly_trades_log.csv")
        with open("weekly_trades_log.csv", "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "date",
                    "ticker",
                    "sector",
                    "current_price",
                    "entry",
                    "stop",
                    "tp1",
                    "tp2",
                    "conviction_score",
                    "rr_ratio",
                    "max_drawdown",
                    "market_cap_b",
                ],
            )
            if not file_exists:
                writer.writeheader()

            for t in trades:
                writer.writerow(
                    {
                        "date": t["screening_date"],
                        "ticker": t["ticker"],
                        "sector": t["sector"],
                        "current_price": f"{t['current_price']:.2f}",
                        "entry": f"{t['entry']:.2f}",
                        "stop": f"{t['stop']:.2f}",
                        "tp1": f"{t['tp1']:.2f}",
                        "tp2": f"{t['tp2']:.2f}",
                        "conviction_score": f"{t['conviction_score']:.2f}",
                        "rr_ratio": f"{t['rr_ratio']:.2f}",
                        "max_drawdown": f"{t['max_drawdown']:.4f}",
                        "market_cap_b": f"{t['market_cap_b']:.2f}",
                    }
                )
        logger.info(f"Logged {len(trades)} trades to CSV")
    except Exception as exc:
        logger.error(f"CSV log error: {exc}")


def export_to_markdown(trades: List[Dict], macro: Dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"weekly_report_{timestamp}.md"

    content = f"""# Weekly High-Conviction Trading Report

**Generated:** {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p ET')}

**Universe:** Prevalidated US equity universe from `universe.json`

**Parameters:**
| Filter | Value |
|--------|-------|
| Min Conviction Score | {MIN_CONVICTION_SCORE}/100 |
| Max Drawdown | {MAX_DRAWDOWN_FILTER:.0%} |
| Min Market Cap | ${MIN_MARKET_CAP_BILLIONS:.0f}B |
| Min Avg Volume | {MIN_AVG_VOLUME:,} |

**SPY Last Week:** {macro.get('spy_weekly_perf', 0.0):+.2f}%

---

## Top {min(5, len(trades)) if trades else 0} Trades

"""

    if not trades:
        content += f"No qualifying trades this week at the {MIN_CONVICTION_SCORE}/100 threshold.\n"
    else:
        content += "| # | Ticker | Sector | Current | Conviction | R:R | Entry | Stop | TP1 | TP2 |\n"
        content += "|---|--------|--------|---------|------------|-----|-------|------|-----|-----|\n"
        for i, t in enumerate(trades[:5], 1):
            content += (
                f"| {i} | {t['ticker']} | {t['sector'][:15]} | ${t['current_price']:.2f} | "
                f"{t['conviction_score']:.1f} | {t['rr_ratio']:.2f}x | ${t['entry']:.2f} | "
                f"${t['stop']:.2f} | ${t['tp1']:.2f} | ${t['tp2']:.2f} |\n"
            )

    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(content)

    logger.info(f"Markdown report saved: {filename}")
    return filename


def build_email_body(trades: List[Dict], macro: Dict) -> str:
    body = "=" * 85 + "\n"
    body += "WEEKLY HIGH-CONVICTION TRADING REPORT\n"
    body += "=" * 85 + "\n\n"
    body += f"Date: {datetime.now().strftime('%A, %B %d, %Y')}\n"
    body += "Universe: Prevalidated US equity universe from universe.json\n"
    body += f"Conviction Threshold: {MIN_CONVICTION_SCORE}/100\n"
    body += f"Max Drawdown Filter: {MAX_DRAWDOWN_FILTER:.0%}\n"
    body += f"Total Qualified Trades: {len(trades)}\n\n"
    body += f"SPY last week: {macro.get('spy_weekly_perf', 0.0):+.2f}%\n\n"

    if not trades:
        body += "NO HIGH-CONVICTION TRADES THIS WEEK.\n"
        body += "This is normal - not every week has setups meeting the threshold.\n"
        return body

    body += f"{'#':<3} {'Ticker':<7} {'Sector':<14} {'Current':<9} {'Score':<8} {'R:R':<6}\n"
    body += "-" * 85 + "\n"
    for i, t in enumerate(trades[:5], 1):
        sector_short = t["sector"][:12] if len(t["sector"]) > 12 else t["sector"]
        body += (
            f"{i:<3} {t['ticker']:<7} {sector_short:<14} ${t['current_price']:>7.2f} "
            f"{t['conviction_score']:>5.1f}   {t['rr_ratio']:.2f}x\n"
        )

    return body


def send_weekly_email(trades: List[Dict], macro: Dict) -> None:
    try:
        if not EMAIL_ADDRESS or not EMAIL_PASSWORD or not RECIPIENT_EMAIL:
            logger.warning("Email credentials missing - skipping email send")
            return

        if not trades:
            subject = f"[WEEKLY HIGH CONVICTION] No trades this week. Threshold: {MIN_CONVICTION_SCORE:.0f}/100"
        else:
            subject = f"[WEEKLY HIGH CONVICTION] Top {min(5, len(trades))} trades"

        body = build_email_body(trades, macro)

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECIPIENT_EMAIL
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        logger.info(f"Email sent to {RECIPIENT_EMAIL}")
    except Exception as exc:
        logger.error(f"Email failed: {exc}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    logger.info("=" * 85)
    logger.info("WEEKLY HIGH-CONVICTION TRADING SYSTEM")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 85)

    init_alpaca_backup()

    universe = build_universe()
    if not universe:
        logger.error("No usable universe loaded")
        return

    macro = build_macro_context()

    global MIN_CONVICTION_SCORE
    if ENABLE_SPY_TREND_FILTER and not macro.get("spy_bullish", True):
        original_threshold = MIN_CONVICTION_SCORE
        MIN_CONVICTION_SCORE = min(95, MIN_CONVICTION_SCORE + SPY_BEARISH_CONVICTION_BOOST)
        logger.warning(
            f"SPY below 200-day MA ({macro.get('spy_pct_above_ma200', 0.0):+.1f}%) - "
            f"raising threshold from {original_threshold} to {MIN_CONVICTION_SCORE}"
        )

    qualified_trades = screen_stocks(universe, macro)

    if qualified_trades:
        log_to_csv(qualified_trades)
        export_to_markdown(qualified_trades, macro)

    send_weekly_email(qualified_trades[:5] if qualified_trades else [], macro)

    logger.info("=" * 85)
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Qualified trades: {len(qualified_trades)}")
    logger.info("=" * 85)


if __name__ == "__main__":
    main()
