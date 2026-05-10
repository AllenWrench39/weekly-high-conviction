# Weekly High Conviction Screener Framework

## Purpose

This job is designed to find a small set of weekly long setups with:

- strong fundamentals
- constructive technical structure
- reasonable macro alignment
- controlled recent drawdown
- defined trade levels for entry, stop, and targets

## End-to-End Flow

1. Initialize Alpaca clients for asset discovery and market data.
2. Build a stock universe from active US equities.
3. Fetch daily bars in batches from Alpaca.
4. Loop through each symbol and apply filters:
   - upcoming earnings exclusion
   - minimum market cap
   - minimum average volume
   - conviction score threshold
   - recent drawdown filter
5. Calculate trade levels and risk/reward.
6. Rank qualified names.
7. Output results to:
   - email
   - CSV log
   - markdown report

## Code Map

- `init_alpaca()`
  - connects to Alpaca trading and data clients

- `get_universe()`
  - builds the eligible symbol list

- `get_bars_batch()`
  - downloads historical daily bars in batches

- `get_fundamentals()`
  - pulls company-level fundamentals from `yfinance`

- `check_earnings_soon()`
  - removes names with near-term earnings risk

- `score_fundamental()`
  - rates ROE, debt-to-equity, and earnings growth

- `score_technical()`
  - rates RSI, moving averages, and ATR behavior

- `score_macro()`
  - applies a sector-aware macro tilt

- `score_risk_adjusted()`
  - rates Sortino ratio and recent drawdown

- `calculate_conviction()`
  - combines the four score buckets into one weighted score

- `calculate_rtr()`
  - sets entry, stop, target levels, and risk/reward

- `generate_perspectives()`
  - creates narrative comments for the email and markdown report

- `screen_stocks()`
  - coordinates filtering, scoring, ranking, and final selection

- `send_weekly_email()` / `export_to_markdown()` / `log_to_csv()`
  - publish results

## Current Strengths

- Clear pipeline from universe to ranked ideas
- Good use of Alpaca batch bar requests
- Scoring model is easy to reason about
- Outputs are useful for both review and automation

## Main Reliability Risks

- Weekend quote lookup can remove otherwise valid candidates.
- Per-symbol `yfinance` calls will make GitHub runs slow and fragile.
- Macro and ETF checks currently add a lot of repeated external requests.
- The script mixes configuration, data access, scoring, and reporting in one file.

## Improvement Roadmap

### Phase 1: Reliability

- use the latest bar close when live quotes are unavailable
- precompute shared macro inputs once per run
- stop making repeated ETF detection calls through `yfinance`
- make environment variable names consistent across scripts and workflows

### Phase 2: Performance

- cache `yfinance` responses per symbol
- separate universe filtering from deep fundamental checks
- add retry and timeout handling around external APIs
- measure runtime per stage

### Phase 3: Maintainability

- split into modules:
  - `config.py`
  - `data_sources.py`
  - `scoring.py`
  - `reporting.py`
  - `main.py`
- replace globals with a config object
- add unit tests for scoring and risk calculations

### Phase 4: Signal Quality

- validate score thresholds with historical results
- add relative strength versus SPY or sector ETF
- add sector concentration guardrails
- add basic result tracking to compare picks versus benchmark

## GitHub Actions Notes

The weekly workflow runs `weekly_high_conviction_ver2.py` and expects these repository secrets:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `RECIPIENT_EMAIL`
- `SENDER_EMAIL`
- `SENDER_PASSWORD`

The workflow maps `SENDER_EMAIL` and `SENDER_PASSWORD` into the variable names that the weekly script expects.
