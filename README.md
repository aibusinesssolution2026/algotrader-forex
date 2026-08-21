# AlgoTrader — Paper-First Event-Driven Trading Platform

46 unit + integration tests. Paper trading is the hard default; live execution
is triple-gated behind environment variables and never reachable from `run.py`.

## Quick start
    pip install -r requirements.txt
    python -m pytest tests/ -q                    # verify everything
    python run.py fetch    --symbols AAPL MSFT INFY.NS   # real Yahoo data
    python run.py backtest --symbols AAPL MSFT
    python run.py paper    --symbols AAPL MSFT    # simulated live session
    # add --offline anywhere to use synthetic data (no network needed)

## Docker
    docker build -t algotrader .        # build fails if any test fails
    docker run --rm -it algotrader      # paper session, synthetic data

## Architecture
    data/fetcher.py       Yahoo Finance fetch + synthetic generator + parquet cache
    data/preprocess.py    cleaning, alignment, MACD/ATR/RSI/vol-bracket features
    backtest/engine.py    bar-by-bar backtester with slippage + fee penalties
    models/ensemble.py    3-tier predictor: AI API -> boosted trees -> MACD rule
    models/evaluate.py    walk-forward accuracy / precision / hit-rate
    state/portfolio.py    thread-safe positions, cash, daily PnL anchor
    risk/engine.py        THE gatekeeper: ATR sizing, caps, latching circuit breaker
    execution/broker.py   PaperBroker (default) + env-gated Alpaca stub
    execution/executor.py bounded event queue, stop monitoring, token-gated orders
    dashboard/            rich terminal panel + optional Streamlit app

## Risk rules (frozen at startup, `algotrader/config.py`)
- 1% of equity risked per trade, stops at 2×ATR
- ≤10% equity notional per position, ≤5 concurrent positions
- ≤5% total open portfolio risk
- 3% daily drawdown → latching circuit breaker: flatten everything, halt process

## Enabling live trading (deliberately inconvenient)
Set ALL of: `ALGO_TRADING_MODE=live`, `ALGO_LIVE_CONFIRM=I_UNDERSTAND_THE_RISKS`,
`ALPACA_API_KEY`, `ALPACA_API_SECRET` — and even then orders route to Alpaca's
**paper** endpoint unless `ALPACA_REAL_MONEY=yes`. Nothing in this repo profits
reliably out of the box; validate for months in paper before considering it.

## Plugging in your AI predictor
Pass any callable `(feature_row) -> {-1,0,1}` as `EnsemblePredictor(primary=...)`.
Timeouts, rate limits, and invalid outputs automatically fail over to the
boosted-tree tier, then the MACD rule. Failover is tested at <1s.
