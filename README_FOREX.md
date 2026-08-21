# AlgoTrader Forex — Deployment Guide

61 tests. Paper-first: nothing in this repo can touch real money without
three explicit environment overrides you would have to set yourself.

## What "deploy it live" means here
"Live" = the continuous loop running 24/5 against **real market data** with
**simulated money** (paper broker, or an OANDA *practice* account). That is
the correct and only sensible first deployment. Judge it for 2-3 months of
real market conditions before any thought of real capital.

## Local verification
    pip install -r requirements.txt
    python -m pytest tests/ -q                       # expect 61 passed
    python run_forex.py backtest --pairs EURUSD GBPUSD USDJPY
    python run_forex.py paper --pairs EURUSD USDJPY  # replay stream
    # add --offline anywhere for synthetic data

## Deploy: Docker (recommended)
    docker compose up -d --build
    docker compose logs -f trader
Dashboard: pip install streamlit, then
    streamlit run algotrader/dashboard/streamlit_app.py
(reads ./runtime/metrics.json written by the loop).

## Deploy: bare VPS
    sudo useradd -r trader && sudo mkdir -p /opt/algotrader
    # copy repo to /opt/algotrader, pip install -r requirements.txt
    sudo cp algotrader.service /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now algotrader
    journalctl -u algotrader -f

## Routing to an OANDA practice account
1. Free practice account: https://www.oanda.com -> demo/practice signup.
2. Generate an API token (Manage API Access) for the PRACTICE environment.
3. Set: ALGO_TRADING_MODE=live, ALGO_LIVE_CONFIRM=I_UNDERSTAND_THE_RISKS,
   OANDA_API_TOKEN, OANDA_ACCOUNT_ID.
4. The broker targets api-fxpractice.oanda.com. The real-money endpoint
   additionally requires OANDA_REAL_MONEY=yes — do not set it.

## Forex-specific protections (on top of the core risk engine)
- Hard 5x leverage cap (brokers offer 30-50x; that is how accounts die)
- Margin-aware sizing clamp — never sized beyond affordable margin
- Weekend gap lockout (no orders Sat / Sun before 21:00 UTC / Fri after)
- Rollover-window entry block (22:00 UTC spread blow-outs)
- Minimum 5-pip stops (tighter is spread noise)
- Spread-crossing cost model in the paper broker (pips, not commissions)
- Same latching 3% daily circuit breaker: flatten everything, halt process

## Expectations, honestly
The bundled signals are placeholders. Offline backtests show ~50% hit-rate
and slightly negative returns after spreads — that is the true baseline for
naive strategies in FX, and roughly 70-80% of retail forex traders lose
money. This platform's job is to keep losses small and measurable while you
search for an edge, not to generate one.

## Multi-timeframe agent swarm (Phase 7)
Eight agents watch 1min/3min/5min/15min/30min/1h/2h/3h bars. Signals fire
only on confluence: slow "anchor" timeframes gate direction and hold veto
power; a weighted vote (slower = heavier) must clear 60% agreement. All
resampling is strictly causal — agents never see a bar still forming.

Compare solo vs swarm on YOUR data (Yahoo serves ~5 days of 1m FX bars):
    python run_forex.py mtf --pairs EURUSD GBPUSD USDJPY

What confluence does: cuts trade count ~20x (less spread bleed) and filters
counter-trend noise. What it does NOT do: create edge. On synthetic
mean-reverting data the solo fast agent actually wins — run the comparison
on real data repeatedly across weeks before concluding anything. Tune via
MultiTimeframeSwarm(timeframes=..., n_anchors=..., min_agreement=...).

## Auto-sweep (Phase 8)
    python run_forex.py sweep --pairs EURUSD GBPUSD USDJPY
Sweeps all 48 combinations of timeframe sets x anchors {1,2,3} x agreement
{50,60,75,90}%. Agent signals are computed once and cached, so the whole
sweep costs little more than a single evaluation. Walk-forward discipline:
models train on the first 60% of data, configs are RANKED on the next 20%,
and only the top-5 are re-scored on the final untouched 20%. The printed
"overfit gap" (rank-1 validation return minus its test return) tells you
how much of the winner's ranking was luck. Re-run the sweep across several
different weeks; a config that stays top-5 repeatedly is worth deploying —
a config that won once is noise.

## TradingView webhook bridge (Phase 9)
    export TV_WEBHOOK_SECRET=$(python -c "import secrets; print(secrets.token_hex(24))")
    python run_forex.py serve --pairs EURUSD GBPUSD USDJPY --port 8422
Expose the port (TradingView needs a public URL — a $5 VPS or an ngrok/
cloudflared tunnel works). In TradingView: Alert -> Notifications ->
Webhook URL = http://YOUR_HOST:8422/webhook, message body:
    {"secret": "YOUR_SECRET", "symbol": "{{ticker}}",
     "action": "buy", "price": {{close}}}
(one alert per condition; use "sell" and "close" bodies for the others; add
"atr": <value> if your script computes it, else a conservative default stop
is applied).

Protections: constant-time secret check, strict schema, unknown-symbol
rejection, 60s per-symbol cooldown against alert storms, and every alert
passes the full risk engine — TradingView proposes, the risk engine
disposes. GET /health shows halt state and open positions.

This makes TradingView a *strategy source competing on equal terms* with
the swarm in paper trading. After some weeks, compare their trade logs:
whichever survives with better risk-adjusted results earns the capital.
The usual truth applies: public TradingView indicators are seen by
millions simultaneously and carry no inherent edge — measure, don't trust.
