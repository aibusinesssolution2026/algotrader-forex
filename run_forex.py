#!/usr/bin/env python3
"""Forex entrypoints.

    python run_forex.py backtest --pairs EURUSD GBPUSD USDJPY
    python run_forex.py paper    --pairs EURUSD USDJPY          # replay stream
    python run_forex.py deploy   --pairs EURUSD USDJPY          # CONTINUOUS
                                                                 # practice loop

'deploy' is the long-running mode you point at a server: every POLL_SECONDS
it pulls the latest hourly bars from Yahoo, updates features, and routes
signals through the forex risk engine to the PaperForexBroker. To route to an
OANDA practice account instead, set the env vars described in README_FOREX.md
— real-money endpoints stay locked behind the same triple gate as always.
Add --offline to run everything on synthetic data.
"""
from __future__ import annotations

import argparse
import logging
import signal as os_signal
import sys
import time

from algotrader.config import AppConfig
from algotrader.backtest.engine import Backtester
from algotrader.data.forex import (fetch_forex, synthetic_fx, pair_spec,
                                   market_open)
from algotrader.data.preprocess import clean, engineer_features
from algotrader.models.ensemble import EnsemblePredictor, StatisticalFallback
from algotrader.models.evaluate import evaluate_predictor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_forex")

POLL_SECONDS = 300          # deploy mode: refresh data every 5 minutes


def get_fx_data(pairs, period, offline, interval="1h"):
    if not offline:
        try:
            data = fetch_forex(pairs, period=period, interval=interval)
            if data:
                return data
        except Exception as exc:
            log.warning("Forex fetch failed (%s); using synthetic", exc)
    return {pair_spec(p).symbol: synthetic_fx(p, n=2000) for p in pairs}


def build_models(feats, train_frac=0.7):
    models = {}
    for sym, f in feats.items():
        fb = StatisticalFallback()
        fb.fit(f.iloc[:int(len(f) * train_frac)])
        models[sym] = fb
    return models


def cmd_backtest(args):
    cfg = AppConfig()
    data = get_fx_data(args.pairs, args.period, args.offline)
    for sym, raw in data.items():
        feats = engineer_features(clean(raw))
        split = int(len(feats) * 0.7)
        fb = StatisticalFallback()
        fb.fit(feats.iloc[:split])
        with EnsemblePredictor(primary=None, fallback=fb) as ens:
            test = feats.iloc[split:]
            rep = evaluate_predictor(test, ens.predict)
            res = Backtester(cfg.backtest).run(
                test, lambda ts, row, s: ens.predict(row))
        log.info("%s | hit-rate %.1f%% | return %.2f%% | sharpe %.2f | "
                 "maxDD %.2f%% | trades %d",
                 sym, rep.hit_rate * 100, res.total_return * 100,
                 res.sharpe, res.max_drawdown * 100, len(res.trades))


def _make_stack(cfg, symbols):
    from algotrader.execution.forex_broker import PaperForexBroker
    from algotrader.execution.forex_executor import ForexExecutor
    from algotrader.risk.engine import RiskEngine
    from algotrader.risk.forex import ForexRiskEngine
    from algotrader.state.portfolio import Portfolio

    specs = {pair_spec(p).symbol: pair_spec(p) for p in symbols}
    pf = Portfolio(cash=cfg.backtest.initial_capital)
    risk = ForexRiskEngine(RiskEngine(cfg.risk, pf))
    ex = ForexExecutor(risk, PaperForexBroker(specs), pf, specs)
    return ex, pf, risk, specs


def cmd_paper(args):
    """Replay the held-out 30% of history through the live stack."""
    from algotrader.execution.executor import PriceEvent, SignalEvent
    from algotrader.dashboard.metrics import MetricsTracker
    from algotrader.dashboard.terminal import render
    from algotrader.dashboard.streamlit_app import write_metrics_file
    from rich.console import Console

    cfg = AppConfig()
    assert cfg.is_paper
    data = get_fx_data(args.pairs, args.period, args.offline)
    feats = {s: engineer_features(clean(df)) for s, df in data.items()}
    models = build_models(feats)
    ex, pf, risk, _ = _make_stack(cfg, list(feats))
    tracker, console = MetricsTracker(ex), Console()
    ens = EnsemblePredictor(primary=None)

    try:
        n = min(len(f) for f in feats.values())
        for i in range(int(n * 0.7), n):
            for sym, f in feats.items():
                row = f.iloc[i]
                ex.process(PriceEvent(sym, float(row["close"]), ts=row.name))
                if risk.halted:
                    break
                ens.fallback = models[sym]
                sig = ens.predict(row)
                if sig != 0 and sym not in pf.positions:
                    ex.process(SignalEvent(sym, sig, float(row["close"]),
                                           float(row["atr_14"]), ts=row.name))
            m = tracker.snapshot()
            write_metrics_file({**m.__dict__,
                                "equity_history":
                                    list(tracker.equity_history)[-500:]})
            if i % 25 == 0 or risk.halted:
                console.print(render(tracker))
            if risk.halted:
                log.critical("Session terminated by circuit breaker.")
                break
    finally:
        ens.close()
        ex.stop()
    console.print(render(tracker))


def cmd_deploy(args):
    """Continuous paper loop on live Yahoo data. Ctrl-C or SIGTERM to stop."""
    from algotrader.execution.executor import PriceEvent, SignalEvent
    from algotrader.dashboard.metrics import MetricsTracker
    from algotrader.dashboard.terminal import render
    from algotrader.dashboard.streamlit_app import write_metrics_file
    from rich.console import Console

    cfg = AppConfig()
    log.info("Trading mode: %s (paper=%s)", cfg.trading_mode, cfg.is_paper)
    data = get_fx_data(args.pairs, args.period, args.offline)
    feats = {s: engineer_features(clean(df)) for s, df in data.items()}
    models = build_models(feats, train_frac=1.0)   # train on all history
    ex, pf, risk, _ = _make_stack(cfg, list(feats))
    tracker, console = MetricsTracker(ex), Console()
    ens = EnsemblePredictor(primary=None)
    running = {"on": True}
    os_signal.signal(os_signal.SIGTERM, lambda *_: running.update(on=False))
    os_signal.signal(os_signal.SIGINT, lambda *_: running.update(on=False))

    seen_bars: dict[str, object] = {}
    try:
        while running["on"] and not risk.halted:
            if not market_open() and not args.offline:
                log.info("Market closed (weekend); sleeping.")
                time.sleep(POLL_SECONDS)
                continue
            fresh = get_fx_data(args.pairs, "5d", args.offline)
            for sym, raw in fresh.items():
                f = engineer_features(clean(raw))
                if f.empty:
                    continue
                row = f.iloc[-1]
                if seen_bars.get(sym) == row.name:
                    ex.process(PriceEvent(sym, float(row["close"]),
                                          ts=row.name.to_pydatetime()))
                    continue
                seen_bars[sym] = row.name
                ex.process(PriceEvent(sym, float(row["close"]),
                                      ts=row.name.to_pydatetime()))
                if risk.halted:
                    break
                ens.fallback = models.get(sym)
                sig = ens.predict(row)
                if sig != 0 and sym not in pf.positions:
                    ex.process(SignalEvent(sym, sig, float(row["close"]),
                                           float(row["atr_14"]),
                                           ts=row.name.to_pydatetime()))
            m = tracker.snapshot()
            write_metrics_file({**m.__dict__,
                                "equity_history":
                                    list(tracker.equity_history)[-500:]})
            console.print(render(tracker))
            if risk.halted:
                log.critical("Circuit breaker: deployment halted.")
                break
            time.sleep(POLL_SECONDS if not args.offline else 1)
            if args.offline:            # offline deploy = short demo, not 24/7
                break
    finally:
        ens.close()
        ex.stop()
    console.print(render(tracker))
    log.info("Deploy loop exited cleanly.")




def cmd_mtf(args):
    """Head-to-head: solo 5min agent vs multi-timeframe swarm, held-out data,
    identical spread cost model. This is the honest accuracy comparison."""
    import numpy as np
    from algotrader.backtest.engine import Backtester
    from algotrader.config import BacktestConfig
    from algotrader.data.forex import pair_spec
    from algotrader.models.multitf import MultiTimeframeSwarm, TimeframeAgent
    from algotrader.models.evaluate import evaluate_predictor
    from algotrader.data.preprocess import engineer_features

    for pair in args.pairs:
        if args.offline:
            base = synthetic_fx(pair, n=20_000, freq="min")
        else:
            data = get_fx_data([pair], "5d", False, interval="1m")
            base = next(iter(data.values())) if data else \
                synthetic_fx(pair, n=20_000, freq="min")
        split_ts = base.index[int(len(base) * 0.7)]
        train, test = base[base.index < split_ts], base[base.index >= split_ts]

        # prepare on the FULL series (indicators are causal); models train
        # only on the first 70% so test-window predictions are out-of-sample
        swarm = MultiTimeframeSwarm(
            timeframes=["1min", "3min", "5min", "15min", "30min",
                        "60min", "120min", "180min"])
        swarm.prepare(base, train_frac=0.7)
        solo = TimeframeAgent("5min")
        solo.prepare(base, train_frac=0.7)

        spec = pair_spec(pair)
        spread_bps = (spec.typical_spread_pips * spec.pip_size
                      / test["close"].mean()) * 10_000
        cfg = BacktestConfig(slippage_bps=spread_bps / 2, fee_bps=0.0,
                             min_fee=0.0)
        bt = Backtester(cfg)
        # evaluate on 5min bars of the held-out window
        from algotrader.models.multitf import resample_ohlcv
        test5 = engineer_features(resample_ohlcv(test, "5min"))

        res = {}
        for name, fn in (("solo-5min", lambda ts, row, s:
                          solo.signal_at(ts)),
                         ("mtf-swarm", lambda ts, row, s:
                          swarm.decide(ts).signal)):
            r = bt.run(test5, fn)
            rep = evaluate_predictor(
                test5, (lambda row: solo.signal_at(row.name)) if
                name == "solo-5min" else
                (lambda row: swarm.decide(row.name).signal))
            res[name] = (rep, r)
            log.info("%s | %-9s | hit-rate %5.1f%% | directional calls %4d | "
                     "trades %3d | net return %+.2f%% | maxDD %.2f%%",
                     spec.symbol, name, rep.hit_rate * 100, rep.n_directional,
                     len(r.trades), r.total_return * 100,
                     r.max_drawdown * 100)
        s_rep, s_bt = res["solo-5min"]; m_rep, m_bt = res["mtf-swarm"]
        log.info("%s | swarm vs solo: hit-rate %+0.1f pts, trades x%.2f, "
                 "net %+.2f pts", spec.symbol,
                 (m_rep.hit_rate - s_rep.hit_rate) * 100,
                 (len(m_bt.trades) / max(len(s_bt.trades), 1)),
                 (m_bt.total_return - s_bt.total_return) * 100)



def cmd_sweep(args):
    """Sweep 48 anchor/threshold/timeframe configs, rank on validation,
    re-score winners on an untouched test window."""
    from algotrader.data.forex import pair_spec
    from algotrader.models.sweep import run_sweep, report

    for pair in args.pairs:
        if args.offline:
            base = synthetic_fx(pair, n=25_000, freq="min")
        else:
            data = get_fx_data([pair], "5d", False, interval="1m")
            base = next(iter(data.values())) if data else \
                synthetic_fx(pair, n=25_000, freq="min")
        spec = pair_spec(pair)
        spread_bps = (spec.typical_spread_pips * spec.pip_size
                      / base["close"].mean()) * 10_000
        log.info("=== %s: sweeping 48 configs (spread %.2f bps) ===",
                 spec.symbol, spread_bps)
        results = run_sweep(base, spread_bps)
        print(f"\n{spec.symbol} — top configs by VALIDATION net return, "
              "re-scored on untouched TEST window:")
        print(report(results))
        print()



def cmd_serve(args):
    """TradingView webhook bridge: FastAPI server routing alerts through the
    forex risk engine to the paper broker. Requires TV_WEBHOOK_SECRET."""
    import os
    import uvicorn
    from algotrader.execution.tv_webhook import create_app, WebhookState

    cfg = AppConfig()
    assert cfg.is_paper, "serve runs paper-only"
    ex, pf, risk, specs = _make_stack(cfg, args.pairs)
    secret = os.environ.get("TV_WEBHOOK_SECRET", "")
    state = WebhookState(ex, secret)
    app = create_app(state)
    log.info("TradingView bridge on :%d for %s (paper). "
             "Point alerts at POST /webhook", args.port, list(specs))
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

def main(argv=None):
    p = argparse.ArgumentParser(description="AlgoTrader Forex (paper-first)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("backtest", cmd_backtest), ("paper", cmd_paper),
                     ("deploy", cmd_deploy), ("mtf", cmd_mtf),
                     ("sweep", cmd_sweep), ("serve", cmd_serve)):
        sp = sub.add_parser(name)
        sp.add_argument("--pairs", nargs="+",
                        default=["EURUSD", "GBPUSD", "USDJPY"])
        sp.add_argument("--period", default="1y")
        sp.add_argument("--offline", action="store_true")
        sp.add_argument("--port", type=int, default=8422)
        sp.set_defaults(fn=fn)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
