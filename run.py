#!/usr/bin/env python3
"""AlgoTrader entrypoint.

    python run.py backtest --symbols AAPL MSFT           # historical backtest
    python run.py paper    --symbols AAPL MSFT           # simulated live loop
    python run.py fetch    --symbols AAPL MSFT --period 2y

Paper mode replays the latest fetched bars through the full event-driven
stack (ensemble -> risk engine -> executor) with the terminal dashboard.
There is deliberately NO 'live' subcommand: enabling real execution requires
editing your environment per algotrader/config.py, by design.
"""
from __future__ import annotations

import argparse
import logging
import sys

from algotrader.config import AppConfig
from algotrader.data.fetcher import fetch_yahoo, synthetic_ohlcv, save_parquet
from algotrader.data.preprocess import clean, engineer_features
from algotrader.backtest.engine import Backtester
from algotrader.models.ensemble import (EnsemblePredictor, StatisticalFallback,
                                        macd_convergence_signal)
from algotrader.models.evaluate import evaluate_predictor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run")


def get_data(symbols, period, offline):
    if offline:
        return {s: synthetic_ohlcv(s, n=500) for s in symbols}
    try:
        data = fetch_yahoo(symbols, period=period)
        if data:
            return data
    except Exception as exc:
        log.warning("Yahoo fetch failed (%s); using synthetic data", exc)
    return {s: synthetic_ohlcv(s, n=500) for s in symbols}


def cmd_fetch(args):
    data = get_data(args.symbols, args.period, args.offline)
    save_parquet(data, "data_cache")
    for s, df in data.items():
        log.info("%s: %d bars %s -> %s", s, len(df), df.index[0].date(),
                 df.index[-1].date())


def cmd_backtest(args):
    cfg = AppConfig()
    data = get_data(args.symbols, args.period, args.offline)
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
                 "maxDD %.2f%% | trades %d | win-rate %.0f%%",
                 sym, rep.hit_rate * 100, res.total_return * 100, res.sharpe,
                 res.max_drawdown * 100, len(res.trades), res.win_rate * 100)


def cmd_paper(args):
    from algotrader.execution.broker import PaperBroker
    from algotrader.execution.executor import Executor, PriceEvent, SignalEvent
    from algotrader.risk.engine import RiskEngine
    from algotrader.state.portfolio import Portfolio
    from algotrader.dashboard.metrics import MetricsTracker
    from algotrader.dashboard.terminal import render
    from algotrader.dashboard.streamlit_app import write_metrics_file
    from rich.console import Console

    cfg = AppConfig()
    log.info("Trading mode: %s", cfg.trading_mode)
    assert cfg.is_paper, "This entrypoint only runs paper mode."

    data = get_data(args.symbols, args.period, args.offline)
    feats = {s: engineer_features(clean(df)) for s, df in data.items()}
    models = {}
    for s, f in feats.items():
        fb = StatisticalFallback()
        fb.fit(f.iloc[:int(len(f) * 0.7)])
        models[s] = fb

    pf = Portfolio(cash=cfg.backtest.initial_capital)
    risk = RiskEngine(cfg.risk, pf)
    ex = Executor(risk, PaperBroker(cfg.backtest), pf)
    tracker = MetricsTracker(ex)
    console = Console()
    ens = EnsemblePredictor(primary=None)  # plug your AI API callable here

    try:
        n = min(len(f) for f in feats.values())
        start = int(n * 0.7)
        for i in range(start, n):          # replay held-out bars as a stream
            for s, f in feats.items():
                row = f.iloc[i]
                ex.process(PriceEvent(s, float(row["close"]), ts=row.name))
                if risk.halted:
                    break
                ens.fallback = models[s]
                sig = ens.predict(row)
                if sig != 0 and s not in pf.positions:
                    ex.process(SignalEvent(s, sig, float(row["close"]),
                                           float(row["atr_14"])))
            m = tracker.snapshot()
            write_metrics_file({**m.__dict__,
                                "equity_history": list(tracker.equity_history)[-500:]})
            if i % 10 == 0 or risk.halted:
                console.print(render(tracker))
            if risk.halted:
                log.critical("Session terminated by circuit breaker.")
                break
    finally:
        ens.close()
        ex.stop()
    console.print(render(tracker))


def main(argv=None):
    p = argparse.ArgumentParser(description="AlgoTrader (paper-first)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fetch", cmd_fetch), ("backtest", cmd_backtest),
                     ("paper", cmd_paper)):
        sp = sub.add_parser(name)
        sp.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"])
        sp.add_argument("--period", default="2y")
        sp.add_argument("--offline", action="store_true",
                        help="use synthetic data (no network)")
        sp.set_defaults(fn=fn)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
