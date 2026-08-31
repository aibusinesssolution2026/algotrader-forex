"""Central configuration. SAFETY FIRST: paper trading is the hard default.

Live trading requires BOTH environment variables to be set explicitly:
    ALGO_TRADING_MODE=live
    ALGO_LIVE_CONFIRM=I_UNDERSTAND_THE_RISKS
Anything else resolves to paper mode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


LIVE_CONFIRM_PHRASE = "I_UNDERSTAND_THE_RISKS"


def resolve_trading_mode() -> str:
    """Return 'paper' unless the explicit double-opt-in for live mode is present."""
    mode = os.environ.get("ALGO_TRADING_MODE", "paper").strip().lower()
    confirm = os.environ.get("ALGO_LIVE_CONFIRM", "")
    if mode == "live" and confirm == LIVE_CONFIRM_PHRASE:
        return "live"
    return "paper"


@dataclass(frozen=True)
class RiskConfig:
    """Hardcoded risk parameters. Frozen: cannot be mutated at runtime."""
    max_risk_per_trade_pct: float = 0.01      # 1% of equity risked per trade
    max_position_pct: float = 0.10            # no single position > 10% of equity
    max_daily_drawdown_pct: float = 0.03      # 3% daily loss => circuit breaker
    max_open_positions: int = 5
    atr_stop_multiplier: float = 2.0          # stop distance = 2 x ATR
    max_total_open_risk_pct: float = 0.05     # sum of open risk <= 5% equity


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    slippage_bps: float = 5.0                 # 5 basis points per side
    fee_bps: float = 2.0                      # 2 bps commission per side
    min_fee: float = 1.0                      # broker minimum fee per order


@dataclass
class AppConfig:
    trading_mode: str = field(default_factory=resolve_trading_mode)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @property
    def is_paper(self) -> bool:
        return self.trading_mode != "live"
