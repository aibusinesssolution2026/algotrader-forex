"""Optional Streamlit dashboard. Run with:
    streamlit run algotrader/dashboard/streamlit_app.py
Reads metrics from a JSON file the trading loop writes (decoupled: the
dashboard never touches trading state directly, so a UI crash can never
affect execution).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

METRICS_FILE = Path("runtime/metrics.json")


def write_metrics_file(metrics: dict, path: Path = METRICS_FILE) -> None:
    """Called by the trading loop each refresh; atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(metrics))
    tmp.replace(path)


def main() -> None:  # pragma: no cover - UI shell
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="AlgoTrader", layout="wide")
    st.title("AlgoTrader — Paper Trading Dashboard")
    placeholder = st.empty()

    while True:
        if METRICS_FILE.exists():
            m = json.loads(METRICS_FILE.read_text())
            with placeholder.container():
                if m.get("halted"):
                    st.error("⛔ CIRCUIT BREAKER — trading halted")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Equity", f"${m['equity']:,.0f}")
                c2.metric("Win/Loss", f"{m['win_loss_ratio']:.2f}")
                c3.metric("Sharpe", f"{m['sharpe']:.2f}")
                c4.metric("Max DD", f"{m['max_drawdown']:.2%}")
                c5.metric("Open Risk", f"${m['current_open_risk']:,.0f}")
                curve = m.get("equity_history", [])
                if curve:
                    st.line_chart(pd.Series(curve, name="equity"))
        else:
            placeholder.info("Waiting for trading loop to write metrics…")
        time.sleep(1)


if __name__ == "__main__":
    main()
