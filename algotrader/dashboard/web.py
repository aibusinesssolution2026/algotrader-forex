"""Mobile-friendly web dashboard, served from the SAME process as the
trading loop (no separate service, no shared volume needed). One page,
auto-refreshing via JS polling of a small JSON endpoint — light enough to
load fine on mobile data.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .live_state import LiveState
from .sweep_runner import SweepRunState, trigger_sweep

PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlgoTrader — Live</title>
  <style>
    body { background:#0b0e14; color:#e6e6e6; font-family:-apple-system,
           BlinkMacSystemFont,Segoe UI,Roboto,sans-serif; margin:0;
           padding:16px; }
    h1 { font-size:1.1rem; color:#8ab4f8; margin:0 0 4px; }
    .sub { color:#888; font-size:0.8rem; margin-bottom:16px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px;
            margin-bottom:16px; }
    .card { background:#151a24; border-radius:10px; padding:12px; }
    .label { color:#888; font-size:0.72rem; text-transform:uppercase;
              letter-spacing:0.04em; }
    .value { font-size:1.35rem; font-weight:600; margin-top:2px; }
    .pos { color:#4ade80; } .neg { color:#f87171; } .neutral{color:#e6e6e6;}
    .status { padding:10px 12px; border-radius:10px; font-weight:600;
               margin-bottom:16px; text-align:center; }
    .status.running { background:#14301f; color:#4ade80; }
    .status.halted { background:#3a1414; color:#f87171; }
    canvas { width:100%; background:#151a24; border-radius:10px;
              margin-bottom:16px; }
    .events { background:#151a24; border-radius:10px; padding:12px;
               font-size:0.78rem; max-height:220px; overflow-y:auto; }
    .events div { padding:3px 0; border-bottom:1px solid #222; color:#aaa; }
    .stale { color:#f87171; font-size:0.75rem; margin-top:8px; }
  </style>
</head>
<body>
  <h1>AlgoTrader — Live Paper Session</h1>
  <div class="sub" id="asof">loading…</div>
  <div id="statusBox" class="status running">● connecting…</div>
  <canvas id="chart" height="140"></canvas>
  <div class="grid" id="metrics"></div>
  <div class="label" style="margin-bottom:6px;">Recent events</div>
  <div class="events" id="events"></div>
  <div class="stale" id="staleWarn" style="display:none;">
    ⚠ No update in a while — check Railway logs.
  </div>
  <div style="margin-top:20px;">
    <a href="/positions" style="color:#8ab4f8; text-decoration:none;
       font-size:0.9rem; display:block; margin-bottom:8px;">→ View open position details (entry, stop, P&L)</a>
    <a href="/sweep" style="color:#8ab4f8; text-decoration:none;
       font-size:0.9rem;">→ Run weekly config sweep</a>
  </div>

<script>
function fmtMoney(v){ return '$' + Number(v).toLocaleString(undefined,
  {minimumFractionDigits:2, maximumFractionDigits:2}); }
function fmtPct(v){ return (v*100).toFixed(2) + '%'; }

let lastUpdate = 0;

async function refresh() {
  try {
    const r = await fetch('/api/state', {cache: 'no-store'});
    const d = await r.json();
    lastUpdate = Date.now();
    document.getElementById('staleWarn').style.display = 'none';

    const m = d.metrics || {};
    document.getElementById('asof').textContent =
      'Started ' + new Date(d.started_at).toLocaleString();

    const halted = m.halted;
    const box = document.getElementById('statusBox');
    box.className = 'status ' + (halted ? 'halted' : 'running');
    box.textContent = halted ? '⛔ HALTED (circuit breaker)' : '● RUNNING (paper)';

    const cards = [
      ['Equity', fmtMoney(m.equity ?? 0), 'neutral'],
      [m.win_loss_ratio === null || m.win_loss_ratio === undefined
        ? 'Win/Loss Ratio' : `Win/Loss (${m.win_count}W / ${m.loss_count}L)`,
       m.win_loss_ratio === null || m.win_loss_ratio === undefined
        ? 'N/A' : m.win_loss_ratio.toFixed(2), 'neutral'],
      ['Sharpe (ann.)', (m.sharpe ?? 0).toFixed(2), 'neutral'],
      ['Max Drawdown', fmtPct(m.max_drawdown ?? 0), 'neg'],
      ['Daily Drawdown', fmtPct(m.daily_drawdown ?? 0),
        (m.daily_drawdown ?? 0) < -0.01 ? 'neg' : 'neutral'],
      ['Open Risk', fmtMoney(m.current_open_risk ?? 0), 'neutral'],
      ['Closed Trades', (m.n_trades ?? 0).toString(), 'neutral'],
    ];
    document.getElementById('metrics').innerHTML = cards.map(([l,v,c]) =>
      `<div class="card"><div class="label">${l}</div>
        <div class="value ${c}">${v}</div></div>`).join('');

    document.getElementById('events').innerHTML =
      (d.events || []).slice(0, 30).map(e => `<div>${e}</div>`).join('')
      || '<div>No events yet.</div>';

    drawChart(d.equity_history || []);
  } catch (e) {
    console.error(e);
  }
}

function drawChart(hist) {
  const c = document.getElementById('chart');
  const ctx = c.getContext('2d');
  const w = c.width = c.clientWidth * 2, h = c.height = 280;
  ctx.clearRect(0,0,w,h);
  if (hist.length < 2) return;
  const vals = hist.map(p => p[1]);
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = (max - min) || 1;
  ctx.beginPath();
  ctx.strokeStyle = '#8ab4f8';
  ctx.lineWidth = 3;
  hist.forEach((p, i) => {
    const x = (i / (hist.length - 1)) * (w - 20) + 10;
    const y = h - 20 - ((p[1] - min) / range) * (h - 40);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// staleness watchdog: warn if no update for 90s (poll is 15s)
setInterval(() => {
  if (lastUpdate && Date.now() - lastUpdate > 90000) {
    document.getElementById('staleWarn').style.display = 'block';
  }
}, 15000);

refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>"""


SWEEP_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlgoTrader — Sweep</title>
  <style>
    body { background:#0b0e14; color:#e6e6e6; font-family:-apple-system,
           BlinkMacSystemFont,Segoe UI,Roboto,sans-serif; margin:0;
           padding:16px; }
    h1 { font-size:1.1rem; color:#8ab4f8; margin:0 0 4px; }
    a { color:#8ab4f8; }
    .sub { color:#888; font-size:0.8rem; margin-bottom:16px; }
    button { background:#8ab4f8; color:#0b0e14; border:none;
              border-radius:8px; padding:12px 20px; font-size:1rem;
              font-weight:600; width:100%; margin-bottom:16px; }
    button:disabled { background:#3a4252; color:#888; }
    .status { padding:10px 12px; border-radius:10px; margin-bottom:16px;
               font-size:0.85rem; }
    .status.idle { background:#151a24; color:#888; }
    .status.running { background:#2a2410; color:#facc15; }
    .status.done { background:#14301f; color:#4ade80; }
    .status.error { background:#3a1414; color:#f87171; }
    pre { background:#151a24; border-radius:10px; padding:12px;
           font-size:0.72rem; overflow-x:auto; white-space:pre-wrap;
           word-break:break-word; margin-bottom:16px; }
    .pair-title { color:#8ab4f8; font-size:0.9rem; margin:16px 0 4px; }
    .warn { color:#facc15; font-size:0.78rem; margin-top:4px; }
  </style>
</head>
<body>
  <h1>Weekly Config Sweep</h1>
  <div class="sub"><a href="/">← back to live dashboard</a></div>
  <button id="runBtn" onclick="runSweep()">Run Sweep Now</button>
  <div id="statusBox" class="status idle">Idle — no sweep run yet this session.</div>
  <div id="results"></div>
  <div class="warn">Note: ranks a fixed grid of timeframe/anchor/agreement
    settings on recent data — a low trade count or few weeks of history
    means the ranking is not yet reliable. Re-run weekly and watch for
    configs that stay near the top consistently.</div>

<script>
async function runSweep() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  await fetch('/api/sweep/run', {method: 'POST'});
  poll();
}

async function poll() {
  const r = await fetch('/api/sweep/status', {cache: 'no-store'});
  const d = await r.json();
  const box = document.getElementById('statusBox');
  const btn = document.getElementById('runBtn');

  if (d.status === 'running') {
    box.className = 'status running';
    box.textContent = '⏳ Running — this can take 30-90 seconds…';
    btn.disabled = true;
    setTimeout(poll, 2000);
    return;
  }
  btn.disabled = false;
  if (d.status === 'error') {
    box.className = 'status error';
    box.textContent = '✖ Sweep failed: ' + d.error;
    return;
  }
  if (d.status === 'done') {
    box.className = 'status done';
    box.textContent = '✓ Last run finished ' +
      new Date(d.finished_at).toLocaleString();
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = Object.entries(d.results).map(([pair, text]) =>
      `<div class="pair-title">${pair}</div><pre>${text}</pre>`).join('');
  } else {
    box.className = 'status idle';
    box.textContent = 'Idle — no sweep run yet this session.';
  }
}

poll();
</script>
</body>
</html>"""


POSITIONS_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlgoTrader — Open Positions</title>
  <style>
    body { background:#0b0e14; color:#e6e6e6; font-family:-apple-system,
           BlinkMacSystemFont,Segoe UI,Roboto,sans-serif; margin:0;
           padding:16px; }
    h1 { font-size:1.1rem; color:#8ab4f8; margin:0 0 4px; }
    a { color:#8ab4f8; }
    .sub { color:#888; font-size:0.8rem; margin-bottom:16px; }
    .card { background:#151a24; border-radius:10px; padding:14px;
             margin-bottom:12px; }
    .row { display:flex; justify-content:space-between;
            padding:4px 0; font-size:0.85rem; }
    .row .k { color:#888; }
    .sym { font-size:1.1rem; font-weight:700; }
    .side-long { color:#4ade80; } .side-short { color:#f87171; }
    .pos { color:#4ade80; } .neg { color:#f87171; }
    .badge { display:inline-block; background:#1e2534; color:#8ab4f8;
              border-radius:6px; padding:2px 8px; font-size:0.72rem;
              margin-left:6px; }
    .empty { color:#888; font-size:0.85rem; padding:20px 0; }
    .note { color:#facc15; font-size:0.78rem; margin-top:16px;
             line-height:1.4; }
  </style>
</head>
<body>
  <h1>Open Positions</h1>
  <div class="sub"><a href="/">← back to live dashboard</a></div>
  <div id="list"><div class="empty">Loading…</div></div>
  <div class="note">"Decided by" shows which model actually chose this
    trade: <b>statistical</b> = a boosted-tree model trained on recent
    price/volatility features, <b>macd</b> = a simple trend-following rule
    used when the model is unavailable. Note: the live bot currently runs
    this simpler model, not the 8-agent multi-timeframe swarm used in the
    weekly sweep comparison.</div>

<script>
function fmt(v, d=5) { return Number(v).toFixed(d); }
function fmtMoney(v) {
  const s = v >= 0 ? '+' : '';
  return s + '$' + v.toFixed(2);
}

async function refresh() {
  try {
    const r = await fetch('/api/state', {cache: 'no-store'});
    const d = await r.json();
    const positions = d.open_positions || [];
    const list = document.getElementById('list');
    if (!positions.length) {
      list.innerHTML = '<div class="empty">No open positions right now.</div>';
      return;
    }
    list.innerHTML = positions.map(p => `
      <div class="card">
        <div class="row">
          <span class="sym">${p.symbol}
            <span class="${p.side === 'long' ? 'side-long' : 'side-short'}">
              ${p.side.toUpperCase()}</span></span>
          <span class="badge">${p.decided_by}</span>
        </div>
        <div class="row"><span class="k">Entry</span><span>${fmt(p.entry_price)}</span></div>
        <div class="row"><span class="k">Current</span><span>${fmt(p.current_price)}</span></div>
        <div class="row"><span class="k">Stop-loss</span><span>${fmt(p.stop_price)}</span></div>
        <div class="row"><span class="k">Unrealized P&L</span>
          <span class="${p.unrealized_pnl >= 0 ? 'pos' : 'neg'}">${fmtMoney(p.unrealized_pnl)}</span></div>
        <div class="row"><span class="k">Risk if stopped</span><span>$${p.risk_amount.toFixed(2)}</span></div>
      </div>`).join('');
  } catch (e) { console.error(e); }
}
refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>"""


def create_dashboard_app(state: LiveState, pairs: list[str] | None = None,
                         offline: bool = False) -> FastAPI:
    app = FastAPI(title="AlgoTrader Live Dashboard", docs_url=None,
                  redoc_url=None)
    sweep_state = SweepRunState()
    sweep_pairs = pairs or ["EURUSD", "GBPUSD", "USDJPY"]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/state")
    def api_state():
        return state.snapshot()

    @app.get("/positions", response_class=HTMLResponse)
    def positions_page():
        return POSITIONS_PAGE

    @app.get("/sweep", response_class=HTMLResponse)
    def sweep_page():
        return SWEEP_PAGE

    @app.post("/api/sweep/run")
    def api_sweep_run():
        started = trigger_sweep(sweep_pairs, offline, sweep_state)
        return {"triggered": started,
               "reason": None if started else "already running"}

    @app.get("/api/sweep/status")
    def api_sweep_status():
        return sweep_state.snapshot()

    @app.get("/health")
    def health():
        m = state.last_metrics
        return {"halted": m.get("halted", False),
               "has_data": bool(m)}

    return app
