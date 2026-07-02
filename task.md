# Pairs Trading System — Task Tracker

## Architecture: Monitor → Build → Version → Deploy (monthly ML cycle)

---

## Backtest Fixes (Forward Bias)

- [x] **FIX-1** Fix pair selection forward bias — clustering + cointegration now run inside each fold on train-window data only (inside `walk_forward_backtest`)
- [x] **FIX-2** Fix `PairSignal` dataclass defined after first use — moved to top of file (after imports)
- [x] **FIX-3** Build walk-forward orchestrator — `walk_forward_backtest()` added; pair selection runs inside each fold on train window only

---

## Missing Backtest Pieces

- [x] **BT-1** Portfolio-level P&L aggregation — fold pairs summed daily inside `walk_forward_backtest`
- [x] **BT-2** Borrow cost / short rebate simulation — `annual_borrow_rate_bps` param added to `backtest_pair_v2`; accrues daily on lagged short notional; surfaces as `daily_borrow_cost` column
- [x] **BT-3** Max holding stop — `_apply_holding_stop()` zeros P&L after `max_holding_bars` (default 20); configurable via endpoint param
- [x] **BT-4** Performance report — `_compute_backtest_metrics()` returns Sharpe, Calmar, max drawdown, win rate, total P&L

---

## 4-Stage ML Cycle

- [x] **CYCLE-1** **Monitor** (daily) — check data freshness, validate open positions are within bounds, detect regime shift, alert on ETB→HTB flips
- [x] **CYCLE-2** **Build** (monthly, 1st trading day) — rerun clustering + cointegration on trailing 24-month window (ETB-only filter), run walk-forward backtest, Sharpe gate before promoting
- [x] **CYCLE-3** **Version** — save model artifact to DB/disk: cluster assignments, valid pair list, hyperparams, backtest stats, timestamp tag
- [x] **CYCLE-4** **Deploy** — swap active pair universe, update params in DB, run on Alpaca paper for N days, promote to IBKR live after gate passes

---

## Broker / Infrastructure

- [x] **INFRA-1** Add Alpaca as paper-trading / staging broker — validate new model versions on Alpaca paper before promoting to IBKR live
- [x] **INFRA-2** ETB (Easy-to-Borrow) filter — before any pair enters the active universe, confirm both legs are ETB; re-check daily in Monitor stage

---

## Order of Work

1. FIX-2 (quick, unblocks clean imports)
2. FIX-1 + FIX-3 (core walk-forward engine — do together)
3. BT-1 → BT-4 (portfolio backtest layer)
4. CYCLE-1 → CYCLE-4 (operationalize the monthly ML cycle)
5. INFRA-1 → INFRA-2 (broker integration)
