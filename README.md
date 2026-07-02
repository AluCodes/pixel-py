# FastAPI Microservice Backend

Python-based microservice architecture built with FastAPI for AI, ML, and analytics workloads.

**Simple, modular, NestJS-inspired structure** - each service is a single Python file!

## Architecture

- **Module-Based**: Each service is a single file that exports a FastAPI router
- **Easy to Add Services**: Just create a new file in `services/` and register it
- **Multi-Protocol Support**: REST APIs with plans for gRPC, WebSocket, Message Queue
- **Docker Ready**: Containerized deployment optimized for Hetzner Cloud

## Project Structure

```
.
├── main.py                       # Main FastAPI application
├── services/                     # Service modules (like NestJS modules)
│   ├── __init__.py
│   ├── deskew_service.py        # Deskew service module
│   └── [future_service].py      # Add more services here
│
├── tests/                        # Tests
│   ├── test_deskew.py
│   └── test_main.py
│
├── requirements.txt              # Python dependencies
├── requirements-dev.txt          # Development dependencies
├── .env.example                  # Environment variables template
└── README.md
```

### Simple Module-Based Architecture

Each service is a single Python file that exports a FastAPI `APIRouter`:

```python
# services/my_service.py
from fastapi import APIRouter

router = APIRouter(prefix="/my-service", tags=["my-service"])

@router.get("/")
async def my_endpoint():
    return {"message": "Hello from my service"}
```

Then register it in `main.py`:

```python
from services import my_service
app.include_router(my_service.router, prefix="/api/v1")
```

## Getting Started

### Prerequisites

- Python 3.13+ (or 3.11+)
- Docker & Docker Compose
- Virtual environment tool (venv)
- Tesseract OCR (for deskew service)

### Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the application
python main.py

# Or with uvicorn (with auto-reload)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run linting
ruff check .
mypy .
```

### Quick Start

```bash
# Start the server
python main.py

# Test deskew endpoint
curl -X POST "http://localhost:8000/api/v1/deskew/" \
  -F "image=@test_image.jpg"

# Check health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

### Docker (Recommended)

```bash
# Build and run with Docker Compose
./docker-build.sh
docker-compose up

# Or run in production mode
docker-compose -f docker-compose.prod.yml up -d

# Test
curl http://localhost:8000/health
```

See [DOCKER.md](DOCKER.md) for complete Docker documentation.

## Documentation

- [Simple Guide](SIMPLE_GUIDE.md) - **Start here!** Learn the module-based architecture
- [Requirements](.kiro/specs/fastapi-microservice-backend/requirements.md)
- [Design](.kiro/specs/fastapi-microservice-backend/design.md)
- [Tasks](.kiro/specs/fastapi-microservice-backend/tasks.md)

## Current Services

- **Deskew**: Image deskew and perspective correction (`/api/v1/deskew/`)
- **Algo Trading**: Statistical pairs trading on S&P 500 stocks (`/api/v1/algo_trading/`)

---

## Algo Trading Service

A statistical pairs-trading engine that finds cointegrated S&P 500 stock pairs, sizes positions using a rolling z-score signal, and executes through IBKR (live) or Alpaca (paper/staging).

All endpoints are under `/api/v1/algo_trading/`. Interactive docs at `http://localhost:8000/docs`.

### Strategy Overview

The strategy is **end-of-day signal, open-market execution** — it does **not** monitor prices intraday. Each morning it reads yesterday's closing prices, computes signals, and submits orders at market open.

**Pair nomenclature**: "Tom" = first leg, "Jerry" = second leg (named after the cartoon).

**Signal logic (computed from prior-day closes only):**

1. Rolling OLS over 90 days → hedge ratio (beta)
2. `spread = log(Tom) - beta × log(Jerry)`
3. 22-day rolling z-score of the spread
4. **Entry** when `|z| > 2.0`:
   - `z < −2.0` → LONG spread: BUY Tom / SELL Jerry
   - `z > +2.0` → SHORT spread: SELL Tom / BUY Jerry
5. **Exit** when `|z| ≤ exit_z` — **data-driven**: the monthly build sweeps `exit_z ∈ [0.0, 0.5, 1.0, 1.5]`, runs the full walk-forward backtest for each candidate, and picks the single value that maximises `total_n_trades × Sharpe` across all pairs. Stored in the model version under `hyperparams.exit_z`.
6. **Position sizing**: risk-based, `0.5% of AUM` per trade, scaled by conviction (z vs max_z=4.0)

**Brokers:**
- IBKR via `ib_async` — live trading (port 7496 = live TWS, 7497 = paper TWS)
- Alpaca — paper/staging environment (configured via `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`)

**Data sources**: Yahoo Finance (default), Stooq, Polygon/Massive (premium, rate-limited)

---

### Four Operational Cycles

#### CYCLE-1 — Daily Monitor (`/run_daily_monitor`)

Run every trading day to check system health **before** executing trades.

| Check | What it does |
|---|---|
| `_check_data_freshness` | Compares most recent DB bar to expected previous trading date |
| `_detect_regime_shift` | Flags if 22d annualized vol > 1.5× the 252d baseline on the equal-weighted universe |
| `_check_position_bounds` | For every OPEN trade: recomputes z-score; alerts if `|z| > 4.0` or held > 20 trading days |
| `_check_etb_status` | Queries IBKR shortable-shares; flags symbols with < 1000 shares as HTB (Hard-to-Borrow) |

Returns a `summary.ok` boolean and a list of `alerts`. **If `ok` is false, do not trade.**

#### CYCLE-2 — Monthly Build (`/run_monthly_build`)

Re-trains the full model. Run once a month (or after a market regime change):

1. Re-cleans price data on a trailing 24-month window
2. Re-clusters all S&P 500 stocks using AffinityPropagation (on annualized return + vol)
3. Runs Engle-Granger cointegration within each cluster
4. Sweeps `exit_z ∈ [0.0, 0.5, 1.0, 1.5]` at **portfolio level** — runs a full walk-forward backtest per candidate, picks the single `exit_z` that maximises `total_n_trades × Sharpe` across all pairs
5. **Sharpe gate**: only marks the build `approved` if annualized Sharpe ≥ 0.5
7. Automatically calls CYCLE-3 on completion

#### CYCLE-3 — Version (`/save_model_version`, auto-called from CYCLE-2)

Persists the current pair universe + hyperparameters + backtest metrics to the `model_versions` table in Postgres. Tracks lifecycle status: `pending → approved → staging → deployed`.

View all versions: `GET /model_versions`

#### CYCLE-4 — Deploy (`/run_deploy` → `/promote_to_live`)

Two-step promotion with a staging gate:

1. **`/run_deploy`** — Stages the latest approved version on Alpaca paper; writes `data/df_staging_pairs.parquet`
2. **`/promote_to_live`** — After ≥ 5 staging days and Alpaca paper loss < 2% of gross notional, promotes to IBKR live; writes `data/df_all_pairs.parquet`

Use `?skip_gate=true` to force-promote during testing.

---

### How Trades Are Triggered

There is **no real-time price monitoring**. The signal is computed from yesterday's closing prices each morning, and a single market order is placed at open. Nothing runs during the trading day.

**Best execution window**: 9:30–9:45 AM ET. The signal does not change intraday so there is no benefit to waiting past this window.

---

### Daily Runbook (every trading day)

**Pre-market — run steps 1–4 in sequence before 9:15 AM ET**

| Step | Time | Method | URL | Notes |
|------|------|--------|-----|-------|
| 1 | 7:00 AM | GET | `/api/v1/algo_trading/sync_recent_data` | Pull latest daily closes (yfinance) |
| 2 | after 1 | GET | `/api/v1/algo_trading/clean_data` | Clean 2-year price history → `df_clean.parquet` |
| 3 | after 2 | GET | `/api/v1/algo_trading/calculate_candidates_etb` | Score all pairs + filter HTB → `df_candidate_{date}.parquet` |
| 4 | 9:00 AM | GET | `/api/v1/algo_trading/run_daily_monitor` | Health check — **stop here if `summary.ok = false`** |
| 5 | 9:15 AM | GET | `/api/v1/algo_trading/preview_best_pair_live` | Dry-run: review signal + sizes before committing |

> Steps 1–3 can be combined into a single call: `GET /api/v1/algo_trading/sync_clean_calculate_candidate`

**Market open — 9:30–9:45 AM ET**

| Step | Time | Method | URL | Notes |
|------|------|--------|-----|-------|
| 6 | 9:30–9:45 AM | POST | `/api/v1/algo_trading/trade_best_pair_live` | Execute top candidate via IBKR live |

**Optional — anytime**

| Action | Method | URL |
|--------|--------|-----|
| Check IBKR account + positions | GET | `/api/v1/algo_trading/get_account_summary` |
| Check Alpaca paper positions | GET | `/api/v1/algo_trading/alpaca_paper/positions` |

---

### Monthly Runbook (first trading day of the month)

Steps 1–3 run before market open on day 1. Steps 4–5 are separated by 5 trading days.

| Step | When | Method | URL | Notes |
|------|------|--------|-----|-------|
| 1 | 7:00 AM, day 1 | GET | `/api/v1/algo_trading/run_monthly_build` | Rebuild model — takes 30–60 min |
| 2 | after 1 | GET | `/api/v1/algo_trading/model_versions` | Confirm new version shows `approved: true` |
| 3 | after 2 | GET | `/api/v1/algo_trading/run_deploy` | Stage on Alpaca paper |
| — | *wait 5 trading days* | — | `/api/v1/algo_trading/alpaca_paper/positions` | Monitor paper PnL daily |
| 4 | day 6 morning | GET | `/api/v1/algo_trading/promote_to_live` | Promote to IBKR live |

> If step 1 returns `approved: false` (Sharpe < 0.5), skip steps 3–4 and keep the existing deployed version.
> After step 4, the next `trade_best_pair_live` call automatically uses the new pair universe.
> Use `?skip_gate=true` on `/promote_to_live` to force-promote during testing.

---

### What `trade_best_pair_live` Does

`POST /api/v1/algo_trading/trade_best_pair_live` calls `manage_live_pair()` which:

1. Reads the top candidate from `data/df_candidate_{today}.parquet` (highest |z-score|)
2. Connects to IBKR and fetches current positions + open orders
3. Recomputes the latest signal from `df_clean.parquet`
4. Decides action:

| Action | Condition | Result |
|--------|-----------|--------|
| **ENTER** | No position + signal ≠ 0 | Submit both legs |
| **EXIT** | In position + z reverts to 0 | Close both legs |
| **EXIT_FIRST** | Signal flipped direction | Close first, re-enter next cycle |
| **REBALANCE** | Same direction, sizing drifted | Adjust quantities (only if enabled) |
| **SKIP** | No signal or open orders exist | Do nothing |
| **MANUAL_REVIEW** | Partial / mismatched position | Requires human intervention |

Trade records are persisted to the `trades` table with full IBKR order IDs.

---

### All Algo Trading Endpoints

**Data & Model**

| Endpoint | Method | Description |
|---|---|---|
| `/sync_recent_data` | GET | Download latest daily bars (yfinance) |
| `/clean_data` | GET | Clean 2-year price history → `df_clean.parquet` |
| `/calculate_cluster_and_cointegration` | GET | Cluster + cointegrate → `df_all_pairs.parquet` |
| `/calculate_candidates` | GET | Score all pairs → `trade_candidates` DB table |
| `/calculate_candidates_etb` | GET | Same + IBKR HTB filter (preferred over above) |
| `/walk_forward_backtest` | GET | Run walk-forward backtest on current `df_clean.parquet` |
| `/sync_clean_calculate_candidate` | GET | Runs sync → clean → candidates in one call |

**Monitor & Governance**

| Endpoint | Method | Description |
|---|---|---|
| `/run_daily_monitor` | GET | CYCLE-1: data freshness, regime, positions, ETB |
| `/run_monthly_build` | GET | CYCLE-2: full rebuild + Sharpe gate + auto-version |
| `/save_model_version` | GET | CYCLE-3: manually version current pair universe |
| `/model_versions` | GET | List all model versions with status |

**Deploy**

| Endpoint | Method | Description |
|---|---|---|
| `/run_deploy` | GET | CYCLE-4a: stage latest approved version on Alpaca paper |
| `/promote_to_live` | GET | CYCLE-4b: promote staged → IBKR live after gate check |

**Execution**

| Endpoint | Method | Description |
|---|---|---|
| `/preview_best_pair_live` | GET | Dry-run: compute signal + trade plan, no orders submitted |
| `/trade_best_pair_live` | POST | Execute live pair trade via IBKR |
| `/get_account_summary` | GET | IBKR account summary, positions, and open orders |

**Alpaca Paper**

| Endpoint | Method | Description |
|---|---|---|
| `/alpaca_paper/positions` | GET | Current Alpaca paper positions |
| `/alpaca_paper/manage_pair` | GET | Evaluate + execute signal for a pair on Alpaca paper |

---

### Environment Variables (algo trading)

Add these to `.env.local`:

```env
# Database
POSTGRES_USER_ALGO_TRADING=
POSTGRES_PASSWORD_ALGO_TRADING=
POSTGRES_DB_ALGO_TRADING=
POSTGRES_HOST_ALGO_TRADING=
POSTGRES_PORT_ALGO_TRADING=5432

# IBKR (TWS or IB Gateway must be running)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497          # 7497 = TWS paper, 7496 = TWS live
IBKR_CLIENT_ID=108

# Alpaca (paper trading)
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_PAPER=true

# Polygon/Massive (optional premium data)
MASSIVE_API_KEY=
```

**IBKR ports**: TWS paper = `7497`, TWS live = `7496`, IB Gateway paper = `4002`, IB Gateway live = `4001`.

---

## Adding a New Service

1. Create `services/my_service.py`
2. Export a FastAPI `APIRouter`
3. Register it in `main.py`
4. Done!

See [SIMPLE_GUIDE.md](SIMPLE_GUIDE.md) for details.

## License

Proprietary
