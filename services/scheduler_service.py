"""
Scheduler Service — APScheduler equivalent of NestJS @nestjs/schedule.

Daily jobs (Mon–Fri, America/New_York):
  07:00  Steps 1-3 — sync + clean + score candidates
  09:00  Step 4    — daily monitor (gate: stops preview/trade if not ok)
  09:15  Step 5    — preview signal (dry-run, no orders)
  09:30  Step 6    — execute trade via IBKR

Monthly jobs (America/New_York):
  First Monday  07:00 — monthly build → Sharpe gate → deploy to Alpaca staging
  Second Monday 07:00 — promote Alpaca staging to IBKR live (after gate check)
"""

import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("scheduler")

# Gate flag: set by the monitor job each morning; cleared at midnight.
# Preview and trade jobs skip execution if this is False.
_monitor_ok: bool = False

_scheduler = AsyncIOScheduler(timezone="America/New_York")


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Daily Step 1-3: sync + clean + score candidates (07:00 AM ET)
# ---------------------------------------------------------------------------
async def _job_sync_clean_candidates():
    global _monitor_ok
    _monitor_ok = False  # reset gate at the start of each trading day

    from services.algo_trading_service import sync_clean_calculate_candidate

    print(f"\n[{_ts()}] → STEP 1-3 — sync_clean_calculate_candidate starting...")
    try:
        result = await sync_clean_calculate_candidate()
        print(f"[{_ts()}] ✓ STEP 1-3 PASSED — data synced, cleaned, candidates scored.")
        print(f"           Result: {result}")
        print(f"           Next:   monitor runs at 09:00 AM ET")
    except Exception as exc:
        logger.exception("STEP 1-3 failed")
        print(f"[{_ts()}] ✗ STEP 1-3 FAILED — {exc}")
        print(f"           Fix the data pipeline, then retry:")
        print(f"           GET /api/v1/algo_trading/sync_clean_calculate_candidate")


# ---------------------------------------------------------------------------
# Daily Step 4: monitor (09:00 AM ET) — gate for preview + trade
# ---------------------------------------------------------------------------
async def _job_daily_monitor():
    global _monitor_ok

    from services.algo_trading_service import daily_monitor_endpoint

    print(f"\n[{_ts()}] → STEP 4 — run_daily_monitor starting...")
    try:
        result = await daily_monitor_endpoint()
        summary = result.get("summary", {})
        ok = summary.get("ok", False)
        _monitor_ok = ok

        if ok:
            print(f"[{_ts()}] ✓ STEP 4 PASSED — monitor ok. Safe to preview and trade.")
            print(f"           Next: preview at 09:15 AM ET")
        else:
            alerts = summary.get("alerts", [])
            print(f"[{_ts()}] ✗ STEP 4 FAILED — monitor returned ok=false. DO NOT TRADE today.")
            for alert in alerts:
                print(f"           ALERT: {alert}")
            print(f"           → Resolve alerts, then re-run manually:")
            print(f"             GET /api/v1/algo_trading/run_daily_monitor")
            print(f"           → Steps 5 and 6 are blocked until monitor passes.")

    except Exception as exc:
        _monitor_ok = False
        logger.exception("STEP 4 failed")
        print(f"[{_ts()}] ✗ STEP 4 ERROR — {exc}")
        print(f"           → Check server, then retry:")
        print(f"             GET /api/v1/algo_trading/run_daily_monitor")


# ---------------------------------------------------------------------------
# Daily Step 5: preview signal (09:15 AM ET)
# ---------------------------------------------------------------------------
async def _job_preview():
    if not _monitor_ok:
        print(f"\n[{_ts()}] ✗ STEP 5 SKIPPED — monitor did not pass this morning.")
        print(f"           → Re-run monitor first: GET /api/v1/algo_trading/run_daily_monitor")
        return

    from services.algo_trading_service import preview_best_pair_live

    print(f"\n[{_ts()}] → STEP 5 — preview_best_pair_live starting...")
    try:
        result = await preview_best_pair_live()
        print(f"[{_ts()}] ✓ STEP 5 PASSED — trade plan computed (no orders submitted).")
        print(f"           Plan: {result}")
        print(f"           Next: trade executes at 09:30 AM ET")
        print(f"           To cancel: stop the server before 09:30 AM ET")
    except Exception as exc:
        logger.exception("STEP 5 failed")
        print(f"[{_ts()}] ✗ STEP 5 FAILED — {exc}")
        print(f"           → Investigate, then retry:")
        print(f"             GET /api/v1/algo_trading/preview_best_pair_live")


# ---------------------------------------------------------------------------
# Daily Step 6: execute trade (09:30 AM ET)
# ---------------------------------------------------------------------------
async def _job_trade():
    if not _monitor_ok:
        print(f"\n[{_ts()}] ✗ STEP 6 SKIPPED — monitor did not pass this morning. Not trading.")
        print(f"           → Do not trade until monitor passes.")
        return

    from services.algo_trading_service import trade_best_pair_live

    print(f"\n[{_ts()}] → STEP 6 — trade_best_pair_live starting...")
    try:
        result = await trade_best_pair_live()
        print(f"[{_ts()}] ✓ STEP 6 PASSED — orders submitted via IBKR.")
        print(f"           Result: {result}")
    except Exception as exc:
        logger.exception("STEP 6 failed")
        print(f"[{_ts()}] ✗ STEP 6 FAILED — {exc}")
        print(f"           → Check IBKR account:")
        print(f"             GET /api/v1/algo_trading/get_account_summary")
        print(f"           → Retry trade if safe:")
        print(f"             POST /api/v1/algo_trading/trade_best_pair_live")


# ---------------------------------------------------------------------------
# Monthly Step 1+2: build model → deploy to Alpaca staging
# First Monday of the month, 07:00 AM ET
# ---------------------------------------------------------------------------
async def _job_monthly_build_and_stage():
    from services.algo_trading_service import monthly_build_endpoint, run_deploy

    # Step 1: monthly build (30-60 min)
    print(f"\n[{_ts()}] → MONTHLY STEP 1 — run_monthly_build starting (30-60 min)...")
    try:
        build_result = await monthly_build_endpoint()
    except Exception as exc:
        logger.exception("MONTHLY STEP 1 failed")
        print(f"[{_ts()}] ✗ MONTHLY STEP 1 FAILED — {exc}")
        print(f"           → Retry: GET /api/v1/algo_trading/run_monthly_build")
        return

    approved = build_result.get("approved", False)
    sharpe = build_result.get("sharpe_gate", {}).get("sharpe", "N/A")

    if not approved:
        print(f"[{_ts()}] ✗ MONTHLY STEP 1 NOT APPROVED — Sharpe={sharpe} < 0.5.")
        print(f"           Keeping existing deployed version. No deploy this month.")
        print(f"           → Check versions: GET /api/v1/algo_trading/model_versions")
        return

    print(f"[{_ts()}] ✓ MONTHLY STEP 1 PASSED — build approved (Sharpe={sharpe}).")
    print(f"           version_id={build_result.get('version_id')}")

    # Step 2: deploy approved version to Alpaca staging
    print(f"\n[{_ts()}] → MONTHLY STEP 2 — run_deploy: staging on Alpaca paper...")
    try:
        deploy_result = await run_deploy()
    except Exception as exc:
        logger.exception("MONTHLY STEP 2 failed")
        print(f"[{_ts()}] ✗ MONTHLY STEP 2 FAILED — {exc}")
        print(f"           → Retry: GET /api/v1/algo_trading/run_deploy")
        return

    if not deploy_result.get("staged"):
        reason = deploy_result.get("reason", "unknown")
        print(f"[{_ts()}] ✗ MONTHLY STEP 2 FAILED — {reason}")
        print(f"           → Retry: GET /api/v1/algo_trading/run_deploy")
        return

    print(f"[{_ts()}] ✓ MONTHLY STEP 2 PASSED — staged on Alpaca paper.")
    print(f"           tag={deploy_result.get('tag')}  n_pairs={deploy_result.get('n_pairs')}")
    print(f"           → Monitor paper PnL daily for 5 trading days:")
    print(f"             GET /api/v1/algo_trading/alpaca_paper/positions")
    print(f"           → Promote job runs automatically on second Monday at 07:00 AM ET")


# ---------------------------------------------------------------------------
# Monthly Step 3: promote Alpaca staging to IBKR live
# Second Monday of the month, 07:00 AM ET
# ---------------------------------------------------------------------------
async def _job_monthly_promote():
    from services.algo_trading_service import promote_to_live

    print(f"\n[{_ts()}] → MONTHLY PROMOTE — checking staging gate (≥5 days + paper loss < 2%)...")
    try:
        result = await promote_to_live()
    except Exception as exc:
        logger.exception("MONTHLY PROMOTE failed")
        print(f"[{_ts()}] ✗ MONTHLY PROMOTE FAILED — {exc}")
        print(f"           → Retry: GET /api/v1/algo_trading/promote_to_live")
        return

    if result.get("promoted"):
        tag = result.get("tag", "?")
        print(f"[{_ts()}] ✓ MONTHLY PROMOTE PASSED — {tag} promoted to IBKR live.")
        print(f"           → Verify account: GET /api/v1/algo_trading/get_account_summary")
    else:
        gate = result.get("gate", {})
        reason = gate.get("reason") or result.get("reason", "unknown")
        print(f"[{_ts()}] ✗ MONTHLY PROMOTE BLOCKED — gate not passed.")
        print(f"           Reason: {reason}")
        print(f"           → Check paper positions: GET /api/v1/algo_trading/alpaca_paper/positions")
        print(f"           → Re-run when ready: GET /api/v1/algo_trading/promote_to_live")
        print(f"           → Force-promote (testing only):")
        print(f"             GET /api/v1/algo_trading/promote_to_live?skip_gate=true")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def start():
    """Register all jobs and start the scheduler. Called from FastAPI lifespan."""
    ET = "America/New_York"

    # Daily jobs — Mon through Fri
    _scheduler.add_job(_job_sync_clean_candidates, CronTrigger(day_of_week="mon-fri", hour=7, minute=0, timezone=ET), id="daily_sync")
    _scheduler.add_job(_job_daily_monitor,         CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=ET), id="daily_monitor")
    _scheduler.add_job(_job_preview,               CronTrigger(day_of_week="mon-fri", hour=9, minute=15, timezone=ET), id="daily_preview")
    _scheduler.add_job(_job_trade,                 CronTrigger(day_of_week="mon-fri", hour=9, minute=30, timezone=ET), id="daily_trade")

    # Monthly jobs — first and second Monday of the month
    _scheduler.add_job(_job_monthly_build_and_stage, CronTrigger(day_of_week="mon", day="1-7",  hour=7, minute=0, timezone=ET), id="monthly_build")
    _scheduler.add_job(_job_monthly_promote,         CronTrigger(day_of_week="mon", day="8-14", hour=7, minute=0, timezone=ET), id="monthly_promote")

    _scheduler.start()
    logger.info("Scheduler started — 4 daily jobs + 2 monthly jobs registered.")
    print(f"[{_ts()}] Scheduler started (America/New_York):")
    print(f"  Mon-Fri  07:00  sync + clean + candidates")
    print(f"  Mon-Fri  09:00  daily monitor (gate)")
    print(f"  Mon-Fri  09:15  preview signal")
    print(f"  Mon-Fri  09:30  execute trade")
    print(f"  1st Mon  07:00  monthly build + stage")
    print(f"  2nd Mon  07:00  monthly promote to live")


def stop():
    """Shut down the scheduler gracefully. Called from FastAPI lifespan."""
    _scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped.")
