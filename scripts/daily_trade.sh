#!/usr/bin/env bash
# Daily trading runbook — run Mon–Fri before 9:15 AM ET
# Cron: 0 11 * * 1-5  (7:00 AM ET = 11:00 UTC, adjust for DST)
#
# Steps 1–3 are combined; step 4 is a hard gate before execution.
# Step 5 (preview) and step 6 (trade) run at market open via separate cron entries.

set -euo pipefail

BASE_URL="${ALGO_BASE_URL:-http://localhost:8000/api/v1/algo_trading}"
DATE=$(date '+%Y-%m-%d %H:%M:%S')
STEP="${1:-all}"  # pass "monitor", "preview", or "trade" to run a single step

log()  { echo "[${DATE}] $*"; }
ok()   { echo "[${DATE}] ✓ $*"; }
fail() { echo "[${DATE}] ✗ $*" >&2; }

# ---------------------------------------------------------------------------
# Helper: call an endpoint, print status, return body
# ---------------------------------------------------------------------------
call() {
    local method="$1" url="$2"
    local response http_code body

    response=$(curl -s -w "\n%{http_code}" -X "${method}" "${url}" \
        -H "Content-Type: application/json" 2>&1)
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [[ "$http_code" -ge 200 && "$http_code" -lt 300 ]]; then
        echo "$body"
        return 0
    else
        echo "$body"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# STEP 1-3: Sync + Clean + Calculate Candidates (combined)
# ---------------------------------------------------------------------------
run_sync_clean_candidates() {
    log "STEP 1-3 — sync_clean_calculate_candidate: syncing data, cleaning, scoring pairs..."
    local body
    if body=$(call GET "${BASE_URL}/sync_clean_calculate_candidate"); then
        ok "STEP 1-3 PASSED — data synced, cleaned, candidates scored."
        echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
    else
        fail "STEP 1-3 FAILED — sync/clean/candidates did not complete."
        echo "    Response: $body"
        echo ""
        echo "  → Fix the data pipeline, then re-run:"
        echo "    curl ${BASE_URL}/sync_clean_calculate_candidate"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# STEP 4: Daily Monitor (hard gate — do not trade if this fails)
# ---------------------------------------------------------------------------
run_daily_monitor() {
    log "STEP 4 — run_daily_monitor: checking data freshness, regime, positions, ETB..."
    local body ok_flag
    if body=$(call GET "${BASE_URL}/run_daily_monitor"); then
        ok_flag=$(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(str(d.get("summary",{}).get("ok", d.get("ok", ""))).lower())' 2>/dev/null || echo "")

        if [[ "$ok_flag" == "true" ]]; then
            ok "STEP 4 PASSED — monitor ok. Safe to proceed to preview + trade."
            echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
        else
            fail "STEP 4 FAILED — monitor returned ok=false. DO NOT TRADE today."
            echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
            echo ""
            echo "  → Review alerts above, resolve issues, then re-run monitor:"
            echo "    curl ${BASE_URL}/run_daily_monitor"
            echo "  → Do NOT run step 5 or 6 until monitor passes."
            exit 2
        fi
    else
        fail "STEP 4 ERROR — could not reach run_daily_monitor endpoint."
        echo "    Response: $body"
        echo ""
        echo "  → Check that the API server is running, then retry:"
        echo "    curl ${BASE_URL}/run_daily_monitor"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# STEP 5: Preview (dry-run, 9:15 AM ET)
# ---------------------------------------------------------------------------
run_preview() {
    log "STEP 5 — preview_best_pair_live: computing signal + trade plan (no orders)..."
    local body
    if body=$(call GET "${BASE_URL}/preview_best_pair_live"); then
        ok "STEP 5 PASSED — preview complete. Review the plan below before trading."
        echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
        echo ""
        echo "  → If the plan looks correct, execute at 9:30–9:45 AM ET:"
        echo "    curl -X POST ${BASE_URL}/trade_best_pair_live"
    else
        fail "STEP 5 FAILED — preview did not return a valid plan."
        echo "    Response: $body"
        echo ""
        echo "  → Investigate before trading. Re-run preview:"
        echo "    curl ${BASE_URL}/preview_best_pair_live"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# STEP 6: Execute trade (9:30–9:45 AM ET)
# ---------------------------------------------------------------------------
run_trade() {
    log "STEP 6 — trade_best_pair_live: submitting orders via IBKR..."
    local body
    if body=$(call POST "${BASE_URL}/trade_best_pair_live"); then
        ok "STEP 6 PASSED — trade submitted."
        echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
    else
        fail "STEP 6 FAILED — trade execution error."
        echo "    Response: $body"
        echo ""
        echo "  → Check IBKR connection and account status:"
        echo "    curl ${BASE_URL}/get_account_summary"
        echo "  → Retry trade if safe:"
        echo "    curl -X POST ${BASE_URL}/trade_best_pair_live"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$STEP" in
    all)
        run_sync_clean_candidates
        run_daily_monitor
        ;;
    monitor)
        run_daily_monitor
        ;;
    preview)
        run_preview
        ;;
    trade)
        run_trade
        ;;
    *)
        echo "Usage: $0 [all|monitor|preview|trade]"
        exit 1
        ;;
esac
