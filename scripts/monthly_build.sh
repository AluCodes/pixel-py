#!/usr/bin/env bash
# Monthly build runbook — first trading day of the month, 7:00 AM ET
# Cron: 0 11 1-7 * 1  (7:00 AM ET = 11:00 UTC on first Mon of month, adjust for DST)
#
# Steps run sequentially: build → confirm version → deploy to Alpaca staging.
# promote_to_live runs 5 trading days later via monthly_promote.sh.

set -euo pipefail

BASE_URL="${ALGO_BASE_URL:-http://localhost:8000/api/v1/algo_trading}"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

log()  { echo "[${DATE}] $*"; }
ok()   { echo "[${DATE}] ✓ $*"; }
fail() { echo "[${DATE}] ✗ $*" >&2; }

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
# STEP 1: Monthly Build (takes 30–60 min)
# ---------------------------------------------------------------------------
log "STEP 1 — run_monthly_build: rebuilding model (30–60 min, please wait)..."
body=""
if body=$(call GET "${BASE_URL}/run_monthly_build"); then
    approved=$(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(str(d.get("approved", "")).lower())' 2>/dev/null || echo "")

    if [[ "$approved" == "true" ]]; then
        ok "STEP 1 PASSED — monthly build approved (Sharpe ≥ 0.5)."
        echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
    else
        fail "STEP 1 NOT APPROVED — build completed but Sharpe < 0.5. Keeping existing deployed version."
        echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
        echo ""
        echo "  → No deploy this month. Resume daily trading with current version."
        echo "  → Check model versions: curl ${BASE_URL}/model_versions"
        exit 2
    fi
else
    fail "STEP 1 FAILED — run_monthly_build returned an error."
    echo "    Response: $body"
    echo ""
    echo "  → Investigate, then retry:"
    echo "    curl ${BASE_URL}/run_monthly_build"
    exit 1
fi

# ---------------------------------------------------------------------------
# STEP 2: Confirm approved version exists
# ---------------------------------------------------------------------------
log "STEP 2 — model_versions: confirming new approved version..."
body=""
if body=$(call GET "${BASE_URL}/model_versions"); then
    ok "STEP 2 PASSED — model versions retrieved."
    echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
else
    fail "STEP 2 FAILED — could not retrieve model versions."
    echo "    Response: $body"
    echo ""
    echo "  → Check versions manually, then continue to step 3 if an approved version exists:"
    echo "    curl ${BASE_URL}/model_versions"
    echo "  → If confirmed, proceed: curl ${BASE_URL}/run_deploy"
    exit 1
fi

# ---------------------------------------------------------------------------
# STEP 3: Deploy to Alpaca staging
# ---------------------------------------------------------------------------
log "STEP 3 — run_deploy: staging latest approved version on Alpaca paper..."
body=""
if body=$(call GET "${BASE_URL}/run_deploy"); then
    ok "STEP 3 PASSED — staged on Alpaca paper. Monitor paper PnL for 5 trading days."
    echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
    echo ""
    echo "  → Monitor Alpaca paper positions daily:"
    echo "    curl ${BASE_URL}/alpaca_paper/positions"
    echo "  → After 5 trading days with paper loss < 2%%, run:"
    echo "    scripts/monthly_promote.sh"
    echo "    (or: curl '${BASE_URL}/promote_to_live')"
else
    fail "STEP 3 FAILED — run_deploy returned an error."
    echo "    Response: $body"
    echo ""
    echo "  → Retry deploy (model version is still approved):"
    echo "    curl ${BASE_URL}/run_deploy"
    exit 1
fi
