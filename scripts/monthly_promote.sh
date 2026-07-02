#!/usr/bin/env bash
# Monthly promote — 5 trading days after monthly_build.sh succeeds, ~7:00 AM ET
# Cron: manually triggered or: 0 11 6-12 * 1  (6th Mon of month at 7 AM ET)
#
# Promotes the staged Alpaca paper version to IBKR live after the gate check.
# Pass --skip-gate to force-promote (testing only).

set -euo pipefail

BASE_URL="${ALGO_BASE_URL:-http://localhost:8000/api/v1/algo_trading}"
DATE=$(date '+%Y-%m-%d %H:%M:%S')
SKIP_GATE="${1:-}"

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
# Check Alpaca paper positions before promoting
# ---------------------------------------------------------------------------
log "PRE-CHECK — alpaca_paper/positions: reviewing paper PnL before promoting..."
body=""
if body=$(call GET "${BASE_URL}/alpaca_paper/positions"); then
    ok "PRE-CHECK PASSED — Alpaca paper positions retrieved. Review PnL below."
    echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
else
    fail "PRE-CHECK FAILED — could not fetch Alpaca paper positions."
    echo "    Response: $body"
    echo ""
    echo "  → Check Alpaca connectivity, then retry:"
    echo "    curl ${BASE_URL}/alpaca_paper/positions"
    exit 1
fi

# ---------------------------------------------------------------------------
# Promote to live
# ---------------------------------------------------------------------------
PROMOTE_URL="${BASE_URL}/promote_to_live"
if [[ "$SKIP_GATE" == "--skip-gate" ]]; then
    PROMOTE_URL="${PROMOTE_URL}?skip_gate=true"
    log "PROMOTE — promote_to_live?skip_gate=true (gate bypassed for testing)..."
else
    log "PROMOTE — promote_to_live: checking ≥5 staging days + paper loss < 2%%..."
fi

body=""
if body=$(call GET "${PROMOTE_URL}"); then
    ok "PROMOTE PASSED — staged version promoted to IBKR live."
    echo "    Response: $(echo "$body" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))' 2>/dev/null || echo "$body")"
    echo ""
    echo "  → Next trade_best_pair_live will use the new pair universe."
    echo "  → Verify IBKR account: curl ${BASE_URL}/get_account_summary"
else
    fail "PROMOTE FAILED — promote_to_live returned an error (gate check may have failed)."
    echo "    Response: $body"
    echo ""
    echo "  → If paper loss ≥ 2%% or < 5 staging days, wait longer and retry:"
    echo "    scripts/monthly_promote.sh"
    echo "  → To force-promote during testing only:"
    echo "    scripts/monthly_promote.sh --skip-gate"
    exit 1
fi
