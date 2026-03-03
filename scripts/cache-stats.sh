#!/usr/bin/env bash
#
# Monitor NLM Proxy response cache statistics.
#
# Usage:
#   ./scripts/cache-stats.sh              # One-shot stats
#   ./scripts/cache-stats.sh --watch      # Refresh every 5s
#   ./scripts/cache-stats.sh --watch 10   # Refresh every 10s

set -euo pipefail

# ── Read .env file ──────────────────────────────────────────────────────

load_env() {
    local env_file=""
    for f in .env "$HOME/.nlm-proxy/.env"; do
        [ -f "$f" ] && env_file="$f" && break
    done
    [ -z "$env_file" ] && return
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        # Trim whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        [ -n "$key" ] && [ -n "$value" ] && export "$key=$value"
    done < "$env_file"
}

load_env

# ── Configuration ───────────────────────────────────────────────────────

API_KEY="${NLM_PROXY_OPENAI_API_KEY:-}"
HOST="${NLM_PROXY_OPENAI_HOST:-localhost}"
[ "$HOST" = "0.0.0.0" ] && HOST="localhost"
PORT="${NLM_PROXY_OPENAI_PORT:-8080}"
BASE_URL="http://${HOST}:${PORT}"

if [ -z "$API_KEY" ]; then
    echo "ERROR: NLM_PROXY_OPENAI_API_KEY not set in .env or environment" >&2
    exit 1
fi

# ── Colors ──────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m'  # No Color

# ── Fetch stats ─────────────────────────────────────────────────────────

fetch_stats() {
    curl -s -f -H "Authorization: Bearer $API_KEY" "$BASE_URL/v1/cache/stats" 2>/dev/null
}

# ── Parse JSON (portable, no jq required) ───────────────────────────────

get_json_val() {
    # Simple JSON value extraction: get_json_val '{"key":123}' key
    local json="$1" key="$2"
    echo "$json" | grep -oP "\"${key}\"\s*:\s*\K[^,}]+" | tr -d ' "'
}

# ── Display stats ───────────────────────────────────────────────────────

show_stats() {
    local json="$1"

    local enabled=$(get_json_val "$json" "enabled")
    local total_hits=$(get_json_val "$json" "total_hits")
    local total_misses=$(get_json_val "$json" "total_misses")
    local total_bypasses=$(get_json_val "$json" "total_bypasses")
    local hit_rate=$(get_json_val "$json" "hit_rate")
    local l1_hits=$(get_json_val "$json" "l1_hits")
    local l2_hits=$(get_json_val "$json" "l2_hits")
    local l3_hits=$(get_json_val "$json" "l3_hits")
    local l3_misses=$(get_json_val "$json" "l3_misses")
    local entry_count=$(get_json_val "$json" "entry_count")
    local notebook_count=$(get_json_val "$json" "notebook_count")
    local max_entries=$(get_json_val "$json" "max_entries")
    local ttl_seconds=$(get_json_val "$json" "ttl_seconds")
    local semantic=$(get_json_val "$json" "semantic_enabled")

    clear

    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       NLM Proxy — Cache Monitor             ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
    echo -e "  ${GRAY}${ts}    ${BASE_URL}${NC}"
    echo ""

    if [ "$enabled" = "false" ]; then
        echo -e "  ${YELLOW}Cache is DISABLED${NC}"
        return
    fi

    # ── Hit Rate Bar ────────────────────────────────────────────────────
    local bar_len=30
    local rate_int=${hit_rate%.*}  # truncate decimal
    local filled=$((rate_int * bar_len / 100))
    local empty=$((bar_len - filled))

    local color="$RED"
    [ "$rate_int" -ge 40 ] && color="$YELLOW"
    [ "$rate_int" -ge 70 ] && color="$GREEN"

    local bar=""
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=0; i<empty; i++)); do bar+="░"; done

    echo -e "  ${WHITE}Hit Rate${NC}   ${color}${bar} ${hit_rate}%${NC}"
    echo ""

    # ── Summary ─────────────────────────────────────────────────────────
    local total=$((total_hits + total_misses))
    echo -e "  ${WHITE}SUMMARY${NC}"
    echo -e "  ${GRAY}─────────────────────────────────────────${NC}"
    printf "  Total Lookups     %8s\n" "$total"
    echo -e "  Total Hits        ${GREEN}$(printf '%8s' "$total_hits")${NC}"
    echo -e "  Total Misses      ${RED}$(printf '%8s' "$total_misses")${NC}"
    echo -e "  Total Bypasses    ${YELLOW}$(printf '%8s' "$total_bypasses")${NC}"
    echo ""

    # ── Layer Breakdown ─────────────────────────────────────────────────
    echo -e "  ${WHITE}LAYER BREAKDOWN${NC}"
    echo -e "  ${GRAY}─────────────────────────────────────────${NC}"
    echo -e "  L1 Exact Match    ${GREEN}$(printf '%8s' "$l1_hits")${NC}  hits"
    echo -e "  L2 Near-Exact     ${CYAN}$(printf '%8s' "$l2_hits")${NC}  hits"
    echo -e "  L3 LLM-Verified   ${BLUE}$(printf '%8s' "$l3_hits")${NC}  hits"
    echo -e "  L3 LLM-Rejected   ${YELLOW}$(printf '%8s' "$l3_misses")${NC}  misses"
    echo ""

    # ── Storage ─────────────────────────────────────────────────────────
    local pct=0
    [ "$max_entries" -gt 0 ] && pct=$((entry_count * 100 / max_entries))
    local ttl_hours=$(echo "scale=1; $ttl_seconds / 3600" | bc 2>/dev/null || echo "?")

    local sem_label="disabled"
    local sem_color="$GRAY"
    if [ "$semantic" = "true" ]; then sem_label="enabled"; sem_color="$GREEN"; fi

    echo -e "  ${WHITE}STORAGE${NC}"
    echo -e "  ${GRAY}─────────────────────────────────────────${NC}"
    printf "  Entries           %8s  / %s (%s%%)\n" "$entry_count" "$max_entries" "$pct"
    printf "  Notebooks         %8s\n" "$notebook_count"
    printf "  TTL               %8ss (%sh)\n" "$ttl_seconds" "$ttl_hours"
    echo -e "  Semantic          ${sem_color}$(printf '%8s' "$sem_label")${NC}"

    if [ -n "$WATCH_MODE" ]; then
        echo ""
        echo -e "  ${GRAY}Refreshing every ${INTERVAL}s  (Ctrl+C to stop)${NC}"
    fi
}

# ── Main ────────────────────────────────────────────────────────────────

WATCH_MODE=""
INTERVAL=5

if [ "${1:-}" = "--watch" ] || [ "${1:-}" = "-w" ]; then
    WATCH_MODE=1
    [ -n "${2:-}" ] && INTERVAL="$2"
fi

if [ -n "$WATCH_MODE" ]; then
    while true; do
        json=$(fetch_stats) || { echo -e "${RED}ERROR: Cannot reach ${BASE_URL}/v1/cache/stats${NC}"; sleep "$INTERVAL"; continue; }
        show_stats "$json"
        sleep "$INTERVAL"
    done
else
    json=$(fetch_stats) || { echo -e "${RED}ERROR: Cannot reach ${BASE_URL}/v1/cache/stats${NC}" >&2; exit 1; }
    show_stats "$json"
fi
