#!/usr/bin/env python3
"""Extract response cache events from nlm-proxy logs for analysis.

Reads log path from NLM_PROXY_LOG_FILE env var or .env file,
falls back to ~/.nlm-proxy/logs/nlm-proxy.log.

Usage:
    python scripts/cache-log-analyzer.py                  # all cache events
    python scripts/cache-log-analyzer.py --since 13:20    # events after HH:MM
    python scripts/cache-log-analyzer.py --today           # today's events only
    python scripts/cache-log-analyzer.py -o report.txt    # save to file
    python scripts/cache-log-analyzer.py --summary         # summary stats only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def find_log_path() -> Path:
    """Resolve log file path from env, .env files, or default."""
    # 1. Environment variable
    env_path = os.environ.get("NLM_PROXY_LOG_FILE")
    if env_path:
        return Path(os.path.expanduser(env_path))

    # 2. .env files (simple key=value parsing, no dependency needed)
    for env_file in [".env", Path.home() / ".nlm-proxy" / ".env"]:
        env_file = Path(env_file)
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    if key.strip() == "NLM_PROXY_LOG_FILE":
                        value = value.strip().strip("\"'")
                        if value:
                            return Path(os.path.expanduser(value))
            except Exception:
                pass

    # 3. Default
    return Path.home() / ".nlm-proxy" / "logs" / "nlm-proxy.log"


# ── Log Line Patterns ────────────────────────────────────────────────────

CACHE_PATTERNS = [
    # response_cache module events (core cache logic)
    (r"response_cache.*\[CACHE\]", "cache_core"),
    # server-level cache events (pre-routing, checking, storing)
    (r"openai\.server.*\[CACHE\]", "cache_server"),
    # notebook_cache events (invalidation triggers)
    (r"notebook_cache.*\[CACHE\].*(?:Sources changed|Invalidat)", "cache_invalidation"),
]

EVENT_CLASSIFIERS = [
    # Pre-routing
    (r"Global L1 HIT", "pre_routing_l1_hit"),
    (r"Pre-routing L1 HIT", "pre_routing_l1_hit_server"),
    # L1
    (r"L1 HIT for", "l1_hit"),
    (r"L1 MISS for", "l1_miss"),
    (r"L1 EXPIRED", "l1_expired"),
    (r"L1 BYPASS", "l1_bypass"),
    # L2
    (r"L2 HIT.*skip-LLM", "l2_hit_exact"),
    (r"L2 HIT", "l2_hit"),
    (r"L2 near-exact match", "l2_near_exact"),
    (r"L2 found \d+ candidates", "l2_candidates"),
    (r"L2 MISS", "l2_miss"),
    (r"L2 no entries", "l2_empty"),
    (r"L2 computing embedding", "l2_computing"),
    # L3
    (r"L3 HIT", "l3_hit"),
    (r"L3 MISS", "l3_miss"),
    (r"L3 verifying", "l3_verifying"),
    (r"LLM verified semantic match", "l3_verified"),
    (r"LLM: no semantic match", "l3_no_match"),
    (r"LLM verification timed out", "l3_timeout"),
    (r"LLM verification failed", "l3_error"),
    # Store/Alias
    (r"STORED", "stored"),
    (r"Alias created", "alias_created"),
    # Invalidation
    (r"Invalidated", "invalidated"),
    (r"Sources changed", "sources_changed"),
    (r"All entries cleared", "cleared"),
    # Model
    (r"Embedding model loaded", "model_loaded"),
    (r"L2 threshold.*below 0\.7", "threshold_warning"),
    # Server-level
    (r"Checking cache", "checking_cache"),
    (r"BYPASS \(async\)", "bypass"),
]

LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([\w.]+) - (\w+) - (.+)$"
)


def classify_event(message: str) -> str:
    """Classify a log message into an event type."""
    for pattern, event_type in EVENT_CLASSIFIERS:
        if re.search(pattern, message):
            return event_type
    return "other"


def extract_query(message: str) -> str | None:
    """Extract query text from a log message."""
    # Match: for 'query text' or query='query text'
    m = re.search(r"(?:for |query=)'([^']+)'", message)
    if m:
        return m.group(1)
    # Match: STORED 'query text'
    m = re.search(r"STORED '([^']+)'", message)
    if m:
        return m.group(1)
    # Match: Alias created: 'new' → 'old'
    m = re.search(r"Alias created: '([^']+)'", message)
    if m:
        return m.group(1)
    return None


def extract_similarity(message: str) -> float | None:
    """Extract similarity score from a log message."""
    m = re.search(r"sim=(\d+\.\d+)", message)
    return float(m.group(1)) if m else None


def extract_notebook(message: str) -> str | None:
    """Extract notebook_id prefix from a log message."""
    m = re.search(r"notebook=(\S+?)(?:\)|,|$)", message)
    return m.group(1) if m else None


# ── Main Logic ────────────────────────────────────────────────────────────


def parse_cache_lines(log_path: Path, since: str | None = None, today_only: bool = False):
    """Parse log file and yield structured cache events."""
    today_str = datetime.now().strftime("%Y-%m-%d") if today_only else None

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            # Parse structured log line
            m = LOG_LINE_RE.match(line)
            if not m:
                continue

            timestamp_str, logger_name, level, message = m.groups()

            # Check if this is a cache-related line
            is_cache = False
            for pattern, _ in CACHE_PATTERNS:
                if re.search(pattern, f"{logger_name}.*{message}") or re.search(pattern, line):
                    is_cache = True
                    break
            if not is_cache:
                continue

            # Time filters
            if today_only and today_str and not timestamp_str.startswith(today_str):
                continue
            if since:
                time_part = timestamp_str.split(" ")[1][:len(since)]
                if time_part < since:
                    continue

            event_type = classify_event(message)
            yield {
                "timestamp": timestamp_str,
                "logger": logger_name,
                "level": level,
                "message": message,
                "event": event_type,
                "query": extract_query(message),
                "similarity": extract_similarity(message),
                "notebook": extract_notebook(message),
                "raw": line,
            }


def format_event(evt: dict) -> str:
    """Format a single cache event for display."""
    ts = evt["timestamp"].split(" ")[1]  # time only
    level = evt["level"][:1]  # I/D/W/E
    event = evt["event"]

    # Color-code by event type
    ICONS = {
        "pre_routing_l1_hit": "🚀",
        "pre_routing_l1_hit_server": "🚀",
        "l1_hit": "✅",
        "l2_hit": "✅",
        "l2_hit_exact": "⚡",
        "l2_near_exact": "⚡",
        "l3_hit": "✅",
        "l3_verified": "✅",
        "stored": "💾",
        "alias_created": "🔗",
        "l1_miss": "·",
        "l2_miss": "·",
        "l2_empty": "·",
        "l3_miss": "❌",
        "l3_no_match": "❌",
        "l3_timeout": "⏳",
        "l3_error": "💥",
        "invalidated": "🗑️",
        "sources_changed": "⚠️",
        "cleared": "🗑️",
        "model_loaded": "📦",
        "threshold_warning": "⚠️",
        "bypass": "⏭️",
        "l1_bypass": "⏭️",
        "checking_cache": "🔍",
        "l2_computing": "🧮",
        "l2_candidates": "📋",
        "l3_verifying": "🤖",
    }
    icon = ICONS.get(event, " ")

    # Build compact display
    parts = [f"{ts} {icon} {event:.<28s}"]

    if evt["query"]:
        q = evt["query"][:60]
        parts.append(f"q='{q}'")
    if evt["similarity"] is not None:
        parts.append(f"sim={evt['similarity']:.4f}")
    if evt["notebook"]:
        parts.append(f"nb={evt['notebook'][:12]}")

    return " ".join(parts)


def print_summary(events: list[dict]) -> str:
    """Generate summary statistics."""
    counter = Counter(e["event"] for e in events)
    queries_seen = set()
    aliases = []
    sims = []

    for e in events:
        if e["query"]:
            queries_seen.add(e["query"])
        if e["event"] == "alias_created" and e["query"]:
            aliases.append(e["query"])
        if e["similarity"] is not None:
            sims.append(e["similarity"])

    lines = []
    lines.append("=" * 60)
    lines.append("CACHE PERFORMANCE SUMMARY")
    lines.append("=" * 60)

    # Time range
    if events:
        t0 = events[0]["timestamp"]
        t1 = events[-1]["timestamp"]
        lines.append(f"Period:  {t0}  →  {t1}")

    lines.append("")

    # Hit/miss breakdown
    lines.append("── Hit/Miss Breakdown ──")
    hit_events = [
        ("pre_routing_l1_hit", "Pre-routing L1 HITs (skipped routing)"),
        ("l1_hit", "Post-routing L1 HITs"),
        ("l2_hit_exact", "L2 HITs (near-exact, skip LLM)"),
        ("l2_hit", "L2 HITs"),
        ("l3_hit", "L3 HITs (LLM verified)"),
    ]
    miss_events = [
        ("l1_miss", "L1 MISSes"),
        ("l2_miss", "L2 MISSes"),
        ("l3_miss", "L3 MISSes (LLM rejected)"),
    ]
    other_events = [
        ("stored", "Responses stored"),
        ("alias_created", "Aliases created"),
        ("bypass", "Cache bypasses"),
        ("invalidated", "Notebooks invalidated"),
    ]

    total_hits = sum(counter.get(e, 0) for e, _ in hit_events)
    total_misses = sum(counter.get(e, 0) for e, _ in miss_events)
    total_lookups = total_hits + total_misses
    hit_rate = (total_hits / total_lookups * 100) if total_lookups > 0 else 0

    for event, label in hit_events:
        count = counter.get(event, 0)
        if count:
            lines.append(f"  {count:>4d}  {label}")
    for event, label in miss_events:
        count = counter.get(event, 0)
        if count:
            lines.append(f"  {count:>4d}  {label}")

    lines.append(f"\n  Hit rate: {hit_rate:.1f}% ({total_hits}/{total_lookups} lookups)")

    lines.append("")
    lines.append("── Storage ──")
    for event, label in other_events:
        count = counter.get(event, 0)
        if count:
            lines.append(f"  {count:>4d}  {label}")

    lines.append(f"  {len(queries_seen):>4d}  Unique queries seen")

    if sims:
        lines.append("")
        lines.append("── L2 Similarity Scores ──")
        lines.append(f"  Min: {min(sims):.4f}  Max: {max(sims):.4f}  Avg: {sum(sims)/len(sims):.4f}")
        for e in events:
            if e["similarity"] is not None and e["query"]:
                lines.append(f"  {e['similarity']:.4f}  '{e['query'][:60]}'")

    if aliases:
        lines.append("")
        lines.append("── Aliases Created ──")
        for alias in aliases:
            lines.append(f"  → '{alias[:70]}'")

    # Query timeline (grouped)
    query_events = defaultdict(list)
    for e in events:
        if e["query"] and e["event"] not in ("l2_computing", "checking_cache"):
            query_events[e["query"]].append(e["event"])

    if query_events:
        lines.append("")
        lines.append("── Query Timeline ──")
        for query, evts in query_events.items():
            flow = " → ".join(evts)
            lines.append(f"  '{query[:50]}': {flow}")

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract response cache events from nlm-proxy logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-file", "-f",
        help="Path to log file (overrides env/default)",
    )
    parser.add_argument(
        "--since", "-s",
        help="Show events after HH:MM (e.g., 13:20)",
    )
    parser.add_argument(
        "--today", "-t",
        action="store_true",
        help="Show only today's events",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics only",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw log lines instead of formatted",
    )
    parser.add_argument(
        "-o", "--output",
        help="Save output to file",
    )
    args = parser.parse_args()

    # Resolve log path
    if args.log_file:
        log_path = Path(os.path.expanduser(args.log_file))
    else:
        log_path = find_log_path()

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        print(f"Set NLM_PROXY_LOG_FILE env var or use --log-file", file=sys.stderr)
        sys.exit(1)

    print(f"Reading: {log_path}", file=sys.stderr)

    # Parse events
    events = list(parse_cache_lines(log_path, since=args.since, today_only=args.today))

    if not events:
        print("No cache events found.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(events)} cache events", file=sys.stderr)

    # Format output
    output_lines = []

    if not args.summary:
        for evt in events:
            if args.raw:
                output_lines.append(evt["raw"])
            else:
                output_lines.append(format_event(evt))

    output_lines.append("")
    output_lines.append(print_summary(events))

    output = "\n".join(output_lines)

    # Write output
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output, encoding="utf-8")
        print(f"Saved to: {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
