#!/usr/bin/env python3
"""Extract response cache events from nlm-proxy logs for analysis.

Reads log path from NLM_PROXY_LOG_FILE env var or .env file,
falls back to ~/.nlm-proxy/logs/nlm-proxy.log.

Usage:
    python scripts/cache-log-analyzer.py                     # all cache events
    python scripts/cache-log-analyzer.py --since 13:20       # events after HH:MM
    python scripts/cache-log-analyzer.py --today              # today's events only
    python scripts/cache-log-analyzer.py --summary            # summary stats only
    python scripts/cache-log-analyzer.py --queries            # per-query grouped view
    python scripts/cache-log-analyzer.py --json               # JSON output (for AI/scripts)
    python scripts/cache-log-analyzer.py --json --queries     # JSON per-query groups
    python scripts/cache-log-analyzer.py -o report.txt        # save to file
"""

from __future__ import annotations

import argparse
import json
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
    (r"Global L1 EXPIRED", "pre_routing_l1_expired"),
    (r"Pre-routing L1 HIT", "pre_routing_l1_hit_server"),
    (r"Pre-routing L1 HIT but notebook.*not in allowed", "pre_routing_l1_acl_blocked"),
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
    (r"Rebuilt embedding matrix", "l2_matrix_rebuild"),
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
    if m:
        return float(m.group(1))
    # Also match: best=0.7216
    m = re.search(r"best=(\d+\.\d+)", message)
    return float(m.group(1)) if m else None


def extract_notebook(message: str) -> str | None:
    """Extract notebook_id prefix from a log message."""
    m = re.search(r"notebook=(\S+?)(?:\)|,|$)", message)
    return m.group(1) if m else None


def extract_details(message: str) -> dict:
    """Extract additional details from a log message."""
    details = {}
    # hits=N
    m = re.search(r"hits=(\d+)", message)
    if m:
        details["hit_count"] = int(m.group(1))
    # age=Ns
    m = re.search(r"age=(\d+)s", message)
    if m:
        details["age_seconds"] = int(m.group(1))
    # answer_len=N
    m = re.search(r"answer_len=(\d+)", message)
    if m:
        details["answer_len"] = int(m.group(1))
    # total_entries=N
    m = re.search(r"total_entries=(\d+)", message)
    if m:
        details["total_entries"] = int(m.group(1))
    # threshold=N
    m = re.search(r"threshold=(\d+\.\d+)", message)
    if m:
        details["threshold"] = float(m.group(1))
    # N entries checked
    m = re.search(r"(\d+) entries checked", message)
    if m:
        details["entries_checked"] = int(m.group(1))
    # N candidates
    m = re.search(r"(?:found |verifying )(\d+) candidates", message)
    if m:
        details["num_candidates"] = int(m.group(1))
    # best=N
    m = re.search(r"best=(\d+\.\d+)", message)
    if m:
        details["best_similarity"] = float(m.group(1))
    # shape=(N, M)
    m = re.search(r"shape=\((\d+), (\d+)\)", message)
    if m:
        details["matrix_rows"] = int(m.group(1))
        details["embedding_dims"] = int(m.group(2))
    # Alias target: → 'target'
    m = re.search(r"→ '([^']+)'", message)
    if m:
        details["alias_target"] = m.group(1)
    # is_first_turn
    m = re.search(r"is_first_turn=(\w+)", message)
    if m:
        details["is_first_turn"] = m.group(1) == "True"
    # hash prefix
    m = re.search(r"hash=([a-f0-9]+)", message)
    if m:
        details["hash_prefix"] = m.group(1)
    return details


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
                "details": extract_details(message),
                "raw": line,
            }


# ── Output Formatters ────────────────────────────────────────────────────

ICONS = {
    "pre_routing_l1_hit": "🚀",
    "pre_routing_l1_hit_server": "🚀",
    "pre_routing_l1_expired": "⏰",
    "pre_routing_l1_acl_blocked": "🔒",
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
    "l2_matrix_rebuild": "🔄",
    "l3_verifying": "🤖",
}


def format_event(evt: dict) -> str:
    """Format a single cache event for display."""
    ts = evt["timestamp"].split(" ")[1]  # time only
    event = evt["event"]
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

    # Append relevant details
    d = evt.get("details", {})
    if "hit_count" in d:
        parts.append(f"hits={d['hit_count']}")
    if "age_seconds" in d:
        parts.append(f"age={d['age_seconds']}s")
    if "answer_len" in d:
        parts.append(f"len={d['answer_len']}")
    if "num_candidates" in d:
        parts.append(f"cands={d['num_candidates']}")
    if "alias_target" in d:
        parts.append(f"→ '{d['alias_target'][:40]}'")

    return " ".join(parts)


def group_by_query(events: list[dict]) -> list[dict]:
    """Group events into per-query lifecycle groups.

    Each group represents a single cache lookup sequence:
    checking_cache → l1_miss → l2 → l3 → stored/hit

    Returns list of dicts with query, events, outcome, timing.
    """
    groups = []
    current_group = None

    # Events that start a new group
    GROUP_STARTERS = {"checking_cache", "pre_routing_l1_hit", "pre_routing_l1_hit_server",
                      "pre_routing_l1_expired", "pre_routing_l1_acl_blocked"}
    # Events that end a group
    GROUP_ENDERS = {"stored", "l1_hit", "l2_hit", "l2_hit_exact", "l3_hit",
                    "l2_miss", "l3_miss", "alias_created",
                    "pre_routing_l1_hit_server", "bypass"}

    for evt in events:
        event_type = evt["event"]

        # Skip model-loading, threshold warnings, matrix rebuilds
        if event_type in ("model_loaded", "threshold_warning", "l2_matrix_rebuild", "other"):
            # Attribute to current group if exists
            if current_group and event_type not in ("model_loaded", "threshold_warning"):
                current_group["events"].append(evt)
            continue

        # Start new group on checking_cache or pre-routing hit
        if event_type in GROUP_STARTERS:
            if current_group:
                _finalize_group(current_group)
                groups.append(current_group)
            current_group = {
                "query": evt.get("query"),
                "notebook": evt.get("notebook"),
                "start_time": evt["timestamp"],
                "end_time": evt["timestamp"],
                "events": [evt],
                "outcome": None,
                "similarity": None,
                "layers_hit": [],
            }
            continue

        # Add to current group
        if current_group:
            current_group["events"].append(evt)
            current_group["end_time"] = evt["timestamp"]
            if evt.get("query") and not current_group["query"]:
                current_group["query"] = evt["query"]
            if evt.get("notebook") and not current_group["notebook"]:
                current_group["notebook"] = evt["notebook"]
            if evt.get("similarity") is not None:
                current_group["similarity"] = evt["similarity"]

            # Finalize on group enders
            if event_type in GROUP_ENDERS:
                _finalize_group(current_group)
                groups.append(current_group)
                current_group = None
        else:
            # Standalone event (e.g., stored after streaming)
            standalone = {
                "query": evt.get("query"),
                "notebook": evt.get("notebook"),
                "start_time": evt["timestamp"],
                "end_time": evt["timestamp"],
                "events": [evt],
                "outcome": event_type,
                "similarity": evt.get("similarity"),
                "layers_hit": [],
            }
            _finalize_group(standalone)
            groups.append(standalone)

    # Flush last group
    if current_group:
        _finalize_group(current_group)
        groups.append(current_group)

    return groups


def _finalize_group(group: dict):
    """Determine outcome and layers from a group's events."""
    event_types = [e["event"] for e in group["events"]]

    # Determine outcome
    if "pre_routing_l1_hit" in event_types or "pre_routing_l1_hit_server" in event_types:
        group["outcome"] = "PRE_ROUTING_L1_HIT"
    elif "l1_hit" in event_types:
        group["outcome"] = "L1_HIT"
    elif "l2_hit_exact" in event_types or "l2_hit" in event_types:
        group["outcome"] = "L2_HIT"
    elif "l3_hit" in event_types or "l3_verified" in event_types:
        group["outcome"] = "L3_HIT"
    elif "stored" in event_types:
        group["outcome"] = "MISS_STORED"
    elif "l3_miss" in event_types or "l3_no_match" in event_types:
        group["outcome"] = "L3_MISS"
    elif "l2_miss" in event_types or "l2_empty" in event_types:
        group["outcome"] = "L2_MISS"
    elif "alias_created" in event_types:
        group["outcome"] = "ALIAS_CREATED"
    elif "bypass" in event_types:
        group["outcome"] = "BYPASS"
    else:
        group["outcome"] = "UNKNOWN"

    # Track layers touched
    layers = []
    if any(e.startswith("pre_routing") for e in event_types):
        layers.append("pre_routing_l1")
    if "l1_miss" in event_types or "l1_hit" in event_types:
        layers.append("l1")
    if any(e.startswith("l2_") for e in event_types):
        layers.append("l2")
    if any(e.startswith("l3_") for e in event_types):
        layers.append("l3")
    group["layers_hit"] = layers

    # Compute duration (ms)
    try:
        fmt = "%Y-%m-%d %H:%M:%S,%f"
        t0 = datetime.strptime(group["start_time"], fmt)
        t1 = datetime.strptime(group["end_time"], fmt)
        group["duration_ms"] = int((t1 - t0).total_seconds() * 1000)
    except Exception:
        group["duration_ms"] = None


def format_query_group(group: dict) -> str:
    """Format a per-query group for display."""
    lines = []
    q = group.get("query", "?")[:70]
    outcome = group.get("outcome", "?")
    dur = group.get("duration_ms")
    sim = group.get("similarity")
    nb = (group.get("notebook") or "")[:12]
    ts = group["start_time"].split(" ")[1]

    # Outcome icon
    OUTCOME_ICONS = {
        "PRE_ROUTING_L1_HIT": "🚀", "L1_HIT": "✅", "L2_HIT": "⚡",
        "L3_HIT": "✅", "MISS_STORED": "💾", "L3_MISS": "❌",
        "L2_MISS": "·", "ALIAS_CREATED": "🔗", "BYPASS": "⏭️",
    }
    icon = OUTCOME_ICONS.get(outcome, "?")

    header = f"  {ts} {icon} {outcome:<22s} q='{q}'"
    if sim is not None:
        header += f"  sim={sim:.4f}"
    if dur is not None and dur > 0:
        header += f"  ({dur}ms)"
    if nb:
        header += f"  nb={nb}"
    lines.append(header)

    # Show event flow on second line
    flow = [e["event"] for e in group["events"]
            if e["event"] not in ("checking_cache", "l2_computing")]
    if flow:
        lines.append(f"           flow: {' → '.join(flow)}")

    # Show details from events
    for evt in group["events"]:
        d = evt.get("details", {})
        if d.get("alias_target"):
            lines.append(f"           alias: '{q}' → '{d['alias_target'][:60]}'")
        if d.get("num_candidates") and evt["event"] == "l2_candidates":
            lines.append(
                f"           L2: {d['num_candidates']} candidates, "
                f"best={d.get('best_similarity', '?')}, "
                f"threshold={d.get('threshold', '?')}"
            )

    return "\n".join(lines)


def build_summary(events: list[dict], groups: list[dict] | None = None) -> dict:
    """Build summary statistics as a dict (for both text and JSON output)."""
    counter = Counter(e["event"] for e in events)
    queries_seen = set()
    aliases = []
    sims = []

    for e in events:
        if e["query"]:
            queries_seen.add(e["query"])
        if e["event"] == "alias_created" and e["query"]:
            target = e.get("details", {}).get("alias_target", "?")
            aliases.append({"from": e["query"], "to": target})
        if e["similarity"] is not None:
            sims.append({"query": e.get("query", "?"), "score": e["similarity"],
                         "event": e["event"]})

    # Hit/miss counts
    hit_events = {
        "pre_routing_l1_hits": counter.get("pre_routing_l1_hit", 0),
        "l1_hits": counter.get("l1_hit", 0),
        "l2_hits_exact": counter.get("l2_hit_exact", 0),
        "l2_hits": counter.get("l2_hit", 0),
        "l3_hits": counter.get("l3_hit", 0),
    }
    miss_events = {
        "l1_misses": counter.get("l1_miss", 0),
        "l2_misses": counter.get("l2_miss", 0),
        "l3_misses": counter.get("l3_miss", 0),
    }
    total_hits = sum(hit_events.values())
    total_misses = sum(miss_events.values())
    total_lookups = total_hits + total_misses

    summary = {
        "period": {
            "start": events[0]["timestamp"] if events else None,
            "end": events[-1]["timestamp"] if events else None,
        },
        "total_events": len(events),
        "hits": hit_events,
        "misses": miss_events,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_lookups": total_lookups,
        "hit_rate_pct": round(total_hits / total_lookups * 100, 1) if total_lookups > 0 else 0,
        "stored": counter.get("stored", 0),
        "aliases_created": counter.get("alias_created", 0),
        "bypasses": counter.get("bypass", 0),
        "invalidations": counter.get("invalidated", 0),
        "unique_queries": len(queries_seen),
        "similarity_scores": sims,
        "aliases": aliases,
    }

    # Group-level stats
    if groups:
        outcomes = Counter(g["outcome"] for g in groups)
        durations = [g["duration_ms"] for g in groups if g.get("duration_ms") is not None]
        summary["query_groups"] = {
            "total": len(groups),
            "outcomes": dict(outcomes),
            "durations_ms": {
                "min": min(durations) if durations else None,
                "max": max(durations) if durations else None,
                "avg": round(sum(durations) / len(durations)) if durations else None,
            },
        }

    return summary


def format_summary_text(summary: dict) -> str:
    """Format summary dict as human-readable text."""
    lines = []
    lines.append("=" * 64)
    lines.append("CACHE PERFORMANCE SUMMARY")
    lines.append("=" * 64)

    p = summary["period"]
    if p["start"]:
        lines.append(f"Period: {p['start']}  →  {p['end']}")

    lines.append("")
    lines.append("── Hit/Miss Breakdown ──")

    labels = {
        "pre_routing_l1_hits": "Pre-routing L1 HITs (skipped routing)",
        "l1_hits": "Post-routing L1 HITs",
        "l2_hits_exact": "L2 HITs (near-exact, skip LLM)",
        "l2_hits": "L2 HITs (LLM verified)",
        "l3_hits": "L3 HITs (LLM verified)",
        "l1_misses": "L1 MISSes",
        "l2_misses": "L2 MISSes",
        "l3_misses": "L3 MISSes (LLM rejected)",
    }
    for key, label in labels.items():
        bucket = "hits" if "hits" in key else "misses"
        count = summary[bucket].get(key, 0)
        if count:
            lines.append(f"  {count:>4d}  {label}")

    lines.append(f"\n  Hit rate: {summary['hit_rate_pct']:.1f}% "
                 f"({summary['total_hits']}/{summary['total_lookups']} lookups)")

    lines.append("")
    lines.append("── Storage ──")
    for key, label in [("stored", "Responses stored"), ("aliases_created", "Aliases created"),
                       ("bypasses", "Cache bypasses"), ("invalidations", "Notebooks invalidated")]:
        count = summary.get(key, 0)
        if count:
            lines.append(f"  {count:>4d}  {label}")
    lines.append(f"  {summary['unique_queries']:>4d}  Unique queries seen")

    sims = summary.get("similarity_scores", [])
    if sims:
        scores = [s["score"] for s in sims]
        lines.append("")
        lines.append("── L2 Similarity Scores ──")
        lines.append(f"  Min: {min(scores):.4f}  Max: {max(scores):.4f}  "
                     f"Avg: {sum(scores)/len(scores):.4f}")
        for s in sims:
            lines.append(f"  {s['score']:.4f}  [{s['event']:<16s}]  '{(s['query'] or '?')[:50]}'")

    aliases = summary.get("aliases", [])
    if aliases:
        lines.append("")
        lines.append("── Aliases Created ──")
        for a in aliases:
            lines.append(f"  '{a['from'][:40]}' → '{a['to'][:40]}'")

    # Group-level stats
    gstats = summary.get("query_groups")
    if gstats:
        lines.append("")
        lines.append("── Query Group Outcomes ──")
        for outcome, count in sorted(gstats["outcomes"].items(), key=lambda x: -x[1]):
            lines.append(f"  {count:>4d}  {outcome}")
        d = gstats["durations_ms"]
        if d["avg"] is not None:
            lines.append(f"\n  Lookup duration: min={d['min']}ms avg={d['avg']}ms max={d['max']}ms")

    lines.append("=" * 64)
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
        help="Show summary statistics only (no event list)",
    )
    parser.add_argument(
        "--queries", "-q",
        action="store_true",
        help="Show per-query grouped lifecycle view",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON (for programmatic parsing)",
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
        print("Set NLM_PROXY_LOG_FILE env var or use --log-file", file=sys.stderr)
        sys.exit(1)

    print(f"Reading: {log_path}", file=sys.stderr)

    # Parse events
    events = list(parse_cache_lines(log_path, since=args.since, today_only=args.today))

    if not events:
        print("No cache events found.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(events)} cache events", file=sys.stderr)

    # Group by query
    groups = group_by_query(events)

    # ── JSON output ──
    if args.json:
        result = {
            "log_file": str(log_path),
            "summary": build_summary(events, groups),
        }
        if not args.summary:
            if args.queries:
                # Per-query grouped output
                result["query_groups"] = [
                    {
                        "query": g.get("query"),
                        "notebook": g.get("notebook"),
                        "outcome": g.get("outcome"),
                        "similarity": g.get("similarity"),
                        "duration_ms": g.get("duration_ms"),
                        "start_time": g.get("start_time"),
                        "end_time": g.get("end_time"),
                        "layers": g.get("layers_hit"),
                        "events": [
                            {k: v for k, v in e.items() if k != "raw"}
                            for e in g["events"]
                        ],
                    }
                    for g in groups
                ]
            else:
                result["events"] = [
                    {k: v for k, v in e.items() if k != "raw"}
                    for e in events
                ]
        output = json.dumps(result, ensure_ascii=False, indent=2)

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Saved to: {args.output}", file=sys.stderr)
        else:
            print(output)
        return

    # ── Text output ──
    output_lines = []

    if not args.summary:
        if args.queries:
            output_lines.append("── Per-Query Cache Lookups ──")
            output_lines.append("")
            for group in groups:
                output_lines.append(format_query_group(group))
                output_lines.append("")
        else:
            for evt in events:
                if args.raw:
                    output_lines.append(evt["raw"])
                else:
                    output_lines.append(format_event(evt))

    output_lines.append("")
    output_lines.append(format_summary_text(build_summary(events, groups)))

    output = "\n".join(output_lines)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output, encoding="utf-8")
        print(f"Saved to: {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
