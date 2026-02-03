#!/usr/bin/env python3
"""
Capture and display raw JSON response from NotebookLM Query Endpoint.

This script collects the complete streaming response and displays it in raw JSON format,
useful for documenting the API response structure and understanding citations, sources,
and metadata.

Usage:
    python tests/test_query_raw_response.py <notebook_id> [query]

Example:
    python tests/test_query_raw_response.py abc123-def456 "What are the main themes?"
    python tests/test_query_raw_response.py abc123 "Compare the sources" --save response.json

Requirements:
    - Must have authenticated using: notebooklm-mcp-auth
    - Notebook must exist and have at least one source
"""

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import AsyncIterator

import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlm_proxy.core.auth import load_cached_tokens
from nlm_proxy.core import NotebookLMClient


async def collect_raw_response(
    client: NotebookLMClient,
    notebook_id: str,
    query_text: str,
    source_ids: list[str],
) -> dict:
    """
    Collect complete raw response from streaming query endpoint.

    Returns dict with:
    - raw_chunks: List of all raw JSON chunks received
    - parsed_chunks: List of parsed chunk data
    - complete_response: Assembled complete response
    - metadata: Request/response metadata
    """
    import uuid

    # Get HTTP client
    http_client = await client._get_client()

    # Generate conversation ID
    conversation_id = str(uuid.uuid4())

    # Build source IDs structure: [[[sid]]] for each source
    sources_array = [[[sid]] for sid in source_ids]

    # Query params structure
    params = [
        sources_array,
        query_text,
        None,  # No conversation history for new conversation
        [2, None, [1]],
        conversation_id,
    ]

    # Build request body
    params_json = json.dumps(params, separators=(",", ":"))
    f_req = [None, params_json]
    f_req_json = json.dumps(f_req, separators=(",", ":"))

    body_parts = [f"f.req={urllib.parse.quote(f_req_json, safe='')}"]
    if client.csrf_token:
        body_parts.append(f"at={urllib.parse.quote(client.csrf_token, safe='')}")
    body = "&".join(body_parts) + "&"

    # Build URL with query parameters
    url_params = {
        "bl": os.environ.get("NOTEBOOKLM_BL", "boq_labs-tailwind-frontend_20260108.06_p0"),
        "hl": "en",
        "_reqid": str(100000),
        "rt": "c",
    }
    if client._session_id:
        url_params["f.sid"] = client._session_id

    query_string = urllib.parse.urlencode(url_params)
    url = f"{client.BASE_URL}{client.QUERY_ENDPOINT}?{query_string}"

    # Track raw data
    raw_lines = []
    raw_chunks = []
    parsed_chunks = []

    start_time = time.time()

    print("📡 Sending request...")
    print(f"   URL: {client.QUERY_ENDPOINT}")
    print(f"   Conversation ID: {conversation_id}")
    print(f"   Sources: {len(source_ids)}")
    print(f"   Query: {query_text[:100]}{'...' if len(query_text) > 100 else ''}\n")

    async with http_client.stream("POST", url, content=body, timeout=120.0) as response:
        response.raise_for_status()

        print(f"✅ Response status: {response.status_code}")
        print(f"📥 Collecting streaming data...\n")

        # Process line by line
        buffer = ""
        chunk_count = 0
        async for chunk in response.aiter_text():
            buffer += chunk

            # Process complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue

                raw_lines.append(line)

                # Skip anti-XSSI prefix
                if line.startswith(")]}'"):
                    continue

                # Try to parse as byte count
                try:
                    byte_count = int(line)
                    continue
                except ValueError:
                    pass

                # Try to parse as JSON chunk
                try:
                    chunk_json = json.loads(line)
                    raw_chunks.append(chunk_json)

                    # Parse the chunk
                    parsed = parse_response_chunk(chunk_json)
                    if parsed:
                        parsed_chunks.append(parsed)
                        chunk_count += 1
                        chunk_type = parsed.get('type', 'unknown')
                        text_len = len(parsed.get('text', ''))
                        has_citations = '[' in parsed.get('text', '') and ']' in parsed.get('text', '')
                        citation_marker = " 📎" if has_citations else ""
                        print(f"   Chunk {chunk_count}: {chunk_type} - {text_len} chars{citation_marker}")
                    else:
                        # Chunk didn't parse as expected, might be citation data
                        print(f"   Chunk {chunk_count + 1}: UNPARSED - might contain citation details")
                        raw_chunks[-1]['_unparsed'] = True

                except json.JSONDecodeError:
                    # Not JSON, store as raw
                    raw_chunks.append({"_raw_line": line})

    elapsed = time.time() - start_time

    print(f"\n✅ Collection complete in {elapsed:.2f}s")
    print(f"   Total chunks: {chunk_count}")
    print(f"   Raw lines: {len(raw_lines)}")

    # Assemble complete response
    complete_response = {
        "metadata": {
            "conversation_id": conversation_id,
            "notebook_id": notebook_id,
            "query": query_text,
            "source_ids": source_ids,
            "timestamp": time.time(),
            "duration_seconds": elapsed,
            "chunk_count": chunk_count,
        },
        "thinking": [c for c in parsed_chunks if c["type"] == "thinking"],
        "answer": [c for c in parsed_chunks if c["type"] == "answer"],
        "complete_answer_text": " ".join(c["text"] for c in parsed_chunks if c["type"] == "answer"),
        "complete_thinking_text": " ".join(c["text"] for c in parsed_chunks if c["type"] == "thinking"),
    }

    return {
        "raw_lines": raw_lines,
        "raw_chunks": raw_chunks,
        "parsed_chunks": parsed_chunks,
        "complete_response": complete_response,
    }


def parse_response_chunk(chunk_json: dict | list) -> dict | None:
    """
    Parse a raw JSON chunk from the streaming response.

    Extracts all available data including text, citations, sources, and metadata.
    """
    try:
        if not isinstance(chunk_json, list) or len(chunk_json) == 0:
            return None

        outer = chunk_json[0]
        if not isinstance(outer, list) or len(outer) < 3:
            return None

        # The third element is the nested JSON string
        nested_json_str = outer[2]
        if not isinstance(nested_json_str, str):
            return None

        # Parse nested JSON
        nested = json.loads(nested_json_str)
        if not isinstance(nested, list) or len(nested) == 0:
            return None

        # Get the content array
        content = nested[0]
        if not isinstance(content, list):
            return None

        # Extract all positions (expand to capture more positions if they exist)
        result = {
            "text": content[0] if len(content) > 0 else None,
            "position_1": content[1] if len(content) > 1 else None,
            "source_ids_array": content[2] if len(content) > 2 else None,  # Renamed for clarity
            "position_3": content[3] if len(content) > 3 else None,
            "type_info": content[4] if len(content) > 4 else None,
        }

        # Capture any additional positions beyond 4
        if len(content) > 5:
            result["extra_positions"] = {
                f"position_{i}": content[i] for i in range(5, len(content))
            }

        # Determine type
        if isinstance(result["type_info"], list) and len(result["type_info"]) > 0:
            type_indicator = result["type_info"][-1]
            if type_indicator == 1:
                result["type"] = "answer"
            elif type_indicator == 2:
                result["type"] = "thinking"
            else:
                result["type"] = "unknown"
        else:
            result["type"] = "unknown"

        # Parse source IDs from metadata array
        source_ids_array = result.get("source_ids_array")
        if isinstance(source_ids_array, list):
            # Extract UUIDs (first N elements before the numeric value)
            source_uuids = [
                item for item in source_ids_array
                if isinstance(item, str) and len(item) == 36  # UUID format
            ]
            if source_uuids:
                result["referenced_source_ids"] = source_uuids
                result["source_count"] = len(source_uuids)

        # Add full content array for complete documentation
        result["_full_content_array"] = content
        result["_array_length"] = len(content)
        result["_full_outer_array"] = outer  # Capture entire outer structure

        return result

    except (json.JSONDecodeError, IndexError, TypeError, KeyError) as e:
        return {"_parse_error": str(e), "_raw": chunk_json}


async def get_notebook_sources(client: NotebookLMClient, notebook_id: str) -> list[str]:
    """Get all source IDs from a notebook."""
    notebook_data = await client.get_notebook(notebook_id)
    source_ids = client._extract_source_ids_from_notebook(notebook_data)

    if not source_ids:
        print(f"❌ No sources found in notebook {notebook_id}")
        print("   Add sources to the notebook before querying.")
        sys.exit(1)

    return source_ids


async def main_async(notebook_id: str, query: str, save_file: str = None, pretty: bool = True):
    """
    Main async function to collect and display raw response.

    Args:
        notebook_id: The notebook UUID to query
        query: The question to ask
        save_file: Optional file path to save JSON output
        pretty: Use pretty-printed JSON (default: True)
    """
    print("=" * 80)
    print("NotebookLM Raw Response Collector")
    print("=" * 80)

    # Load authentication tokens
    print("\n🔑 Loading authentication tokens...")
    tokens = load_cached_tokens()
    if not tokens:
        print("❌ No cached authentication tokens found.")
        print("   Run 'notebooklm-mcp-auth' to authenticate.")
        sys.exit(1)

    print(f"✅ Loaded tokens (extracted {int(time.time() - tokens.extracted_at)}s ago)")

    # Initialize client
    print("\n🔧 Initializing API client...")
    client = NotebookLMClient(
        cookies=tokens.cookies,
        csrf_token=tokens.csrf_token,
        session_id=tokens.session_id,
    )

    try:
        await client._ensure_initialized()
        print("✅ Client initialized")

        # Get notebook sources
        print(f"\n📚 Fetching sources from notebook {notebook_id}...")
        source_ids = await get_notebook_sources(client, notebook_id)
        print(f"✅ Found {len(source_ids)} source(s)")
        print(f"   Source IDs: {', '.join(s[:8] + '...' for s in source_ids)}")

        # Collect raw response
        print("\n" + "=" * 80)
        result = await collect_raw_response(client, notebook_id, query, source_ids)
        print("=" * 80)

        # Display summary
        print("\n📊 Response Summary:")
        print(f"   Total raw lines: {len(result['raw_lines'])}")
        print(f"   Total raw chunks: {len(result['raw_chunks'])}")
        print(f"   Parsed chunks: {len(result['parsed_chunks'])}")
        print(f"   Thinking chunks: {len(result['complete_response']['thinking'])}")
        print(f"   Answer chunks: {len(result['complete_response']['answer'])}")

        # Show metadata presence
        chunks_with_source_ids = sum(
            1 for c in result['parsed_chunks']
            if c.get('referenced_source_ids')
        )
        chunks_with_citations = sum(
            1 for c in result['parsed_chunks']
            if '[1]' in c.get('text', '') or '[2]' in c.get('text', '')
        )
        unparsed_chunks = sum(
            1 for c in result['raw_chunks']
            if isinstance(c, dict) and c.get('_unparsed')
        )

        print(f"\n🔍 Metadata Analysis:")
        print(f"   Chunks with source IDs: {chunks_with_source_ids}/{len(result['parsed_chunks'])}")
        print(f"   Chunks with [N] markers: {chunks_with_citations}/{len(result['parsed_chunks'])}")
        print(f"   Unparsed chunks: {unparsed_chunks}")

        # Show source ID mapping
        if chunks_with_source_ids > 0:
            print(f"\n📚 Source References Found:")
            for i, chunk in enumerate(result['parsed_chunks']):
                if chunk.get('referenced_source_ids'):
                    sources = chunk['referenced_source_ids']
                    print(f"   Chunk {i+1} ({chunk['type']}): {len(sources)} source(s)")
                    for sid in sources:
                        print(f"      - {sid}")

        # Save to file if requested
        if save_file:
            print(f"\n💾 Saving to {save_file}...")
            with open(save_file, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(result, f, ensure_ascii=False)
            print(f"✅ Saved {os.path.getsize(save_file):,} bytes")
        else:
            # Display JSON to console
            print("\n" + "=" * 80)
            print("RAW JSON RESPONSE")
            print("=" * 80)
            if pretty:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(json.dumps(result, ensure_ascii=False))

        # Display answer
        print("\n" + "=" * 80)
        print("COMPLETE ANSWER")
        print("=" * 80)
        print(result['complete_response']['complete_answer_text'])

        print("\n" + "=" * 80)
        print("✅ Collection completed successfully")
        print("=" * 80)

    finally:
        await client.close()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Collect raw JSON response from NotebookLM query endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display raw response to console
  python tests/test_query_raw_response.py abc123 "What are the main themes?"

  # Save to file
  python tests/test_query_raw_response.py abc123 "Compare sources" --save response.json

  # Compact JSON (no pretty printing)
  python tests/test_query_raw_response.py abc123 "Analyze this" --save data.json --compact
        """,
    )

    parser.add_argument(
        "notebook_id",
        help="Notebook UUID to query",
    )

    parser.add_argument(
        "query",
        nargs="?",
        default="What are the main themes discussed in these sources?",
        help="Question to ask (default: generic themes question)",
    )

    parser.add_argument(
        "-s", "--save",
        metavar="FILE",
        help="Save JSON output to file instead of printing to console",
    )

    parser.add_argument(
        "-c", "--compact",
        action="store_true",
        help="Use compact JSON format (no pretty printing)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main_async(
            args.notebook_id,
            args.query,
            args.save,
            not args.compact
        ))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
