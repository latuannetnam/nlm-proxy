# OpenAI Proxy Citation Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add source citation support to the OpenAI proxy so Open WebUI displays clickable citation references for NotebookLM responses.

**Architecture:** Extract source UUIDs from NotebookLM streaming chunks (position 2 of content array), collect them during response streaming, resolve UUIDs to source titles using notebook metadata, then emit Open WebUI-compatible `{"type": "source", ...}` SSE events before `[DONE]`.

**Tech Stack:** Python, FastAPI, httpx streaming, SSE (Server-Sent Events)

---

## Background Context

### NotebookLM Citation Format (from `docs/CITATION_ANALYSIS.md`)
- Citations appear inline as `[1]`, `[2]` in answer text
- Source UUIDs are in position 2 of streaming chunk content array:
  ```python
  ["Answer text [1]...", null, ["uuid-1", "uuid-2", timestamp], null, [type_info]]
  ```
- Citation `[N]` maps to Nth UUID in the array (1-indexed)

### Open WebUI Citation Format (from source code analysis)
- Expects `sources` array in response OR `{"type": "source", ...}` SSE events
- Format: `{"source": {"name": "Title", "id": "uuid"}, "document": []}`
- Maps `[1]` → `sources[0]`, `[2]` → `sources[1]`

---

## Task 1: Extract Source IDs in Core Client

**Files:**
- Modify: `src/nlm_proxy/core/client.py:1680-1730` (`_parse_stream_chunk` method)
- Test: `tests/test_citation_extraction.py` (create)

**Step 1: Write the failing test**

Create `tests/test_citation_extraction.py`:

```python
"""Tests for citation extraction from NotebookLM streaming chunks."""

import pytest
from nlm_proxy.core.client import NotebookLMClient


class TestParseStreamChunk:
    """Test _parse_stream_chunk method extracts source IDs."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return NotebookLMClient(cookies="test", csrf_token="test", session_id="test")

    def test_extracts_source_ids_from_answer_chunk(self, client):
        """Should extract source UUIDs from position 2 of content array."""
        # Real chunk format from NotebookLM API
        chunk_json = '''[[
            "wrb.fr",
            null,
            "[[\\\"Answer text with [1] and [2] citations.\\\",null,[\\\\"d458c47d-6b1e-463e-9cf4-47d716230f0a\\\\",\\\\"689bd968-0864-4019-92f8-ce61db5852b0\\\\",3975011549],null,[null,null,null,null,1]]]"
        ]]'''

        result = client._parse_stream_chunk(chunk_json)

        assert result is not None
        assert result["type"] == "answer"
        assert result["text"] == "Answer text with [1] and [2] citations."
        assert result["source_ids"] == [
            "d458c47d-6b1e-463e-9cf4-47d716230f0a",
            "689bd968-0864-4019-92f8-ce61db5852b0",
        ]

    def test_extracts_source_ids_from_thinking_chunk(self, client):
        """Should extract source UUIDs from thinking chunks too."""
        chunk_json = '''[[
            "wrb.fr",
            null,
            "[[\\\"**Analyzing** the question...\\\",null,[\\\\"abc12345-1234-5678-90ab-cdef01234567\\\\",9876543210],null,[null,null,null,null,2]]]"
        ]]'''

        result = client._parse_stream_chunk(chunk_json)

        assert result is not None
        assert result["type"] == "thinking"
        assert result["source_ids"] == ["abc12345-1234-5678-90ab-cdef01234567"]

    def test_returns_empty_source_ids_when_none_present(self, client):
        """Should return empty list when no sources in chunk."""
        chunk_json = '''[[
            "wrb.fr",
            null,
            "[[\\\"Some text without citations.\\\",null,null,null,[null,null,null,null,1]]]"
        ]]'''

        result = client._parse_stream_chunk(chunk_json)

        assert result is not None
        assert result["source_ids"] == []

    def test_filters_out_timestamp_from_source_ids(self, client):
        """Should only include UUID strings, not the trailing timestamp."""
        chunk_json = '''[[
            "wrb.fr",
            null,
            "[[\\\"Text [1].\\\",null,[\\\\"uuid-string\\\\",3975011549],null,[null,null,null,null,1]]]"
        ]]'''

        result = client._parse_stream_chunk(chunk_json)

        assert result is not None
        # Should only include the UUID string, not the integer timestamp
        assert result["source_ids"] == ["uuid-string"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_citation_extraction.py -v`
Expected: FAIL with `KeyError: 'source_ids'` or similar

**Step 3: Implement source ID extraction**

Modify `src/nlm_proxy/core/client.py` - update `_parse_stream_chunk` method (around line 1680):

```python
def _parse_stream_chunk(self, json_str: str) -> dict | None:
    """Parse a single streaming chunk and extract text, type, and source IDs.

    Args:
        json_str: A single JSON line from the streaming response

    Returns:
        Dict with type ("thinking" or "answer"), text, raw_type, and source_ids,
        or None if parsing fails
    """
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list) or len(parsed) == 0:
        return None

    for item in parsed:
        if not isinstance(item, list) or len(item) < 3:
            continue
        if item[0] != "wrb.fr":
            continue

        inner_json_str = item[2]
        if not isinstance(inner_json_str, str):
            continue

        try:
            inner_data = json.loads(inner_json_str)
        except json.JSONDecodeError:
            continue

        if isinstance(inner_data, list) and len(inner_data) > 0:
            first_elem = inner_data[0]
            if isinstance(first_elem, list) and len(first_elem) > 0:
                text = first_elem[0]
                if isinstance(text, str) and len(text) > 10:
                    # Extract source IDs from position 2
                    source_ids = []
                    if len(first_elem) > 2 and isinstance(first_elem[2], list):
                        for source_item in first_elem[2]:
                            # Only include UUID strings, skip integer timestamps
                            if isinstance(source_item, str):
                                source_ids.append(source_item)

                    # Extract type indicator
                    raw_type = 2  # Default to thinking
                    if len(first_elem) > 4 and isinstance(first_elem[4], list):
                        type_info = first_elem[4]
                        if len(type_info) > 0 and isinstance(type_info[-1], int):
                            raw_type = type_info[-1]

                    return {
                        "type": "answer" if raw_type == 1 else "thinking",
                        "text": text,
                        "raw_type": raw_type,
                        "source_ids": source_ids,
                    }

    return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_citation_extraction.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add tests/test_citation_extraction.py src/nlm_proxy/core/client.py
git commit -m "feat(core): extract source IDs from streaming chunks for citation support"
```

---

## Task 2: Add Source Metadata Extraction Helper

**Files:**
- Modify: `src/nlm_proxy/openai/server.py` (add helper function)
- Test: `tests/test_source_metadata.py` (create)

**Step 1: Write the failing test**

Create `tests/test_source_metadata.py`:

```python
"""Tests for source metadata extraction from notebook data."""

import pytest


class TestExtractSourceMetadata:
    """Test _extract_source_metadata helper function."""

    def test_extracts_source_titles_and_ids(self):
        """Should extract source ID -> metadata mapping."""
        from nlm_proxy.openai.server import _extract_source_metadata

        notebook_data = {
            "sources": [
                {"id": "uuid-1", "title": "NetNam Company Profile", "type": "pdf"},
                {"id": "uuid-2", "title": "Vietnam ISP History", "type": "web_page", "url": "https://example.com"},
            ]
        }

        result = _extract_source_metadata(notebook_data)

        assert result == {
            "uuid-1": {"title": "NetNam Company Profile", "type": "pdf", "url": None},
            "uuid-2": {"title": "Vietnam ISP History", "type": "web_page", "url": "https://example.com"},
        }

    def test_handles_empty_sources(self):
        """Should return empty dict when no sources."""
        from nlm_proxy.openai.server import _extract_source_metadata

        result = _extract_source_metadata({"sources": []})
        assert result == {}

        result = _extract_source_metadata({})
        assert result == {}

    def test_handles_missing_fields(self):
        """Should use defaults for missing fields."""
        from nlm_proxy.openai.server import _extract_source_metadata

        notebook_data = {
            "sources": [
                {"id": "uuid-1"},  # Only ID, no title/type
            ]
        }

        result = _extract_source_metadata(notebook_data)

        assert result == {
            "uuid-1": {"title": "Unknown Source", "type": None, "url": None},
        }
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_source_metadata.py -v`
Expected: FAIL with `ImportError: cannot import name '_extract_source_metadata'`

**Step 3: Implement the helper function**

Add to `src/nlm_proxy/openai/server.py` after the imports (around line 30):

```python
def _extract_source_metadata(notebook_data: dict) -> dict[str, dict]:
    """Extract source ID -> metadata mapping from notebook data.

    Args:
        notebook_data: The notebook data from get_notebook() API call

    Returns:
        Dict mapping source_id to {"title": str, "type": str, "url": str|None}
    """
    sources = {}
    source_list = notebook_data.get("sources", [])

    for source in source_list:
        source_id = source.get("id")
        if source_id:
            sources[source_id] = {
                "title": source.get("title", "Unknown Source"),
                "type": source.get("type"),
                "url": source.get("url"),
            }

    return sources
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_source_metadata.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add tests/test_source_metadata.py src/nlm_proxy/openai/server.py
git commit -m "feat(openai): add helper to extract source metadata from notebook"
```

---

## Task 3: Add Citation Events to Streaming Response

**Files:**
- Modify: `src/nlm_proxy/openai/server.py:332-411` (`stream_response` function)

**Step 1: Read current implementation**

Read: `src/nlm_proxy/openai/server.py` lines 332-411 to understand current structure.

**Step 2: Modify stream_response to collect and emit citations**

Update the `stream_response` function in `src/nlm_proxy/openai/server.py`:

```python
async def stream_response(client, notebook_id: str, query_text: str, request: ChatCompletionRequest, chat_id: str = None):
    """Generate OpenAI-compatible SSE stream from NotebookLM query_stream."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    conversation_id = None
    chunk_count = 0

    # Track previous text to compute deltas (NotebookLM sends cumulative text)
    previous_thinking = ""
    previous_answer = ""

    # NEW: Track source IDs for citation emission
    collected_source_ids: list[str] = []  # Ordered, unique source IDs
    notebook_sources: dict[str, dict] = {}  # source_id -> metadata

    # NEW: Pre-fetch notebook source metadata for citation resolution
    try:
        notebook_data = await client.get_notebook(notebook_id)
        notebook_sources = _extract_source_metadata(notebook_data)
        logger.debug(f"[PROXY] Loaded {len(notebook_sources)} source metadata entries for citations")
    except Exception as e:
        logger.warning(f"[PROXY] Could not fetch notebook sources for citations: {e}")

    logger.debug(f"[NOTEBOOKLM] Starting stream query: notebook_id={notebook_id}, query={query_text[:100]}..., conversation_id={request.conversation_id}, chat_id={chat_id}")

    try:
        async for chunk in client.query_stream(
            notebook_id=notebook_id,
            query_text=query_text,
            conversation_id=request.conversation_id
        ):
            chunk_count += 1
            chunk_type = chunk.get("type")
            full_text = chunk.get("text", "")

            # NEW: Collect source IDs as they appear (maintain order)
            for sid in chunk.get("source_ids", []):
                if sid not in collected_source_ids:
                    collected_source_ids.append(sid)

            logger.debug(f"[NOTEBOOKLM] Received chunk #{chunk_count}: type={chunk_type}, text_len={len(full_text)}")

            # Filter thinking unless requested
            if chunk_type == "thinking" and not request.include_thinking:
                logger.debug(f"[PROXY] Filtering thinking chunk (include_thinking={request.include_thinking})")
                previous_thinking = full_text  # Still track it for delta computation
                continue

            new_conv_id = chunk.get("conversation_id")
            if new_conv_id and not conversation_id:
                conversation_id = new_conv_id
                # Save to session store if we have a chat_id
                if chat_id and app.state.session_store:
                    app.state.session_store.set(chat_id, conversation_id)

            # Compute delta: NotebookLM sends cumulative text, we need only the new part
            if chunk_type == "thinking":
                delta_text = full_text[len(previous_thinking):]
                previous_thinking = full_text
            else:  # answer
                delta_text = full_text[len(previous_answer):]
                previous_answer = full_text

            logger.debug(f"[PROXY] Delta text length: {len(delta_text)} chars (full={len(full_text)}, previous={len(previous_answer if chunk_type == 'answer' else previous_thinking)})")

            # Only yield if there's new content
            if delta_text:
                # Send thinking as reasoning_content, answers as content (OpenAI o1/o3 format)
                if chunk_type == "thinking":
                    delta = DeltaContent(reasoning_content=delta_text)
                else:  # answer
                    delta = DeltaContent(content=delta_text)

                openai_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=notebook_id,
                    choices=[Choice(delta=delta)],
                    system_fingerprint=f"conv_{conversation_id}" if conversation_id else None
                )
                logger.debug(f"[PROXY] Yielding OpenAI chunk with {len(delta_text)} chars (type={chunk_type})")
                yield f"data: {openai_chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        logger.debug(f"[NOTEBOOKLM] Stream complete: {chunk_count} total chunks, conversation_id={conversation_id}")
        final_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=notebook_id,
            choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"

        # NEW: Emit citation events for Open WebUI (after content, before [DONE])
        if collected_source_ids:
            logger.debug(f"[PROXY] Emitting {len(collected_source_ids)} citation events")
            for source_id in collected_source_ids:
                source_meta = notebook_sources.get(source_id, {})
                citation_event = {
                    "type": "source",
                    "source": {
                        "name": source_meta.get("title", "Unknown Source"),
                        "id": source_id,
                    },
                    "document": [],  # NotebookLM doesn't provide passage-level refs
                }
                # Add URL if available (for web sources)
                if source_meta.get("url"):
                    citation_event["source"]["url"] = source_meta["url"]

                yield f"data: {json.dumps(citation_event)}\n\n"

        yield "data: [DONE]\n\n"
        logger.debug("[PROXY] Stream finished")
    finally:
        await client.close()
```

**Step 3: Run existing tests to ensure no regressions**

Run: `uv run pytest tests/ -v -k "not integration"`
Expected: All existing tests PASS

**Step 4: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(openai): emit citation events in streaming response for Open WebUI"
```

---

## Task 4: Add Citations to Non-Streaming Response

**Files:**
- Modify: `src/nlm_proxy/openai/server.py:471-512` (non-streaming path in `chat_completions`)
- Modify: `src/nlm_proxy/core/client.py` (`query` method to return source_ids)

**Step 1: Update core client query() to return source_ids**

First, we need to check how `query()` is implemented and update it to return source_ids.

Read `src/nlm_proxy/core/client.py` to find the `query()` method and understand how it collects the response.

The `query()` method likely calls `query_stream()` internally and collects results. We need to ensure it also collects and returns `source_ids`.

Add source_ids collection to the `query()` method:

```python
# In the query() method, after collecting answer text:
# Collect source_ids from all chunks
all_source_ids = []
async for chunk in self.query_stream(...):
    for sid in chunk.get("source_ids", []):
        if sid not in all_source_ids:
            all_source_ids.append(sid)
    # ... existing logic ...

return {
    "answer": combined_answer,
    "conversation_id": conversation_id,
    "source_ids": all_source_ids,  # NEW
}
```

**Step 2: Update non-streaming response in server.py**

Modify the non-streaming path in `chat_completions` (around line 471):

```python
# Non-streaming path
logger.debug("[PROXY] Using non-streaming response")
try:
    # NEW: Fetch notebook metadata for citation resolution
    notebook_data = await client.get_notebook(request.model)
    notebook_sources = _extract_source_metadata(notebook_data)

    logger.debug(f"[NOTEBOOKLM] Calling query: notebook_id={request.model}, query={query_text[:100]}..., conversation_id={request.conversation_id}")
    result = await client.query(
        notebook_id=request.model,
        query_text=query_text,
        conversation_id=request.conversation_id,
    )

    answer = result.get("answer", "") if result else ""
    conv_id = result.get("conversation_id", "") if result else ""
    source_ids = result.get("source_ids", []) if result else []

    # Save conversation_id to session store if we have a chat_id
    if chat_id and conv_id and app.state.session_store:
        app.state.session_store.set(chat_id, conv_id)

    logger.debug(f"[NOTEBOOKLM] Response received: answer_len={len(answer)}, conversation_id={conv_id}, sources={len(source_ids)}")
    logger.debug(f"[NOTEBOOKLM] Answer preview: {answer[:200]}{'...' if len(answer) > 200 else ''}")

    # Handle empty responses gracefully
    if not answer or not answer.strip():
        logger.warning(f"[NOTEBOOKLM] Empty answer received for query: {query_text[:100]}...")
        answer = "I apologize, but I couldn't generate a response for that query. This might happen when the question doesn't relate to the notebook content or uses unsupported formatting. Please try rephrasing your question."

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[ResponseChoice(
            index=0,
            message=ResponseMessage(role="assistant", content=answer),
            finish_reason="stop"
        )],
        usage=Usage(prompt_tokens=len(query_text), completion_tokens=len(answer), total_tokens=len(query_text) + len(answer)),
        system_fingerprint=f"conv_{conv_id}" if conv_id else None
    )

    # NEW: Build sources array for Open WebUI
    sources = []
    for source_id in source_ids:
        source_meta = notebook_sources.get(source_id, {})
        source_entry = {
            "source": {
                "name": source_meta.get("title", "Unknown Source"),
                "id": source_id,
            },
            "document": [],
        }
        if source_meta.get("url"):
            source_entry["source"]["url"] = source_meta["url"]
        sources.append(source_entry)

    # Return response with sources for Open WebUI
    response_dict = response.model_dump()
    if sources:
        response_dict["sources"] = sources
        logger.debug(f"[PROXY] Added {len(sources)} sources to response")

    logger.debug(f"[PROXY] Returning response with {len(answer)} characters")
    return response_dict
finally:
    await client.close()
```

**Step 3: Run tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/nlm_proxy/core/client.py src/nlm_proxy/openai/server.py
git commit -m "feat(openai): add citations to non-streaming response"
```

---

## Task 5: Update Smart Router to Support Citations

**Files:**
- Modify: `src/nlm_proxy/openai/server.py:135-230` (`stream_smart_response` function)
- Modify: `src/nlm_proxy/openai/server.py:233-329` (`handle_smart_routing` function)

**Step 1: Update stream_smart_response for citations**

Apply similar changes to `stream_smart_response` function - collect source_ids during NotebookLM streaming and emit citation events at the end.

**Step 2: Update handle_smart_routing non-streaming path**

Apply similar changes to the non-streaming path in `handle_smart_routing` - fetch notebook metadata and add sources to response.

**Step 3: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(openai): add citation support to smart router"
```

---

## Task 6: Integration Testing

**Files:**
- Test: `tests/test_openai_citations_integration.py` (create)

**Step 1: Write integration test**

Create `tests/test_openai_citations_integration.py`:

```python
"""Integration tests for OpenAI proxy citation support."""

import pytest
import json


class TestStreamingCitations:
    """Test citation events in streaming responses."""

    @pytest.mark.asyncio
    async def test_citation_events_emitted_after_content(self):
        """Citation events should be emitted after content chunks, before [DONE]."""
        # This test requires a running server or mocked client
        # For now, document the expected behavior

        # Expected SSE stream order:
        # 1. Content chunks: data: {"choices": [{"delta": {"content": "..."}}]}
        # 2. Citation events: data: {"type": "source", "source": {"name": "...", "id": "..."}}
        # 3. Final: data: [DONE]
        pass


class TestNonStreamingCitations:
    """Test sources field in non-streaming responses."""

    def test_sources_included_in_response(self):
        """Response should include sources array when citations present."""
        # Expected response structure:
        # {
        #     "id": "chatcmpl-xxx",
        #     "choices": [...],
        #     "sources": [
        #         {"source": {"name": "Title", "id": "uuid"}, "document": []}
        #     ]
        # }
        pass
```

**Step 2: Manual testing with Open WebUI**

1. Start the proxy: `nlm-proxy serve openai --port 8080 --debug`
2. Configure Open WebUI to use `http://localhost:8080/v1` as the API endpoint
3. Send a query to a notebook with sources
4. Verify:
   - Answer contains `[1]`, `[2]` citations inline
   - "Sources" button appears below the message
   - Clicking a source shows the source title
   - Citation badges `[1]` in text are interactive

**Step 3: Commit**

```bash
git add tests/test_openai_citations_integration.py
git commit -m "test: add integration tests for citation support"
```

---

## Verification Checklist

### Automated Tests
- [ ] `uv run pytest tests/test_citation_extraction.py -v` - All pass
- [ ] `uv run pytest tests/test_source_metadata.py -v` - All pass
- [ ] `uv run pytest tests/ -v` - No regressions

### Manual Testing
- [ ] Start proxy: `nlm-proxy serve openai --port 8080 --debug`
- [ ] Configure Open WebUI with proxy endpoint
- [ ] Send query that produces citations
- [ ] Verify streaming: Citation events appear in debug logs
- [ ] Verify UI: "Sources" section appears in Open WebUI
- [ ] Verify clickable: Citation numbers `[1]` are interactive
- [ ] Test non-streaming mode (if Open WebUI supports toggling)

### Debug Commands
```bash
# Watch streaming events
curl -N -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"NOTEBOOK_ID","messages":[{"role":"user","content":"What is NetNam?"}],"stream":true}' \
  http://localhost:8080/v1/chat/completions

# Check non-streaming response
curl -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"NOTEBOOK_ID","messages":[{"role":"user","content":"What is NetNam?"}],"stream":false}' \
  http://localhost:8080/v1/chat/completions | jq '.sources'
```

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `src/nlm_proxy/core/client.py` | Extract `source_ids` in `_parse_stream_chunk()`, return in `query()` |
| `src/nlm_proxy/openai/server.py` | Add `_extract_source_metadata()`, emit citation SSE events, add `sources` to response |
| `tests/test_citation_extraction.py` | New: Unit tests for source ID extraction |
| `tests/test_source_metadata.py` | New: Unit tests for metadata extraction |
| `tests/test_openai_citations_integration.py` | New: Integration test documentation |
