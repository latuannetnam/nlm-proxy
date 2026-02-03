# NotebookLM Citation Analysis

This document describes the citation and source reference structure in NotebookLM's query responses, based on analysis of real streaming API responses. This information is crucial for implementing citation support in the OpenAI proxy.

---

## Executive Summary

NotebookLM uses a **positional citation system** where:
1. Citation markers (`[1]`, `[2]`, etc.) are embedded **inline** in the markdown answer text
2. Source UUIDs are provided in a **metadata array** (position 2 of content array)
3. Citation number `[N]` maps to the **Nth source UUID** in the metadata array
4. **No detailed passage references** are included in the streaming response
5. Client-side resolution handles showing specific passages in the UI

---

## Response Structure

### Streaming Endpoint

```
POST /_/LabsTailwindUi/data/google.internal.labs.tailwind.orchestration.v1.LabsTailwindOrchestrationService/GenerateFreeFormStreamed
```

### Chunk Format

Each streaming chunk follows this structure:

```json
[
  [
    "wrb.fr",
    null,
    "<nested_json_string>",
    ...
  ]
]
```

The nested JSON string contains:

```json
[
  [
    "Text content with inline [1], [2] citations",  // Position 0: Text
    null,                                            // Position 1: Unknown
    [                                                // Position 2: Source metadata
      "d458c47d-6b1e-463e-9cf4-47d716230f0a",       //   Source 1 UUID
      "689bd968-0864-4019-92f8-ce61db5852b0",       //   Source 2 UUID
      3975011549                                     //   Timestamp/hash
    ],
    null,                                            // Position 3: Unknown
    [                                                // Position 4: Type info
      ...,
      1  // or 2 (1=answer, 2=thinking)
    ]
  ]
]
```

---

## Citation System Deep Dive

### 1. Inline Citation Markers

Citations appear directly in the markdown text:

```markdown
NetNam là doanh nghiệp tiên phong cung cấp dịch vụ Internet (ISP)
tại Việt Nam từ năm 1994 [1], chuyên cung cấp các dịch vụ công nghệ
cao và giải pháp mạng trong thị trường ngách với chiến lược đặt khách
hàng làm trọng tâm [2].
```

### 2. Source ID Mapping

The metadata array at position 2 contains:

```python
[
  "d458c47d-6b1e-463e-9cf4-47d716230f0a",  # [1] maps to this source
  "689bd968-0864-4019-92f8-ce61db5852b0",  # [2] maps to this source
  3975011549                               # Metadata (timestamp/hash)
]
```

**Mapping Rule**: Citation `[N]` refers to the **Nth UUID** in the array (1-indexed).

### 3. Source Metadata Structure

The third element in the array appears to be:
- **Timestamp** (Unix epoch in milliseconds)
- **Hash** (for cache invalidation)
- **Version identifier**

**Example**: `3975011549` could represent:
- Timestamp: `3975011549` ms = ~66.25 minutes = timestamp from a base point
- Or a hash/checksum of the sources used

---

## Complete Response Structure

### Sample Parsed Chunk (Answer with Citations)

```json
{
  "type": "answer",
  "text": "NetNam là doanh nghiệp tiên phong... [1], [2]",
  "source_ids_array": [
    "d458c47d-6b1e-463e-9cf4-47d716230f0a",
    "689bd968-0864-4019-92f8-ce61db5852b0",
    3975011549
  ],
  "referenced_source_ids": [
    "d458c47d-6b1e-463e-9cf4-47d716230f0a",
    "689bd968-0864-4019-92f8-ce61db5852b0"
  ],
  "source_count": 2,
  "type_info": [..., 1],
  "_array_length": 5
}
```

### Sample Parsed Chunk (Thinking Step)

```json
{
  "type": "thinking",
  "text": "**Initiating the Analysis** I've initiated the analysis...",
  "source_ids_array": [
    "d458c47d-6b1e-463e-9cf4-47d716230f0a",
    "689bd968-0864-4019-92f8-ce61db5852b0",
    3975011549
  ],
  "referenced_source_ids": [
    "d458c47d-6b1e-463e-9cf4-47d716230f0a",
    "689bd968-0864-4019-92f8-ce61db5852b0"
  ],
  "source_count": 2,
  "type_info": [..., 2],
  "_array_length": 5
}
```

---

## Complete Response Example

### Assembled Response

```json
{
  "metadata": {
    "conversation_id": "f8a7c6d4-1234-5678-90ab-cdef01234567",
    "notebook_id": "abc123-def456",
    "query": "NetNam là gì?",
    "source_ids": [
      "d458c47d-6b1e-463e-9cf4-47d716230f0a",
      "689bd968-0864-4019-92f8-ce61db5852b0"
    ],
    "timestamp": 1738569840.123,
    "duration_seconds": 3.45,
    "chunk_count": 11
  },
  "thinking": [
    {
      "text": "**Initiating the Analysis** I've initiated the analysis...",
      "referenced_source_ids": ["d458c47d-...", "689bd968-..."],
      "source_count": 2
    },
    {
      "text": "**Pinpointing NetNam's Identity** I've homed in on NetNam...",
      "referenced_source_ids": ["d458c47d-...", "689bd968-..."],
      "source_count": 2
    }
  ],
  "answer": [
    {
      "text": "NetNam là doanh nghiệp tiên phong... [1], [2]",
      "referenced_source_ids": ["d458c47d-...", "689bd968-..."],
      "source_count": 2
    }
  ],
  "complete_answer_text": "NetNam là doanh nghiệp tiên phong cung cấp dịch vụ Internet (ISP) tại Việt Nam từ năm 1994 [1], chuyên cung cấp các dịch vụ công nghệ cao và giải pháp mạng trong thị trường ngách với chiến lược đặt khách hàng làm trọng tâm [2]."
}
```

---

## OpenAI Proxy Integration Strategies

### Strategy 1: Inline Citations (Recommended)

**Keep citations in the text** and add metadata to track sources.

**OpenAI Chat Completion Format**:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "model": "notebooklm",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "NetNam là doanh nghiệp tiên phong... [1], [2]"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  },
  "metadata": {
    "sources": [
      {
        "index": 1,
        "source_id": "d458c47d-6b1e-463e-9cf4-47d716230f0a",
        "title": "NetNam Company Profile"
      },
      {
        "index": 2,
        "source_id": "689bd968-0864-4019-92f8-ce61db5852b0",
        "title": "Vietnam ISP History"
      }
    ]
  }
}
```

**Pros**:
- Preserves citation context
- Standard OpenAI format compatibility
- Easy to implement

**Cons**:
- Non-standard `metadata` field
- Requires custom client parsing for rich citation UI

### Strategy 2: Function Call Format

**Use OpenAI's function/tool calling** to represent citations.

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "NetNam là doanh nghiệp tiên phong...",
        "tool_calls": [
          {
            "id": "cite_1",
            "type": "function",
            "function": {
              "name": "cite_source",
              "arguments": "{\"source_id\":\"d458c47d-6b1e-463e-9cf4-47d716230f0a\",\"citation_number\":1}"
            }
          },
          {
            "id": "cite_2",
            "type": "function",
            "function": {
              "name": "cite_source",
              "arguments": "{\"source_id\":\"689bd968-0864-4019-92f8-ce61db5852b0\",\"citation_number\":2}"
            }
          }
        ]
      }
    }
  ]
}
```

**Pros**:
- Uses standard OpenAI structures
- Clear separation between content and citations
- Supports rich client-side processing

**Cons**:
- More complex response structure
- Requires stripping `[N]` markers or mapping them
- Non-standard use of tool_calls

### Strategy 3: Streaming with SSE Metadata

**Send citation data as separate SSE events** during streaming.

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"NetNam là doanh nghiệp"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" [1]"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"citation.reference","citation_number":1,"source_id":"d458c47d-6b1e-463e-9cf4-47d716230f0a"}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":", [2]"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"citation.reference","citation_number":2,"source_id":"689bd968-0864-4019-92f8-ce61db5852b0"}

data: [DONE]
```

**Pros**:
- Real-time citation tracking during streaming
- Flexible for rich UIs
- Preserves streaming performance

**Cons**:
- Custom event types (non-standard)
- Requires custom SSE parser
- More complex implementation

### Strategy 4: Annotations (OpenAI Assistants API Style)

**Use the Assistants API annotation format** for citations.

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": {
              "value": "NetNam là doanh nghiệp tiên phong cung cấp dịch vụ Internet (ISP) tại Việt Nam từ năm 1994 [1], chuyên cung cấp các dịch vụ công nghệ cao và giải pháp mạng trong thị trường ngách với chiến lược đặt khách hàng làm trọng tâm [2].",
              "annotations": [
                {
                  "type": "citation",
                  "text": "[1]",
                  "start_index": 87,
                  "end_index": 90,
                  "source": {
                    "source_id": "d458c47d-6b1e-463e-9cf4-47d716230f0a",
                    "title": "NetNam Company Profile"
                  }
                },
                {
                  "type": "citation",
                  "text": "[2]",
                  "start_index": 198,
                  "end_index": 201,
                  "source": {
                    "source_id": "689bd968-0864-4019-92f8-ce61db5852b0",
                    "title": "Vietnam ISP History"
                  }
                }
              ]
            }
          }
        ]
      }
    }
  ]
}
```

**Pros**:
- Most semantically correct
- Aligns with OpenAI Assistants API
- Supports precise citation location tracking

**Cons**:
- Requires parsing text to find citation positions
- Heavier response payload
- More complex implementation

---

## Recommended Implementation

### Phase 1: Basic Citations (Inline)
**Strategy 1** - Keep `[N]` in text, add `metadata.sources`

```python
def convert_to_openai_format(notebooklm_response):
    """Convert NotebookLM response to OpenAI format with citation metadata."""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "model": "notebooklm",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": notebooklm_response["complete_answer_text"]
            },
            "finish_reason": "stop"
        }],
        "metadata": {
            "sources": extract_sources(notebooklm_response),
            "thinking_steps": len(notebooklm_response["thinking"])
        }
    }
```

### Phase 2: Rich Citations (Annotations)
**Strategy 4** - Parse `[N]` markers and create annotations

```python
def create_annotations(text, source_ids_map):
    """Create OpenAI-style annotations from citation markers."""
    annotations = []
    pattern = r'\[(\d+)\]'

    for match in re.finditer(pattern, text):
        citation_num = int(match.group(1))
        source_id = source_ids_map.get(citation_num)

        if source_id:
            annotations.append({
                "type": "citation",
                "text": match.group(0),
                "start_index": match.start(),
                "end_index": match.end(),
                "source": {
                    "source_id": source_id,
                    # Fetch source details from notebook metadata
                }
            })

    return annotations
```

### Phase 3: Streaming Citations
**Strategy 3** - Send citations as SSE events

```python
async def stream_with_citations(notebooklm_chunks):
    """Stream response with citation metadata."""
    citation_buffer = {}

    async for chunk in notebooklm_chunks:
        # Emit text chunk
        yield create_sse_event({
            "delta": {"content": chunk["text"]},
            "finish_reason": None
        })

        # Emit citation references
        if chunk.get("referenced_source_ids"):
            for idx, source_id in enumerate(chunk["referenced_source_ids"], 1):
                if source_id not in citation_buffer:
                    citation_buffer[source_id] = idx
                    yield create_sse_event({
                        "object": "citation.reference",
                        "citation_number": idx,
                        "source_id": source_id
                    })
```

---

## Source Resolution

### Mapping Source IDs to Titles

To provide meaningful citation information, you need to resolve source IDs to titles/metadata:

```python
async def resolve_sources(notebook_id, source_ids):
    """Resolve source IDs to full metadata."""
    # Fetch notebook details
    notebook = await client.get_notebook(notebook_id)

    # Extract source info
    sources = []
    for source in notebook["sources"]:
        if source["id"] in source_ids:
            sources.append({
                "id": source["id"],
                "title": source["title"],
                "type": source["type"],  # pdf, web_page, youtube, etc.
                "url": source.get("url"),
            })

    return sources
```

### Citation Index Building

```python
def build_citation_index(answer_chunks):
    """Build a unified citation index from all chunks."""
    citation_map = {}
    citation_counter = 1

    for chunk in answer_chunks:
        for source_id in chunk.get("referenced_source_ids", []):
            if source_id not in citation_map:
                citation_map[source_id] = citation_counter
                citation_counter += 1

    return citation_map
```

---

## Testing Citation Parsing

### Test Cases

```python
import re

def test_citation_extraction():
    text = "NetNam was founded in 1994 [1], specializing in ISP services [2], [3]."

    # Extract all citations
    citations = re.findall(r'\[(\d+)\]', text)
    assert citations == ['1', '2', '3']

    # Extract with positions
    for match in re.finditer(r'\[(\d+)\]', text):
        print(f"Citation [{match.group(1)}] at position {match.start()}-{match.end()}")

def test_source_mapping():
    source_ids = [
        "d458c47d-6b1e-463e-9cf4-47d716230f0a",
        "689bd968-0864-4019-92f8-ce61db5852b0",
        "abc12345-1234-5678-90ab-cdef01234567"
    ]

    # Build mapping: [N] -> source_id
    citation_map = {i+1: sid for i, sid in enumerate(source_ids)}

    assert citation_map[1] == "d458c47d-6b1e-463e-9cf4-47d716230f0a"
    assert citation_map[2] == "689bd968-0864-4019-92f8-ce61db5852b0"
    assert citation_map[3] == "abc12345-1234-5678-90ab-cdef01234567"
```

---

## API Reference Updates

Add to `docs/API_REFERENCE.md`:

### Query Response Structure

```python
# Response chunks contain citations in position 2
content_array = [
    "Answer text with [1], [2] citations",  # Position 0: Text
    null,                                     # Position 1: Unknown
    [                                         # Position 2: Source metadata
        "source_id_1",  # Maps to [1]
        "source_id_2",  # Maps to [2]
        timestamp       # Metadata timestamp/hash
    ],
    null,             # Position 3: Unknown
    [type_info]       # Position 4: Type (1=answer, 2=thinking)
]
```

### Citation Mapping Rule

**Citation `[N]` maps to the Nth source UUID in position 2.**

Example:
- Text: `"This is a fact [1] and another [2]."`
- Sources: `["uuid-abc", "uuid-def", 1234567890]`
- Mapping: `[1]` → `"uuid-abc"`, `[2]` → `"uuid-def"`

---

## Future Enhancements

### 1. Passage-Level Citations

Currently, citations only reference source documents. Future enhancement could:
- Extract the exact passage being cited
- Search source content for matching text
- Return passage snippets with citations

### 2. Citation Confidence Scores

Track which sources are most relevant:
```json
{
  "citations": [
    {
      "number": 1,
      "source_id": "uuid-1",
      "confidence": 0.95,
      "relevance_score": 0.88
    }
  ]
}
```

### 3. Interactive Citation Resolution

Enable clients to request citation details on-demand:
```
GET /v1/notebooks/{notebook_id}/citations/{citation_number}
```

---

## References

- **Test Script**: `tests/test_query_raw_response.py` - Captures raw responses
- **Streaming Test**: `tests/test_streaming_query.py` - Real-time citation monitoring
- **API Reference**: `docs/API_REFERENCE.md` - Complete API documentation
- **Sample Response**: `response.json` - Real NotebookLM response with citations

---

**Document Version**: 1.0
**Last Updated**: 2026-02-03
**Analysis Based On**: Real NotebookLM API responses (Vietnamese query about NetNam ISP)
