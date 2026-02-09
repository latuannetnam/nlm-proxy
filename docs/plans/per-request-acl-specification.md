# Per-Request ACL Support for Smart Router

> **Type**: Feature Enhancement
> **Status**: Ready for Implementation
> **Priority**: Required for Chatbot Integration
> **Estimated Effort**: ~2 hours
> **Files Changed**: 3 files, ~25 lines added

---

## Summary

Add support for per-request notebook filtering in the Smart Router, enabling external applications (like the NotebookLM Chatbot) to pass a list of allowed notebook IDs based on user permissions (ACL).

## Motivation

The current Smart Router filters notebooks at two levels:
1. **Server startup** (`NLM_PROXY_ROUTING_ALLOWED_NOTEBOOKS` env var) - applies to ALL requests
2. **No per-request filtering** - every user sees the same notebooks

For the NotebookLM Chatbot integration, we need per-request filtering where:
- Each user has different notebook access based on their AD group membership
- The chatbot's ACL service determines which notebooks each user can query
- nlm-proxy filters notebook selection to only those the user can access

## Design

### API Contract

**Request:**
```json
POST /v1/chat/completions
{
  "model": "knowledge-finder",
  "messages": [{"role": "user", "content": "How do I deploy?"}],
  "stream": true,
  "metadata": {
    "allowed_notebooks": ["notebook-id-1", "notebook-id-2"]
  }
}
```

**Behavior:**
- If `metadata.allowed_notebooks` is provided, Smart Router only considers those notebooks
- If `metadata.allowed_notebooks` is `null` or not provided, all cached notebooks are considered (existing behavior)
- If `metadata.allowed_notebooks` is an empty list `[]`, returns "No accessible notebooks" error
- Filtering happens AFTER cache lookup, so all notebooks remain cached for other users

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Request with ACL                              │
│  metadata.allowed_notebooks = ["tech-docs", "public-kb"]        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 handle_smart_routing()                           │
│                                                                  │
│  allowed_notebooks = request.metadata.get("allowed_notebooks")  │
│  decision = await router.route(query, allowed_notebooks)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SmartRouter.route()                              │
│                                                                  │
│  → classify_request(query)                                       │
│  → select_notebook(query, allowed_notebooks)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│             SmartRouter.select_notebook()                        │
│                                                                  │
│  notebooks = cache.get_all()     # All cached: [A, B, C, D]     │
│                                                                  │
│  if allowed_notebooks:           # Filter for this request      │
│      notebooks = [nb for nb in notebooks                        │
│                   if nb.id in allowed_notebooks]                │
│                                  # Result: [A, C] only          │
│                                                                  │
│  → LLM selects best from filtered list                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### File 1: `src/nlm_proxy/openai/router.py`

#### Change 1: Update `select_notebook()` signature and add filtering

```python
# BEFORE (line 100-108)
@record_span("smart_router.select_notebook")
async def select_notebook(self, query: str) -> tuple[str | None, str]:
    """Select best notebook for query. Returns (notebook_id, reasoning)."""
    logger.debug(f"[ROUTER] Selecting notebook for query: {query[:100]}...")
    notebooks = await self._ensure_notebooks_cached()

    if not notebooks:
        logger.warning("[ROUTER] No notebooks available for selection")
        add_span_attributes(candidates_count=0)
        return None, "No notebooks available"
```

```python
# AFTER
@record_span("smart_router.select_notebook")
async def select_notebook(
    self,
    query: str,
    allowed_notebooks: list[str] | None = None,
) -> tuple[str | None, str]:
    """Select best notebook for query. Returns (notebook_id, reasoning).

    Args:
        query: User's question text
        allowed_notebooks: Optional list of notebook IDs to filter candidates.
                          If provided, only notebooks in this list are considered.
                          If None, all cached notebooks are considered.
    """
    logger.debug(f"[ROUTER] Selecting notebook for query: {query[:100]}...")
    notebooks = await self._ensure_notebooks_cached()

    if not notebooks:
        logger.warning("[ROUTER] No notebooks available for selection")
        add_span_attributes(candidates_count=0)
        return None, "No notebooks available"

    # Per-request ACL filtering
    if allowed_notebooks is not None:
        original_count = len(notebooks)
        notebooks = [nb for nb in notebooks if nb.id in allowed_notebooks]
        logger.debug(
            f"[ROUTER] ACL filter: {original_count} → {len(notebooks)} notebooks "
            f"(allowed: {allowed_notebooks})"
        )
        add_span_attributes(
            acl_filter_applied=True,
            acl_allowed_count=len(allowed_notebooks),
            acl_matched_count=len(notebooks),
        )

        if not notebooks:
            logger.warning("[ROUTER] No notebooks match ACL filter")
            return None, "No accessible notebooks for this user"

    add_span_attributes(candidates_count=len(notebooks))
```

#### Change 2: Update `route()` to pass `allowed_notebooks`

```python
# BEFORE (line 210-240)
@record_span("smart_router.route")
async def route(self, query: str) -> RoutingDecision:
    """Classify and route the request."""
    logger.info(f"[ROUTER] Starting routing for query: {query[:50]}...")

    request_type = await self.classify_request(query)

    if request_type == RequestType.LLM_TASK:
        logger.info("[ROUTER] Routing to external LLM")
        add_span_attributes(
            request_type="LLM_TASK",
            notebook_id=None
        )
        return RoutingDecision(
            request_type=RequestType.LLM_TASK,
            reasoning="Classified as LLM task (not a notebook query)"
        )

    notebook_id, reasoning = await self.select_notebook(query)
    # ... rest of method
```

```python
# AFTER
@record_span("smart_router.route")
async def route(
    self,
    query: str,
    allowed_notebooks: list[str] | None = None,
) -> RoutingDecision:
    """Classify and route the request.

    Args:
        query: User's question text
        allowed_notebooks: Optional list of notebook IDs for per-request ACL filtering.
                          Passed through to select_notebook().
    """
    logger.info(f"[ROUTER] Starting routing for query: {query[:50]}...")

    request_type = await self.classify_request(query)

    if request_type == RequestType.LLM_TASK:
        logger.info("[ROUTER] Routing to external LLM")
        add_span_attributes(
            request_type="LLM_TASK",
            notebook_id=None
        )
        return RoutingDecision(
            request_type=RequestType.LLM_TASK,
            reasoning="Classified as LLM task (not a notebook query)"
        )

    notebook_id, reasoning = await self.select_notebook(query, allowed_notebooks)
    # ... rest of method unchanged
```

---

### File 2: `src/nlm_proxy/openai/server.py`

#### Change 3: Extract `allowed_notebooks` from request metadata

```python
# BEFORE (line 257-311, handle_smart_routing function)
async def handle_smart_routing(request: ChatCompletionRequest, http_request: Request):
    """Handle requests to the smart router model."""
    routing_settings = get_routing_settings()
    tracing_settings = get_tracing_settings()
    tracer = get_tracer(__name__)

    client = await get_client()

    # ... existing cache check and router creation ...

    router = SmartRouter(
        nlm_client=client,
        notebook_cache=app.state.notebook_cache,
        llm_base_url=routing_settings.llm_base_url,
        llm_api_key=routing_settings.llm_api_key,
        llm_model=routing_settings.llm_model,
        allowed_notebooks=routing_settings.allowed_notebooks
    )

    # Extract chat_id from headers or request metadata
    chat_id = http_request.headers.get("X-OpenWebUI-Chat-Id")
    if not chat_id and hasattr(request, 'metadata') and request.metadata:
        chat_id = request.metadata.get("chat_id")

    logger.debug(f"[SMART-ROUTER] Extracted chat_id: {chat_id}")

    # ... rest of function ...
```

```python
# AFTER
async def handle_smart_routing(request: ChatCompletionRequest, http_request: Request):
    """Handle requests to the smart router model."""
    routing_settings = get_routing_settings()
    tracing_settings = get_tracing_settings()
    tracer = get_tracer(__name__)

    client = await get_client()

    # ... existing cache check and router creation (unchanged) ...

    router = SmartRouter(
        nlm_client=client,
        notebook_cache=app.state.notebook_cache,
        llm_base_url=routing_settings.llm_base_url,
        llm_api_key=routing_settings.llm_api_key,
        llm_model=routing_settings.llm_model,
        allowed_notebooks=routing_settings.allowed_notebooks
    )

    # Extract chat_id from headers or request metadata
    chat_id = http_request.headers.get("X-OpenWebUI-Chat-Id")
    if not chat_id and hasattr(request, 'metadata') and request.metadata:
        chat_id = request.metadata.get("chat_id")

    logger.debug(f"[SMART-ROUTER] Extracted chat_id: {chat_id}")

    # NEW: Extract per-request ACL filter from metadata
    request_allowed_notebooks: list[str] | None = None
    if hasattr(request, 'metadata') and request.metadata:
        request_allowed_notebooks = request.metadata.get("allowed_notebooks")
        if request_allowed_notebooks:
            logger.info(
                f"[SMART-ROUTER] Per-request ACL: {len(request_allowed_notebooks)} allowed notebooks"
            )

    # ... rest of function, but update router.route() calls ...
```

#### Change 4: Pass `allowed_notebooks` to `router.route()` (streaming path)

```python
# BEFORE (around line 309)
            query = user_messages[-1].content
            decision = await router.route(query)
```

```python
# AFTER
            query = user_messages[-1].content
            decision = await router.route(query, allowed_notebooks=request_allowed_notebooks)
```

#### Change 5: Pass `allowed_notebooks` to `router.route()` (non-streaming path)

```python
# BEFORE (around line 336)
            decision = await router.route(query)
```

```python
# AFTER
            decision = await router.route(query, allowed_notebooks=request_allowed_notebooks)
```

---

### File 3: `src/nlm_proxy/openai/types.py`

#### Change 6: Ensure `metadata` field exists with proper typing (verify only)

```python
# Verify this exists in ChatCompletionRequest class
# If not present, add it:

from typing import Any

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    conversation_id: str | None = None
    include_thinking: bool = False

    # Ensure this field exists for ACL and other metadata
    metadata: dict[str, Any] | None = None
```

---

## Testing

### Unit Tests

```python
# tests/test_router_acl.py

import pytest
from nlm_proxy.openai.router import SmartRouter, RequestType

@pytest.fixture
def mock_notebooks():
    """Create mock NotebookInfo objects."""
    from nlm_proxy.openai.notebook_cache import NotebookInfo
    return [
        NotebookInfo(id="nb-1", title="HR Docs", summary="", topics=[], cached_at=0),
        NotebookInfo(id="nb-2", title="Tech Docs", summary="", topics=[], cached_at=0),
        NotebookInfo(id="nb-3", title="Public KB", summary="", topics=[], cached_at=0),
    ]

@pytest.mark.asyncio
async def test_select_notebook_no_acl_filter(router, mock_notebooks, mocker):
    """Without ACL filter, all notebooks are considered."""
    mocker.patch.object(router, '_ensure_notebooks_cached', return_value=mock_notebooks)
    mocker.patch.object(router.llm_client, 'complete', return_value="nb-2")

    notebook_id, reasoning = await router.select_notebook("How to deploy?")

    assert notebook_id == "nb-2"

@pytest.mark.asyncio
async def test_select_notebook_with_acl_filter(router, mock_notebooks, mocker):
    """With ACL filter, only allowed notebooks are considered."""
    mocker.patch.object(router, '_ensure_notebooks_cached', return_value=mock_notebooks)
    mocker.patch.object(router.llm_client, 'complete', return_value="nb-1")

    # Only allow HR Docs and Public KB
    notebook_id, reasoning = await router.select_notebook(
        "What is vacation policy?",
        allowed_notebooks=["nb-1", "nb-3"]
    )

    assert notebook_id == "nb-1"
    # nb-2 (Tech Docs) should not have been in the LLM prompt

@pytest.mark.asyncio
async def test_select_notebook_acl_filters_all(router, mock_notebooks, mocker):
    """When ACL filter matches no notebooks, return appropriate error."""
    mocker.patch.object(router, '_ensure_notebooks_cached', return_value=mock_notebooks)

    # Allow notebooks that don't exist in cache
    notebook_id, reasoning = await router.select_notebook(
        "Some question",
        allowed_notebooks=["nb-999"]
    )

    assert notebook_id is None
    assert "No accessible notebooks" in reasoning

@pytest.mark.asyncio
async def test_select_notebook_empty_acl_list(router, mock_notebooks, mocker):
    """Empty ACL list means no access."""
    mocker.patch.object(router, '_ensure_notebooks_cached', return_value=mock_notebooks)

    notebook_id, reasoning = await router.select_notebook(
        "Some question",
        allowed_notebooks=[]
    )

    assert notebook_id is None
    assert "No accessible notebooks" in reasoning

@pytest.mark.asyncio
async def test_route_passes_acl_to_select_notebook(router, mock_notebooks, mocker):
    """route() should pass allowed_notebooks to select_notebook()."""
    mocker.patch.object(router, 'classify_request', return_value=RequestType.NOTEBOOKLM)
    mock_select = mocker.patch.object(
        router, 'select_notebook',
        return_value=("nb-1", "Selected HR Docs")
    )

    await router.route("vacation policy", allowed_notebooks=["nb-1"])

    mock_select.assert_called_once_with("vacation policy", ["nb-1"])
```

### Integration Tests

```python
# tests/test_server_acl.py

import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_chat_completions_with_acl_metadata(test_client: AsyncClient, auth_headers):
    """Test that metadata.allowed_notebooks is processed."""
    response = await test_client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "What is vacation policy?"}],
            "stream": False,
            "metadata": {
                "allowed_notebooks": ["hr-notebook-id"]
            }
        }
    )

    assert response.status_code == 200
    data = response.json()
    # Verify response came from allowed notebook
    assert "reasoning_content" in data["choices"][0]["message"]

@pytest.mark.asyncio
async def test_chat_completions_acl_blocks_unauthorized(test_client: AsyncClient, auth_headers):
    """Test that requests with no matching notebooks are handled gracefully."""
    response = await test_client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "Some question"}],
            "stream": False,
            "metadata": {
                "allowed_notebooks": ["nonexistent-notebook-id"]
            }
        }
    )

    assert response.status_code == 200
    data = response.json()
    # Should contain message about no accessible notebooks
    assert "No accessible" in data["choices"][0]["message"]["content"] or \
           "No accessible" in data["choices"][0]["message"].get("reasoning_content", "")
```

### Manual Testing

```bash
# 1. Start nlm-proxy with smart routing enabled
NLM_PROXY_DEBUG=true nlm-proxy serve openai --port 8080

# 2. Test without ACL (should use all notebooks)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "knowledge-finder",
    "messages": [{"role": "user", "content": "What is the vacation policy?"}],
    "stream": false
  }'

# 3. Test with ACL filter (should only consider specified notebooks)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "knowledge-finder",
    "messages": [{"role": "user", "content": "What is the vacation policy?"}],
    "stream": false,
    "metadata": {
      "allowed_notebooks": ["YOUR-HR-NOTEBOOK-ID"]
    }
  }'

# 4. Test with empty ACL (should return "no accessible notebooks")
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "knowledge-finder",
    "messages": [{"role": "user", "content": "What is the vacation policy?"}],
    "stream": false,
    "metadata": {
      "allowed_notebooks": []
    }
  }'

# Expected log output for test #3:
# INFO  [SMART-ROUTER] Per-request ACL: 1 allowed notebooks
# DEBUG [ROUTER] ACL filter: 5 → 1 notebooks (allowed: ['YOUR-HR-NOTEBOOK-ID'])
```

---

## OpenTelemetry Span Attributes

New span attributes added for observability:

| Span | Attribute | Type | Description |
|------|-----------|------|-------------|
| `smart_router.select_notebook` | `acl_filter_applied` | bool | Whether per-request ACL was applied |
| `smart_router.select_notebook` | `acl_allowed_count` | int | Number of notebooks in ACL list |
| `smart_router.select_notebook` | `acl_matched_count` | int | Number of cached notebooks matching ACL |

Example Grafana query:
```sql
SELECT
    SpanAttributes['acl_filter_applied'] as acl_applied,
    avg(SpanAttributes['acl_matched_count']) as avg_matched,
    count() as requests
FROM nlm_traces
WHERE SpanName = 'smart_router.select_notebook'
GROUP BY acl_applied
```

---

## Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| No `metadata` field in request | All notebooks considered (existing behavior) |
| `metadata` without `allowed_notebooks` | All notebooks considered (existing behavior) |
| `metadata.allowed_notebooks = null` | All notebooks considered (existing behavior) |
| `metadata.allowed_notebooks = []` | No notebooks (returns error) |
| `metadata.allowed_notebooks = ["id1", "id2"]` | Only specified notebooks considered |

**Breaking changes:** None. This is a purely additive change.

---

## Documentation Updates

### Update `docs/smart-routing-architecture.md`

Add new section after "Configuration":

```markdown
## Per-Request ACL Filtering

The smart router supports per-request notebook filtering via request metadata,
enabling integration with external access control systems.

### Usage

Pass `allowed_notebooks` in the request metadata:

```python
response = client.chat.completions.create(
    model="knowledge-finder",
    messages=[{"role": "user", "content": "..."}],
    extra_body={
        "metadata": {
            "allowed_notebooks": ["notebook-id-1", "notebook-id-2"]
        }
    }
)
```

### Behavior

- If `allowed_notebooks` is provided, only those notebooks are considered for routing
- If not provided or `null`, all cached notebooks are considered
- If empty list `[]`, returns "No accessible notebooks" error
- Filtering happens at request time; cache remains unchanged

### Integration Example (Chatbot with AD-based ACL)

```python
# 1. Look up user's AD groups
user_groups = await graph_client.get_user_groups(user_id)

# 2. Map groups to allowed notebooks via ACL config
allowed_notebooks = acl_service.get_allowed_notebooks(user_groups)

# 3. Pass to nlm-proxy
response = await nlm_client.chat.completions.create(
    model="knowledge-finder",
    messages=[{"role": "user", "content": user_message}],
    extra_body={"metadata": {"allowed_notebooks": allowed_notebooks}}
)
```
```

### Update `.env.example`

No changes needed - this is a per-request feature, not a configuration option.

---

## Checklist

- [ ] Update `router.py`: Add `allowed_notebooks` param to `select_notebook()`
- [ ] Update `router.py`: Add filtering logic with logging
- [ ] Update `router.py`: Add `allowed_notebooks` param to `route()`
- [ ] Update `server.py`: Extract `allowed_notebooks` from `request.metadata`
- [ ] Update `server.py`: Pass to `router.route()` in streaming path
- [ ] Update `server.py`: Pass to `router.route()` in non-streaming path
- [ ] Verify `types.py`: Ensure `metadata` field exists
- [ ] Add unit tests for ACL filtering
- [ ] Add integration tests for API
- [ ] Update `smart-routing-architecture.md` documentation
- [ ] Manual testing with curl

---

## Commit Message

```
feat(router): add per-request ACL filtering for notebook selection

Add support for filtering notebook candidates via request metadata,
enabling external applications to implement access control:

- Add `allowed_notebooks` parameter to `route()` and `select_notebook()`
- Extract `metadata.allowed_notebooks` from chat completion requests
- Filter cached notebooks to only those in the allowed list
- Add OpenTelemetry span attributes for ACL observability
- Maintain backward compatibility (no ACL = all notebooks)

This enables the NotebookLM Chatbot to implement Azure AD group-based
access control by passing user-specific notebook lists per request.

Co-Authored-By: Claude <noreply@anthropic.com>
```
