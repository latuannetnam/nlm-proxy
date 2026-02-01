# OpenAI Proxy Authentication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add required API key authentication to the OpenAI proxy for secure remote deployment.

**Architecture:** Add `api_key` field to `OpenAISettings` (required, no default). Create FastAPI dependency `verify_api_key` that validates `Authorization: Bearer <key>` header using timing-safe comparison. Apply dependency to all `/v1/*` routes. Return OpenAI-compatible error responses.

**Tech Stack:** FastAPI, pydantic-settings, secrets module

---

### Task 1: Add api_key to OpenAISettings

**Files:**
- Modify: `src/nlm_proxy/core/config.py:86-100`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

Add to `tests/test_config.py` in `TestOpenAISettings` class:

```python
def test_api_key_required(self):
    """OpenAISettings should require api_key."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("NLM_PROXY_OPENAI_")}
    with patch.dict(os.environ, env, clear=True):
        from pydantic import ValidationError
        from nlm_proxy.core.config import OpenAISettings
        with pytest.raises(ValidationError) as exc_info:
            OpenAISettings()
        assert "api_key" in str(exc_info.value)

def test_api_key_from_env(self):
    """NLM_PROXY_OPENAI_API_KEY should set api_key."""
    with patch.dict(os.environ, {"NLM_PROXY_OPENAI_API_KEY": "test-key-123"}, clear=False):
        from nlm_proxy.core.config import OpenAISettings
        settings = OpenAISettings()
        assert settings.api_key == "test-key-123"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestOpenAISettings::test_api_key_required -v`
Expected: FAIL with `AttributeError: 'OpenAISettings' object has no attribute 'api_key'`

**Step 3: Write minimal implementation**

In `src/nlm_proxy/core/config.py`, modify `OpenAISettings` class (lines 86-100):

```python
class OpenAISettings(BaseSettings):
    """OpenAI proxy server settings."""

    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8080, description="Port to listen on")
    session_ttl: int = Field(
        default=86400, description="Session TTL in seconds (default: 24h)"
    )
    api_key: str = Field(description="API key for authentication (required)")

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_OPENAI_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py::TestOpenAISettings::test_api_key_required tests/test_config.py::TestOpenAISettings::test_api_key_from_env -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/config.py tests/test_config.py
git commit -m "feat(config): add required api_key to OpenAISettings"
```

---

### Task 2: Fix existing OpenAISettings tests

**Files:**
- Modify: `tests/test_config.py`

**Step 1: Update existing tests to provide api_key**

The existing `TestOpenAISettings` tests will now fail because `api_key` is required. Update them:

```python
class TestOpenAISettings:
    """Test OpenAISettings class."""

    def test_default_values(self):
        """Default values for OpenAI settings (with required api_key)."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("NLM_PROXY_OPENAI_")}
        env["NLM_PROXY_OPENAI_API_KEY"] = "test-key"  # Required
        with patch.dict(os.environ, env, clear=True):
            from nlm_proxy.core.config import OpenAISettings
            settings = OpenAISettings()

            assert settings.host == "0.0.0.0"
            assert settings.port == 8080
            assert settings.session_ttl == 86400
            assert settings.api_key == "test-key"

    def test_env_override_all(self):
        """All OpenAI env vars should work."""
        env_vars = {
            "NLM_PROXY_OPENAI_HOST": "127.0.0.1",
            "NLM_PROXY_OPENAI_PORT": "3000",
            "NLM_PROXY_OPENAI_SESSION_TTL": "3600",
            "NLM_PROXY_OPENAI_API_KEY": "my-secret-key",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            from nlm_proxy.core.config import OpenAISettings
            settings = OpenAISettings()

            assert settings.host == "127.0.0.1"
            assert settings.port == 3000
            assert settings.session_ttl == 3600
            assert settings.api_key == "my-secret-key"

    def test_api_key_required(self):
        """OpenAISettings should require api_key."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("NLM_PROXY_OPENAI_")}
        with patch.dict(os.environ, env, clear=True):
            from pydantic import ValidationError
            from nlm_proxy.core.config import OpenAISettings
            with pytest.raises(ValidationError) as exc_info:
                OpenAISettings()
            assert "api_key" in str(exc_info.value)

    def test_api_key_from_env(self):
        """NLM_PROXY_OPENAI_API_KEY should set api_key."""
        with patch.dict(os.environ, {"NLM_PROXY_OPENAI_API_KEY": "test-key-123"}, clear=False):
            from nlm_proxy.core.config import OpenAISettings
            settings = OpenAISettings()
            assert settings.api_key == "test-key-123"
```

**Step 2: Run all OpenAISettings tests**

Run: `uv run pytest tests/test_config.py::TestOpenAISettings -v`
Expected: PASS (all 4 tests)

**Step 3: Commit**

```bash
git add tests/test_config.py
git commit -m "test(config): update OpenAISettings tests for required api_key"
```

---

### Task 3: Add authentication dependency to server

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`
- Test: `tests/test_openai_proxy.py`

**Step 1: Write the failing tests**

Add to `tests/test_openai_proxy.py`:

```python
@pytest.mark.openai
def test_missing_auth_header_returns_401():
    """Request without Authorization header should return 401."""
    with patch.dict(os.environ, {"NLM_PROXY_OPENAI_API_KEY": "test-secret"}, clear=False):
        # Need to reimport to pick up new settings
        import importlib
        import nlm_proxy.core.config as config
        config._openai = None
        import nlm_proxy.openai.server as server_module
        importlib.reload(server_module)

        client = TestClient(server_module.app)
        response = client.get("/v1/models")

        assert response.status_code == 401
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "invalid_api_key"


@pytest.mark.openai
def test_invalid_api_key_returns_401():
    """Request with wrong API key should return 401."""
    with patch.dict(os.environ, {"NLM_PROXY_OPENAI_API_KEY": "correct-key"}, clear=False):
        import importlib
        import nlm_proxy.core.config as config
        config._openai = None
        import nlm_proxy.openai.server as server_module
        importlib.reload(server_module)

        client = TestClient(server_module.app)
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"}
        )

        assert response.status_code == 401


@pytest.mark.openai
def test_valid_api_key_allows_request():
    """Request with correct API key should succeed."""
    from nlm_proxy.core import Notebook

    mock_notebooks = [
        Notebook(id="nb-123", title="Test", source_count=1, sources=[]),
    ]

    with patch.dict(os.environ, {"NLM_PROXY_OPENAI_API_KEY": "valid-key"}, clear=False):
        import importlib
        import nlm_proxy.core.config as config
        config._openai = None
        import nlm_proxy.openai.server as server_module
        importlib.reload(server_module)

        with patch.object(server_module, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.list_notebooks = AsyncMock(return_value=mock_notebooks)
            mock_client.close = AsyncMock()
            mock_get_client.return_value = mock_client

            client = TestClient(server_module.app)
            response = client.get(
                "/v1/models",
                headers={"Authorization": "Bearer valid-key"}
            )

            assert response.status_code == 200
```

Add this import at the top of the file:

```python
import os
from unittest.mock import patch
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_proxy.py::test_missing_auth_header_returns_401 -v`
Expected: FAIL (currently returns 200 or different error)

**Step 3: Write the implementation**

Modify `src/nlm_proxy/openai/server.py`:

Add imports at top:

```python
import secrets
from typing import Annotated

from fastapi import Depends, Header
```

Add after the `app` definition (around line 34):

```python
from nlm_proxy.core.config import get_openai_settings


def verify_api_key(authorization: Annotated[str | None, Header()] = None) -> None:
    """Verify the API key from Authorization header."""
    settings = get_openai_settings()

    error_response = {
        "error": {
            "message": "Invalid API key",
            "type": "invalid_request_error",
            "code": "invalid_api_key"
        }
    }

    if not authorization:
        raise HTTPException(status_code=401, detail=error_response)

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail=error_response)

    token = authorization.removeprefix("Bearer ")
    if not secrets.compare_digest(token, settings.api_key):
        raise HTTPException(status_code=401, detail=error_response)
```

Update route decorators to include dependency:

```python
@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    ...

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    ...

@app.post("/v1/embeddings", dependencies=[Depends(verify_api_key)])
async def embeddings():
    ...

@app.get("/v1/sessions", dependencies=[Depends(verify_api_key)])
async def list_sessions():
    ...

@app.delete("/v1/sessions/{chat_id}", dependencies=[Depends(verify_api_key)])
async def delete_session(chat_id: str):
    ...

@app.get("/v1/sessions/stats", dependencies=[Depends(verify_api_key)])
async def session_stats():
    ...
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_proxy.py::test_missing_auth_header_returns_401 tests/test_openai_proxy.py::test_invalid_api_key_returns_401 tests/test_openai_proxy.py::test_valid_api_key_allows_request -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/server.py tests/test_openai_proxy.py
git commit -m "feat(openai): add API key authentication to all /v1/* routes"
```

---

### Task 4: Fix existing openai proxy tests

**Files:**
- Modify: `tests/test_openai_proxy.py`

**Step 1: Update existing tests to include auth header**

All existing tests that call `/v1/*` endpoints need the auth header. Create a fixture and update tests:

Add near top of file:

```python
import os
from unittest.mock import patch

# Test API key used across all tests
TEST_API_KEY = "test-api-key-for-tests"

@pytest.fixture(autouse=True)
def setup_test_api_key():
    """Set up test API key for all tests."""
    with patch.dict(os.environ, {"NLM_PROXY_OPENAI_API_KEY": TEST_API_KEY}, clear=False):
        import nlm_proxy.core.config as config
        config._openai = None  # Reset singleton
        yield
```

Update test functions to include auth header:

```python
@pytest.mark.openai
def test_models_list_returns_notebooks():
    from nlm_proxy.openai.server import app
    from nlm_proxy.core import Notebook

    mock_notebooks = [
        Notebook(id="nb-123", title="Research Notes", source_count=3, sources=[]),
        Notebook(id="nb-456", title="Project Docs", source_count=1, sources=[]),
    ]

    with patch("nlm_proxy.openai.server.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.list_notebooks = AsyncMock(return_value=mock_notebooks)
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 200
        # ... rest of assertions
```

Apply similar changes to:
- `test_embeddings_returns_501`
- `test_chat_completions_non_streaming`
- `test_chat_completions_streaming`
- `test_chat_completions_streaming_with_thinking`

**Step 2: Run all openai tests**

Run: `uv run pytest tests/test_openai_proxy.py -v`
Expected: PASS (all tests)

**Step 3: Commit**

```bash
git add tests/test_openai_proxy.py
git commit -m "test(openai): update all tests to include auth header"
```

---

### Task 5: Update .env.example

**Files:**
- Modify: `.env.example`

**Step 1: Add api_key to .env.example**

Add to the OpenAI Proxy section:

```bash
# =============================================================================
# OpenAI Proxy Server (nlm-proxy serve openai)
# =============================================================================
# REQUIRED: API key for authentication (no default - must be set)
# NLM_PROXY_OPENAI_API_KEY=your-secret-key-here
# NLM_PROXY_OPENAI_HOST=0.0.0.0
# NLM_PROXY_OPENAI_PORT=8080
# NLM_PROXY_OPENAI_SESSION_TTL=86400
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add NLM_PROXY_OPENAI_API_KEY to .env.example"
```

---

### Task 6: Update memory documentation

**Files:**
- Modify: `.claude/memory/openai-proxy.md`

**Step 1: Read current file**

Read `.claude/memory/openai-proxy.md` to understand current structure.

**Step 2: Add authentication section**

Add after any existing configuration section:

```markdown
## Authentication

The OpenAI proxy requires API key authentication on all `/v1/*` endpoints.

### Setup

Set the required environment variable:

```bash
export NLM_PROXY_OPENAI_API_KEY="your-secret-key-here"
nlm-proxy serve openai
```

Or in `.env`:

```
NLM_PROXY_OPENAI_API_KEY=your-secret-key-here
```

### Client Configuration

**OpenAI Python SDK:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://your-server:8080/v1",
    api_key="your-nlm-proxy-key"
)
```

**curl:**
```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer your-nlm-proxy-key"
```

**Open WebUI:**
```
OPENAI_API_KEY=your-nlm-proxy-key
OPENAI_API_BASE_URL=http://your-server:8080/v1
```

### Error Responses

Missing or invalid API key returns 401:
```json
{
  "error": {
    "message": "Invalid API key",
    "type": "invalid_request_error",
    "code": "invalid_api_key"
  }
}
```
```

**Step 3: Commit**

```bash
git add .claude/memory/openai-proxy.md
git commit -m "docs: add authentication section to openai-proxy memory"
```

---

### Task 7: Run full test suite

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 2: Manual verification**

```bash
# Should fail - no API key
NLM_PROXY_OPENAI_API_KEY= nlm-proxy serve openai
# Expected: ValidationError about api_key

# Should start
export NLM_PROXY_OPENAI_API_KEY="test123"
nlm-proxy serve openai &

# Should return 401
curl http://localhost:8080/v1/models
# Expected: {"error":{"message":"Invalid API key",...}}

# Should work (if authenticated with NotebookLM)
curl http://localhost:8080/v1/models -H "Authorization: Bearer test123"
```

**Step 3: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address issues from integration testing"
```
