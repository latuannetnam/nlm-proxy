# Stage 3: Config + NotebookCache Move

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `AgentSettings` configuration class and move `NotebookCache` from `openai/` to `core/` for shared access.

**Architecture:** Additive changes only — no existing classes are modified. `AgentSettings` uses `NLM_PROXY_AGENT_` prefix, keeping backward compatibility with existing env vars. `NotebookCache` move uses a re-export for backward compat.

**Inputs:** None — this stage has no dependencies.

**Outputs:** `AgentSettings` accessible via `get_agent_settings()`. `NotebookCache` importable from both `core/notebook_cache` and `openai/notebook_cache`.

---

## Task 3.1: Add AgentSettings to config.py

**Files:**
- Modify: `src/nlm_proxy/core/config.py`
- Modify: `tests/test_config.py`

**Step 1: Write failing test**

Add to `tests/test_config.py`:

```python
def test_agent_settings_defaults():
    """Test AgentSettings has expected defaults."""
    from nlm_proxy.core.config import get_agent_settings
    settings = get_agent_settings()
    assert settings.llm_provider == "openai"
    assert settings.embedding_provider == "huggingface"
    assert settings.memory_backend == "memory"
    assert settings.agent_max_iterations == 10
    assert settings.agent_fallback_on_error is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k "test_agent_settings_defaults" -v`
Expected: FAIL — `ImportError: cannot import name 'get_agent_settings'`

**Step 3: Implement AgentSettings**

Add to `config.py` after `CacheSettings`:

```python
class AgentSettings(BaseSettings):
    """LangChain/LangGraph agent configuration (additive — does not replace existing)."""
    llm_provider: str = Field(default="openai", description="LLM provider")
    embedding_provider: str = Field(default="huggingface", description="Embedding provider")
    memory_backend: str = Field(default="memory", description="memory | sqlite | postgres")
    memory_db_path: str = Field(default="~/.nlm-proxy/memory.db")
    agent_max_iterations: int = Field(default=10)
    agent_verbose: bool = Field(default=False)
    agent_fallback_on_error: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_AGENT_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

# Add singleton:
_agent: AgentSettings | None = None

def get_agent_settings() -> AgentSettings:
    """Get the agent settings instance."""
    global _agent
    if _agent is None:
        _agent = AgentSettings()
    return _agent
```

**Step 4: Run test, commit**

Run: `uv run pytest tests/test_config.py -v`

```bash
git add src/nlm_proxy/core/config.py tests/test_config.py
git commit -m "feat: add AgentSettings with NLM_PROXY_AGENT_ prefix"
```

---

## Task 3.2: Move NotebookCache to core/

**Files:**
- Copy: `src/nlm_proxy/openai/notebook_cache.py` → `src/nlm_proxy/core/notebook_cache.py`
- Modify: `src/nlm_proxy/openai/notebook_cache.py` (re-export only)
- Test: `tests/test_openai_module/test_notebook_cache.py`

**Step 1: Copy file and add re-export**

```bash
# Copy the file
cp src/nlm_proxy/openai/notebook_cache.py src/nlm_proxy/core/notebook_cache.py
```

Replace `src/nlm_proxy/openai/notebook_cache.py` with:
```python
"""Backward-compatible re-export. Actual implementation moved to core/."""
from nlm_proxy.core.notebook_cache import (  # noqa: F401
    NotebookCache, NotebookInfo, SourceInfo, _extract_first_sentence,
)
```

**Step 2: Update `core/__init__.py`** to export `NotebookCache`

**Step 3: Run ALL tests**

Run: `uv run pytest -v`
Expected: ALL PASS — re-export maintains backward compatibility

**Step 4: Commit**

```bash
git add src/nlm_proxy/core/notebook_cache.py src/nlm_proxy/openai/notebook_cache.py
git commit -m "refactor: move NotebookCache to core/ with backward-compat re-export"
```

---

## 🔒 Stage 3 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS
