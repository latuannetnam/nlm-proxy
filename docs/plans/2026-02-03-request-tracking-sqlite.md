# Request/Response Tracking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add database-agnostic request/response tracking for post-hoc analysis and debugging of NotebookLM proxy requests.

**Architecture:** SQLAlchemy Core 2.0 with async batching via `asyncio.Queue`. Records are pushed to a queue (zero latency), and a background task batch-writes every 1 second or 10 records. Swap SQLite → PostgreSQL/MySQL by changing one connection string.

**Tech Stack:** SQLAlchemy Core 2.0, aiosqlite (SQLite driver), asyncio.Queue for batching, Pydantic settings.

**New Dependencies:** `sqlalchemy>=2.0`, `aiosqlite`

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add sqlalchemy and aiosqlite to dependencies**

In `pyproject.toml`, add to the dependencies list:

```toml
dependencies = [
    # ... existing deps ...
    "sqlalchemy>=2.0",
    "aiosqlite>=0.19.0",
]
```

**Step 2: Install dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

**Step 3: Verify imports work**

Run: `uv run python -c "from sqlalchemy.ext.asyncio import create_async_engine; print('OK')"`
Expected: "OK"

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add sqlalchemy and aiosqlite for tracking"
```

---

## Task 2: Add TrackingSettings Configuration

**Files:**
- Modify: `src/nlm_proxy/core/config.py`

**Step 1: Write the test for TrackingSettings**

```python
# tests/test_config.py (add to existing file)

def test_tracking_settings_defaults():
    """TrackingSettings should have sensible defaults."""
    from nlm_proxy.core.config import TrackingSettings

    settings = TrackingSettings()
    assert settings.enabled is False
    assert "sqlite+aiosqlite" in settings.db_url
    assert settings.max_response_length == 10000
    assert settings.batch_size == 10
    assert settings.flush_interval == 1.0


def test_tracking_settings_from_env(monkeypatch):
    """TrackingSettings should read from environment."""
    from nlm_proxy.core.config import TrackingSettings

    monkeypatch.setenv("NLM_PROXY_TRACKING_ENABLED", "true")
    monkeypatch.setenv("NLM_PROXY_TRACKING_DB_URL", "postgresql+asyncpg://user:pass@localhost/db")
    monkeypatch.setenv("NLM_PROXY_TRACKING_BATCH_SIZE", "20")

    settings = TrackingSettings()
    assert settings.enabled is True
    assert "postgresql" in settings.db_url
    assert settings.batch_size == 20
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_tracking_settings_defaults -v`
Expected: FAIL with "cannot import name 'TrackingSettings'"

**Step 3: Implement TrackingSettings**

Add to `src/nlm_proxy/core/config.py` after existing settings classes:

```python
class TrackingSettings(BaseSettings):
    """Request/response tracking settings.

    Supports SQLite (default), PostgreSQL, and MySQL by changing db_url:
    - SQLite:    sqlite+aiosqlite:///~/.nlm-proxy/tracking.db
    - Postgres:  postgresql+asyncpg://user:pass@host/db
    - MySQL:     mysql+aiomysql://user:pass@host/db
    """

    enabled: bool = Field(
        default=False,
        description="Enable request/response tracking"
    )
    db_url: str = Field(
        default="sqlite+aiosqlite:///~/.nlm-proxy/tracking.db",
        description="Database connection URL (SQLAlchemy async format)"
    )
    max_response_length: int = Field(
        default=10000,
        description="Max response text to store (longer responses are truncated)"
    )
    batch_size: int = Field(
        default=10,
        description="Number of records to batch before writing"
    )
    flush_interval: float = Field(
        default=1.0,
        description="Max seconds to wait before flushing batch"
    )

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_TRACKING_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_tracking_settings() -> TrackingSettings:
    """Get cached tracking settings."""
    return TrackingSettings()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py::test_tracking_settings_defaults tests/test_config.py::test_tracking_settings_from_env -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/config.py tests/test_config.py
git commit -m "feat(tracking): add TrackingSettings with db_url for swappable backends"
```

---

## Task 3: Create RequestRecord Dataclass

**Files:**
- Create: `src/nlm_proxy/openai/tracking.py`
- Create: `tests/test_tracking.py`

**Step 1: Write the test for RequestRecord**

```python
# tests/test_tracking.py

import time
from nlm_proxy.openai.tracking import RequestRecord


def test_request_record_creation():
    """RequestRecord should store request metadata."""
    record = RequestRecord(
        id="chatcmpl-abc123",
        query_text="What is NetNam?",
        model="knowledge-finder",
        stream=True
    )

    assert record.id == "chatcmpl-abc123"
    assert record.query_text == "What is NetNam?"
    assert record.model == "knowledge-finder"
    assert record.stream is True
    assert record.created_at is not None
    assert record.request_type is None  # Not set yet


def test_request_record_to_dict():
    """RequestRecord should convert to dict for DB insertion."""
    record = RequestRecord(
        id="chatcmpl-abc123",
        query_text="Test query",
        model="test-model",
        stream=False
    )
    record.request_type = "notebooklm"
    record.selected_notebook_id = "nb-123"
    record.latency_ms = 500

    data = record.to_dict()

    assert data["id"] == "chatcmpl-abc123"
    assert data["request_type"] == "notebooklm"
    assert data["selected_notebook_id"] == "nb-123"
    assert data["latency_ms"] == 500
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracking.py::test_request_record_creation -v`
Expected: FAIL with "No module named 'nlm_proxy.openai.tracking'"

**Step 3: Implement RequestRecord**

Create `src/nlm_proxy/openai/tracking.py`:

```python
"""Request/response tracking for analysis and debugging.

Uses SQLAlchemy Core for database-agnostic storage. Supports:
- SQLite (default, zero config)
- PostgreSQL (change db_url)
- MySQL (change db_url)
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RequestRecord:
    """Accumulates data during request lifecycle for tracking."""

    # Required fields
    id: str
    query_text: str
    model: str
    stream: bool

    # Auto-set
    created_at: float = field(default_factory=time.time)

    # Request context (optional)
    chat_id: Optional[str] = None
    conversation_id: Optional[str] = None

    # Routing decision (populated by router)
    request_type: Optional[str] = None  # "notebooklm" | "llm_task"
    selected_notebook_id: Optional[str] = None
    selected_notebook_title: Optional[str] = None
    routing_reasoning: Optional[str] = None

    # Response (populated after completion)
    response_text: Optional[str] = None
    response_length: Optional[int] = None
    finish_reason: Optional[str] = None

    # Performance
    latency_ms: Optional[int] = None
    routing_latency_ms: Optional[int] = None

    # Error tracking
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return asdict(self)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/tracking.py tests/test_tracking.py
git commit -m "feat(tracking): add RequestRecord dataclass"
```

---

## Task 4: Define SQLAlchemy Table Schema

**Files:**
- Modify: `src/nlm_proxy/openai/tracking.py`
- Modify: `tests/test_tracking.py`

**Step 1: Write the test for table schema**

```python
# tests/test_tracking.py (add to existing)

from sqlalchemy import inspect
from nlm_proxy.openai.tracking import requests_table, metadata


def test_requests_table_columns():
    """requests_table should have all required columns."""
    columns = {c.name for c in requests_table.columns}

    required = {
        "id", "created_at", "query_text", "model", "stream",
        "chat_id", "conversation_id",
        "request_type", "selected_notebook_id", "selected_notebook_title", "routing_reasoning",
        "response_text", "response_length", "finish_reason",
        "latency_ms", "routing_latency_ms",
        "error_type", "error_message"
    }

    assert required.issubset(columns)


def test_requests_table_primary_key():
    """requests_table should have id as primary key."""
    pk_columns = [c.name for c in requests_table.primary_key.columns]
    assert pk_columns == ["id"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracking.py::test_requests_table_columns -v`
Expected: FAIL with "cannot import name 'requests_table'"

**Step 3: Implement SQLAlchemy table schema**

Add to `src/nlm_proxy/openai/tracking.py`:

```python
from sqlalchemy import (
    Table, Column, String, Float, Integer, Boolean, Text, MetaData, Index
)

metadata = MetaData()

requests_table = Table(
    "requests",
    metadata,
    # Primary key
    Column("id", String(64), primary_key=True),
    Column("created_at", Float, nullable=False),

    # Request
    Column("query_text", Text, nullable=False),
    Column("model", String(128), nullable=False),
    Column("chat_id", String(64)),
    Column("conversation_id", String(64)),
    Column("stream", Boolean, nullable=False, default=False),

    # Routing
    Column("request_type", String(32)),
    Column("selected_notebook_id", String(64)),
    Column("selected_notebook_title", String(256)),
    Column("routing_reasoning", Text),

    # Response
    Column("response_text", Text),
    Column("response_length", Integer),
    Column("finish_reason", String(32)),

    # Performance
    Column("latency_ms", Integer),
    Column("routing_latency_ms", Integer),

    # Errors
    Column("error_type", String(128)),
    Column("error_message", Text),
)

# Indexes for common queries
Index("idx_requests_created_at", requests_table.c.created_at)
Index("idx_requests_chat_id", requests_table.c.chat_id)
Index("idx_requests_selected_notebook", requests_table.c.selected_notebook_id)
Index("idx_requests_request_type", requests_table.c.request_type)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracking.py::test_requests_table_columns tests/test_tracking.py::test_requests_table_primary_key -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/tracking.py tests/test_tracking.py
git commit -m "feat(tracking): add SQLAlchemy table schema"
```

---

## Task 5: Implement RequestTracker with Async Batching

**Files:**
- Modify: `src/nlm_proxy/openai/tracking.py`
- Modify: `tests/test_tracking.py`

**Step 1: Write the test for RequestTracker**

```python
# tests/test_tracking.py (add to existing)

import pytest
import asyncio
import tempfile
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_tracker_creates_database(tmp_path):
    """RequestTracker should create database and tables on start."""
    from nlm_proxy.openai.tracking import RequestTracker

    db_path = tmp_path / "test_tracking.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    tracker = RequestTracker(db_url=db_url, enabled=True)
    await tracker.start()

    assert db_path.exists()

    # Verify table exists
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        result = await conn.execute(
            select(requests_table).limit(1)
        )
        # Should not raise - table exists

    await tracker.shutdown()
    await engine.dispose()


@pytest.mark.asyncio
async def test_tracker_writes_batched_records(tmp_path):
    """RequestTracker should batch and write records."""
    from nlm_proxy.openai.tracking import RequestTracker, RequestRecord

    db_path = tmp_path / "test_tracking.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    tracker = RequestTracker(db_url=db_url, enabled=True, batch_size=2, flush_interval=0.1)
    await tracker.start()

    # Queue 2 records (triggers batch write)
    record1 = RequestRecord(id="test-1", query_text="Q1", model="m1", stream=False)
    record2 = RequestRecord(id="test-2", query_text="Q2", model="m2", stream=True)

    await tracker.queue_record(record1)
    await tracker.queue_record(record2)

    # Wait for batch to flush
    await asyncio.sleep(0.3)

    # Verify records in database
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        result = await conn.execute(select(requests_table))
        rows = result.fetchall()

    assert len(rows) == 2
    ids = {row.id for row in rows}
    assert ids == {"test-1", "test-2"}

    await tracker.shutdown()
    await engine.dispose()


@pytest.mark.asyncio
async def test_tracker_disabled_no_database(tmp_path):
    """Disabled tracker should not create database."""
    from nlm_proxy.openai.tracking import RequestTracker

    db_path = tmp_path / "test_tracking.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    tracker = RequestTracker(db_url=db_url, enabled=False)
    await tracker.start()

    assert not db_path.exists()

    await tracker.shutdown()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracking.py::test_tracker_creates_database -v`
Expected: FAIL with "cannot import name 'RequestTracker'"

**Step 3: Implement RequestTracker with async batching**

Add to `src/nlm_proxy/openai/tracking.py`:

```python
import asyncio
import os
from pathlib import Path
from typing import List

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from nlm_proxy.core.logging import get_logger

logger = get_logger(__name__)


class RequestTracker:
    """Tracks request/response pairs with async batched writes.

    Records are queued and batch-written for minimal latency impact.
    Supports SQLite, PostgreSQL, MySQL via SQLAlchemy.
    """

    def __init__(
        self,
        db_url: str = None,
        enabled: bool = None,
        batch_size: int = None,
        flush_interval: float = None,
        max_response_length: int = None
    ):
        """Initialize tracker.

        Args:
            db_url: SQLAlchemy async database URL. If None, uses settings.
            enabled: Whether tracking is enabled. If None, uses settings.
            batch_size: Records per batch. If None, uses settings.
            flush_interval: Max seconds between flushes. If None, uses settings.
            max_response_length: Truncate responses to this length.
        """
        from nlm_proxy.core.config import get_tracking_settings
        settings = get_tracking_settings()

        self.enabled = enabled if enabled is not None else settings.enabled
        self.batch_size = batch_size or settings.batch_size
        self.flush_interval = flush_interval or settings.flush_interval
        self.max_response_length = max_response_length or settings.max_response_length

        # Expand ~ in SQLite path
        db_url = db_url or settings.db_url
        if db_url.startswith("sqlite"):
            # Extract path and expand ~
            parts = db_url.split("///")
            if len(parts) == 2:
                path = os.path.expanduser(parts[1])
                db_url = f"{parts[0]}:///{path}"

        self.db_url = db_url
        self._engine: AsyncEngine = None
        self._queue: asyncio.Queue = None
        self._writer_task: asyncio.Task = None
        self._shutdown_event: asyncio.Event = None

    async def start(self):
        """Start the tracker and create database tables."""
        if not self.enabled:
            logger.debug("[TRACKING] Tracker disabled, skipping initialization")
            return

        # Ensure parent directory exists for SQLite
        if "sqlite" in self.db_url:
            parts = self.db_url.split("///")
            if len(parts) == 2:
                db_path = Path(parts[1])
                db_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_async_engine(self.db_url, echo=False)

        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

        logger.info(f"[TRACKING] Database initialized: {self.db_url}")

        # Start background writer
        self._queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        self._writer_task = asyncio.create_task(self._batch_writer())

        logger.debug("[TRACKING] Background writer started")

    async def queue_record(self, record: RequestRecord):
        """Queue a record for batch writing.

        This is non-blocking and returns immediately.
        """
        if not self.enabled or self._queue is None:
            return

        await self._queue.put(record)

    async def _batch_writer(self):
        """Background task that batch-writes records."""
        buffer: List[RequestRecord] = []

        while not self._shutdown_event.is_set():
            try:
                # Wait for record with timeout
                try:
                    record = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=self.flush_interval
                    )
                    buffer.append(record)
                except asyncio.TimeoutError:
                    pass  # Timeout - flush whatever we have

                # Flush if batch size reached or timeout
                if len(buffer) >= self.batch_size or (buffer and self._queue.empty()):
                    await self._flush(buffer)
                    buffer = []

            except asyncio.CancelledError:
                # Flush remaining on shutdown
                if buffer:
                    await self._flush(buffer)
                raise
            except Exception as e:
                logger.warning(f"[TRACKING] Batch writer error: {e}")

    async def _flush(self, records: List[RequestRecord]):
        """Write a batch of records to the database."""
        if not records:
            return

        try:
            data = [r.to_dict() for r in records]

            async with self._engine.begin() as conn:
                await conn.execute(requests_table.insert(), data)

            logger.debug(f"[TRACKING] Wrote {len(records)} records")
        except Exception as e:
            logger.warning(f"[TRACKING] Failed to write batch: {e}")

    async def shutdown(self):
        """Shutdown tracker and flush pending records."""
        if not self.enabled:
            return

        if self._shutdown_event:
            self._shutdown_event.set()

        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass

        if self._engine:
            await self._engine.dispose()

        logger.debug("[TRACKING] Tracker shutdown complete")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/tracking.py tests/test_tracking.py
git commit -m "feat(tracking): implement RequestTracker with async batching"
```

---

## Task 6: Add Helper Methods to RequestTracker

**Files:**
- Modify: `src/nlm_proxy/openai/tracking.py`

**Step 1: Add start_request and finish_request helpers**

Add to `RequestTracker` class:

```python
    def start_request(
        self,
        request_id: str,
        query_text: str,
        model: str,
        stream: bool,
        chat_id: str = None,
        conversation_id: str = None
    ) -> RequestRecord:
        """Create a new request record.

        Returns:
            RequestRecord to accumulate data during request lifecycle.
        """
        return RequestRecord(
            id=request_id,
            query_text=query_text[:self.max_response_length],  # Truncate query too
            model=model,
            stream=stream,
            chat_id=chat_id,
            conversation_id=conversation_id
        )

    async def finish_request(self, record: RequestRecord):
        """Finalize and queue record for writing.

        Truncates response_text if needed before queuing.
        """
        if not self.enabled:
            return

        # Truncate response if needed
        if record.response_text and len(record.response_text) > self.max_response_length:
            record.response_text = record.response_text[:self.max_response_length]

        await self.queue_record(record)
```

**Step 2: Commit**

```bash
git add src/nlm_proxy/openai/tracking.py
git commit -m "feat(tracking): add start_request and finish_request helpers"
```

---

## Task 7: Integrate Tracker into Server Initialization

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Add tracker initialization to server startup**

In `src/nlm_proxy/openai/server.py`, add import at top:

```python
from nlm_proxy.core.config import get_tracking_settings
```

Add FastAPI lifespan handler (or modify existing startup/shutdown):

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    tracking_settings = get_tracking_settings()
    if tracking_settings.enabled:
        from nlm_proxy.openai.tracking import RequestTracker
        app.state.tracker = RequestTracker()
        await app.state.tracker.start()
        logger.info(f"Request tracking enabled: {tracking_settings.db_url}")
    else:
        app.state.tracker = None

    yield

    # Shutdown
    if app.state.tracker:
        await app.state.tracker.shutdown()
```

Update FastAPI app initialization:

```python
app = FastAPI(
    title="NotebookLM OpenAI Proxy",
    description="OpenAI-compatible API for NotebookLM",
    version="0.1.0",
    lifespan=lifespan  # Add this
)
```

**Note:** If existing startup logic is in `main()`, move async initialization to lifespan handler.

**Step 2: Run server to verify no errors**

Run: `NLM_PROXY_TRACKING_ENABLED=true uv run python -c "from nlm_proxy.openai.server import app; print('Import OK')"`
Expected: "Import OK" with no errors

**Step 3: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(tracking): integrate tracker with FastAPI lifespan"
```

---

## Task 8: Track Non-Streaming Requests

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Add tracking to chat_completions endpoint**

In `chat_completions()` function, add tracking logic:

```python
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    start_time = time.time()
    record = None

    # ... existing query extraction logic ...

    # Start tracking if enabled
    if app.state.tracker:
        record = app.state.tracker.start_request(
            request_id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            query_text=query_text,
            model=request.model,
            stream=request.stream,
            chat_id=chat_id,
            conversation_id=request.conversation_id
        )

    try:
        # ... existing request handling ...

        # For non-streaming, after getting answer:
        if record:
            record.response_text = answer
            record.response_length = len(answer) if answer else 0
            record.finish_reason = "stop"

        return response

    except Exception as e:
        if record:
            record.error_type = type(e).__name__
            record.error_message = str(e)[:500]
        raise
    finally:
        if record:
            record.latency_ms = int((time.time() - start_time) * 1000)
            await app.state.tracker.finish_request(record)
```

**Step 2: Run server and test manually**

Run: `NLM_PROXY_TRACKING_ENABLED=true nlm-proxy serve openai --port 8080`

Test: Make a request and verify record in database.

**Step 3: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(tracking): track non-streaming requests"
```

---

## Task 9: Track Routing Decisions

**Files:**
- Modify: `src/nlm_proxy/openai/router.py`
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Modify SmartRouter.route() to accept record**

In `src/nlm_proxy/openai/router.py`:

```python
async def route(self, query: str, record=None) -> RoutingDecision:
    """Route request to appropriate handler."""
    routing_start = time.time()

    # ... existing logic ...

    # Record routing decision
    if record:
        record.request_type = decision.request_type.value
        record.selected_notebook_id = decision.notebook_id
        record.selected_notebook_title = getattr(decision, 'notebook_title', None)
        record.routing_reasoning = decision.reasoning
        record.routing_latency_ms = int((time.time() - routing_start) * 1000)

    return decision
```

**Step 2: Pass record from handle_smart_routing()**

In `handle_smart_routing()`:

```python
decision = await router.route(query, record=record)
```

**Step 3: Commit**

```bash
git add src/nlm_proxy/openai/router.py src/nlm_proxy/openai/server.py
git commit -m "feat(tracking): capture routing decisions"
```

---

## Task 10: Track Streaming Responses

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Create async wrapper for streaming with tracking**

```python
async def stream_response_with_tracking(
    stream_generator,
    record: RequestRecord,
    tracker,
    start_time: float
):
    """Wrap stream generator to capture response for tracking."""
    full_content = []

    try:
        async for chunk_data in stream_generator:
            # Extract content from SSE data
            if isinstance(chunk_data, str) and chunk_data.startswith("data: "):
                if not chunk_data.startswith("data: [DONE]"):
                    try:
                        chunk_json = json.loads(chunk_data[6:])
                        if chunk_json.get("choices"):
                            delta = chunk_json["choices"][0].get("delta", {})
                            if delta.get("content"):
                                full_content.append(delta["content"])
                    except json.JSONDecodeError:
                        pass
            yield chunk_data

        # Stream completed
        if record:
            record.response_text = "".join(full_content)
            record.response_length = len(record.response_text)
            record.finish_reason = "stop"

    except Exception as e:
        if record:
            record.error_type = type(e).__name__
            record.error_message = str(e)[:500]
        raise
    finally:
        if record:
            record.latency_ms = int((time.time() - start_time) * 1000)
            await tracker.finish_request(record)
```

**Step 2: Use wrapper in chat_completions**

```python
if request.stream:
    if record and app.state.tracker:
        return StreamingResponse(
            stream_response_with_tracking(
                stream_response(client, request.model, query_text, request, chat_id),
                record,
                app.state.tracker,
                start_time
            ),
            media_type="text/event-stream"
        )
    else:
        return StreamingResponse(
            stream_response(client, request.model, query_text, request, chat_id),
            media_type="text/event-stream"
        )
```

**Step 3: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(tracking): capture streaming responses"
```

---

## Task 11: Add Documentation

**Files:**
- Create: `.claude/memory/tracking.md`

**Step 1: Document the tracking feature**

```markdown
# Request Tracking

Track request/response pairs for post-hoc analysis and debugging.

## Enable Tracking

```bash
export NLM_PROXY_TRACKING_ENABLED=true
nlm-proxy serve openai --port 8080
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NLM_PROXY_TRACKING_ENABLED` | `false` | Enable tracking |
| `NLM_PROXY_TRACKING_DB_URL` | `sqlite+aiosqlite:///~/.nlm-proxy/tracking.db` | Database URL |
| `NLM_PROXY_TRACKING_BATCH_SIZE` | `10` | Records per batch write |
| `NLM_PROXY_TRACKING_FLUSH_INTERVAL` | `1.0` | Seconds between flushes |
| `NLM_PROXY_TRACKING_MAX_RESPONSE_LENGTH` | `10000` | Truncation limit |

## Database Backends

**SQLite (default):**
```bash
NLM_PROXY_TRACKING_DB_URL="sqlite+aiosqlite:///~/.nlm-proxy/tracking.db"
```

**PostgreSQL:**
```bash
pip install asyncpg
NLM_PROXY_TRACKING_DB_URL="postgresql+asyncpg://user:pass@localhost/nlm_tracking"
```

**MySQL:**
```bash
pip install aiomysql
NLM_PROXY_TRACKING_DB_URL="mysql+aiomysql://user:pass@localhost/nlm_tracking"
```

## Query Examples

```bash
# Recent requests
sqlite3 ~/.nlm-proxy/tracking.db "
SELECT datetime(created_at, 'unixepoch', 'localtime') as time,
       substr(query_text, 1, 50) as query,
       request_type, selected_notebook_title, latency_ms
FROM requests ORDER BY created_at DESC LIMIT 20;"

# Requests per notebook
sqlite3 ~/.nlm-proxy/tracking.db "
SELECT selected_notebook_title, COUNT(*) as count, AVG(latency_ms) as avg_latency
FROM requests WHERE request_type = 'notebooklm'
GROUP BY selected_notebook_id ORDER BY count DESC;"

# Latency percentiles
sqlite3 ~/.nlm-proxy/tracking.db "
SELECT
    MIN(latency_ms) as p0,
    MAX(latency_ms) as p100,
    AVG(latency_ms) as avg
FROM requests;"
```
```

**Step 2: Commit**

```bash
git add .claude/memory/tracking.md
git commit -m "docs: add tracking documentation"
```

---

## Verification Checklist

- [ ] `NLM_PROXY_TRACKING_ENABLED=true` creates database
- [ ] Non-streaming requests tracked with response
- [ ] Streaming requests tracked with accumulated response
- [ ] Routing decisions captured (request_type, selected_notebook, reasoning)
- [ ] Latency measured (total and routing separately)
- [ ] Errors captured with type and message
- [ ] Batch writes work (check logs for "[TRACKING] Wrote N records")
- [ ] Disabled tracking has no overhead
- [ ] Can swap to PostgreSQL by changing `DB_URL` (optional test)

## Swapping to PostgreSQL Example

```bash
# Install driver
pip install asyncpg

# Configure
export NLM_PROXY_TRACKING_DB_URL="postgresql+asyncpg://user:pass@localhost/tracking"
export NLM_PROXY_TRACKING_ENABLED=true

# Run - tables auto-created
nlm-proxy serve openai --port 8080
```
