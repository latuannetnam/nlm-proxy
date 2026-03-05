"""Tests for tracing spans in the routing graph nodes.

Verifies that @record_span decorators on classify_node and
select_notebook_node produce correct spans with expected attributes.
Uses a custom TracerProvider per test to capture spans.
"""

import pytest
from typing import Sequence
from unittest.mock import AsyncMock, MagicMock

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult


class _InMemoryExporter(SpanExporter):
    """Simple in-memory span exporter for testing."""

    def __init__(self):
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def get_finished_spans(self) -> list[ReadableSpan]:
        return list(self.spans)


# Add our exporter to the existing global TracerProvider.
# OTel only allows set_tracer_provider once per process ─ if another
# test module already set it, we hook into that provider instead.
_exporter = _InMemoryExporter()
_provider = trace.get_tracer_provider()
if not isinstance(_provider, TracerProvider):
    # No SDK provider set yet – create one
    _provider = TracerProvider()
    trace.set_tracer_provider(_provider)
_provider.add_span_processor(SimpleSpanProcessor(_exporter))


@pytest.fixture(autouse=True)
def _clear_spans():
    """Clear collected spans before each test."""
    _exporter.spans.clear()
    yield


def _mock_llm_response(content: str):
    msg = MagicMock()
    msg.content = content
    return msg


def _mock_notebook_cache():
    nb1 = MagicMock()
    nb1.id = "nb-aaa"
    nb1.title = "ML Research"
    nb1.summary = "Machine learning papers"
    nb1.topics = ["ML"]
    nb1.source_count = 3
    nb1.source_types = ["pdf"]
    nb1.source_titles = ["Paper A", "Paper B"]

    nb2 = MagicMock()
    nb2.id = "nb-bbb"
    nb2.title = "History Notes"
    nb2.summary = "World history notes"
    nb2.topics = ["History"]
    nb2.source_count = 2
    nb2.source_types = ["doc"]
    nb2.source_titles = ["Book 1"]

    cache = MagicMock()
    cache.get_all.return_value = [nb1, nb2]
    return cache


def _routing_settings():
    settings = MagicMock()
    settings.max_source_titles = 15
    settings.source_descriptions_enabled = False
    settings.source_max_keywords = 5
    settings.source_summary_max_chars = 200
    settings.source_descriptions_max_sources = 10
    return settings


# --- classify_node tests ---


@pytest.mark.asyncio
async def test_classify_creates_span_with_attributes():
    """classify_node creates smart_router.classify span with classification_result."""
    from nlm_proxy.core.routing_graph import classify_node

    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response("NOTEBOOKLM"))

    state = {"query": "What is ML?", "allowed_notebooks": None}
    result = await classify_node(state, chat_model=chat_model, notebook_cache=_mock_notebook_cache())

    assert result["request_type"] == "notebooklm"

    spans = _exporter.get_finished_spans()
    classify_spans = [s for s in spans if s.name == "smart_router.classify"]
    assert len(classify_spans) == 1

    attrs = dict(classify_spans[0].attributes)
    assert attrs["classification_result"] == "NOTEBOOKLM"


@pytest.mark.asyncio
async def test_classify_records_llm_model_attribute():
    """classify_node records llm_model attribute when model_name is available."""
    from nlm_proxy.core.routing_graph import classify_node

    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response("LLM_TASK"))
    chat_model.model_name = "gpt-4o-mini"

    state = {"query": "Write a poem", "allowed_notebooks": None}
    result = await classify_node(state, chat_model=chat_model, notebook_cache=_mock_notebook_cache())

    assert result["request_type"] == "llm_task"

    spans = _exporter.get_finished_spans()
    classify_spans = [s for s in spans if s.name == "smart_router.classify"]
    attrs = dict(classify_spans[0].attributes)
    assert attrs["llm_model"] == "gpt-4o-mini"
    assert attrs["classification_result"] == "LLM_TASK"


@pytest.mark.asyncio
async def test_classify_llm_model_fallback_to_class_name():
    """Without model_name attr, llm_model falls back to class name."""
    from nlm_proxy.core.routing_graph import classify_node

    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response("NOTEBOOKLM"))
    # Remove the model_name attribute so getattr returns None
    del chat_model.model_name

    state = {"query": "What is ML?", "allowed_notebooks": None}
    await classify_node(state, chat_model=chat_model, notebook_cache=_mock_notebook_cache())

    spans = _exporter.get_finished_spans()
    classify_spans = [s for s in spans if s.name == "smart_router.classify"]
    attrs = dict(classify_spans[0].attributes)
    assert "llm_model" in attrs
    assert attrs["llm_model"]  # not empty


@pytest.mark.asyncio
async def test_classify_span_records_error_on_exception():
    """classify_node records error status on span when LLM call fails."""
    from nlm_proxy.core.routing_graph import classify_node
    from opentelemetry.trace import StatusCode

    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("Connection error"))

    state = {"query": "test", "allowed_notebooks": None}
    with pytest.raises(RuntimeError, match="Connection error"):
        await classify_node(state, chat_model=chat_model, notebook_cache=_mock_notebook_cache())

    spans = _exporter.get_finished_spans()
    classify_spans = [s for s in spans if s.name == "smart_router.classify"]
    assert len(classify_spans) == 1
    assert classify_spans[0].status.status_code == StatusCode.ERROR


# --- select_notebook_node tests ---


@pytest.mark.asyncio
async def test_select_notebook_creates_span_with_attributes():
    """select_notebook_node creates span with selection attributes."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response("nb-aaa"))

    state = {"query": "What is ML?", "request_type": "notebooklm", "allowed_notebooks": None}

    result = await select_notebook_node(
        state, chat_model=chat_model, notebook_cache=_mock_notebook_cache(),
        routing_settings=_routing_settings(),
    )

    assert result["notebook_id"] == "nb-aaa"

    spans = _exporter.get_finished_spans()
    select_spans = [s for s in spans if s.name == "smart_router.select_notebook"]
    assert len(select_spans) == 1

    attrs = dict(select_spans[0].attributes)
    assert attrs["candidates_count"] == 2
    assert attrs["selected_notebook_id"] == "nb-aaa"
    assert attrs["selected_notebook_title"] == "ML Research"
    assert attrs["acl_filter_applied"] is False


@pytest.mark.asyncio
async def test_select_notebook_acl_filter_attributes():
    """select_notebook_node records ACL filter attributes."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response("nb-aaa"))

    state = {
        "query": "What is ML?", "request_type": "notebooklm",
        "allowed_notebooks": ["nb-aaa"],
    }

    result = await select_notebook_node(
        state, chat_model=chat_model, notebook_cache=_mock_notebook_cache(),
        routing_settings=_routing_settings(),
    )

    spans = _exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert attrs["acl_filter_applied"] is True
    assert attrs["acl_allowed_count"] == 1
    assert attrs["acl_matched_count"] == 1


@pytest.mark.asyncio
async def test_select_notebook_fallback_attribute():
    """Fallback selection sets selection_fallback=True attribute."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    chat_model = AsyncMock()
    # Return something that doesn't match any notebook ID
    chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response("no-match-id"))

    state = {"query": "test", "request_type": "notebooklm", "allowed_notebooks": None}

    result = await select_notebook_node(
        state, chat_model=chat_model, notebook_cache=_mock_notebook_cache(),
        routing_settings=_routing_settings(),
    )

    # Should fallback to first notebook
    assert result["notebook_id"] == "nb-aaa"

    spans = _exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert attrs["selection_fallback"] is True
