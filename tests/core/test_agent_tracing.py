"""Tests for tracing spans in AgentCore.route() and server handlers.

Verifies:
- AgentCore.route() creates smart_router.route span with attributes
- Error fallback records error on span
- stream_smart_response() creates smart_router.handle_request span
- _handle_non_streaming() creates smart_router.handle_request span

Uses a module-level TracerProvider (since OTel only allows setting it once).
"""

import pytest
from typing import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from nlm_proxy.core.agent import AgentCore, RequestOptions, RoutingDecision


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


def _make_agent(routing_graph_result=None, routing_graph_error=None, notebooks=None):
    """Create AgentCore with mocked dependencies."""
    nlm_client = MagicMock()
    notebook_cache = MagicMock()
    response_cache = MagicMock()
    response_cache.lookup_global.return_value = (None, None)  # No pre-routing cache hit
    chat_model = MagicMock()

    if notebooks:
        notebook_cache.get_all.return_value = notebooks
    else:
        nb = MagicMock()
        nb.id = "nb-fallback"
        notebook_cache.get_all.return_value = [nb]

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        if routing_graph_error:
            mock_graph.ainvoke = AsyncMock(side_effect=routing_graph_error)
        else:
            mock_graph.ainvoke = AsyncMock(return_value=routing_graph_result or {
                "request_type": "notebooklm",
                "notebook_id": "nb-123",
                "reasoning": "Selected ML notebook",
            })
        mock_build.return_value = mock_graph
        agent = AgentCore(nlm_client, notebook_cache, response_cache, chat_model)

    return agent


# --- AgentCore.route() span tests ---


class TestRouteSpan:
    """Test that AgentCore.route() creates smart_router.route span."""

    @pytest.mark.asyncio
    async def test_route_creates_span(self):
        """AgentCore.route() creates smart_router.route span."""
        agent = _make_agent(routing_graph_result={
            "request_type": "notebooklm",
            "notebook_id": "nb-123",
            "reasoning": "Selected notebook",
        })

        decision = await agent.route("What is ML?", RequestOptions())
        assert decision.request_type == "notebooklm"

        spans = _exporter.get_finished_spans()
        route_spans = [s for s in spans if s.name == "smart_router.route"]
        assert len(route_spans) == 1

    @pytest.mark.asyncio
    async def test_route_span_has_attributes(self):
        """Span records request_type, notebook_id, routing_reasoning."""
        agent = _make_agent(routing_graph_result={
            "request_type": "notebooklm",
            "notebook_id": "nb-xyz",
            "reasoning": "Best match for query",
        })

        await agent.route("test query", RequestOptions())

        spans = _exporter.get_finished_spans()
        route_span = [s for s in spans if s.name == "smart_router.route"][0]
        attrs = dict(route_span.attributes)

        assert attrs["request_type"] == "notebooklm"
        assert attrs["notebook_id"] == "nb-xyz"
        assert attrs["routing_reasoning"] == "Best match for query"

    @pytest.mark.asyncio
    async def test_route_span_ok_status(self):
        """Successful route sets span status to OK."""
        agent = _make_agent()

        await agent.route("test", RequestOptions())

        spans = _exporter.get_finished_spans()
        route_span = [s for s in spans if s.name == "smart_router.route"][0]
        assert route_span.status.status_code == StatusCode.OK

    @pytest.mark.asyncio
    async def test_route_span_llm_task(self):
        """LLM_TASK routing records correct request_type."""
        agent = _make_agent(routing_graph_result={
            "request_type": "llm_task",
            "reasoning": "General task",
        })

        decision = await agent.route("Write a poem", RequestOptions())
        assert decision.request_type == "llm_task"

        spans = _exporter.get_finished_spans()
        route_span = [s for s in spans if s.name == "smart_router.route"][0]
        assert dict(route_span.attributes)["request_type"] == "llm_task"


# --- Error and fallback span tests ---


class TestRouteErrorSpan:
    """Test error recording on span during routing fallback."""

    @pytest.mark.asyncio
    async def test_route_error_records_on_span(self):
        """Routing error records error status and exception on span."""
        agent = _make_agent(routing_graph_error=RuntimeError("LLM connection failed"))

        # Should fallback, not raise
        decision = await agent.route("test", RequestOptions())
        assert decision.request_type == "notebooklm"
        assert "fallback" in decision.reasoning.lower()

        spans = _exporter.get_finished_spans()
        route_span = [s for s in spans if s.name == "smart_router.route"][0]
        assert route_span.status.status_code == StatusCode.ERROR

        # Check exception is recorded as span event
        events = route_span.events
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) >= 1

    @pytest.mark.asyncio
    async def test_route_fallback_records_attributes(self):
        """Fallback decision still records attributes on span."""
        agent = _make_agent(routing_graph_error=RuntimeError("fail"))

        await agent.route("test", RequestOptions())

        spans = _exporter.get_finished_spans()
        route_span = [s for s in spans if s.name == "smart_router.route"][0]
        attrs = dict(route_span.attributes)

        assert attrs["request_type"] == "notebooklm"
        assert attrs["notebook_id"] == "nb-fallback"
        assert "fallback" in attrs["routing_reasoning"].lower()

    @pytest.mark.asyncio
    async def test_route_error_no_notebooks_raises(self):
        """Routing error with no notebooks re-raises and records error."""
        nb_cache = MagicMock()
        nb_cache.get_all.return_value = []  # No notebooks

        response_cache = MagicMock()
        response_cache.lookup_global.return_value = (None, None)

        with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))
            mock_build.return_value = mock_graph
            agent = AgentCore(MagicMock(), nb_cache, response_cache, MagicMock())

        with pytest.raises(RuntimeError, match="fail"):
            await agent.route("test", RequestOptions())

        spans = _exporter.get_finished_spans()
        route_span = [s for s in spans if s.name == "smart_router.route"][0]
        assert route_span.status.status_code == StatusCode.ERROR


# --- Pre-routing cache hit doesn't create route span ---


class TestPreRoutingCacheSkipsSpan:
    """Pre-routing cache hit should NOT create a route span."""

    @pytest.mark.asyncio
    async def test_pre_routing_cache_hit_no_route_span(self):
        """Phase 0 cache hit returns before LangGraph, no route span created."""
        cached = MagicMock()
        cached.notebook_id = "nb-cached"
        response_cache = MagicMock()
        response_cache.lookup_global.return_value = (cached, "exact")

        with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
            mock_build.return_value = AsyncMock()
            agent = AgentCore(MagicMock(), MagicMock(), response_cache, MagicMock())

        decision = await agent.route("cached query", RequestOptions())
        assert decision.cache_result is not None

        spans = _exporter.get_finished_spans()
        route_spans = [s for s in spans if s.name == "smart_router.route"]
        assert len(route_spans) == 0  # No route span for cache hit


# --- Server span ownership tests ---


class TestServerSpanOwnership:
    """Test that streaming/non-streaming handlers create handle_request span."""

    @pytest.mark.asyncio
    async def test_stream_smart_response_creates_span(self):
        """stream_smart_response() creates smart_router.handle_request span."""
        from nlm_proxy.openai.server import stream_smart_response
        from nlm_proxy.openai.types import ChatCompletionRequest, Message

        agent = MagicMock(spec=AgentCore)
        agent.response_cache = None
        agent.save_conversation_id = MagicMock()

        chunk = MagicMock()
        chunk.content = "Hello"

        async def mock_astream(messages):
            yield chunk
        agent.chat_model = MagicMock()
        agent.chat_model.astream = mock_astream

        decision = RoutingDecision(request_type="llm_task", reasoning="LLM task")
        request = ChatCompletionRequest(
            model="knowledge-finder",
            messages=[Message(role="user", content="test")],
            stream=True,
        )

        async for _ in stream_smart_response(agent, decision, "test", request):
            pass

        spans = _exporter.get_finished_spans()
        handle_spans = [s for s in spans if s.name == "smart_router.handle_request"]
        assert len(handle_spans) == 1

    @pytest.mark.asyncio
    async def test_stream_span_has_response_attributes(self):
        """Streaming span records user_query, response_content, response_source."""
        from nlm_proxy.openai.server import stream_smart_response
        from nlm_proxy.openai.types import ChatCompletionRequest, Message
        from nlm_proxy.core.config import TracingSettings

        agent = MagicMock(spec=AgentCore)
        agent.response_cache = None
        agent.save_conversation_id = MagicMock()

        chunk = MagicMock()
        chunk.content = "Hello World"

        async def mock_astream(messages):
            yield chunk
        agent.chat_model = MagicMock()
        agent.chat_model.astream = mock_astream

        decision = RoutingDecision(request_type="llm_task", reasoning="LLM task")
        request = ChatCompletionRequest(
            model="knowledge-finder",
            messages=[Message(role="user", content="test query")],
            stream=True,
        )
        tracing_settings = TracingSettings(
            enabled=True,
            request_max_length=500,
            response_max_length=1000,
        )

        async for _ in stream_smart_response(
            agent, decision, "test query", request, tracing_settings=tracing_settings
        ):
            pass

        spans = _exporter.get_finished_spans()
        handle_span = [s for s in spans if s.name == "smart_router.handle_request"][0]
        attrs = dict(handle_span.attributes)

        assert attrs["user_query"] == "test query"
        assert "Hello World" in attrs["response_content"]
        assert attrs["response_source"] == "llm"

    @pytest.mark.asyncio
    async def test_non_streaming_creates_span(self):
        """_handle_non_streaming() creates smart_router.handle_request span."""
        from nlm_proxy.openai.server import _handle_non_streaming
        from nlm_proxy.openai.types import ChatCompletionRequest, Message

        mock_response = MagicMock()
        mock_response.content = "Generated poem"

        agent = MagicMock(spec=AgentCore)
        agent.chat_model = AsyncMock()
        agent.chat_model.ainvoke = AsyncMock(return_value=mock_response)
        agent.response_cache = None

        decision = RoutingDecision(request_type="llm_task", reasoning="LLM task")
        request = ChatCompletionRequest(
            model="knowledge-finder",
            messages=[Message(role="user", content="Write a poem")],
            stream=False,
        )

        await _handle_non_streaming(agent, decision, "Write a poem", request)

        spans = _exporter.get_finished_spans()
        handle_spans = [s for s in spans if s.name == "smart_router.handle_request"]
        assert len(handle_spans) == 1

    @pytest.mark.asyncio
    async def test_non_streaming_span_has_attributes(self):
        """Non-streaming span records response attributes."""
        from nlm_proxy.openai.server import _handle_non_streaming
        from nlm_proxy.openai.types import ChatCompletionRequest, Message
        from nlm_proxy.core.config import TracingSettings

        mock_response = MagicMock()
        mock_response.content = "The answer is 42"

        agent = MagicMock(spec=AgentCore)
        agent.chat_model = AsyncMock()
        agent.chat_model.ainvoke = AsyncMock(return_value=mock_response)
        agent.response_cache = None

        decision = RoutingDecision(request_type="llm_task", reasoning="LLM task")
        request = ChatCompletionRequest(
            model="knowledge-finder",
            messages=[Message(role="user", content="meaning of life")],
            stream=False,
        )
        tracing_settings = TracingSettings(
            enabled=True,
            request_max_length=500,
            response_max_length=1000,
        )

        await _handle_non_streaming(
            agent, decision, "meaning of life", request,
            tracing_settings=tracing_settings,
        )

        spans = _exporter.get_finished_spans()
        handle_span = [s for s in spans if s.name == "smart_router.handle_request"][0]
        attrs = dict(handle_span.attributes)

        assert attrs["user_query"] == "meaning of life"
        assert attrs["response_content"] == "The answer is 42"
        assert attrs["response_source"] == "llm"

    @pytest.mark.asyncio
    async def test_no_tracing_settings_no_error(self):
        """When tracing_settings=None, no span attribute errors."""
        from nlm_proxy.openai.server import stream_smart_response
        from nlm_proxy.openai.types import ChatCompletionRequest, Message

        agent = MagicMock(spec=AgentCore)
        agent.response_cache = None
        agent.save_conversation_id = MagicMock()
        agent.chat_model = MagicMock()

        async def mock_astream(messages):
            chunk = MagicMock()
            chunk.content = "ok"
            yield chunk
        agent.chat_model.astream = mock_astream

        decision = RoutingDecision(request_type="llm_task", reasoning="test")
        request = ChatCompletionRequest(
            model="knowledge-finder",
            messages=[Message(role="user", content="test")],
            stream=True,
        )

        # Should not raise even with tracing_settings=None
        async for _ in stream_smart_response(agent, decision, "test", request, tracing_settings=None):
            pass
