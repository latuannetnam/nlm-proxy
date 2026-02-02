"""Tests for smart router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_router_classify_notebooklm():
    """Test classification of NotebookLM queries."""
    from nlm_proxy.openai.router import SmartRouter, RequestType

    mock_nlm_client = MagicMock()

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="notebooklm")
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        result = await router.classify_request("What does my research notebook say about AI?")

        assert result == RequestType.NOTEBOOKLM


@pytest.mark.asyncio
async def test_router_classify_llm_task():
    """Test classification of LLM tasks."""
    from nlm_proxy.openai.router import SmartRouter, RequestType

    mock_nlm_client = MagicMock()

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="llm_task")
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        result = await router.classify_request("Summarize our conversation so far")

        assert result == RequestType.LLM_TASK


@pytest.mark.asyncio
async def test_router_route_decision():
    """Test full routing decision."""
    from nlm_proxy.openai.router import SmartRouter, RequestType

    mock_nlm_client = AsyncMock()
    mock_nlm_client.list_notebooks = AsyncMock(return_value=[
        MagicMock(id="nb-123", title="Research Notes", source_count=5)
    ])
    mock_nlm_client.get_notebook_summary = AsyncMock(return_value={
        "summary": "AI research notes",
        "suggested_topics": ["AI", "ML"]
    })

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        # First call: classify as notebooklm
        # Second call: select notebook
        mock_llm.complete = AsyncMock(side_effect=["notebooklm", "nb-123"])
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        decision = await router.route("What does my research say?")

        assert decision.request_type == RequestType.NOTEBOOKLM
        assert decision.notebook_id == "nb-123"
        assert "Research Notes" in decision.reasoning
