"""Tests for per-request ACL filtering in smart router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_select_notebook_no_acl_filter():
    """Test that all notebooks are considered when no ACL filter is provided."""
    from nlm_proxy.openai.router import SmartRouter
    from nlm_proxy.openai.notebook_cache import NotebookCache

    mock_nlm_client = MagicMock()
    mock_nlm_client.list_notebooks = AsyncMock(return_value=[])

    # Create mock cache with multiple notebooks
    mock_cache = NotebookCache(nlm_client=mock_nlm_client, ttl_seconds=3600)
    mock_cache.set("nb-1", "Notebook 1", "Summary 1", ["AI"])
    mock_cache.set("nb-2", "Notebook 2", "Summary 2", ["ML"])
    mock_cache.set("nb-3", "Notebook 3", "Summary 3", ["Data"])

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="nb-2")
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            notebook_cache=mock_cache,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        # Call without allowed_notebooks parameter (None)
        notebook_id, reasoning = await router.select_notebook("test query")

        # Should select from all 3 notebooks
        assert notebook_id == "nb-2"
        assert "Notebook 2" in reasoning

        # Verify LLM was called with all 3 notebooks
        assert mock_llm.complete.called
        call_args = mock_llm.complete.call_args[0][0]
        assert "nb-1" in call_args
        assert "nb-2" in call_args
        assert "nb-3" in call_args

        await router.close()


@pytest.mark.asyncio
async def test_select_notebook_with_acl_filter():
    """Test that only allowed notebooks are included in LLM selection prompt."""
    from nlm_proxy.openai.router import SmartRouter
    from nlm_proxy.openai.notebook_cache import NotebookCache

    mock_nlm_client = MagicMock()
    mock_nlm_client.list_notebooks = AsyncMock(return_value=[])

    # Create mock cache with multiple notebooks
    mock_cache = NotebookCache(nlm_client=mock_nlm_client, ttl_seconds=3600)
    mock_cache.set("nb-1", "Notebook 1", "Summary 1", ["AI"])
    mock_cache.set("nb-2", "Notebook 2", "Summary 2", ["ML"])
    mock_cache.set("nb-3", "Notebook 3", "Summary 3", ["Data"])

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="nb-2")
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            notebook_cache=mock_cache,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        # Call with ACL filter allowing only nb-2 and nb-3
        notebook_id, reasoning = await router.select_notebook(
            "test query",
            allowed_notebooks=["nb-2", "nb-3"]
        )

        # Should select from filtered notebooks
        assert notebook_id == "nb-2"
        assert "Notebook 2" in reasoning

        # Verify LLM was called with only allowed notebooks
        assert mock_llm.complete.called
        call_args = mock_llm.complete.call_args[0][0]
        assert "nb-1" not in call_args  # nb-1 filtered out
        assert "nb-2" in call_args
        assert "nb-3" in call_args

        await router.close()


@pytest.mark.asyncio
async def test_select_notebook_acl_filters_all():
    """Test that ACL filter returns error when no notebooks match."""
    from nlm_proxy.openai.router import SmartRouter
    from nlm_proxy.openai.notebook_cache import NotebookCache

    mock_nlm_client = MagicMock()
    mock_nlm_client.list_notebooks = AsyncMock(return_value=[])

    # Create mock cache with notebooks
    mock_cache = NotebookCache(nlm_client=mock_nlm_client, ttl_seconds=3600)
    mock_cache.set("nb-1", "Notebook 1", "Summary 1", ["AI"])
    mock_cache.set("nb-2", "Notebook 2", "Summary 2", ["ML"])

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            notebook_cache=mock_cache,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        # Call with ACL filter that matches no notebooks
        notebook_id, reasoning = await router.select_notebook(
            "test query",
            allowed_notebooks=["nb-999", "nb-888"]
        )

        # Should return None with error message
        assert notebook_id is None
        assert "No accessible notebooks" in reasoning

        # LLM should NOT be called (no notebooks to select from)
        assert not mock_llm.complete.called

        await router.close()


@pytest.mark.asyncio
async def test_select_notebook_empty_acl_list():
    """Test that empty ACL list returns error."""
    from nlm_proxy.openai.router import SmartRouter
    from nlm_proxy.openai.notebook_cache import NotebookCache

    mock_nlm_client = MagicMock()
    mock_nlm_client.list_notebooks = AsyncMock(return_value=[])

    # Create mock cache with notebooks
    mock_cache = NotebookCache(nlm_client=mock_nlm_client, ttl_seconds=3600)
    mock_cache.set("nb-1", "Notebook 1", "Summary 1", ["AI"])

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            notebook_cache=mock_cache,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        # Call with empty ACL list
        notebook_id, reasoning = await router.select_notebook(
            "test query",
            allowed_notebooks=[]
        )

        # Should return None with error message
        assert notebook_id is None
        assert "No accessible notebooks" in reasoning

        # LLM should NOT be called
        assert not mock_llm.complete.called

        await router.close()


@pytest.mark.asyncio
async def test_route_passes_acl_to_select_notebook():
    """Test that route() forwards allowed_notebooks to select_notebook()."""
    from nlm_proxy.openai.router import SmartRouter, RequestType
    from nlm_proxy.openai.notebook_cache import NotebookCache

    mock_nlm_client = MagicMock()
    mock_nlm_client.list_notebooks = AsyncMock(return_value=[])

    # Create mock cache
    mock_cache = NotebookCache(nlm_client=mock_nlm_client, ttl_seconds=3600)
    mock_cache.set("nb-1", "Notebook 1", "Summary 1", ["AI"])
    mock_cache.set("nb-2", "Notebook 2", "Summary 2", ["ML"])

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        # First call: classify as NOTEBOOKLM
        # Second call: select notebook
        mock_llm.complete = AsyncMock(side_effect=["notebooklm", "nb-1"])
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            notebook_cache=mock_cache,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        # Call route with ACL
        decision = await router.route("test query", allowed_notebooks=["nb-1"])

        # Should classify as NOTEBOOKLM and select nb-1
        assert decision.request_type == RequestType.NOTEBOOKLM
        assert decision.notebook_id == "nb-1"

        # Verify LLM was called twice (classify + select)
        assert mock_llm.complete.call_count == 2

        # Verify second call (select_notebook) only had nb-1
        select_call_args = mock_llm.complete.call_args_list[1][0][0]
        assert "nb-1" in select_call_args
        assert "nb-2" not in select_call_args  # Filtered out by ACL

        await router.close()


@pytest.mark.asyncio
async def test_route_llm_task_ignores_acl():
    """Test that LLM_TASK classification bypasses ACL filtering."""
    from nlm_proxy.openai.router import SmartRouter, RequestType
    from nlm_proxy.openai.notebook_cache import NotebookCache

    mock_nlm_client = MagicMock()
    mock_nlm_client.list_notebooks = AsyncMock(return_value=[])

    # Create mock cache
    mock_cache = NotebookCache(nlm_client=mock_nlm_client, ttl_seconds=3600)

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        # Classify as LLM_TASK
        mock_llm.complete = AsyncMock(return_value="llm_task")
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            notebook_cache=mock_cache,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        # Call route with ACL (should be ignored for LLM_TASK)
        decision = await router.route("Summarize this", allowed_notebooks=[])

        # Should classify as LLM_TASK
        assert decision.request_type == RequestType.LLM_TASK
        assert decision.notebook_id is None

        # Verify LLM was only called once (classify), not for select_notebook
        assert mock_llm.complete.call_count == 1

        await router.close()
