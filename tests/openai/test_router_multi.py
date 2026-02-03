import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nlm_proxy.openai.router import SmartRouter, RequestType, NotebookRole, MultiRoutingDecision
import json
import os

@pytest.mark.asyncio
async def test_route_multi_notebooklm():
    mock_llm_client = AsyncMock()
    # Mock successful JSON response
    # The implementation will call this method
    mock_llm_client.chat_completion.return_value = json.dumps({
        "type": "notebooklm",
        "reasoning": "test reasoning",
        "notebooks": [
            {"id": "nb1", "role": "primary", "reason": "main", "title": "NB1"},
            {"id": "nb2", "role": "secondary", "reason": "supp", "title": "NB2"}
        ]
    })

    # Initialize with required args
    router = SmartRouter(
        nlm_client=MagicMock(),
        notebook_cache=MagicMock(),
        llm_base_url="http://mock",
        llm_api_key="mock",
        llm_model="mock"
    )
    # Inject mock client
    router.llm_client = mock_llm_client
    
    # Mock cache
    router.notebook_cache.get_all.return_value = [] # Content mocked in prompt logic effectively

    # Mock environment variable for settings if needed, or just rely on default
    # The test in prompt used settings.cross_notebook_max_secondary.
    # If implementation uses settings, we need to handle it. 
    # If implementation uses os.environ, we patch it.
    # Implementation instruction: "Add route_multi method... Get notebooks from cache... Load prompt... Call..."
    # It doesn't explicitly mention using settings for max_secondary, but typically it would.
    
    with patch.dict(os.environ, {"NLM_PROXY_CROSS_NOTEBOOK_MAX_SECONDARY": "2"}):
        decision = await router.route_multi("test query")

    assert decision.request_type == RequestType.NOTEBOOKLM
    assert decision.reasoning == "test reasoning"
    assert decision.primary_notebook.notebook_id == "nb1"
    assert len(decision.secondary_notebooks) == 1
    assert decision.secondary_notebooks[0].notebook_id == "nb2"

@pytest.mark.asyncio
async def test_route_multi_llm_task():
    mock_llm_client = AsyncMock()
    mock_llm_client.chat_completion.return_value = json.dumps({
        "type": "llm_task",
        "reasoning": "general query",
        "notebooks": []
    })

    router = SmartRouter(
        nlm_client=MagicMock(),
        notebook_cache=MagicMock(),
        llm_base_url="http://mock",
        llm_api_key="mock",
        llm_model="mock"
    )
    router.llm_client = mock_llm_client
    router.notebook_cache.get_all.return_value = []

    decision = await router.route_multi("hello")
    assert decision.request_type == RequestType.LLM_TASK
    assert decision.primary_notebook is None

@pytest.mark.asyncio
async def test_route_multi_fallback_on_json_error():
    mock_llm_client = AsyncMock()
    mock_llm_client.chat_completion.return_value = "INVALID JSON {{"

    router = SmartRouter(
        nlm_client=MagicMock(),
        notebook_cache=MagicMock(),
        llm_base_url="http://mock",
        llm_api_key="mock",
        llm_model="mock"
    )
    router.llm_client = mock_llm_client
    router.notebook_cache.get_all.return_value = []

    # Should fall back to LLM_TASK
    decision = await router.route_multi("query")
    assert decision.request_type == RequestType.LLM_TASK
    assert "Error parsing" in decision.reasoning
