"""Root conftest for tests — reset shared app state between tests."""

import pytest


@pytest.fixture(autouse=True)
def _reset_openai_app_state():
    """Reset the OpenAI server app.state between tests."""
    from nlm_proxy.openai.server import app

    original_agent_core = getattr(app.state, "agent_core", None)
    original_response_cache = getattr(app.state, "response_cache", None)
    original_session_store = getattr(app.state, "session_store", None)

    yield

    app.state.agent_core = original_agent_core
    app.state.response_cache = original_response_cache
    app.state.session_store = original_session_store
