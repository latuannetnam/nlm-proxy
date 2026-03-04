"""Root conftest for tests — reset shared app state between tests."""

import pytest


@pytest.fixture(autouse=True)
def _reset_openai_app_state():
    """Reset the OpenAI server app.state between tests.

    Because the FastAPI `app` is a module-level singleton, state set by one
    test can leak into subsequent tests. This fixture ensures clean state.
    """
    from nlm_proxy.openai.server import app

    # Save original state
    original_agent_core = getattr(app.state, "agent_core", None)
    original_response_cache = getattr(app.state, "response_cache", None)

    yield

    # Restore to None (clean state) after each test
    app.state.agent_core = original_agent_core
    app.state.response_cache = original_response_cache
