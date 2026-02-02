"""Tests for notebook summary cache."""

import time
import pytest


def test_notebook_cache_set_and_get():
    """Test basic set and get operations."""
    from nlm_proxy.openai.notebook_cache import NotebookCache
    from unittest.mock import AsyncMock, MagicMock

    # Create mock client
    mock_client = MagicMock()
    mock_client.list_notebooks = AsyncMock(return_value=[])

    cache = NotebookCache(nlm_client=mock_client, ttl_seconds=3600)
    cache.set(
        notebook_id="nb-123",
        title="Research Notes",
        summary="Notes about AI research",
        topics=["AI", "ML"]
    )

    info = cache.get("nb-123")
    assert info is not None
    assert info.id == "nb-123"
    assert info.title == "Research Notes"
    assert info.summary == "Notes about AI research"
    assert info.topics == ["AI", "ML"]


def test_notebook_cache_expiration():
    """Test that entries expire after TTL."""
    from nlm_proxy.openai.notebook_cache import NotebookCache
    from unittest.mock import AsyncMock, MagicMock

    # Create mock client
    mock_client = MagicMock()
    mock_client.list_notebooks = AsyncMock(return_value=[])

    cache = NotebookCache(nlm_client=mock_client, ttl_seconds=0.1)  # 100ms TTL
    cache.set("nb-123", "Test", "Summary", [])

    assert cache.get("nb-123") is not None
    time.sleep(0.15)
    assert cache.get("nb-123") is None


def test_notebook_cache_get_all():
    """Test getting all non-expired entries."""
    from nlm_proxy.openai.notebook_cache import NotebookCache
    from unittest.mock import AsyncMock, MagicMock

    # Create mock client
    mock_client = MagicMock()
    mock_client.list_notebooks = AsyncMock(return_value=[])

    cache = NotebookCache(nlm_client=mock_client, ttl_seconds=3600)
    cache.set("nb-1", "First", "Summary 1", ["topic1"])
    cache.set("nb-2", "Second", "Summary 2", ["topic2"])

    all_notebooks = cache.get_all()
    assert len(all_notebooks) == 2
    ids = {nb.id for nb in all_notebooks}
    assert ids == {"nb-1", "nb-2"}


def test_notebook_cache_clear():
    """Test clearing the cache."""
    from nlm_proxy.openai.notebook_cache import NotebookCache
    from unittest.mock import AsyncMock, MagicMock

    # Create mock client
    mock_client = MagicMock()
    mock_client.list_notebooks = AsyncMock(return_value=[])

    cache = NotebookCache(nlm_client=mock_client, ttl_seconds=3600)
    cache.set("nb-123", "Test", "Summary", [])
    cache.clear()

    assert cache.get("nb-123") is None
    assert cache.get_all() == []
