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


def test_on_sources_changed_callback_fires():
    """Callback should fire when sources change."""
    from nlm_proxy.openai.notebook_cache import NotebookCache, SourceInfo
    from unittest.mock import AsyncMock, MagicMock

    callback = MagicMock()
    mock_client = MagicMock()
    mock_client.list_notebooks = AsyncMock(return_value=[])

    cache = NotebookCache(
        nlm_client=mock_client, ttl_seconds=3600,
        on_sources_changed=callback,
    )

    # First set — no previous sources, no callback
    cache.set("nb-1", "Test", "Summary", [], sources=[
        SourceInfo(id="src-1", title="Doc 1", source_type="pdf"),
    ])
    callback.assert_not_called()

    # Same sources — no callback
    cache.set("nb-1", "Test", "Summary", [], sources=[
        SourceInfo(id="src-1", title="Doc 1", source_type="pdf"),
    ])
    callback.assert_not_called()

    # Different sources — callback fires
    cache.set("nb-1", "Test", "Summary", [], sources=[
        SourceInfo(id="src-1", title="Doc 1", source_type="pdf"),
        SourceInfo(id="src-2", title="Doc 2", source_type="url"),
    ])
    callback.assert_called_once_with("nb-1")


def test_on_sources_changed_no_callback():
    """Without callback, source changes should not crash."""
    from nlm_proxy.openai.notebook_cache import NotebookCache, SourceInfo
    from unittest.mock import AsyncMock, MagicMock

    mock_client = MagicMock()
    mock_client.list_notebooks = AsyncMock(return_value=[])

    cache = NotebookCache(nlm_client=mock_client, ttl_seconds=3600)  # No callback

    cache.set("nb-1", "Test", "Summary", [], sources=[
        SourceInfo(id="src-1", title="Doc 1", source_type="pdf"),
    ])
    # Change sources — should not crash
    cache.set("nb-1", "Test", "Summary", [], sources=[
        SourceInfo(id="src-2", title="Doc 2", source_type="url"),
    ])
