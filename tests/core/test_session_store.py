"""Tests for SessionStore (core/session.py)."""

import time
import pytest
from nlm_proxy.core.session import SessionStore


class TestSessionStore:
    """Test thread-safe session store with TTL."""

    def test_set_and_get(self):
        """Store → retrieve → correct conversation_id."""
        store = SessionStore(ttl_seconds=3600)
        try:
            store.set("chat-1", "conv-abc")
            assert store.get("chat-1") == "conv-abc"
        finally:
            store.shutdown()

    def test_get_expired_returns_none(self):
        """TTL expired → returns None."""
        store = SessionStore(ttl_seconds=1)
        try:
            store.set("chat-1", "conv-abc")
            time.sleep(1.1)
            assert store.get("chat-1") is None
        finally:
            store.shutdown()

    def test_get_nonexistent_returns_none(self):
        """Non-existent chat_id → returns None."""
        store = SessionStore(ttl_seconds=3600)
        try:
            assert store.get("nonexistent") is None
        finally:
            store.shutdown()

    def test_delete_returns_true_on_existing(self):
        """Delete existing → True."""
        store = SessionStore(ttl_seconds=3600)
        try:
            store.set("chat-1", "conv-abc")
            assert store.delete("chat-1") is True
            assert store.get("chat-1") is None
        finally:
            store.shutdown()

    def test_delete_returns_false_on_missing(self):
        """Delete non-existent → False."""
        store = SessionStore(ttl_seconds=3600)
        try:
            assert store.delete("nonexistent") is False
        finally:
            store.shutdown()

    def test_list_all(self):
        """Multiple sessions → all listed with metadata."""
        store = SessionStore(ttl_seconds=3600)
        try:
            store.set("chat-1", "conv-a")
            store.set("chat-2", "conv-b")
            sessions = store.list_all()
            assert len(sessions) == 2
            assert sessions["chat-1"]["conversation_id"] == "conv-a"
            assert sessions["chat-2"]["conversation_id"] == "conv-b"
            assert "age_seconds" in sessions["chat-1"]
            assert "expires_in_seconds" in sessions["chat-1"]
        finally:
            store.shutdown()

    def test_cleanup_expired(self):
        """Mix of fresh/expired → only expired removed."""
        store = SessionStore(ttl_seconds=1)
        try:
            store.set("old", "conv-old")
            time.sleep(1.1)
            store.set("new", "conv-new")
            removed = store.cleanup_expired()
            assert removed == 1
            assert store.get("old") is None
            assert store.get("new") == "conv-new"
        finally:
            store.shutdown()

    def test_get_stats(self):
        """Returns total_sessions, oldest_session_age_seconds."""
        store = SessionStore(ttl_seconds=3600)
        try:
            store.set("chat-1", "conv-a")
            store.set("chat-2", "conv-b")
            stats = store.get_stats()
            assert stats["total_sessions"] == 2
            assert stats["ttl_seconds"] == 3600
            assert "oldest_session_age_seconds" in stats
        finally:
            store.shutdown()
