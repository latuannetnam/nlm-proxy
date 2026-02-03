"""Tests for citation extraction from NotebookLM streaming chunks."""

import pytest
from nlm_proxy.core.client import NotebookLMClient


class TestParseStreamChunk:
    """Test _parse_stream_chunk method extracts source IDs."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return NotebookLMClient(cookies="test", csrf_token="test", session_id="test")

    def test_extracts_source_ids_from_answer_chunk(self, client):
        """Should extract source UUIDs from position 2 of content array."""
        # Real chunk format from NotebookLM API
        chunk_json = '''[[
            "wrb.fr",
            null,
            "[[\\\"Answer text with [1] and [2] citations.\\\",null,[\\\"d458c47d-6b1e-463e-9cf4-47d716230f0a\\\",\\\"689bd968-0864-4019-92f8-ce61db5852b0\\\",3975011549],null,[null,null,null,null,1]]]"
        ]]'''

        result = client._parse_stream_chunk(chunk_json)

        assert result is not None
        assert result["type"] == "answer"
        assert result["text"] == "Answer text with [1] and [2] citations."
        assert result["source_ids"] == [
            "d458c47d-6b1e-463e-9cf4-47d716230f0a",
            "689bd968-0864-4019-92f8-ce61db5852b0",
        ]

    def test_extracts_source_ids_from_thinking_chunk(self, client):
        """Should extract source UUIDs from thinking chunks too."""
        chunk_json = '''[[
            "wrb.fr",
            null,
            "[[\\\"**Analyzing** the question...\\\",null,[\\\"abc12345-1234-5678-90ab-cdef01234567\\\",9876543210],null,[null,null,null,null,2]]]"
        ]]'''

        result = client._parse_stream_chunk(chunk_json)

        assert result is not None
        assert result["type"] == "thinking"
        assert result["source_ids"] == ["abc12345-1234-5678-90ab-cdef01234567"]

    def test_returns_empty_source_ids_when_none_present(self, client):
        """Should return empty list when no sources in chunk."""
        chunk_json = '''[[
            "wrb.fr",
            null,
            "[[\\\"Some text without citations.\\\",null,null,null,[null,null,null,null,1]]]"
        ]]'''

        result = client._parse_stream_chunk(chunk_json)

        assert result is not None
        assert result["source_ids"] == []

    def test_filters_out_timestamp_from_source_ids(self, client):
        """Should only include UUID strings, not the trailing timestamp."""
        chunk_json = '''[[
            "wrb.fr",
            null,
            "[[\\\"Text [1] with citation.\\\",null,[\\\"uuid-string\\\",3975011549],null,[null,null,null,null,1]]]"
        ]]'''

        result = client._parse_stream_chunk(chunk_json)

        assert result is not None
        # Should only include the UUID string, not the integer timestamp
        assert result["source_ids"] == ["uuid-string"]
