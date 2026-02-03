"""Tests for source metadata extraction from notebook data."""

import pytest


class TestExtractSourceMetadata:
    """Test _extract_source_metadata helper function."""

    def test_extracts_source_titles_and_ids(self):
        """Should extract source ID -> metadata mapping."""
        from nlm_proxy.openai.server import _extract_source_metadata

        notebook_data = {
            "sources": [
                {"id": "uuid-1", "title": "NetNam Company Profile", "type": "pdf"},
                {"id": "uuid-2", "title": "Vietnam ISP History", "type": "web_page", "url": "https://example.com"},
            ]
        }

        result = _extract_source_metadata(notebook_data)

        assert result == {
            "uuid-1": {"title": "NetNam Company Profile", "type": "pdf", "url": None},
            "uuid-2": {"title": "Vietnam ISP History", "type": "web_page", "url": "https://example.com"},
        }

    def test_handles_empty_sources(self):
        """Should return empty dict when no sources."""
        from nlm_proxy.openai.server import _extract_source_metadata

        result = _extract_source_metadata({"sources": []})
        assert result == {}

        result = _extract_source_metadata({})
        assert result == {}

    def test_handles_missing_fields(self):
        """Should use defaults for missing fields."""
        from nlm_proxy.openai.server import _extract_source_metadata

        notebook_data = {
            "sources": [
                {"id": "uuid-1"},  # Only ID, no title/type
            ]
        }

        result = _extract_source_metadata(notebook_data)

        assert result == {
            "uuid-1": {"title": "Unknown Source", "type": None, "url": None},
        }
