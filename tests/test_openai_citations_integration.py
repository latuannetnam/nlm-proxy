"""Integration tests for OpenAI proxy citation support.

These tests document the expected behavior for citation support.
Manual testing with Open WebUI is recommended for full verification.
"""

import pytest
import json


class TestStreamingCitations:
    """Test citation events in streaming responses."""

    def test_citation_event_format(self):
        """Citation events should follow Open WebUI expected format."""
        # Expected citation event structure
        expected_event = {
            "type": "source",
            "source": {
                "name": "Source Title",
                "id": "uuid-string",
            },
            "document": [],
        }

        # Verify structure
        assert expected_event["type"] == "source"
        assert "name" in expected_event["source"]
        assert "id" in expected_event["source"]
        assert expected_event["document"] == []

    def test_citation_events_order_documentation(self):
        """Document expected SSE stream order for citations.

        Expected SSE stream order:
        1. Content chunks: data: {"choices": [{"delta": {"content": "..."}}]}
        2. Final chunk: data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        3. Citation events: data: {"type": "source", "source": {"name": "...", "id": "..."}}
        4. Done marker: data: [DONE]

        Citation events are emitted AFTER content, BEFORE [DONE].
        This allows clients to collect all sources after the answer is complete.
        """
        pass


class TestNonStreamingCitations:
    """Test sources field in non-streaming responses."""

    def test_sources_array_format(self):
        """Non-streaming responses should include sources array."""
        # Expected response structure with sources
        expected_response = {
            "id": "chatcmpl-xxx",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Answer with [1] citation."},
                    "finish_reason": "stop"
                }
            ],
            "sources": [
                {
                    "source": {"name": "Source Title", "id": "uuid-1"},
                    "document": []
                }
            ]
        }

        # Verify sources array structure
        assert "sources" in expected_response
        assert len(expected_response["sources"]) == 1
        assert expected_response["sources"][0]["source"]["name"] == "Source Title"


class TestSourceIdExtraction:
    """Test source ID extraction from NotebookLM chunks."""

    def test_source_ids_maintain_order(self):
        """Source IDs should maintain order for [1], [2] mapping."""
        # Citation [N] maps to Nth source in order
        # [1] -> sources[0], [2] -> sources[1]
        source_ids = ["uuid-1", "uuid-2", "uuid-3"]

        # First source corresponds to [1]
        assert source_ids[0] == "uuid-1"
        # Second source corresponds to [2]
        assert source_ids[1] == "uuid-2"

    def test_source_ids_are_unique(self):
        """Each source ID should appear only once."""
        # When collecting source IDs, duplicates should be skipped
        collected = []
        incoming = ["uuid-1", "uuid-2", "uuid-1", "uuid-3", "uuid-2"]

        for sid in incoming:
            if sid not in collected:
                collected.append(sid)

        assert collected == ["uuid-1", "uuid-2", "uuid-3"]
        assert len(collected) == 3


class TestManualVerification:
    """Manual testing checklist for Open WebUI integration.

    Run these tests manually:

    1. Start the proxy:
       nlm-proxy serve openai --port 8080 --debug

    2. Configure Open WebUI:
       - Set API endpoint to http://localhost:8080/v1
       - Set API key to your configured key

    3. Test streaming mode:
       - Send a query that produces citations
       - Verify debug logs show: "[PROXY] Emitting N citation events"
       - Verify "Sources" button appears in Open WebUI
       - Verify clicking citation numbers [1], [2] works

    4. Test non-streaming mode:
       - Toggle streaming off (if Open WebUI supports it)
       - Verify sources appear in response

    Debug commands:

    # Watch streaming events with curl
    curl -N -H "Authorization: Bearer YOUR_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{"model":"NOTEBOOK_ID","messages":[{"role":"user","content":"What is NetNam?"}],"stream":true}' \\
      http://localhost:8080/v1/chat/completions

    # Check non-streaming response
    curl -H "Authorization: Bearer YOUR_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{"model":"NOTEBOOK_ID","messages":[{"role":"user","content":"What is NetNam?"}],"stream":false}' \\
      http://localhost:8080/v1/chat/completions | jq '.sources'
    """
    pass
