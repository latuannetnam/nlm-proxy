"""
Test suite for NotebookLM MCP Open WebUI Tool.

Run with:
    pytest test_tool.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Import the tool (adjust path as needed)
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from notebooklm_mcp_tool import Tools, MCPClientAdapter


class TestMCPClientAdapter:
    """Test MCP client adapter functionality."""
    
    @pytest.mark.asyncio
    async def test_session_initialization(self):
        """Test MCP session initialization."""
        adapter = MCPClientAdapter("http://localhost:9888", timeout=30.0)
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.headers.get.return_value = "test-session-id"
            mock_response.raise_for_status = MagicMock()
            
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post
            
            session_id = await adapter.initialize_session()
            
            assert session_id == "test-session-id"
            assert adapter.session_id == "test-session-id"
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of connection errors."""
        import httpx
        
        adapter = MCPClientAdapter("http://invalid-server:9999")
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = \
                httpx.ConnectError("Connection refused")
            
            with pytest.raises(ConnectionError, match="Cannot connect to MCP server"):
                await adapter.initialize_session()
    
    @pytest.mark.asyncio
    async def test_streaming_tool_call(self):
        """Test streaming tool call with progress events."""
        adapter = MCPClientAdapter("http://localhost:9888")
        adapter.session_id = "test-session-id"
        
        progress_events = []
        
        async def progress_callback(params):
            progress_events.append(params)
        
        # Mock SSE stream
        sse_events = [
            'data: {"method":"notifications/progress","params":{"message":"Step 1"}}\n\n',
            'data: {"method":"notifications/progress","params":{"message":"Step 2"}}\n\n',
            'data: {"result":{"answer":"Final answer"}}\n\n',
        ]
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            
            async def mock_aiter_text():
                for event in sse_events:
                    yield event
            
            mock_response.aiter_text = mock_aiter_text
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            
            mock_client.return_value.__aenter__.return_value.stream.return_value = mock_stream
            
            result = await adapter.call_tool_streaming(
                "test_tool",
                {"arg": "value"},
                on_progress=progress_callback
            )
            
            assert result == {"answer": "Final answer"}
            assert len(progress_events) == 2
            assert progress_events[0]["message"] == "Step 1"
            assert progress_events[1]["message"] == "Step 2"


class TestTools:
    """Test Open WebUI tool methods."""
    
    def test_tool_initialization(self):
        """Test tool initializes with correct defaults."""
        tool = Tools()
        
        assert tool.valves.mcp_server_url == "http://localhost:9888"
        assert tool.valves.timeout == 120.0
        assert tool.valves.enable_debug == False
        assert tool.citation == False
    
    def test_function_calling_mode_detection(self):
        """Test detection of Native vs Default mode."""
        tool = Tools()
        
        # Test Default mode (or no metadata)
        assert tool._detect_function_calling_mode(None) == False
        assert tool._detect_function_calling_mode({}) == False
        assert tool._detect_function_calling_mode({"params": {}}) == False
        assert tool._detect_function_calling_mode(
            {"params": {"function_calling": "default"}}
        ) == False
        
        # Test Native mode
        assert tool._detect_function_calling_mode(
            {"params": {"function_calling": "native"}}
        ) == True
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test health check with successful connection."""
        tool = Tools()
        
        with patch.object(MCPClientAdapter, 'initialize_session', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = "test-session-id"
            
            result = await tool.health_check()
            
            assert "✅ MCP server is healthy" in result
            assert "test-session-id" in result
            assert tool.valves.mcp_server_url in result
    
    @pytest.mark.asyncio
    async def test_health_check_connection_error(self):
        """Test health check with connection failure."""
        tool = Tools()
        
        with patch.object(MCPClientAdapter, 'initialize_session', new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = ConnectionError("Cannot connect")
            
            result = await tool.health_check()
            
            assert "❌ Cannot connect to MCP server" in result
            assert "Troubleshooting" in result
    
    @pytest.mark.asyncio
    async def test_notebook_list(self):
        """Test listing notebooks."""
        tool = Tools()
        
        mock_result = {
            "notebooks": [
                {"title": "Test Notebook", "id": "nb-123", "source_count": 5},
                {"title": "Another Notebook", "id": "nb-456", "source_count": 3},
            ]
        }
        
        with patch.object(MCPClientAdapter, 'initialize_session', new_callable=AsyncMock):
            with patch.object(MCPClientAdapter, 'call_tool', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = mock_result
                
                result = await tool.notebook_list(max_results=10)
                
                assert "Test Notebook" in result
                assert "nb-123" in result
                assert "5" in result
                assert "Another Notebook" in result
                assert "Showing 2 notebook" in result
    
    @pytest.mark.asyncio
    async def test_notebook_query_stream_with_emitter(self):
        """Test streaming query with event emitter."""
        tool = Tools()
        
        emitted_events = []
        
        async def mock_emitter(event):
            emitted_events.append(event)
        
        mock_result = {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "answer": "Test answer",
                    "conversation_id": "conv-123",
                    "thinking_steps": ["Step 1", "Step 2"]
                })
            }]
        }
        
        with patch.object(MCPClientAdapter, 'initialize_session', new_callable=AsyncMock):
            with patch.object(MCPClientAdapter, 'call_tool_streaming', new_callable=AsyncMock) as mock_stream:
                # Simulate streaming with progress callback
                async def mock_streaming(*args, **kwargs):
                    on_progress = kwargs.get('on_progress')
                    if on_progress:
                        await on_progress({"message": "🤔 Thinking..."})
                        await on_progress({"message": "💡 Receiving answer..."})
                    return mock_result
                
                mock_stream.side_effect = mock_streaming
                
                result = await tool.notebook_query_stream(
                    notebook_id="nb-123",
                    query="Test question",
                    __event_emitter__=mock_emitter,
                    __metadata__={"params": {"function_calling": "default"}}
                )
                
                # Check final answer
                assert "Test answer" in result
                assert "conv-123" in result
                
                # Check emitted events
                assert len(emitted_events) > 0
                
                # Should have status events
                status_events = [e for e in emitted_events if e["type"] == "status"]
                assert len(status_events) > 0
                
                # In Default mode, should have message events
                message_events = [e for e in emitted_events if e["type"] == "message"]
                assert len(message_events) > 0
    
    @pytest.mark.asyncio
    async def test_notebook_query_stream_native_mode_warning(self):
        """Test that Native mode triggers warning."""
        tool = Tools()
        
        emitted_events = []
        
        async def mock_emitter(event):
            emitted_events.append(event)
        
        mock_result = {
            "answer": "Test answer",
            "conversation_id": "conv-123"
        }
        
        with patch.object(MCPClientAdapter, 'initialize_session', new_callable=AsyncMock):
            with patch.object(MCPClientAdapter, 'call_tool_streaming', new_callable=AsyncMock) as mock_stream:
                mock_stream.return_value = mock_result
                
                await tool.notebook_query_stream(
                    notebook_id="nb-123",
                    query="Test question",
                    __event_emitter__=mock_emitter,
                    __metadata__={"params": {"function_calling": "native"}}
                )
                
                # Should have notification warning about Native mode
                notification_events = [
                    e for e in emitted_events 
                    if e["type"] == "notification"
                ]
                assert len(notification_events) > 0
                assert any("Native" in str(e) for e in notification_events)
    
    @pytest.mark.asyncio
    async def test_error_emission(self):
        """Test error event emission."""
        tool = Tools()
        
        emitted_events = []
        
        async def mock_emitter(event):
            emitted_events.append(event)
        
        error = RuntimeError("Test error")
        result = await tool._emit_error(mock_emitter, error, "Test context")
        
        assert "❌ Test context: Test error" in result
        
        # Check error event was emitted
        error_events = [e for e in emitted_events if e["type"] == "chat:message:error"]
        assert len(error_events) == 1


class TestIntegration:
    """Integration tests (require running MCP server)."""
    
    @pytest.mark.skip(reason="Requires running MCP server")
    @pytest.mark.asyncio
    async def test_real_health_check(self):
        """Test health check against real server."""
        tool = Tools()
        tool.valves.mcp_server_url = "http://localhost:9888"
        
        result = await tool.health_check()
        assert "healthy" in result.lower()
    
    @pytest.mark.skip(reason="Requires running MCP server and auth")
    @pytest.mark.asyncio
    async def test_real_notebook_list(self):
        """Test listing real notebooks."""
        tool = Tools()
        tool.valves.mcp_server_url = "http://localhost:9888"
        
        result = await tool.notebook_list()
        assert "Notebook" in result or "No notebooks" in result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
