"""
title: NotebookLM MCP Client
author: NotebookLM MCP Contributors
author_url: https://github.com/jacob-bd/notebooklm-mcp
git_url: https://github.com/jacob-bd/notebooklm-mcp
description: Query NotebookLM notebooks with real-time streaming progress via MCP protocol
required_open_webui_version: 0.4.0
requirements: httpx
version: 1.0.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Callable, Optional, Any
import httpx
import json
import asyncio
import uuid


class MCPClientAdapter:
    """
    Adapter for MCP HTTP client with session management.
    
    Handles MCP protocol communication including:
    - Session initialization and handshake
    - Streaming Server-Sent Events (SSE) for real-time progress
    - Tool invocation with proper request/response handling
    """
    
    def __init__(self, base_url: str, timeout: float = 120.0):
        """
        Initialize MCP client adapter.
        
        Args:
            base_url: Base URL of MCP server (e.g., http://localhost:9888)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None
    
    async def initialize_session(self) -> str:
        """
        Initialize MCP session with handshake.
        
        Returns:
            Session ID from server
            
        Raises:
            httpx.HTTPError: If connection fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "openwebui-mcp-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json"
            }
            
            try:
                response = await client.post(
                    f"{self.base_url}/mcp",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                # Extract session ID from response headers
                self.session_id = response.headers.get("mcp-session-id")
                
                if not self.session_id:
                    # Some servers may not return session ID
                    self.session_id = str(uuid.uuid4())
                
                return self.session_id
                
            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Cannot connect to MCP server at {self.base_url}. "
                    f"Ensure server is running: uv run notebooklm-mcp --transport http --port 9888"
                ) from e
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"MCP server returned error {e.response.status_code}: {e.response.reason_phrase}"
                ) from e
    
    async def call_tool_streaming(
        self,
        tool_name: str,
        arguments: dict,
        on_progress: Optional[Callable[[dict], Any]] = None,
    ) -> dict:
        """
        Call an MCP tool with streaming SSE support.
        
        Handles progress notifications in real-time as they arrive via
        Server-Sent Events (SSE) protocol.
        
        Args:
            tool_name: Name of the MCP tool to invoke
            arguments: Tool arguments as dictionary
            on_progress: Optional async callback for progress notifications
            
        Returns:
            Final response dictionary from server
            
        Raises:
            RuntimeError: If session not initialized
            httpx.HTTPError: If request fails
        """
        if not self.session_id:
            await self.initialize_session()
        
        # Generate unique progress token for this request
        progress_token = str(uuid.uuid4())
        
        payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
                "_meta": {
                    "progressToken": progress_token
                }
            }
        }
        
        headers = {
            "Accept": "text/event-stream, application/json",
            "Content-Type": "application/json",
            "mcp-session-id": self.session_id
        }
        
        final_result = None
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/mcp",
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                
                buffer = ""
                
                # Process SSE stream
                async for chunk in response.aiter_text():
                    buffer += chunk
                    
                    # Process complete SSE events (terminated by \n\n) or data: lines
                    while "\n\n" in buffer or "\ndata: " in buffer:
                        # Find complete events
                        lines = buffer.split("\n")
                        processed_lines = 0
                        
                        for i, line in enumerate(lines):
                            line = line.strip()
                            
                            if line.startswith("data: "):
                                data_json = line[6:]  # Remove "data: " prefix
                                try:
                                    event_data = json.loads(data_json)
                                    
                                    # Handle progress notifications
                                    if event_data.get("method") == "notifications/progress":
                                        if on_progress:
                                            params = event_data.get("params", {})
                                            # Call progress callback
                                            if asyncio.iscoroutinefunction(on_progress):
                                                await on_progress(params)
                                            else:
                                                on_progress(params)
                                    
                                    # Handle final result or error
                                    elif "result" in event_data or "error" in event_data:
                                        if "error" in event_data:
                                            error = event_data["error"]
                                            raise RuntimeError(
                                                f"MCP tool error: {error.get('message', str(error))}"
                                            )
                                        final_result = event_data["result"]
                                        processed_lines = i + 1
                                        break
                                
                                except json.JSONDecodeError:
                                    # Skip malformed JSON
                                    pass
                            
                            processed_lines = i + 1
                        
                        # Remove processed lines from buffer
                        buffer = "\n".join(lines[processed_lines:])
                        
                        # If we found a result, we can stop
                        if final_result is not None:
                            break
                    
                    # If we already have a result, stop processing chunks
                    if final_result is not None:
                        break
                
                # Process any remaining buffer after stream ends
                if final_result is None and buffer.strip():
                    for line in buffer.split("\n"):
                        line = line.strip()
                        if line.startswith("data: "):
                            data_json = line[6:]
                            try:
                                event_data = json.loads(data_json)
                                if "result" in event_data:
                                    final_result = event_data["result"]
                                    break
                                elif "error" in event_data:
                                    error = event_data["error"]
                                    raise RuntimeError(
                                        f"MCP tool error: {error.get('message', str(error))}"
                                    )
                            except json.JSONDecodeError:
                                pass
        
        if final_result is None:
            raise RuntimeError("No result received from MCP server")
        
        # Extract actual content from MCP response structure
        # MCP returns: {"content": [...], "structuredContent": {...}, "isError": false}
        if isinstance(final_result, dict):
            # Check for structuredContent first (most reliable)
            if "structuredContent" in final_result:
                return final_result["structuredContent"]
            # Check for content array
            elif "content" in final_result:
                content_items = final_result["content"]
                if isinstance(content_items, list) and len(content_items) > 0:
                    # Extract text from first content item
                    first_item = content_items[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        return first_item["text"]
        
        return final_result
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """
        Call an MCP tool without streaming (synchronous response).
        
        Args:
            tool_name: Name of the MCP tool to invoke
            arguments: Tool arguments as dictionary
            
        Returns:
            Response dictionary from server
        """
        if not self.session_id:
            await self.initialize_session()
        
        payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": self.session_id
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            # Check if response is SSE (text/event-stream) or JSON
            content_type = response.headers.get("content-type", "")
            
            if "text/event-stream" in content_type:
                # Parse SSE response
                result = None
                for line in response.text.split("\n"):
                    line = line.strip()
                    if line.startswith("data: "):
                        data_json = line[6:]
                        try:
                            event_data = json.loads(data_json)
                            if "result" in event_data:
                                result = event_data["result"]
                            elif "error" in event_data:
                                error = event_data["error"]
                                raise RuntimeError(
                                    f"MCP tool error: {error.get('message', str(error))}"
                                )
                        except json.JSONDecodeError:
                            continue
                
                if result is None:
                    raise RuntimeError("No result received from MCP server")
                
                return result
            else:
                # Parse JSON response
                result = response.json()
                
                if "error" in result:
                    error = result["error"]
                    raise RuntimeError(
                        f"MCP tool error: {error.get('message', str(error))}"
                    )
                
                return result.get("result", {})


class Tools:
    """
    Open WebUI tool for querying NotebookLM via MCP protocol.
    
    Provides real-time streaming of AI thinking steps and answers.
    """
    
    class Valves(BaseModel):
        """
        Configuration settings for the NotebookLM MCP tool.
        
        These can be modified by administrators in the Open WebUI interface.
        """
        mcp_server_url: str = Field(
            default="http://localhost:9888",
            description="MCP Server base URL (HTTP transport). Start server with: uv run notebooklm-mcp --transport http --port 9888"
        )
        timeout: float = Field(
            default=120.0,
            description="Request timeout in seconds (for long-running queries)"
        )
        enable_debug: bool = Field(
            default=False,
            description="Enable debug logging for troubleshooting"
        )
    
    def __init__(self):
        """Initialize the tool with default configuration."""
        self.valves = self.Valves()
        self.citation = False  # Disable automatic citations
        
        # Tool description with usage instructions
        self.description = """
        Query NotebookLM notebooks with real-time AI thinking steps.
        
        ⚠️ IMPORTANT: This tool requires Default function calling mode for full streaming support.
        If thinking steps don't appear in real-time, check:
        Admin Panel > Settings > Models > Advanced Parameters > Function Calling = "Default"
        
        Features:
        - Real-time streaming of AI thinking steps
        - Progressive answer delivery
        - Support for follow-up questions via conversation_id
        - List and manage notebooks
        """
    
    async def _emit_error(
        self,
        __event_emitter__: Optional[Callable],
        error: Any,
        context: str
    ) -> str:
        """
        Emit error notification to Open WebUI UI.
        
        Args:
            __event_emitter__: Open WebUI event emitter
            error: Exception or error that occurred
            context: Context description for the error
            
        Returns:
            Error message string
        """
        error_msg = f"❌ {context}: {str(error)}"
        
        if __event_emitter__:
            await __event_emitter__({
                "type": "chat:message:error",
                "data": {"content": error_msg}
            })
        
        return error_msg
    
    def _detect_function_calling_mode(self, __metadata__: Optional[dict]) -> bool:
        """
        Detect if running in Native (agentic) function calling mode.
        
        Args:
            __metadata__: Metadata from Open WebUI
            
        Returns:
            True if Native mode, False if Default mode
        """
        if not __metadata__:
            return False
        
        return __metadata__.get("params", {}).get("function_calling") == "native"
    
    async def notebook_query_stream(
        self,
        notebook_id: str,
        query: str,
        source_ids: Optional[list[str]] = None,
        conversation_id: Optional[str] = None,
        __event_emitter__: Optional[Callable] = None,
        __metadata__: Optional[dict] = None,
    ) -> str:
        """
        Query NotebookLM with real-time streaming of thinking steps and answer.
        
        This is the primary tool for asking questions to NotebookLM notebooks.
        It provides real-time progress updates as the AI thinks through the query.
        
        Args:
            notebook_id: Notebook UUID (get from notebook_list)
            query: Question to ask the notebook
            source_ids: Optional list of specific source IDs to query (default: all sources)
            conversation_id: For follow-up questions in same conversation (optional)
            __event_emitter__: Injected by Open WebUI for progress updates
            __metadata__: Injected by Open WebUI (contains function_calling mode)
        
        Returns:
            Final answer from NotebookLM AI
        """
        is_native_mode = self._detect_function_calling_mode(__metadata__)
        
        # Warn about Native mode limitations
        # if is_native_mode and __event_emitter__:
        #     await __event_emitter__({
        #         "type": "notification",
        #         "data": {
        #             "content": "⚠️ Native function calling mode detected. Streaming may be limited. "
        #                        "Switch to Default mode for full real-time experience."
        #         }
        #     })
        
        try:
            # Initialize MCP client
            adapter = MCPClientAdapter(
                self.valves.mcp_server_url,
                self.valves.timeout
            )
            
            # Initial status
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "🔗 Connecting to AI...",
                        "done": False
                    }
                })
            
            await adapter.initialize_session()
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "📝 Sending query to AI...",
                        "done": False
                    }
                })
            
            thinking_steps: list[str] = []
            answer_parts: list[str] = []
            progress_count = 0
            final_conversation_id = conversation_id
            has_streamed_answer = False  # Track if we've streamed answer chunks
            full_streamed_content = ""  # Track complete content for final replacement
            
            # Progress callback for streaming updates
            async def handle_progress(params: dict):
                nonlocal progress_count, thinking_steps, answer_parts, has_streamed_answer, full_streamed_content
                progress_count += 1
                
                message = params.get("message", "Processing...")
                progress = params.get("progress", progress_count)
                total = params.get("total", progress_count + 5)
                
                if not __event_emitter__:
                    return
                
                # Always emit status updates (compatible with both modes)
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": message,
                        "done": False,
                        "hidden": False
                    }
                })
                
                # In Default mode, also stream content to message
                # if not is_native_mode:
                if True:
                    # Detect thinking vs answer based on emoji in message
                    if "🤔" in message:
                        # Extract thinking text
                        thinking_text = message.replace("🤔 ", "").strip()
                        thinking_steps.append(thinking_text)
                        
                        # Stream thinking step to chat only if debug is enabled
                        if self.valves.enable_debug:
                            await __event_emitter__({
                                "type": "message",  # Appends to message content
                                "data": {
                                    "content": f"\n\n**💭 Thinking:** {thinking_text}\n"
                                }
                            })
                    
                    elif message.startswith("💡"):
                        # Answer chunk - extract actual text after emoji
                        answer_text = message[1:].strip()  # Remove 💡 prefix
                        if answer_text:
                            # First answer chunk - add header
                            if not answer_parts:
                                full_streamed_content += "**📝 Answer:**\n\n"
                                await __event_emitter__({
                                    "type": "message",
                                    "data": {
                                        "content": "\n\n**📝 Answer:**\n\n"
                                    }
                                })
                            
                            # Track answer parts
                            answer_parts.append(answer_text)
                            has_streamed_answer = True  # Mark that we've streamed answer
                            
                            # Stream answer chunk to chat
                            # Calculate delta (new text since last chunk)
                            if len(answer_parts) == 1:
                                delta = answer_text
                            else:
                                # Answer chunks are cumulative, so calculate the delta
                                prev_len = len(answer_parts[-2]) if len(answer_parts) > 1 else 0
                                delta = answer_text[prev_len:] if len(answer_text) > prev_len else ""
                            
                            if delta:
                                full_streamed_content += delta
                                await __event_emitter__({
                                    "type": "message",
                                    "data": {
                                        "content": delta
                                    }
                                })
            
            # Call MCP tool with streaming
            result = await adapter.call_tool_streaming(
                tool_name="notebook_query_stream",
                arguments={
                    "notebook_id": notebook_id,
                    "query": query,
                    "source_ids": source_ids,
                    "conversation_id": conversation_id
                },
                on_progress=handle_progress
            )
            
            # Extract final answer
            final_answer = ""
            if isinstance(result, dict):
                # Debug: Log raw result structure if debug enabled
                if self.valves.enable_debug and __event_emitter__:
                    import json as json_debug
                    debug_msg = f"DEBUG: Raw MCP result structure: {json_debug.dumps(result, default=str, indent=2)}"
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": debug_msg[:200], "done": False}
                    })
                
                # Handle content array format
                if "content" in result and isinstance(result["content"], list):
                    for item in result["content"]:
                        if item.get("type") == "text":
                            try:
                                # Parse nested JSON
                                data = json.loads(item["text"])
                                final_answer = data.get("answer", "")
                                final_conversation_id = data.get("conversation_id", conversation_id)
                                if "thinking_steps" in data:
                                    thinking_steps = data["thinking_steps"]
                                break
                            except (json.JSONDecodeError, KeyError):
                                final_answer = item.get("text", "")
                
                # Handle direct format
                elif "answer" in result:
                    final_answer = result["answer"]
                    final_conversation_id = result.get("conversation_id", conversation_id)
                    if "thinking_steps" in result:
                        thinking_steps = result["thinking_steps"]
                
                # Fallback to text content
                elif "text" in result:
                    final_answer = result["text"]
                
                # Debug: Log what was extracted
                if self.valves.enable_debug and __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"DEBUG: Extracted answer length: {len(final_answer)} chars",
                            "done": False
                        }
                    })
            
            # Final status update
            if __event_emitter__:
                status_msg = f"✅ Complete - {progress_count} updates"
                if thinking_steps:
                    status_msg += f", {len(thinking_steps)} thinking steps"
                
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": status_msg,
                        "done": True
                    }
                })
            
            # Format final response
            # If we've already streamed the answer chunks, we need to handle differently
            # to prevent the model from appending additional generated content
            if has_streamed_answer and not is_native_mode and __event_emitter__:
                # Add conversation ID footer if available
                # if final_conversation_id:
                #     full_streamed_content += f"\n\n---\n*💬 Conversation ID: `{final_conversation_id}` (use for follow-ups)*"
                
                # Add hidden end marker for the companion filter to detect
                # This marker will be used by notebooklm_mcp_filter.py to strip
                # any model-generated text that appears after it
                # full_streamed_content += "\n<!-- NOTEBOOKLM_STREAM_END -->"
                
                # Use "chat:message" to replace the entire message content with our streamed content
                # await __event_emitter__({
                #     "type": "chat:message",  # This REPLACES the entire message content
                #     "data": {
                #         "content": full_streamed_content
                #     }
                # })
                
                # Return a special signal that the system prompt should instruct the model to stop
                # The model should be configured to NOT generate ANY text when it sees this return value
                return "<<<STREAMING_COMPLETE_NO_RESPONSE_NEEDED>>>"
            
            response = final_answer or "No answer received from NotebookLM."
            
            if final_conversation_id and not is_native_mode:
                response += f"\n\n*💬 Conversation ID: `{final_conversation_id}` (use for follow-ups)*"
            
            return response
        
        except Exception as e:
            return await self._emit_error(__event_emitter__, e, "Query failed")
    
    async def notebook_list(
        self,
        max_results: int = 10,
        __event_emitter__: Optional[Callable] = None,
    ) -> str:
        """
        List available NotebookLM notebooks.
        
        Args:
            max_results: Maximum number of notebooks to return (default: 10)
            __event_emitter__: Injected by Open WebUI for progress updates
        
        Returns:
            Formatted markdown table of notebooks
        """
        try:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "📚 Fetching notebooks...", "done": False}
                })
            
            # Initialize MCP client
            adapter = MCPClientAdapter(
                self.valves.mcp_server_url,
                self.valves.timeout
            )
            await adapter.initialize_session()
            
            # Call non-streaming tool
            result = await adapter.call_tool(
                tool_name="notebook_list",
                arguments={"max_results": max_results}
            )
            
            # Parse response
            notebooks = []
            if isinstance(result, dict):
                # Check for structuredContent (new format)
                if "structuredContent" in result:
                    structured = result["structuredContent"]
                    if isinstance(structured, dict):
                        notebooks = structured.get("notebooks", [])
                # Check for content array (old format)
                elif "content" in result and isinstance(result["content"], list):
                    for item in result["content"]:
                        if item.get("type") == "text":
                            try:
                                data = json.loads(item["text"])
                                notebooks = data.get("notebooks", [])
                                break
                            except json.JSONDecodeError:
                                pass
                # Direct notebooks key
                elif "notebooks" in result:
                    notebooks = result["notebooks"]
            
            if not notebooks:
                return "No notebooks found. Create one at https://notebooklm.google.com"
            
            # Format as markdown table
            output = "## 📚 Your NotebookLM Notebooks\n\n"
            output += "| Title | Notebook ID | Sources |\n"
            output += "|-------|-------------|----------|\n"
            
            for nb in notebooks:
                title = nb.get("title", "Untitled")
                nb_id = nb.get("id", "unknown")
                source_count = nb.get("source_count", 0)
                output += f"| {title} | `{nb_id}` | {source_count} |\n"
            
            output += f"\n*Showing {len(notebooks)} notebook(s)*"
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"✅ Found {len(notebooks)} notebooks", "done": True}
                })
            
            return output
        
        except Exception as e:
            return await self._emit_error(__event_emitter__, e, "Failed to list notebooks")
    
    async def notebook_info(
        self,
        notebook_id: str,
        __event_emitter__: Optional[Callable] = None,
    ) -> str:
        """
        Get detailed information about a specific notebook.
        
        Args:
            notebook_id: Notebook UUID
            __event_emitter__: Injected by Open WebUI for progress updates
        
        Returns:
            Formatted notebook details including sources
        """
        try:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "📖 Fetching notebook details...", "done": False}
                })
            
            adapter = MCPClientAdapter(
                self.valves.mcp_server_url,
                self.valves.timeout
            )
            await adapter.initialize_session()
            
            result = await adapter.call_tool(
                tool_name="notebook_info",
                arguments={"notebook_id": notebook_id}
            )
            
            # Parse response
            notebook_data = {}
            if isinstance(result, dict):
                if "content" in result and isinstance(result["content"], list):
                    for item in result["content"]:
                        if item.get("type") == "text":
                            try:
                                notebook_data = json.loads(item["text"])
                                break
                            except json.JSONDecodeError:
                                pass
                else:
                    notebook_data = result
            
            # Format output
            output = f"## 📖 Notebook: {notebook_data.get('title', 'Unknown')}\n\n"
            output += f"**ID:** `{notebook_id}`\n\n"
            
            sources = notebook_data.get("sources", [])
            if sources:
                output += f"### 📄 Sources ({len(sources)})\n\n"
                for i, source in enumerate(sources, 1):
                    source_title = source.get("title", "Untitled")
                    source_id = source.get("id", "unknown")
                    source_type = source.get("type", "unknown")
                    output += f"{i}. **{source_title}** (`{source_id}`) - *{source_type}*\n"
            else:
                output += "*No sources in this notebook*\n"
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✅ Done", "done": True}
                })
            
            return output
        
        except Exception as e:
            return await self._emit_error(__event_emitter__, e, "Failed to get notebook info")
    
    async def health_check(
        self,
        __event_emitter__: Optional[Callable] = None,
    ) -> str:
        """
        Check if MCP server is running and accessible.
        
        Args:
            __event_emitter__: Injected by Open WebUI for progress updates
        
        Returns:
            Health status message
        """
        try:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "🔍 Checking MCP server...", "done": False}
                })
            
            adapter = MCPClientAdapter(
                self.valves.mcp_server_url,
                self.valves.timeout
            )
            
            # Try to initialize session
            session_id = await adapter.initialize_session()
            
            status_msg = f"✅ MCP server is healthy!\n\n"
            status_msg += f"**Server URL:** {self.valves.mcp_server_url}\n"
            status_msg += f"**Session ID:** `{session_id}`\n"
            status_msg += f"**Timeout:** {self.valves.timeout}s\n"
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✅ Server is healthy", "done": True}
                })
            
            return status_msg
        
        except ConnectionError as e:
            error_msg = f"❌ Cannot connect to MCP server\n\n"
            error_msg += f"**Error:** {str(e)}\n\n"
            error_msg += "**Troubleshooting:**\n"
            error_msg += "1. Ensure server is running:\n"
            error_msg += "   ```bash\n"
            error_msg += "   uv run notebooklm-mcp --transport http --port 9888\n"
            error_msg += "   ```\n"
            error_msg += f"2. Check server URL in tool settings: `{self.valves.mcp_server_url}`\n"
            error_msg += "3. Verify no firewall is blocking the connection\n"
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "chat:message:error",
                    "data": {"content": error_msg}
                })
            
            return error_msg
        
        except Exception as e:
            return await self._emit_error(__event_emitter__, e, "Health check failed")
