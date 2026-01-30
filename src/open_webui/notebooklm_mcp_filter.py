"""
title: NotebookLM MCP Response Filter
author: NotebookLM MCP Contributors  
author_url: https://github.com/jacob-bd/notebooklm-mcp
git_url: https://github.com/jacob-bd/notebooklm-mcp
description: Filter that strips model-generated text after NotebookLM tool streaming completes
required_open_webui_version: 0.5.17
version: 1.0.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Optional
import re


class Filter:
    """
    Filter to handle NotebookLM MCP tool responses.
    
    In Default function calling mode, OpenWebUI passes the tool's return value 
    to the LLM which then generates additional text. This filter intercepts the 
    outlet (post-LLM response) and strips any text that appears AFTER the 
    NotebookLM streamed content.
    
    The filter looks for the hidden end marker <!-- NOTEBOOKLM_STREAM_END -->
    that the tool adds at the end of its streamed content, and removes 
    everything after it (including the marker itself for a clean output).
    
    INSTALLATION:
    1. Add this filter in OpenWebUI: Admin Panel -> Functions -> Create New
    2. Select type "Filter"
    3. Enable the filter for your NotebookLM model
    4. Make sure the NotebookLM MCP tool is also installed
    """
    
    class Valves(BaseModel):
        """Configuration options for the filter."""
        enabled: bool = Field(
            default=True,
            description="Enable/disable the filter"
        )
        # Hidden HTML comment marker added by the tool
        end_marker: str = Field(
            default="<!-- NOTEBOOKLM_STREAM_END -->",
            description="Hidden marker that indicates end of NotebookLM response"
        )
        # Fallback pattern using conversation ID footer
        fallback_pattern: str = Field(
            default=r"\*💬 Conversation ID: `[^`]+` \(use for follow-ups\)\*",
            description="Fallback regex pattern if hidden marker not found"
        )
        strip_streaming_marker: bool = Field(
            default=True,
            description="Strip the <<<STREAMING_COMPLETE>>> marker if present in response"
        )
        remove_end_marker: bool = Field(
            default=True,
            description="Remove the hidden end marker from final output"
        )
        debug_logging: bool = Field(
            default=False,
            description="Enable debug logging to console"
        )
    
    def __init__(self):
        self.valves = self.Valves()
    
    def _debug_log(self, message: str):
        """Print debug message if debugging is enabled."""
        if self.valves.debug_logging:
            print(f"[NotebookLM Filter] {message}")
    
    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Pre-process user input before sending to model.
        
        Currently a pass-through, but can be extended if needed.
        """
        return body
    
    def stream(self, event: dict) -> dict:
        """
        Process streamed chunks from the model.
        
        Currently a pass-through, can be extended for real-time filtering.
        """
        return event
    
    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Post-process model output after LLM response.
        
        This is where we strip any model-generated text that appears after
        the NotebookLM streamed content. The LLM in Default mode will receive
        the tool's return value and may generate additional text - we remove that.
        """
        if not self.valves.enabled:
            return body
        
        messages = body.get("messages", [])
        if not messages:
            return body
        
        # Process the last assistant message
        for i, message in enumerate(reversed(messages)):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                
                if not content:
                    break
                
                self._debug_log(f"Processing message ({len(content)} chars): {content[:200]}...")
                
                original_content = content
                modified = False
                
                # Strategy 1: Look for the hidden end marker
                if self.valves.end_marker in content:
                    marker_pos = content.find(self.valves.end_marker)
                    self._debug_log(f"Found hidden end marker at position {marker_pos}")
                    
                    if self.valves.remove_end_marker:
                        # Remove marker and everything after it
                        content = content[:marker_pos]
                    else:
                        # Keep marker but remove everything after it
                        content = content[:marker_pos + len(self.valves.end_marker)]
                    
                    modified = True
                
                # Strategy 2: Fallback to conversation ID pattern
                elif "Conversation ID:" in content or "📝 Answer:" in content:
                    match = re.search(self.valves.fallback_pattern, content)
                    if match:
                        end_pos = match.end()
                        self._debug_log(f"Found fallback pattern at position {end_pos}")
                        content = content[:end_pos]
                        modified = True
                
                # Strip the streaming complete marker if present anywhere
                if self.valves.strip_streaming_marker:
                    new_content = re.sub(
                        r'<<<STREAMING_COMPLETE[^>]*>>>',
                        '',
                        content,
                        flags=re.IGNORECASE
                    )
                    if new_content != content:
                        content = new_content
                        modified = True
                
                # Clean up any trailing whitespace
                content = content.rstrip()
                
                if modified and content != original_content:
                    stripped_chars = len(original_content) - len(content)
                    self._debug_log(f"Stripped {stripped_chars} characters from response")
                    # Update the message in place
                    messages[len(messages) - 1 - i]["content"] = content
                
                break  # Only process the most recent assistant message
        
        body["messages"] = messages
        return body
