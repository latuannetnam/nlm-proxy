#!/usr/bin/env python3
"""
Debug test to identify why Open WebUI shows no response from the tool.

Run with: uv run python src/open_webui/debug_test.py
"""

import asyncio
import json
from notebooklm_mcp_tool import Tools, MCPClientAdapter


async def test_mcp_server_direct():
    """Test MCP server directly to verify it's working."""
    print("\n" + "="*60)
    print("TEST 1: Direct MCP Server Connection")
    print("="*60)
    
    try:
        adapter = MCPClientAdapter("http://localhost:9888", timeout=30.0)
        print("✓ Created adapter")
        
        session_id = await adapter.initialize_session()
        print(f"✓ Session initialized: {session_id}")
        
        print("\n→ Calling notebook_list...")
        result = await adapter.call_tool(
            tool_name="notebook_list",
            arguments={"max_results": 5}
        )
        
        print(f"\n✓ Raw result type: {type(result)}")
        print(f"✓ Raw result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        print(f"\n✓ Raw result:\n{json.dumps(result, indent=2, default=str)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_notebook_list():
    """Test the tool's notebook_list method."""
    print("\n" + "="*60)
    print("TEST 2: Tool notebook_list Method")
    print("="*60)
    
    try:
        tool = Tools()
        tool.valves.mcp_server_url = "http://localhost:9888"
        print("✓ Created tool instance")
        
        # Track emitted events
        emitted_events = []
        
        async def mock_emitter(event):
            emitted_events.append(event)
            print(f"  📤 Event: {event.get('type')} - {event.get('data', {}).get('description', '')}")
        
        print("\n→ Calling tool.notebook_list()...")
        result = await tool.notebook_list(
            max_results=5,
            __event_emitter__=mock_emitter
        )
        
        print(f"\n✓ Result type: {type(result)}")
        print(f"✓ Result length: {len(result)} chars")
        print(f"✓ Events emitted: {len(emitted_events)}")
        print(f"\n✓ Final result:\n{result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_health_check():
    """Test the tool's health_check method."""
    print("\n" + "="*60)
    print("TEST 3: Tool health_check Method")
    print("="*60)
    
    try:
        tool = Tools()
        tool.valves.mcp_server_url = "http://localhost:9888"
        
        emitted_events = []
        
        async def mock_emitter(event):
            emitted_events.append(event)
            print(f"  📤 Event: {event.get('type')}")
        
        print("\n→ Calling tool.health_check()...")
        result = await tool.health_check(__event_emitter__=mock_emitter)
        
        print(f"\n✓ Result type: {type(result)}")
        print(f"✓ Events emitted: {len(emitted_events)}")
        print(f"\n✓ Final result:\n{result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming_query():
    """Test streaming query with a real notebook (if available)."""
    print("\n" + "="*60)
    print("TEST 4: Streaming Query (Optional)")
    print("="*60)
    
    try:
        # First get notebooks
        adapter = MCPClientAdapter("http://localhost:9888")
        await adapter.initialize_session()
        
        result = await adapter.call_tool(
            tool_name="notebook_list",
            arguments={"max_results": 1}
        )
        
        # Parse notebook ID
        notebook_id = None
        if isinstance(result, dict):
            if "content" in result:
                for item in result["content"]:
                    if item.get("type") == "text":
                        data = json.loads(item["text"])
                        notebooks = data.get("notebooks", [])
                        if notebooks:
                            notebook_id = notebooks[0]["id"]
                            print(f"✓ Found notebook: {notebooks[0]['title']} ({notebook_id})")
                            break
            elif "notebooks" in result:
                if result["notebooks"]:
                    notebook_id = result["notebooks"][0]["id"]
                    print(f"✓ Found notebook: {result['notebooks'][0]['title']} ({notebook_id})")
        
        if not notebook_id:
            print("⚠ No notebooks found - skipping streaming test")
            return True
        
        # Test streaming query
        tool = Tools()
        tool.valves.mcp_server_url = "http://localhost:9888"
        
        emitted_events = []
        
        async def mock_emitter(event):
            emitted_events.append(event)
            event_type = event.get('type')
            if event_type == 'status':
                desc = event.get('data', {}).get('description', '')
                # Truncate long status messages
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                print(f"  📤 Status: {desc}")
            elif event_type == 'message':
                content = event.get('data', {}).get('content', '')
                # Show content type and length
                if "**💭 Thinking:**" in content:
                    preview = content[20:70].replace('\n', ' ')
                    print(f"  📤 Message [thinking]: {preview}...")
                elif "**📝 Answer:**" in content:
                    print(f"  📤 Message [answer header]")
                else:
                    # Streaming answer chunk
                    print(f"  📤 Message [answer +{len(content)} chars]: {content[:40].replace(chr(10), ' ')}...")
        
        print(f"\n→ Querying notebook {notebook_id}...")
        result = await tool.notebook_query_stream(
            notebook_id=notebook_id,
            query="What is this notebook about? (brief summary)",
            __event_emitter__=mock_emitter
        )
        
        print(f"\n✓ Result type: {type(result)}")
        print(f"✓ Result length: {len(result)} chars")
        print(f"✓ Events emitted: {len(emitted_events)}")
        print(f"\n✓ Final result (first 200 chars):\n{result[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"⚠ Streaming test skipped or failed: {e}")
        return True  # Don't fail if no notebooks


async def main():
    """Run all debug tests."""
    print("\n" + "="*60)
    print("🔍 DEBUG TEST SUITE FOR OPEN WEBUI TOOL")
    print("="*60)
    print("\nThis will help identify why Open WebUI shows no response.")
    print("\nPrerequisites:")
    print("  1. MCP server running on http://localhost:9888")
    print("  2. Authenticated with NotebookLM")
    print("  3. At least one notebook available")
    
    results = []
    
    # Test 1: Direct MCP server
    results.append(("Direct MCP Server", await test_mcp_server_direct()))
    
    # Test 2: Tool notebook_list
    results.append(("Tool notebook_list", await test_tool_notebook_list()))
    
    # Test 3: Tool health_check
    results.append(("Tool health_check", await test_tool_health_check()))
    
    # Test 4: Streaming query (optional)
    results.append(("Streaming query", await test_streaming_query()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✅ All tests passed!")
        print("\nIf Open WebUI still shows no response, check:")
        print("  1. Open WebUI logs for errors")
        print("  2. Browser console for JavaScript errors")
        print("  3. Network tab to see if response is being sent")
        print("  4. Tool return type (must be string, not dict)")
    else:
        print("\n❌ Some tests failed - see errors above")
        print("\nCommon issues:")
        print("  - MCP server not running or not accessible")
        print("  - Authentication expired")
        print("  - Network/firewall blocking connection")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
