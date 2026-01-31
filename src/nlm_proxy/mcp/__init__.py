"""MCP server for NotebookLM."""


def create_server():
    """Create and return the MCP server instance."""
    try:
        from .server import mcp
    except ImportError as e:
        raise ImportError(
            "MCP dependencies not installed. Run: pip install nlm-proxy[mcp]"
        ) from e
    return mcp


def run_server(debug: bool = False, transport: str = "stdio", port: int = 8000):
    """Run the MCP server."""
    try:
        from .server import main as server_main
    except ImportError as e:
        raise ImportError(
            "MCP dependencies not installed. Run: pip install nlm-proxy[mcp]"
        ) from e
    server_main(debug=debug, transport=transport, port=port)


__all__ = ["create_server", "run_server"]
