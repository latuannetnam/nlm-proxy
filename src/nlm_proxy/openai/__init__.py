"""OpenAI-compatible proxy for NotebookLM."""


def create_app():
    """Create and return the FastAPI app instance."""
    try:
        from .server import app
    except ImportError as e:
        raise ImportError(
            "OpenAI proxy dependencies not installed. Run: pip install nlm-proxy[openai]"
        ) from e
    return app


def run_server(host: str = "0.0.0.0", port: int = 8080, session_ttl: int = 86400):
    """Run the OpenAI proxy server."""
    try:
        from .server import main as server_main
    except ImportError as e:
        raise ImportError(
            "OpenAI proxy dependencies not installed. Run: pip install nlm-proxy[openai]"
        ) from e
    server_main(host=host, port=port, session_ttl=session_ttl)


__all__ = ["create_app", "run_server"]
