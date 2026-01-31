"""Unified CLI for NLM Proxy."""

import argparse
import sys


def cmd_serve_mcp(args):
    """Run the MCP server."""
    from nlm_proxy.mcp import run_server
    run_server(debug=args.debug, transport=args.transport, port=args.port)


def cmd_serve_openai(args):
    """Run the OpenAI proxy server."""
    from nlm_proxy.openai import run_server
    run_server(host=args.host, port=args.port, session_ttl=args.session_ttl)


def cmd_auth_extract(args):
    """Extract authentication tokens."""
    # Import the existing auth CLI functionality
    from notebooklm_mcp.auth_cli import main as auth_main
    auth_main()


def cmd_auth_test(args):
    """Test if current tokens are valid."""
    from nlm_proxy.core import NotebookLMClient
    from nlm_proxy.core.auth import load_cached_tokens
    from nlm_proxy.core.exceptions import AuthenticationError

    try:
        tokens = load_cached_tokens()
        if not tokens:
            print("No cached tokens found. Run: nlm-proxy auth extract")
            sys.exit(1)
        client = NotebookLMClient(tokens)
        notebooks = client.list_notebooks()
        print(f"Authentication successful! Found {len(notebooks)} notebooks.")
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="nlm-proxy",
        description="NotebookLM client library with MCP and OpenAI interfaces",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run a server")
    serve_subparsers = serve_parser.add_subparsers(dest="server", help="Server type")

    # serve mcp
    mcp_parser = serve_subparsers.add_parser("mcp", help="Run MCP server")
    mcp_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )
    mcp_parser.set_defaults(func=cmd_serve_mcp)

    # serve openai
    openai_parser = serve_subparsers.add_parser("openai", help="Run OpenAI proxy")
    openai_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    openai_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    openai_parser.add_argument(
        "--session-ttl",
        type=int,
        default=86400,
        help="Session TTL in seconds (default: 86400 = 24h)",
    )
    openai_parser.set_defaults(func=cmd_serve_openai)

    # auth command
    auth_parser = subparsers.add_parser("auth", help="Authentication management")
    auth_subparsers = auth_parser.add_subparsers(dest="action", help="Auth action")

    # auth extract
    extract_parser = auth_subparsers.add_parser("extract", help="Extract tokens from browser")
    extract_parser.set_defaults(func=cmd_auth_extract)

    # auth test
    test_parser = auth_subparsers.add_parser("test", help="Test current tokens")
    test_parser.set_defaults(func=cmd_auth_test)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve" and not args.server:
        serve_parser.print_help()
        sys.exit(1)

    if args.command == "auth" and not args.action:
        auth_parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
