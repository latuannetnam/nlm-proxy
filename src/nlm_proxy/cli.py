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
    from nlm_proxy.core.auth_cli import run_auth_flow, run_file_cookie_entry

    try:
        if args.file is not None:  # --file was used (with or without path)
            # File-based cookie import
            tokens = run_file_cookie_entry(cookie_file=args.file if args.file else None)
        else:
            # Automatic extraction via Chrome DevTools
            tokens = run_auth_flow(args.port, auto_launch=not args.no_auto_launch)

        sys.exit(0 if tokens else 1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def cmd_auth_test_async(args):
    """Test if current tokens are valid."""
    from nlm_proxy.core import NotebookLMClient
    from nlm_proxy.core.auth import load_cached_tokens
    from nlm_proxy.core.exceptions import AuthenticationError

    try:
        tokens = load_cached_tokens()
        if not tokens:
            print("No cached tokens found. Run: nlm-proxy auth extract")
            sys.exit(1)
        client = NotebookLMClient(
            cookies=tokens.cookies,
            csrf_token=tokens.csrf_token,
            session_id=tokens.session_id
        )
        notebooks = await client.list_notebooks()
        print(f"Authentication successful! Found {len(notebooks)} notebooks.")
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_auth_test(args):
    """Wrapper to run async auth test."""
    import asyncio
    asyncio.run(cmd_auth_test_async(args))


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
    extract_parser.add_argument(
        "--file",
        nargs="?",
        const="",  # When --file is used without argument, set to empty string
        metavar="PATH",
        help="Import cookies from file (recommended). Shows instructions if no path given."
    )
    extract_parser.add_argument(
        "--port",
        type=int,
        default=9222,
        help="Chrome DevTools port (default: 9222)"
    )
    extract_parser.add_argument(
        "--no-auto-launch",
        action="store_true",
        help="Don't automatically launch Chrome (requires Chrome to be running with debugging)"
    )
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
