"""Unified CLI for NLM Proxy using Typer."""

import sys
from typing import Annotated, Optional

import typer

from nlm_proxy.core.config import (
    get_auth_settings,
    get_mcp_settings,
    get_openai_settings,
    get_shared_settings,
)

# Main app
app = typer.Typer(
    name="nlm-proxy",
    help="NotebookLM client library with MCP and OpenAI interfaces",
    no_args_is_help=True,
)

# Subcommand groups
serve_app = typer.Typer(help="Run a server", no_args_is_help=True)
auth_app = typer.Typer(help="Authentication management", no_args_is_help=True)

app.add_typer(serve_app, name="serve")
app.add_typer(auth_app, name="auth")


def version_callback(value: bool):
    if value:
        from nlm_proxy import __version__

        print(f"nlm-proxy {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version", "-v", callback=version_callback, is_eager=True, help="Show version"
        ),
    ] = None,
):
    """NotebookLM client library with MCP and OpenAI interfaces."""
    pass


# =============================================================================
# serve mcp
# =============================================================================
@serve_app.command("mcp")
def serve_mcp(
    debug: Annotated[
        Optional[bool],
        typer.Option("--debug", help="Enable debug logging"),
    ] = None,
    transport: Annotated[
        Optional[str],
        typer.Option("--transport", help="Transport type (stdio or http)"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option("--port", help="Port for HTTP transport"),
    ] = None,
):
    """Run MCP server."""
    from nlm_proxy.core import setup_logging
    from nlm_proxy.mcp import run_server

    # Resolve: CLI arg > env var > default (via settings)
    shared = get_shared_settings()
    mcp = get_mcp_settings()

    resolved_debug = debug if debug is not None else shared.debug
    resolved_transport = transport if transport is not None else mcp.transport
    resolved_port = port if port is not None else mcp.port

    setup_logging(debug=resolved_debug)
    run_server(debug=resolved_debug, transport=resolved_transport, port=resolved_port)


# =============================================================================
# serve openai
# =============================================================================
@serve_app.command("openai")
def serve_openai(
    debug: Annotated[
        Optional[bool],
        typer.Option("--debug", help="Enable debug logging"),
    ] = None,
    host: Annotated[
        Optional[str],
        typer.Option("--host", help="Host to bind to"),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option("--port", help="Port to listen on"),
    ] = None,
    session_ttl: Annotated[
        Optional[int],
        typer.Option("--session-ttl", help="Session TTL in seconds"),
    ] = None,
):
    """Run OpenAI proxy server."""
    from nlm_proxy.core import setup_logging
    from nlm_proxy.openai import run_server

    shared = get_shared_settings()
    openai = get_openai_settings()

    resolved_debug = debug if debug is not None else shared.debug
    resolved_host = host if host is not None else openai.host
    resolved_port = port if port is not None else openai.port
    resolved_session_ttl = session_ttl if session_ttl is not None else openai.session_ttl

    setup_logging(debug=resolved_debug)
    run_server(host=resolved_host, port=resolved_port, session_ttl=resolved_session_ttl)


# =============================================================================
# auth extract
# =============================================================================
@auth_app.command("extract")
def auth_extract(
    debug: Annotated[
        Optional[bool],
        typer.Option("--debug", help="Enable debug logging"),
    ] = None,
    file: Annotated[
        Optional[str],
        typer.Option(
            "--file",
            help="Import cookies from file. Shows instructions if no path given.",
        ),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option("--port", help="Chrome DevTools port"),
    ] = None,
    no_auto_launch: Annotated[
        bool,
        typer.Option("--no-auto-launch", help="Don't automatically launch Chrome"),
    ] = False,
):
    """Extract tokens from browser."""
    from nlm_proxy.core import setup_logging
    from nlm_proxy.core.auth_cli import run_auth_flow, run_file_cookie_entry

    shared = get_shared_settings()
    auth = get_auth_settings()

    resolved_debug = debug if debug is not None else shared.debug
    resolved_port = port if port is not None else auth.chrome_port
    resolved_auto_launch = auth.auto_launch and not no_auto_launch

    setup_logging(debug=resolved_debug)

    try:
        if file is not None:
            tokens = run_file_cookie_entry(cookie_file=file if file else None)
        else:
            tokens = run_auth_flow(resolved_port, auto_launch=resolved_auto_launch)

        raise typer.Exit(0 if tokens else 1)
    except typer.Exit:
        raise  # Let typer.Exit pass through without error handling
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise typer.Exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# =============================================================================
# auth test
# =============================================================================
@auth_app.command("test")
def auth_test(
    debug: Annotated[
        Optional[bool],
        typer.Option("--debug", help="Enable debug logging"),
    ] = None,
):
    """Test current tokens."""
    import asyncio

    from nlm_proxy.core import setup_logging, NotebookLMClient
    from nlm_proxy.core.auth import load_cached_tokens
    from nlm_proxy.core.exceptions import AuthenticationError

    shared = get_shared_settings()
    resolved_debug = debug if debug is not None else shared.debug

    setup_logging(debug=resolved_debug)

    async def _test():
        tokens = load_cached_tokens()
        if not tokens:
            print("No cached tokens found. Run: nlm-proxy auth extract")
            raise typer.Exit(1)

        client = NotebookLMClient(
            cookies=tokens.cookies,
            csrf_token=tokens.csrf_token,
            session_id=tokens.session_id,
        )
        notebooks = await client.list_notebooks()
        print(f"Authentication successful! Found {len(notebooks)} notebooks.")

    try:
        asyncio.run(_test())
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


def main_cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main_cli()
