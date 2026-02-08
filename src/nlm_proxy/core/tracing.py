"""OpenTelemetry tracing initialization and utilities."""

from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar, ParamSpec, Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import Status, StatusCode

from nlm_proxy.core.config import get_tracing_settings
from nlm_proxy.core.logging import get_logger

logger = get_logger(__name__)

_initialized = False
_export_error_logged = False  # Track if we've already logged export errors

P = ParamSpec('P')
T = TypeVar('T')


class SafeSpanExporter(SpanExporter):
    """Wrapper that catches and logs export exceptions instead of propagating them.

    This prevents noisy "Exception while exporting Span" messages from flooding logs
    when the collector is unreachable or has SSL issues.
    """

    def __init__(self, delegate: SpanExporter):
        self._delegate = delegate

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        global _export_error_logged
        try:
            return self._delegate.export(spans)
        except Exception as e:
            # Only log once to avoid log spam
            if not _export_error_logged:
                error_msg = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in error_msg:
                    logger.warning(
                        "[TRACING] SSL certificate verification failed. "
                        "Set NLM_PROXY_OTEL_VERIFY_CERT=false or provide valid CA cert. "
                        "Tracing will continue but spans may be lost."
                    )
                elif "UNAVAILABLE" in error_msg or "Connection refused" in error_msg:
                    logger.warning(
                        f"[TRACING] Cannot reach collector: {e}. "
                        "Tracing will continue but spans may be lost."
                    )
                else:
                    logger.warning(
                        f"[TRACING] Export failed: {e}. "
                        "Tracing will continue but spans may be lost."
                    )
                _export_error_logged = True
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            return self._delegate.force_flush(timeout_millis)
        except Exception:
            return False


def _build_http_url(endpoint: str, insecure: bool) -> str:
    """Build full HTTP URL from host:port endpoint."""
    scheme = "http" if insecure else "https"
    return f"{scheme}://{endpoint}/v1/traces"


def _create_exporter(settings) -> SpanExporter:
    """Create appropriate exporter based on protocol setting."""
    headers = None
    if settings.api_key:
        headers = {"authorization": f"Bearer {settings.api_key}"}

    if settings.protocol == "http":
        return _create_http_exporter(settings, headers)
    else:
        return _create_grpc_exporter(settings, headers)


def _create_http_exporter(settings, headers: dict | None) -> SpanExporter:
    """Create HTTP exporter with TLS options."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPSpanExporter
    )
    import os
    import urllib3

    # Build full URL
    endpoint_url = _build_http_url(settings.endpoint, settings.insecure)

    # Prepare exporter kwargs
    exporter_kwargs = {
        "endpoint": endpoint_url,
        "headers": headers,
        "timeout": settings.export_timeout,
    }

    if settings.insecure:
        # Plain HTTP, no cert configuration needed
        cert_info = "insecure (HTTP)"
    elif not settings.verify_cert:
        # Skip certificate verification by setting environment variable
        # This is the most reliable way to disable SSL verification
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        # Suppress InsecureRequestWarning
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        cert_info = "verify=disabled (development only)"
        logger.warning(
            "[TRACING] Certificate verification disabled - only use in development!"
        )
    elif settings.ca_cert_path:
        # Use custom CA certificate - expand path and validate
        ca_path = Path(os.path.expanduser(settings.ca_cert_path))
        if not ca_path.is_absolute():
            ca_path = Path.cwd() / ca_path
        if not ca_path.exists():
            raise FileNotFoundError(
                f"[TRACING] CA certificate not found: {ca_path}"
            )
        # Pass certificate to exporter
        exporter_kwargs["certificate_file"] = str(ca_path)
        cert_info = f"ca_cert={ca_path}"
    else:
        # Use system CA bundle
        cert_info = "system CA"

    logger.debug(
        f"[TRACING] Creating HTTP exporter: endpoint={endpoint_url}, "
        f"{cert_info}, auth={'enabled' if headers else 'disabled'}"
    )

    try:
        return HTTPSpanExporter(**exporter_kwargs)
    except Exception as e:
        logger.error(f"[TRACING] Failed to create HTTP exporter: {e}")
        raise


def _create_grpc_exporter(settings, headers: dict | None) -> SpanExporter:
    """Create gRPC exporter with TLS options."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter
    )

    credentials = None

    if not settings.insecure:
        from grpc import ssl_channel_credentials

        # Warn if verify_cert=False (not supported in gRPC)
        if not settings.verify_cert:
            logger.warning(
                "[TRACING] verify_cert=False not supported with gRPC protocol, "
                "certificates will be validated. Use protocol=http for skip-verify."
            )

        ca_cert = None
        if settings.ca_cert_path:
            cert_path = Path(settings.ca_cert_path)
            if not cert_path.exists():
                raise FileNotFoundError(
                    f"[TRACING] CA certificate not found: {settings.ca_cert_path}"
                )
            with open(cert_path, "rb") as f:
                ca_cert = f.read()

        credentials = ssl_channel_credentials(root_certificates=ca_cert)

    # gRPC headers format: list of tuples
    grpc_headers = [(k, v) for k, v in headers.items()] if headers else None

    logger.debug(
        f"[TRACING] Creating gRPC exporter: endpoint={settings.endpoint}, "
        f"insecure={settings.insecure}, auth={'enabled' if headers else 'disabled'}"
    )

    return GRPCSpanExporter(
        endpoint=settings.endpoint,
        insecure=settings.insecure,
        credentials=credentials,
        headers=grpc_headers,
        timeout=settings.export_timeout,
    )


def init_tracing() -> None:
    """Initialize OpenTelemetry tracing based on settings."""
    global _initialized

    if _initialized:
        return

    settings = get_tracing_settings()

    if not settings.enabled:
        logger.debug("[TRACING] OpenTelemetry tracing is disabled")
        _initialized = True
        return

    try:
        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: settings.service_name})

        # Create and configure tracer provider
        provider = TracerProvider(resource=resource)

        # Create exporter using factory (handles protocol, TLS, auth)
        raw_exporter = _create_exporter(settings)

        # Wrap with SafeSpanExporter to catch and log export errors gracefully
        exporter = SafeSpanExporter(raw_exporter)

        # Configure processor to drop spans instead of blocking when queue is full
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=settings.max_queue_size,
            schedule_delay_millis=5000,  # Export every 5s (default)
            max_export_batch_size=512,   # Batch size (default)
            export_timeout_millis=settings.export_timeout * 1000  # Convert to ms
        )

        provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Log configuration summary
        tls_status = "TLS" if not settings.insecure else "plain"
        auth_status = "auth=enabled" if settings.api_key else "auth=disabled"
        logger.info(
            f"[TRACING] OpenTelemetry initialized: protocol={settings.protocol}, "
            f"endpoint={settings.endpoint}, {tls_status}, {auth_status}, "
            f"service={settings.service_name}"
        )
        _initialized = True

    except Exception as e:
        logger.error(f"[TRACING] Failed to initialize OpenTelemetry: {e}")
        logger.warning("[TRACING] Tracing disabled - server will continue without observability")
        _initialized = True  # Don't retry, server continues normally


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance for the given module name."""
    return trace.get_tracer(name)


def shutdown_tracing(timeout_seconds: int = 3) -> None:
    """Shutdown tracing and flush pending spans with timeout.

    Args:
        timeout_seconds: Max time to wait for shutdown (default: 3s)
    """
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, 'shutdown'):
            # Use force_flush with timeout, then shutdown
            if hasattr(provider, 'force_flush'):
                success = provider.force_flush(timeout_millis=timeout_seconds * 1000)
                if not success:
                    logger.warning("[TRACING] force_flush timed out, some spans may be lost")
            provider.shutdown()
            logger.debug("[TRACING] OpenTelemetry shutdown complete")
    except Exception as e:
        # Non-fatal - don't block server shutdown on tracing errors
        logger.warning(f"[TRACING] Shutdown encountered error (non-fatal): {e}")


def record_span(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to record a span for an async function.

    Usage:
        @record_span("smart_router.classify")
        async def classify_request(self, query: str) -> RequestType:
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(name) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator


def add_span_attributes(**attributes) -> None:
    """Add attributes to the current span.

    Safe to call even when no span is active.

    Usage:
        add_span_attributes(
            notebook_id="abc123",
            notebook_title="ML Research",
            candidates_count=5
        )
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)


def instrument_fastapi(app) -> None:
    """Instrument a FastAPI application for automatic tracing.

    Args:
        app: FastAPI application instance
    """
    settings = get_tracing_settings()
    if not settings.enabled:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("[TRACING] FastAPI instrumentation enabled")
    except ImportError:
        logger.warning("[TRACING] FastAPI instrumentation not available")
    except Exception as e:
        logger.error(f"[TRACING] Failed to instrument FastAPI: {e}")


def instrument_httpx() -> None:
    """Instrument httpx for automatic tracing of outgoing HTTP calls."""
    settings = get_tracing_settings()
    if not settings.enabled:
        return

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.info("[TRACING] httpx instrumentation enabled")
    except ImportError:
        logger.warning("[TRACING] httpx instrumentation not available")
    except Exception as e:
        logger.error(f"[TRACING] Failed to instrument httpx: {e}")
