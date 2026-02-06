"""OpenTelemetry tracing initialization and utilities."""

from functools import wraps
from typing import Callable, TypeVar, ParamSpec

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import Status, StatusCode

from nlm_proxy.core.config import get_tracing_settings
from nlm_proxy.core.logging import get_logger

logger = get_logger(__name__)

_initialized = False

P = ParamSpec('P')
T = TypeVar('T')


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
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: settings.service_name})

        # Create and configure tracer provider
        provider = TracerProvider(resource=resource)

        # Configure OTLP exporter with fast timeout to prevent blocking
        exporter = OTLPSpanExporter(
            endpoint=settings.endpoint,
            insecure=True,
            timeout=settings.export_timeout  # Fast fail on connection issues
        )

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

        logger.info(
            f"[TRACING] OpenTelemetry initialized: endpoint={settings.endpoint}, "
            f"service={settings.service_name}, timeout={settings.export_timeout}s, "
            f"queue_size={settings.max_queue_size}"
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
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentation
        HTTPXClientInstrumentation().instrument()
        logger.info("[TRACING] httpx instrumentation enabled")
    except ImportError:
        logger.warning("[TRACING] httpx instrumentation not available")
    except Exception as e:
        logger.error(f"[TRACING] Failed to instrument httpx: {e}")
