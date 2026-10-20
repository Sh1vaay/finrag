"""Enterprise Tracing & Observability Service.

Supports dual-mode tracing:
1. "phoenix" (default): Arize Phoenix via OpenInference and OpenTelemetry standard.
   Instruments LangChain/LangGraph calls, records retrieval, reranker, and tool execution spans.
2. "langfuse": LiteLLM-forwarded traces to self-hosted Langfuse.
3. "none": Disables external tracing.

Graceful degradation: if tracing packages or endpoints are unavailable, the application
continues uninterrupted without raising runtime exceptions.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Iterator

try:
    from src.config.settings import settings
except ImportError:
    settings = None

logger = logging.getLogger(__name__)

_is_tracing_initialized: bool = False
_tracer: Any = None


def init_tracing() -> None:
    """Initialize the configured tracing backend."""
    global _is_tracing_initialized, _tracer

    if _is_tracing_initialized:
        return

    backend = (os.environ.get("TRACING_BACKEND") or getattr(settings, "TRACING_BACKEND", "phoenix")).lower()
    logger.info("Initializing tracing backend: %s", backend)

    if backend == "phoenix":
        _init_phoenix_tracing()
    elif backend == "langfuse":
        host = getattr(settings, "LANGFUSE_HOST", "http://langfuse-web:3000")
        logger.info("Langfuse tracing backend active (host: %s)", host)
        _tracer = None
    elif backend in ("none", "disabled", "off"):
        logger.info("Tracing is disabled (TRACING_BACKEND=%s)", backend)
        _tracer = None
    else:
        logger.warning("Unknown TRACING_BACKEND '%s'; defaulting to none", backend)
        _tracer = None

    _is_tracing_initialized = True


def _init_phoenix_tracing() -> None:
    """Setup Arize Phoenix OpenInference + OpenTelemetry instrumentation."""
    global _tracer
    endpoint = getattr(settings, "PHOENIX_COLLECTOR_ENDPOINT", None) or os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
    )
    project_name = getattr(settings, "PHOENIX_PROJECT_NAME", None) or os.environ.get(
        "PHOENIX_PROJECT_NAME", "finrag-investment"
    )

    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": project_name})
        tracer_provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(span_processor)
        trace_api.set_tracer_provider(tracer_provider)
        _tracer = trace_api.get_tracer("finrag.agent", "0.3.5")

        # Instrument LangChain / LangGraph via OpenInference
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor

            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
            logger.info("Phoenix OpenInference LangChain instrumentation successfully installed")
        except ImportError:
            logger.info("openinference-instrumentation-langchain not installed; custom spans only")

        logger.info("Arize Phoenix OpenTelemetry tracing initialized (endpoint: %s)", endpoint)
    except Exception as exc:
        logger.warning(
            "Could not initialize Arize Phoenix tracing (%s: %s). Continuing without Phoenix.",
            type(exc).__name__,
            exc,
        )
        _tracer = None


def get_tracer() -> Any:
    """Get active OpenTelemetry tracer or None if uninitialized/offline."""
    return _tracer


@contextlib.contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Context manager to record a custom span with financial and execution attributes.

    Usage:
        with trace_span("retrieval.hybrid", {"company": "Apple", "top_k": 10}) as span:
            ...
    """
    if _tracer is None:
        yield None
        return

    try:
        with _tracer.start_as_current_span(name) as span:
            if attributes and hasattr(span, "set_attribute"):
                for key, val in attributes.items():
                    if val is not None:
                        # Convert complex structures to string/primitive
                        if isinstance(val, (int, float, str, bool)):
                            span.set_attribute(str(key), val)
                        else:
                            span.set_attribute(str(key), str(val))
            yield span
    except Exception as exc:
        logger.debug("Exception inside trace_span '%s': %s", name, exc)
        yield None
