"""
Tracing bootstrap. Off unless OTEL_EXPORTER_OTLP_ENDPOINT is set (so tests/CI stay hermetic).
Works with any OTLP/HTTP backend: Jaeger (compose), an OTel collector, or Langfuse
(OTEL_EXPORTER_OTLP_ENDPOINT=https://<langfuse>/api/public/otel and an `Authorization: Basic ...`
header via OTEL_EXPORTER_OTLP_HEADERS). Strands emits its own gen_ai.* spans (model + tool calls)
into the same provider, nested under our `pipeline` / `agent.*` spans.

PHI rule: spans carry ids/codes/flags only. Strands can record prompts/completions as span events;
because prompts here contain chart data, enable that ONLY against a backend covered by your BAA:
set PRIORAUTH_TRACE_CONTENT=1 to opt in (default: off).
"""
from __future__ import annotations

import os


def setup_tracing(service_name: str) -> bool:
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name,
                                                        "deployment.environment": os.getenv("PRIORAUTH_ENV", "dev")}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    if os.getenv("PRIORAUTH_TRACE_CONTENT") != "1":
        os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
    return True
