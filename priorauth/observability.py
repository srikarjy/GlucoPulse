"""
Telemetry hooks. Metrics are always on (Prometheus client, in-process registry);
tracing uses the OpenTelemetry API and is a no-op until an SDK/exporter is configured
(see priorauth/telemetry_setup.py and docs/priorauth/OBSERVABILITY.md).

Rule: labels and span attributes carry ids/codes/flags only -- never clinical text.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from opentelemetry import trace

REGISTRY = CollectorRegistry()
tracer = trace.get_tracer("priorauth")

REQUESTS = Counter("pa_requests_total", "PA requests received", ["urgency"], registry=REGISTRY)
DRAFTS = Counter("pa_drafts_total", "Drafts produced", ["recommendation", "tier"], registry=REGISTRY)
FLAGS = Counter("pa_flags_total", "Safety/quality flags raised", ["flag"], registry=REGISTRY)
AGENT_LAT = Histogram("pa_agent_seconds", "Agent step latency", ["agent"], registry=REGISTRY,
                      buckets=(.005, .01, .05, .1, .5, 1, 2, 5, 10, 30))
TOOL_CALLS = Counter("pa_tool_calls_total", "MCP tool calls", ["tool", "outcome"], registry=REGISTRY)
CONFIDENCE = Histogram("pa_draft_confidence", "Draft confidence", registry=REGISTRY,
                       buckets=(.3, .5, .6, .7, .8, .85, .9, .95, 1.0))
PIPELINE_FAIL = Counter("pa_pipeline_failures_total", "Pipelines that failed closed to human review", ["agent"], registry=REGISTRY)
QUEUE_DEPTH = Gauge("pa_queue_jobs", "Jobs by state", ["state"], registry=REGISTRY)
REVIEW_OPEN = Gauge("pa_review_tasks_open", "Open review tasks", ["tier"], registry=REGISTRY)
REVIEW_OVERRIDE = Counter("pa_review_overrides_total", "Human decisions that disagreed with the draft", registry=REGISTRY)
