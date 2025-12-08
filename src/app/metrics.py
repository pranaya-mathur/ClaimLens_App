# src/app/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# HTTP
REQUEST_COUNT = Counter(
    "claimlens_requests_total",
    "Total HTTP requests processed",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "claimlens_request_latency_seconds",
    "Request processing latency in seconds",
    ["endpoint"],
)

IN_PROGRESS = Gauge(
    "claimlens_requests_inprogress",
    "Number of requests currently in progress"
)

# LLM & pipeline
LLM_CALLS = Counter(
    "claimlens_llm_calls_total",
    "Total LLM invocation attempts",
    ["result"],  # result: success | failure | skipped
)

FALLBACKS = Counter(
    "claimlens_fallbacks_total",
    "Total times fallback heuristic was used"
)

CACHE_HITS = Counter(
    "claimlens_cache_hits_total",
    "Total cache hits for narrative (avoids LLM calls)"
)

ERRORS = Counter(
    "claimlens_errors_total",
    "Total errors encountered in evaluation",
    ["stage"],  # stage: parse | validate | llm | other
)
