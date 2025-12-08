# src/app/api.py
from fastapi import FastAPI, Request
from typing import List
import concurrent.futures
import os
import json
import logging
from pathlib import Path
from time import perf_counter

# prometheus ASGI app
from prometheus_client import make_asgi_app

from src.app.config import settings
from src.app.models import ClaimRequest, ClaimResponse
from src.app.evaluator import evaluate_claim
from src.app import metrics  # import the metrics module

app = FastAPI(title="ClaimLens API", version="0.1")

# mount prometheus /metrics
app.mount("/metrics", make_asgi_app())

PARTIAL_JSONL: Path = settings.PARTIAL_JSONL
PARTIAL_JSONL.parent.mkdir(parents=True, exist_ok=True)

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
logger = logging.getLogger("claimlens-api")


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Middleware to instrument all HTTP requests for Prometheus.
    Adds a histogram for latency and a counter for total requests.
    """
    endpoint = request.url.path
    method = request.method
    metrics.IN_PROGRESS.inc()
    start = perf_counter()
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        elapsed = perf_counter() - start
        metrics.IN_PROGRESS.dec()
        try:
            metrics.REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
            metrics.REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status).inc()
        except Exception:
            # metrics should never crash the app
            logger.exception("Failed to record metrics")


@app.get("/health")
def health():
    return {"status": "ok", "use_groq": settings.USE_GROQ}


@app.post("/evaluate", response_model=ClaimResponse)
def evaluate_endpoint(req: ClaimRequest):
    """
    Evaluate a single claim synchronously (instrumented).
    """
    claim = req.dict()
    endpoint_label = "/evaluate"
    metrics.IN_PROGRESS.inc()
    start = perf_counter()
    try:
        rec = evaluate_claim(claim, llm_timeout=settings.LLM_TIMEOUT)
        # bump counters based on rec meta
        meta = rec.get("meta", {})
        used_fallback = meta.get("used_fallback", True)
        llm_parsed = meta.get("llm_parsed", False)
        cached = meta.get("cached", False)

        if cached:
            metrics.CACHE_HITS.inc()
            metrics.LLM_CALLS.labels(result="skipped_cached").inc()
        else:
            if settings.USE_GROQ:
                # llm_parsed True => LLM success; False => LLM attempted but failed/parse invalid
                if llm_parsed:
                    metrics.LLM_CALLS.labels(result="success").inc()
                else:
                    # if USE_GROQ True but LLM not parsed, that was a failure (or timeouts)
                    metrics.LLM_CALLS.labels(result="failure").inc()

        if used_fallback:
            metrics.FALLBACKS.inc()

        # errors (if present)
        for e in rec.get("errors", []):
            # best-effort parse of stage from message
            stage = "other"
            if isinstance(e, str):
                if "validation_error" in e:
                    stage = "validate"
                elif "llm_call_error" in e:
                    stage = "llm"
                elif "parse_error" in e:
                    stage = "parse"
            metrics.ERRORS.labels(stage=stage).inc()

        append_jsonl(PARTIAL_JSONL, rec)
        return {"claim_id": rec.get("claim_id"), "outputs": rec.get("outputs"), "used_fallback": used_fallback, "errors": rec.get("errors", [])}
    finally:
        elapsed = perf_counter() - start
        metrics.REQUEST_LATENCY.labels(endpoint=endpoint_label).observe(elapsed)
        metrics.REQUEST_COUNT.labels(method="POST", endpoint=endpoint_label, http_status="200").inc()
        metrics.IN_PROGRESS.dec()


@app.post("/evaluate_batch")
def evaluate_batch_endpoint(claims: List[ClaimRequest]):
    results = []
    for c in claims:
        rec = evaluate_claim(c.dict(), llm_timeout=settings.LLM_TIMEOUT)
        append_jsonl(PARTIAL_JSONL, rec)
        # per-claim metrics similar to single evaluate
        meta = rec.get("meta", {})
        used_fallback = meta.get("used_fallback", True)
        cached = meta.get("cached", False)
        if cached:
            metrics.CACHE_HITS.inc()
            metrics.LLM_CALLS.labels(result="skipped_cached").inc()
        else:
            if settings.USE_GROQ:
                metrics.LLM_CALLS.labels(result="success" if meta.get("llm_parsed", False) else "failure").inc()
        if used_fallback:
            metrics.FALLBACKS.inc()
        for e in rec.get("errors", []):
            stage = "other"
            if isinstance(e, str):
                if "validation_error" in e:
                    stage = "validate"
                elif "llm_call_error" in e:
                    stage = "llm"
                elif "parse_error" in e:
                    stage = "parse"
            metrics.ERRORS.labels(stage=stage).inc()

        results.append({"claim_id": rec.get("claim_id"), "outputs": rec.get("outputs"), "used_fallback": used_fallback, "errors": rec.get("errors", [])})

    return {"count": len(results), "results": results}
