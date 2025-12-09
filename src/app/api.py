# src/app/api.py
from fastapi import FastAPI, Request, HTTPException
from typing import List
import concurrent.futures
import asyncio
import os
import json
import logging
from pathlib import Path
from time import perf_counter

# prometheus ASGI app
from prometheus_client import make_asgi_app

# rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.app.config import settings
from src.app.models import ClaimRequest, ClaimResponse
from src.app.evaluator import evaluate_claim
from src.app import metrics  # import the metrics module

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="ClaimLens API", version="0.2")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# mount prometheus /metrics
app.mount("/metrics", make_asgi_app())

PARTIAL_JSONL: Path = settings.PARTIAL_JSONL
PARTIAL_JSONL.parent.mkdir(parents=True, exist_ok=True)

# Increased workers from 2 to 10 for better concurrency
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
logger = logging.getLogger("claimlens-api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def append_jsonl_async(path: Path, obj: dict):
    """Async file writing to avoid blocking request processing"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _append_jsonl_sync, path, obj)


def _append_jsonl_sync(path: Path, obj: dict):
    """Synchronous file append helper"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception as e:
            logger.warning(f"fsync failed: {e}")


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
    status = "500"  # default to error
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception as e:
        logger.exception(f"Request failed: {e}")
        raise
    finally:
        elapsed = perf_counter() - start
        metrics.IN_PROGRESS.dec()
        try:
            metrics.REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
            metrics.REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status).inc()
        except Exception as e:
            # metrics should never crash the app
            logger.exception(f"Failed to record metrics: {e}")


@app.get("/health")
@limiter.limit("10/minute")  # Even health checks get rate limited
def health(request: Request):
    return {"status": "ok", "use_groq": settings.USE_GROQ, "version": "0.2"}


@app.post("/evaluate", response_model=ClaimResponse)
@limiter.limit("60/minute")  # 60 requests per minute per IP
async def evaluate_endpoint(request: Request, req: ClaimRequest):
    """
    Evaluate a single claim synchronously (instrumented).
    Rate limited to 60 requests/minute per IP.
    """
    claim = req.dict()
    endpoint_label = "/evaluate"
    metrics.IN_PROGRESS.inc()
    start = perf_counter()
    try:
        # Run evaluation in executor with timeout
        loop = asyncio.get_event_loop()
        rec = await loop.run_in_executor(
            _executor,
            evaluate_claim,
            claim,
            settings.LLM_TIMEOUT
        )
        
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

        # Async file write to avoid blocking
        await append_jsonl_async(PARTIAL_JSONL, rec)
        
        return {
            "claim_id": rec.get("claim_id"),
            "outputs": rec.get("outputs"),
            "used_fallback": used_fallback,
            "errors": rec.get("errors", [])
        }
    finally:
        elapsed = perf_counter() - start
        metrics.REQUEST_LATENCY.labels(endpoint=endpoint_label).observe(elapsed)
        metrics.REQUEST_COUNT.labels(method="POST", endpoint=endpoint_label, http_status="200").inc()
        metrics.IN_PROGRESS.dec()


@app.post("/evaluate_batch")
@limiter.limit("20/minute")  # Lower rate for batch operations
async def evaluate_batch_endpoint(request: Request, claims: List[ClaimRequest]):
    """
    Evaluate multiple claims in batch.
    Rate limited to 20 requests/minute per IP.
    Maximum 100 claims per batch.
    """
    # Validate batch size
    if len(claims) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(claims)} exceeds maximum of 100 claims"
        )
    
    if len(claims) == 0:
        raise HTTPException(
            status_code=400,
            detail="Batch must contain at least 1 claim"
        )
    
    results = []
    loop = asyncio.get_event_loop()
    
    for c in claims:
        try:
            # Run evaluation in executor
            rec = await loop.run_in_executor(
                _executor,
                evaluate_claim,
                c.dict(),
                settings.LLM_TIMEOUT
            )
            
            # per-claim metrics similar to single evaluate
            meta = rec.get("meta", {})
            used_fallback = meta.get("used_fallback", True)
            cached = meta.get("cached", False)
            
            if cached:
                metrics.CACHE_HITS.inc()
                metrics.LLM_CALLS.labels(result="skipped_cached").inc()
            else:
                if settings.USE_GROQ:
                    metrics.LLM_CALLS.labels(
                        result="success" if meta.get("llm_parsed", False) else "failure"
                    ).inc()
            
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

            # Async file write
            await append_jsonl_async(PARTIAL_JSONL, rec)
            
            results.append({
                "claim_id": rec.get("claim_id"),
                "outputs": rec.get("outputs"),
                "used_fallback": used_fallback,
                "errors": rec.get("errors", [])
            })
        except Exception as e:
            logger.exception(f"Failed to process claim {c.claim_id}: {e}")
            results.append({
                "claim_id": c.claim_id,
                "outputs": {},
                "used_fallback": True,
                "errors": [f"processing_error: {str(e)}"]
            })

    return {"count": len(results), "results": results}
