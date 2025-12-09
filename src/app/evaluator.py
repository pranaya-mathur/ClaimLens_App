# src/app/evaluator.py
import os
import json
import re
import logging
import hashlib
import signal
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from contextlib import contextmanager

from pydantic import ValidationError
from .config import settings
from .models import CombinedOut

# Try to import ChatGroq only if installed in the environment
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

logger = logging.getLogger("claimlens.evaluator")

# Module-level flags / caches
_groq_disabled: bool = False
_llm_client = None

# Simple on-disk cache to avoid repeated LLM calls for same narrative
CACHE_FILE = settings.DATA_DIR / "llm_cache.json"
_CACHE: Dict[str, Dict[str, Any]] = {}


class TimeoutException(Exception):
    """Custom exception for timeout"""
    pass


@contextmanager
def timeout_context(seconds: float):
    """
    Context manager for timeout enforcement (Unix only).
    Falls back to no timeout on Windows.
    """
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    # Check if signal.alarm is available (Unix only)
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows - no timeout enforcement
        logger.warning("Timeout not supported on this platform")
        yield


def _load_cache():
    global _CACHE
    try:
        if CACHE_FILE.exists():
            with CACHE_FILE.open("r", encoding="utf-8") as f:
                _CACHE = json.load(f)
        else:
            _CACHE = {}
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        _CACHE = {}


def _save_cache():
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def _narr_hash(narr: str) -> str:
    """Generate hash of narrative for caching"""
    h = hashlib.sha256()
    h.update((narr or "").encode("utf-8"))
    return h.hexdigest()


# Initialize cache at import
_load_cache()


def sanitize_narrative(narr: str, max_chars: int = 1200) -> str:
    """
    Clean up whitespace and truncate long narratives to protect token usage.
    """
    if not narr:
        return ""
    s = re.sub(r"\s+", " ", narr).strip()
    if len(s) > max_chars:
        return s[:max_chars] + " ... [TRUNCATED]"
    return s


def local_fallback(narr: str) -> Dict[str, Any]:
    """
    Deterministic fallback heuristic used when LLM is disabled / fails / returns invalid JSON.
    Keep the heuristic simple and reproducible.
    """
    text = (narr or "").lower()
    score = 0.1
    flags = []

    if len(text) < 30:
        score += 0.25
        flags.append("short-narrative")

    if any(x in text for x in ["kuch", "will file", "will file later", "later", "file later"]):
        score += 0.25
        flags.append("delayed-report")

    if any(x in text for x in ["no fir", "no police", "not reported", "didn't report"]):
        score += 0.25
        flags.append("no-police-report")

    score = min(score, 1.0)
    return {
        "clarity": 4,
        "clarity_explanation": "local fallback heuristic",
        "completeness": 3,
        "completeness_explanation": "local fallback heuristic",
        "timeline_consistency": 4,
        "timeline_explanation": "local fallback heuristic",
        "fraud_risk": round(score, 2),
        "red_flags": flags,
        "fraud_explanation": "local fallback used due to LLM unavailability or parse/validation failure.",
    }


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Naive extraction of the first JSON object found inside LLM text output.
    Returns dict or None.
    """
    if not isinstance(text, str):
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                cand = text[start : i + 1]
                try:
                    return json.loads(cand)
                except Exception:
                    # minor fixes: single quotes -> double, remove trailing commas
                    cand2 = cand.replace("'", '"')
                    cand2 = re.sub(r",\s*}", "}", cand2)
                    cand2 = re.sub(r",\s*]", "]", cand2)
                    try:
                        return json.loads(cand2)
                    except Exception:
                        return None
    return None


def get_llm_client():
    """
    Lazily instantiate the ChatGroq client.
    Raises if USE_GROQ is False or langchain_groq is not installed.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    if not settings.USE_GROQ:
        raise RuntimeError("Groq usage is disabled (USE_GROQ=False)")

    if ChatGroq is None:
        raise RuntimeError("langchain_groq (ChatGroq) is not installed in this environment")

    _llm_client = ChatGroq(model=settings.GROQ_MODEL, temperature=0, timeout=settings.LLM_TIMEOUT)
    return _llm_client


def invoke_llm(prompt: str, timeout: Optional[float] = None):
    """
    Call the LLM client with timeout enforcement and return raw content (dict or str).
    """
    client = get_llm_client()
    
    if timeout and timeout > 0:
        try:
            with timeout_context(timeout):
                resp = client.invoke(prompt)
                return getattr(resp, "content", resp)
        except TimeoutException as e:
            logger.warning(f"LLM call timed out: {e}")
            raise
    else:
        resp = client.invoke(prompt)
        return getattr(resp, "content", resp)


def parse_and_validate(candidate: Any) -> Tuple[Optional[dict], Optional[str]]:
    """
    Try to parse and validate candidate (dict or string) into CombinedOut schema.
    Returns (validated_dict, None) on success or (None, error_message) on failure.
    """
    if candidate is None:
        return None, "no_candidate"

    if isinstance(candidate, dict):
        try:
            validated = CombinedOut(**candidate)
            return validated.dict(), None
        except ValidationError as e:
            return None, f"validation_error:{e}"

    if isinstance(candidate, str):
        parsed = extract_json_from_text(candidate)
        if parsed:
            try:
                validated = CombinedOut(**parsed)
                return validated.dict(), None
            except ValidationError as e:
                return None, f"validation_error:{e}"
        return None, "parse_error:not_json"

    return None, f"unsupported_candidate_type:{type(candidate)}"


def evaluate_claim(claim: dict, llm_timeout: Optional[float] = None) -> dict:
    """
    Evaluate one claim dictionary with proper timeout enforcement.

    Expected claim shape: {"claim_id": str, "narrative": str, "metadata": {...}}

    Behavior:
      - sanitize narrative
      - check cache (if enabled)
      - build compact prompt
      - try LLM (if enabled) with timeout -> parse -> validate
      - on LLM failure or invalid parse/validation -> use local_fallback
      - return a record dict containing outputs, errors and meta information
    """
    global _groq_disabled

    cid = claim.get("claim_id")
    narr_raw = claim.get("narrative", "")
    narr = sanitize_narrative(narr_raw)
    narr_h = _narr_hash(narr)
    rec: Dict[str, Any] = {"claim_id": cid, "narrative": narr_raw, "outputs": {}, "errors": []}

    # If Groq disabled by config, do not attempt LLM; go fallback directly
    if not settings.USE_GROQ:
        rec["outputs"] = local_fallback(narr)
        rec["meta"] = {"used_fallback": True, "llm_parsed": False, "cached": False}
        return rec

    # At this point, USE_GROQ is True. Check if we have cached validated output.
    cache_entry = _CACHE.get(narr_h)
    if cache_entry:
        rec["outputs"] = cache_entry
        rec["meta"] = {"used_fallback": False, "llm_parsed": True, "cached": True}
        return rec

    parsed_candidate = None
    llm_attempted = False

    # Try LLM if configured and not globally disabled
    if settings.USE_GROQ and not _groq_disabled:
        try:
            llm_attempted = True
            prompt = (
                "You are an insurance claim evaluator. RETURN ONLY a single JSON object with EXACT keys:\n"
                '{"clarity":int,"clarity_explanation":str,'
                '"completeness":int,"completeness_explanation":str,'
                '"timeline_consistency":int,"timeline_explanation":str,'
                '"fraud_risk":float,"red_flags":list,"fraud_explanation":str}\n\n'
                f"Narrative:\n{narr}\n\n"
                "Keep explanations concise (1-2 short sentences). If unsure return 0 for numeric fields and [] for red_flags."
            )

            # Use timeout from parameter or settings
            timeout_val = llm_timeout or settings.LLM_TIMEOUT
            raw = invoke_llm(prompt, timeout=timeout_val)

            if isinstance(raw, dict):
                parsed_candidate = raw
            elif isinstance(raw, str):
                parsed_candidate = extract_json_from_text(raw)
            else:
                parsed_candidate = None

        except TimeoutException as e:
            rec["errors"].append(f"llm_timeout_error:{e}")
            logger.warning(f"LLM timeout for {cid}: {e}")
            parsed_candidate = None
            
        except Exception as e:
            msg = str(e).lower()
            rec["errors"].append(f"llm_call_error:{e}")
            logger.warning(f"LLM call error for {cid}: {e}")

            # If this looks like a rate-limit / token-limit error, disable Groq for remainder of run
            if ("rate limit" in msg) or ("tokens per day" in msg) or ("429" in msg) or ("token limit" in msg):
                _groq_disabled = True
                logger.error("Detected token/rate-limit issue; disabling Groq for remainder of process.")

            parsed_candidate = None

    # Validate parsed candidate (if present)
    validated, val_err = parse_and_validate(parsed_candidate)

    if validated:
        rec["outputs"] = validated
        rec["meta"] = {"used_fallback": False, "llm_parsed": True, "cached": False}
        # store in cache to avoid re-query
        try:
            _CACHE[narr_h] = validated
            _save_cache()
        except Exception as e:
            logger.warning(f"Failed to save cache entry: {e}")
    else:
        # Only append a validation error if we attempted to call the LLM.
        if llm_attempted and val_err:
            rec["errors"].append(f"validation_error:{val_err}")
        # fallback path
        rec["outputs"] = local_fallback(narr)
        rec["meta"] = {"used_fallback": True, "llm_parsed": bool(parsed_candidate), "cached": False}

    return rec
