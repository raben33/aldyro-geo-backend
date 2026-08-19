"""
Aldyro.com — Section 2 "GEO AI Audit Terminal"
==============================================
Dedicated FastAPI proxy engine between the public audit widget and the
Google GenAI inference layer (gemini-2.5-flash-lite).

Design constraints:
  * Zero secrets in source. GEMINI_API_KEY is read from process env only.
  * Origin-locked CORS (no wildcard, credentials disabled).
  * Sliding 24h per-IP quota enforced BEFORE any upstream token spend.
  * Model output is schema-constrained and re-validated server-side,
    so the browser can never receive malformed or unbounded scores.

Runtime target: Vercel Python Serverless Function (ASGI, uvicorn workers).
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------------------
# 1. LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("aldyro.geo-audit")

# ---------------------------------------------------------------------------
# 2. CONFIGURATION (all tunables in one place, all overridable by env)
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Quota boundary: 2 evaluation scans per unique IP per rolling 24h window.
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "2"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", str(24 * 60 * 60)))

# Defensive ceiling on the tracking dict so a spoofed X-Forwarded-For flood
# cannot grow the process heap without bound.
MAX_TRACKED_IPS = int(os.getenv("MAX_TRACKED_IPS", "20000"))

# Upstream timeout in milliseconds. Keep this BELOW your Vercel maxDuration.
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "22000"))

# Origin allow-list. Explicit entries only — never "*" on a credentialed API.
ALLOWED_ORIGINS: List[str] = [
    "[aldyro.com](https://aldyro.com)",
    "[aldyro.com](https://www.aldyro.com)",
    "[localhost](http://localhost:3000)",
    "[127.0.0.1](http://127.0.0.1:3000)",
    "[localhost](http://localhost:5173)",
    "[127.0.0.1](http://127.0.0.1:5173)",
    "[localhost](http://localhost:8000)",
    "[127.0.0.1](http://127.0.0.1:8000)",
]

# Exact wire payload for an exhausted quota (contract with the frontend).
RATE_LIMIT_PAYLOAD: Dict[str, str] = {
    "status": "RATE_LIMIT_EXHAUSTED",
    "message": (
        "Evaluation quota fulfilled for this node. Secure your custom "
        "architecture blueprint via communication credentials row."
    ),
}

# Sub-score ceilings. Server-authoritative: the model cannot exceed these.
SUBSCORE_CAPS: Dict[str, int] = {
    "crawler_accessibility_score": 20,
    "schema_graph_score": 20,
    "entity_resolution_score": 20,
    "llm_citations_score": 40,
}

# Hosts that must never be accepted as an audit target.
BLOCKED_HOST_FRAGMENTS = ("localhost", "metadata.google.internal", ".local", ".internal")


# ---------------------------------------------------------------------------
# 3. SYSTEM INSTRUCTION (the anti-gimmick contract given to the model)
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """\
You are the GEO Audit Engine for Aldyro — a Generative Engine Optimization
analyzer. You assess how discoverable, parseable and citable a brand's web
presence is to LLM-driven answer surfaces (ChatGPT, Gemini, Perplexity,
Copilot, Google AI Overviews).

OUTPUT CONTRACT — non-negotiable:
- Emit exactly ONE raw JSON object. Nothing before it, nothing after it.
- No greeting, no preamble, no explanation, no closing remark.
- No markdown of any kind: no code fences, no asterisks, no bullet glyphs,
  no headings. The first character must be "{" and the last must be "}".
- Use only these keys, spelled exactly: geo_readiness_score,
  crawler_accessibility_score, schema_graph_score, entity_resolution_score,
  llm_citations_score, actionable_quick_fix, system_directive_warning.

WEIGHTED SCORING MODEL — evaluate each axis independently:
- crawler_accessibility_score (integer, max 20): robots.txt and llms.txt
  posture toward AI user-agents, server-side vs client-side render
  dependency, HTTP status hygiene, canonical consistency, sitemap coverage.
- schema_graph_score (integer, max 20): presence and correctness of
  structured data — Organization, WebSite, Product, Article, FAQPage,
  BreadcrumbList — plus @id linkage and sameAs graph completeness.
- entity_resolution_score (integer, max 20): how unambiguously the brand
  resolves as a named entity across authoritative corpora (Wikidata,
  Wikipedia, Crunchbase, LinkedIn, official social profiles) and internal
  about/team/contact signals.
- llm_citations_score (integer, max 40): likelihood of being retrieved and
  quoted verbatim by answer engines — extractable factual passages,
  question-shaped headings, comparison and pricing tables, third-party
  editorial mentions, freshness and content depth.
- geo_readiness_score (integer) MUST equal the exact arithmetic sum of the
  four sub-scores. Never let the headline number contradict the breakdown.

CALIBRATION RULES — anti-gimmick, mandatory:
- Never emit a deflated shock score. If the brand is a globally recognised
  authority (major multinational, household consumer name, top-tier
  platform), constrain geo_readiness_score to 65-82 inclusive and distribute
  the four sub-scores so they sum precisely to that value.
- For smaller, regional or unfamiliar brands, score on observable evidence
  without artificial suppression or inflation.
- Never award 100, and never award below 30. Both read as unserious.

EVIDENCE RULES:
- Every qualitative claim must reference a concrete, verifiable technical
  artifact — for example "Organization node missing a sameAs array",
  "no llms.txt at the domain root", "primary specification table rendered
  client-side after hydration".
- Never fabricate metrics. No invented traffic figures, no invented citation
  counts, no invented Core Web Vitals numbers, no invented crawl logs.

FIELD: actionable_quick_fix
- One genuinely free, copy-pasteable remediation tailored to the supplied
  target_url and brand_name.
- Either a complete, valid JSON-LD block wrapped in a
  <script type="application/ld+json"> tag, or a concrete <head> meta fix.
- It must be syntactically valid and use the real supplied domain — no
  example.com, no placeholder tokens the user has to guess at.

FIELD: system_directive_warning
- Exactly two sentences, written in professional technical risk-register
  tone, describing the retrieval or attribution consequence of the gaps
  identified.
- No clickbait, no urgency theatre, no fear language, no phrases such as
  "shocking", "you are losing", "act now", "critical alert".
"""


# ---------------------------------------------------------------------------
# 4. I/O SCHEMAS
# ---------------------------------------------------------------------------
class AuditRequest(BaseModel):
    """Inbound payload from the Section 2 terminal widget."""

    brand_name: str = Field(..., min_length=1, max_length=120)
    target_url: str = Field(..., min_length=4, max_length=2048)

    @field_validator("brand_name")
    @classmethod
    def _clean_brand(cls, v: str) -> str:
        # Collapse whitespace and strip control chars / markdown injection bait.
        v = re.sub(r"[\x00-\x1f\x7f]", "", v)
        v = re.sub(r"\s+", " ", v).strip()
        if not v:
            raise ValueError("brand_name cannot be blank")
        return v

    @field_validator("target_url")
    @classmethod
    def _clean_url(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^https?://", v, flags=re.IGNORECASE):
            v = f"[{v}](https://{v})"

        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("target_url must be a valid http(s) URL")

        host = parsed.hostname.lower()

        # Block loopback / RFC1918 / link-local targets so this endpoint can
        # never be repurposed as an internal-network reconnaissance oracle.
        try:
            addr = ipaddress.ip_address(host)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                raise ValueError("target_url resolves to a non-public address")
        except ValueError as exc:
            if "non-public" in str(exc):
                raise
            # Not a literal IP — fall through to hostname screening.

        if any(frag in host for frag in BLOCKED_HOST_FRAGMENTS):
            raise ValueError("target_url host is not auditable")
        if "." not in host:
            raise ValueError("target_url must contain a public domain")

        return v


class GeoAuditResult(BaseModel):
    """
    Doubles as the structured-output schema handed to the model AND the
    server-side validation gate for whatever comes back.
    """

    geo_readiness_score: int
    crawler_accessibility_score: int
    schema_graph_score: int
    entity_resolution_score: int
    llm_citations_score: int
    actionable_quick_fix: str
    system_directive_warning: str


# ---------------------------------------------------------------------------
# 5. RATE LIMITER — sliding window, in-memory, thread-safe
# ---------------------------------------------------------------------------
class SlidingWindowRateLimiter:
    """
    Per-key sliding-window counter backed by a plain dict.

    Swap this class for a Redis/Vercel-KV implementation to make the quota
    durable across serverless instances; the call sites below only depend on
    reserve() / refund() / retry_after().
    """

    def __init__(self, max_hits: int, window_seconds: int, max_keys: int) -> None:
        self._max_hits = max_hits
        self._window = window_seconds
        self._max_keys = max_keys
        self._hits: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def _prune_locked(self, key: str, now: float) -> List[float]:
        """Drop timestamps that have aged out of the window."""
        cutoff = now - self._window
        stamps = [t for t in self._hits.get(key, ()) if t > cutoff]
        if stamps:
            self._hits[key] = stamps
        else:
            self._hits.pop(key, None)
        return stamps

    def _evict_locked(self, now: float) -> None:
        """Bound heap growth: sweep expired keys, then trim oldest if needed."""
        if len(self._hits) <= self._max_keys:
            return
        cutoff = now - self._window
        for key in list(self._hits.keys()):
            if not any(t > cutoff for t in self._hits[key]):
                self._hits.pop(key, None)
        if len(self._hits) > self._max_keys:
            ordered = sorted(self._hits.items(), key=lambda kv: max(kv[1]))
            for key, _ in ordered[: len(self._hits) - self._max_keys]:
                self._hits.pop(key, None)

    def reserve(self, key: str) -> Tuple[bool, int]:
        """
        Atomically claim one slot.

        Returns (allowed, remaining_after_this_call). The slot is consumed
        BEFORE the upstream call so a burst of parallel requests cannot race
        past the ceiling.
        """
        now = time.time()
        with self._lock:
            stamps = self._prune_locked(key, now)
            # ---- Quota gate. Change to ">= self._max_hits - 1" to block on
            # ---- the 2nd attempt instead of allowing 2 full scans.
            if len(stamps) >= self._max_hits:
                return False, 0
            stamps.append(now)
            self._hits[key] = stamps
            self._evict_locked(now)
            return True, max(0, self._max_hits - len(stamps))

    def refund(self, key: str) -> None:
        """Return the most recent slot — used when the upstream call fails."""
        with self._lock:
            stamps = self._hits.get(key)
            if stamps:
                stamps.pop()
                if not stamps:
                    self._hits.pop(key, None)

    def retry_after(self, key: str) -> int:
        """Seconds until the oldest recorded hit leaves the window."""
        now = time.time()
        with self._lock:
            stamps = self._prune_locked(key, now)
            if not stamps:
                return 0
            return max(1, int(self._window - (now - min(stamps))))


rate_limiter = SlidingWindowRateLimiter(
    max_hits=RATE_LIMIT_MAX,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    max_keys=MAX_TRACKED_IPS,
)


# ---------------------------------------------------------------------------
# 6. GENAI CLIENT — lazily built singleton, reused across warm invocations
# ---------------------------------------------------------------------------
_client: Optional[genai.Client] = None
_client_lock = threading.Lock()


def get_genai_client() -> genai.Client:
    """
    Build (once) and return the GenAI client.

    The credential is resolved from server configuration state at call time —
    it is never imported, never logged, never returned to the caller.
    """
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not present in the environment")
            _client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
            )
    return _client


# ---------------------------------------------------------------------------
# 7. APP + MIDDLEWARE
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Aldyro GEO AI Audit Engine",
    version="1.0.0",
    description="Proxy engine for the Section 2 GEO AI Audit Terminal.",
    docs_url=None,      # Interactive docs disabled on the public surface.
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,            # Token-free public endpoint: no cookies.
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    expose_headers=["X-RateLimit-Remaining", "Retry-After"],
    max_age=600,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Attach conservative hardening headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# 8. HELPERS
# ---------------------------------------------------------------------------
def resolve_client_ip(request: Request) -> str:
    """
    Identify the caller.

    On Vercel the socket peer is the platform edge, so the left-most entry of
    X-Forwarded-For is the real client. request.client.host is the fallback
    for local/uvicorn runs. Only trust XFF because a controlled proxy sets it.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        candidate = forwarded.split(",")[0].strip()
        if candidate:
            return candidate

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    return request.client.host if request.client else "unknown"


def strip_markdown_artifacts(text: str) -> str:
    """Remove asterisk emphasis and stray fence markers from model strings."""
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("```", "").replace("**", "")
    return text.strip()


def extract_json_object(raw: str) -> Dict[str, Any]:
    """
    Coerce the model's text into a dict.

    Structured output makes this nearly always a straight json.loads, but the
    brace-slice fallback keeps a stray token from breaking the endpoint.
    """
    candidate = strip_markdown_artifacts(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("model response contained no JSON object")
        return json.loads(candidate[start : end + 1])


def reconcile_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Server-authoritative normalisation.

    Sub-scores are clamped to their declared ceilings and the headline score
    is RECOMPUTED as their sum, so the breakdown and the total can never
    disagree on the wire. Note: we deliberately do not inflate a low total
    here — calibration is the model's job, and silently rewriting it upward
    would be exactly the kind of vanity metric this engine avoids.
    """
    clean: Dict[str, Any] = {}

    for key, cap in SUBSCORE_CAPS.items():
        try:
            value = int(round(float(payload.get(key, 0))))
        except (TypeError, ValueError):
            value = 0
        clean[key] = max(0, min(cap, value))

    computed = sum(clean[k] for k in SUBSCORE_CAPS)
    reported = payload.get("geo_readiness_score")
    if isinstance(reported, (int, float)) and int(reported) != computed:
        log.warning(
            "Score mismatch corrected: model reported %s, sub-scores sum to %s",
            reported,
            computed,
        )
    clean["geo_readiness_score"] = computed

    clean["actionable_quick_fix"] = strip_markdown_artifacts(
        str(payload.get("actionable_quick_fix", "")).strip()
    )
    clean["system_directive_warning"] = strip_markdown_artifacts(
        str(payload.get("system_directive_warning", "")).strip()
    )

    # Reorder for a stable, readable wire shape.
    return {
        "geo_readiness_score": clean["geo_readiness_score"],
        "crawler_accessibility_score": clean["crawler_accessibility_score"],
        "schema_graph_score": clean["schema_graph_score"],
        "entity_resolution_score": clean["entity_resolution_score"],
        "llm_citations_score": clean["llm_citations_score"],
        "actionable_quick_fix": clean["actionable_quick_fix"],
        "system_directive_warning": clean["system_directive_warning"],
    }


def build_user_prompt(brand_name: str, target_url: str) -> str:
    """Compose the per-request analysis brief."""
    return (
        "Run a GEO readiness audit on the following target and return the "
        "single JSON object defined by your output contract.\n\n"
        f"brand_name: {brand_name}\n"
        f"target_url: {target_url}\n\n"
        "Derive the four sub-scores from what is technically verifiable about "
        "this brand and this domain, apply the calibration rules, and make "
        "geo_readiness_score the exact sum of the four sub-scores. Tailor "
        "actionable_quick_fix to this exact domain and brand entity."
    )


# ---------------------------------------------------------------------------
# 9. ROUTES
# ---------------------------------------------------------------------------
@app.get("/api/v1/health", include_in_schema=False)
async def health() -> JSONResponse:
    """Liveness probe. Reports credential presence without exposing it."""
    return JSONResponse(
        {
            "status": "OPERATIONAL",
            "model": GEMINI_MODEL,
            "credential_bound": bool(os.getenv("GEMINI_API_KEY")),
        }
    )


@app.post("/api/v1/geo-audit")
async def geo_audit(payload: AuditRequest, request: Request) -> JSONResponse:
    """
    Primary audit route.

    Order of operations is deliberate:
      1. Validate input (Pydantic, already done by the time we get here).
      2. Enforce the per-IP quota — no upstream token is spent past the limit.
      3. Call the inference layer with a constrained response schema.
      4. Re-validate and normalise before anything reaches the browser.
    """
    client_ip = resolve_client_ip(request)

    # --- Step 2: quota boundary --------------------------------------------
    allowed, remaining = rate_limiter.reserve(client_ip)
    if not allowed:
        retry_after = rate_limiter.retry_after(client_ip)
        log.info("Quota exhausted for %s (retry in %ss)", client_ip, retry_after)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=RATE_LIMIT_PAYLOAD,
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Remaining": "0",
            },
        )

    # --- Step 3: inference -------------------------------------------------
    try:
        client = get_genai_client()

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=build_user_prompt(payload.brand_name, payload.target_url),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.35,            # Low variance for repeatable scoring.
                top_p=0.9,
                max_output_tokens=2048,
                response_mime_type="application/json",
                response_schema=GeoAuditResult,   # Hard structural guarantee.
            ),
        )

        raw_text = (response.text or "").strip()
        if not raw_text:
            raise ValueError("empty completion returned by the inference layer")

        result = reconcile_payload(extract_json_object(raw_text))
        GeoAuditResult(**result)   # Final type gate before egress.

    except RuntimeError as exc:
        # Missing credential — configuration fault, not the caller's fault.
        rate_limiter.refund(client_ip)
        log.error("Configuration fault: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "ENGINE_UNCONFIGURED",
                "message": "Audit engine credential is not bound on this node.",
            },
        )

    except genai_errors.APIError as exc:
        # Upstream rejected or throttled us: do not burn the visitor's quota.
        rate_limiter.refund(client_ip)
        log.error("Upstream APIError (code=%s): %s", getattr(exc, "code", "n/a"), exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "status": "UPSTREAM_INFERENCE_FAILURE",
                "message": "The inference layer did not return a usable audit. Retry shortly.",
            },
        )

    except (ValueError, json.JSONDecodeError) as exc:
        rate_limiter.refund(client_ip)
        log.error("Response contract violation: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "status": "MALFORMED_INFERENCE_PAYLOAD",
                "message": "The audit payload failed schema validation and was discarded.",
            },
        )

    except Exception as exc:  # noqa: BLE001 — final containment boundary
        rate_limiter.refund(client_ip)
        log.exception("Unhandled fault in geo-audit: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "ENGINE_FAULT",
                "message": "An unexpected error interrupted the audit run.",
            },
        )

    # --- Step 4: egress ----------------------------------------------------
    log.info(
        "Audit complete | ip=%s brand=%s score=%s remaining=%s",
        client_ip,
        payload.brand_name,
        result["geo_readiness_score"],
        remaining,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=result,
        headers={"X-RateLimit-Remaining": str(remaining)},
    )


# ---------------------------------------------------------------------------
# 10. LOCAL DEVELOPMENT ENTRYPOINT (ignored by Vercel)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
