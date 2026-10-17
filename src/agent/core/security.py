"""
require_api_key - checks the X-API-Key header against SECURITY_API_KEY
  using a constant-time comparison. If no key is configured the check is
  skipped in development; in production the app refuses to start without
  one (see validate_security_config, called from main.py's lifespan).

rate_limit - sliding-window limiter, SECURITY_RATE_LIMIT_PER_MINUTE
  requests per client per minute (0 disables it). Clients are identified
  by API key when one is configured, otherwise by IP. The counters live in
  this process's memory, which is correct for ONE worker; with several
  workers or instances each has its own counters, so move this to Redis
  when you scale out.
"""
import hmac
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request

from agent.core.exception import RateLimitExceededException, UnauthorizedException
from agent.core.settings import settings

WINDOW_SECONDS = 60

_hits: Dict[str, Deque[float]] = defaultdict(deque)

def validate_security_config() -> None:
    if settings.ENV == "production" and not settings.security.api_key:
        raise RuntimeError("SECURITY_API_KEY must be set when ENV=production")

async def require_api_key(request:Request) -> None:
    expected = settings.security.api_key
    if not expected:
        return
    provided = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        raise UnauthorizedException()

def _client_key(request:Request) -> str:
    api_key = request.headers.get("x-api-key")
    if settings.security.api_key and api_key:
        return f"key:{api_key[:8]}"
    return f"ip:{request.client.host if request.client else 'unknown'}"

async def rate_limit(request:Request) -> None:
    limit = settings.security.rate_limit_per_minute
    if limit <= 0:
        return

    now = time.monotonic()
    hits = _hits[_client_key(request)]
    while hits and now - hits[0] > WINDOW_SECONDS:
        hits.popleft()

    if len(hits) >= limit:
        retry_after = max(1, int(WINDOW_SECONDS - (now - hits[0])) + 1)
        raise RateLimitExceededException(retry_after)

    hits.append(now)

def reset_rate_limit() -> None:
    _hits.clear()