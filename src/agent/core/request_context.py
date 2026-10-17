"""
Request ID plumbing. One ID per HTTP request:
- taken from the caller's X-Request-ID header if it looks sane, otherwise
  generated;
- stored on request.state (the exception handlers read it from there);
- stored in a ContextVar so EVERY log line emitted while the request runs
  (including from libraries) can carry it via RequestIDFilter;
- echoed back in the X-Request-ID response header.
"""
import logging

logger = logging.getLogger(__name__)
import re
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        try: 
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response