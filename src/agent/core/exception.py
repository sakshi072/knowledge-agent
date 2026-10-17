from typing import Optional, Dict, Any

class AgentBaseException(Exception):
    """Base for all custom agent exceptions - mirrors rag-mcp-service's
    RetrievalBaseException pattern (message/error_code/status_code/details
    all live on the exception instance) so a single generic handler can
    render any subclass correctly, without a handler per exception type."""

    def __init__(
        self,
        message:str,
        error_code:str = "AGENT_ERROR",
        status_code: int = 500,
        details:Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        internal_reasons: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.headers = headers or {}
        self.internal_reasons = internal_reasons
        super().__init__(message)

class InvalidChatRequestException(AgentBaseException):
    """Raised for malformed chat input caught at the service boundary,
    not just relying on Pydantic's own validation (same philosophy as
    rag-mcp-service's InvalidQueryException - validate explicitly where a
    caller could bypass the HTTP schema, e.g. a future non-HTTP caller)."""
 
    def __init__(self, reason: str):
        super().__init__(
            message="Invalid chat request",
            error_code="INVALID_CHAT_REQUEST",
            status_code=400,
            details={"reason": reason},
        )

class GraphExecutionException(AgentBaseException):
    """Raised when graph.ainvoke() itself fails - an LLM call error, an
    MCP tool call failure propagating up, etc. Wrapping this distinctly
    (rather than letting the raw exception bubble up) means the HTTP
    response is intentional and sanitized, not whatever a third-party
    library's exception happens to str() into."""
 
    def __init__(self, reason: str, thread_id: Optional[str] = None):
        super().__init__(
            message="Agent failed to process the request",
            error_code="GRAPH_EXECUTION_FAILED",
            status_code=502,
            details={"thread_id": thread_id},
            internal_reasons=reason
        )

class RequestTimeoutException(AgentBaseException):
    """The whole request exceeded AGENT_REQUEST_TIMEOUT_SECONDS."""

    def __init__(self, timeout_seconds: float, thread_id: Optional[str]=None):
        super().__init__(
            message="The request took too long to complete",
            error_code="REQUEST_TIMEOUT",
            status_code=504,
            details={"thread_id": thread_id},
            internal_reasons=f"exceeded {timeout_seconds}"
        )
class ServiceUnavailableException(AgentBaseException):
    """Raised when the agent graph hasn't been initialized yet (e.g. MCP
    tool loading failed at startup) - mirrors rag-mcp-service's identical
    exception, same shape, same reason for existing."""
 
    def __init__(self, service_name: str, reason: str):
        super().__init__(
            message=f"{service_name} is not available",
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            details={"reason": reason},
        )

class UnauthorizedException(AgentBaseException):
    """Missing or wrong X-API-Key."""

    def __init__(self):
        super().__init__(
            message="Missing or invalid API key",
            error_code="UNAUTHORIZED",
            status_code=401,
            headers={"WWW-Authenticate": "ApiKey"},
        )

class RateLimitExceededException(AgentBaseException):
    def __init__(self, retry_after_seconds: int):
        super().__init__(
            message="Too many requests",
            error_code="RATE_LIMITED",
            status_code=429,
            details={"retry_after_seconds": retry_after_seconds},
            headers={"Retry-After": str(retry_after_seconds)},
        )