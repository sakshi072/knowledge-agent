from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agent.core.exception import AgentBaseException
from agent.core.settings import settings
import logging
import traceback

logger = logging.getLogger(__name__)

def get_request_id(request: Request):
    return getattr(request.state, "request_id", "unknown")

def sanitize_error_details(details:dict, is_production:bool = None) -> dict:
    if is_production is None:
        is_production = settings.ENV == "production"

    if not is_production:
        return details

    keywords = {"password", "jwt", "token", "api_key", "access_token", "refresh_token"}
    return {k:v for k,v in details.items() if k.lower() not in keywords}

def should_log_traceback(status_code:int) -> bool:
    return status_code >=500

async def agent_exception_handler(request: Request, exc: AgentBaseException):
    request_id = get_request_id(request)

    if should_log_traceback(exc.status_code):
        logger.error(
            f"{exc.error_code}: {exc.message} | internal_reason={exc.internal_reason!r}",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    else:
        logger.warning(f"{exc.error_code}: {exc.message}")

    detail = sanitize_error_details(exc.details) if exc.details else {}
    detail["request_id"] = request_id
    detail["error_code"] = exc.error_code

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.status_code, 
            "message": exc.message, 
            "detail": detail},
        headers=exc.headers or None
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = get_request_id(request)
    logger.warning(f"[{request_id}] HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.status_code,
            "message": str(exc.detail),
            "detail": {"request_id": request_id, "error_code": "HTTP_ERROR"},
        },
    )

async def validation_exception_handler(request:Request, exc:RequestValidationError):
    request_id = get_request_id(request)
    errors = [
        {"field":"->".join(str(loc) for loc in e["loc"]), "message": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]
    logger.warning(f"[{request_id}] Validation error: {len(errors)} field(s) invalid")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error": status.HTTP_422_UNPROCESSABLE_CONTENT,
            "message": "Request validation error",
            "detail": {"request_id": request_id, "error_code": "VALIDATION_ERROR", "errors": errors},
        },
    )

async def general_exception_handler(request: Request, exc: Exception):
    request_id = get_request_id(request)
    logger.critical(
        f"UNHANDLED EXCEPTION: {type(exc).__name__}: {exc}",
        extra={"traceback": traceback.format_exc()},
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "An unexpected error occurred. Please try again later.",
            "detail": {"request_id": request_id, "error_code": "INTERNAL_SERVER_ERROR"},
        },
    )

def register_exception_handler(app) -> None:
    app.add_exception_handler(AgentBaseException, agent_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    logger.info("Exception handlers registered")