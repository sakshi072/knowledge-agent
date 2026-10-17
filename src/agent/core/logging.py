import logging
import sys

from agent.core.request_context import RequestIDFilter

def configure_logging(level:str = "INFO") -> None:
    """
    Called once at process startup. Every line carries the request ID of
    the HTTP request being served ("-" outside a request).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIDFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    logging.getLogger("httpx").setLevel(logging.WARNING)