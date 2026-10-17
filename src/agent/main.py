import logging
from agent.core.logging import configure_logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from agent.core.logging import configure_logging
from agent.core.request_context import RequestIDMiddleware
from agent.core.security import validate_security_config
from agent.core.exception_handler import register_exception_handler
from agent.api.dependencies import set_graph, get_graph
from agent.api.routes import chat
from agent.graph.builder import build_graph
from agent.graph.checkpointer import get_checkpointer, close_checkpointer

from dotenv import load_dotenv
load_dotenv()

configure_logging()

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app:FastAPI):
    """Builds the graph ONCE at startup (loads MCP tools, binds the
    model, compiles with a checkpointer) - not per-request."""

    logger.info("Starting enterprise-knowledge-agent...")
    validate_security_config()
    try:
        checkpointer = await get_checkpointer()
        graph = await build_graph(checkpointer)
        set_graph(graph)
        logger.info("Agent graph initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent graph: {e}")
        raise

    yield

    logger.info("Shutting down knowledge-agent...")
    await close_checkpointer()

app = FastAPI(title="Knowledge Agent", lifespan=lifespan)

app.add_middleware(RequestIDMiddleware)
register_exception_handler(app)
app.include_router(chat.router)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    """Readiness: the graph (MCP tools + model + checkpointer) is built,
    so /chat can actually serve. Use this for deploy health checks."""
    if get_graph() is None:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready"}