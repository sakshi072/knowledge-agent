import logging
from agent.core.logging import configure_logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from agent.core.settings import settings
from agent.core.exception_handler import register_exception_handler
from agent.api.dependencies import set_graph
from agent.api.routes import chat
from agent.graph.builder import build_graph

configure_logging()

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app:FastAPI):
    """Builds the graph ONCE at startup (loads MCP tools, binds the
    model, compiles with a checkpointer) - not per-request."""

    logger.info("Starting enterprise-knowledge-agent...")
    try:
        graph = await build_graph()
        set_graph(graph)
        logger.info("Agent graph initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent graph: {e}")
        raise

    yield

    logger.info("Shutting down knowledge-agent...")

app = FastAPI(title="Knowledge Agent", lifespan=lifespan)

register_exception_handler(app)
app.include_router(chat.router)

@app.get("/health")
async def health():
    return {"status": "healthy"}