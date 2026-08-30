from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from agent.core.settings import settings
import logging

logger = logging.getLogger(__name__)

_pool = None

async def get_checkpointer():
    if settings.postgres.checkpoint_url is None:
        return InMemorySaver()

    global _pool
    _pool = AsyncConnectionPool(
        conninfo=settings.postgres.checkpoint_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )

    await _pool.open()

    checkpointer = AsyncPostgresSaver(conn=_pool)
    await checkpointer.setup()
    logger.info("PostgresSaver initialized with a real connection pool (min=1, max=10)")
    return checkpointer

async def close_checkpointer()->None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Postgres connection pool closed")