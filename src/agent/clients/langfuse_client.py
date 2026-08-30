import logging
 
from agent.core.settings import settings
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)
 
_langfuse_client = None
_langfuse_init_attempted = False

def _get_langfuse_client():
    global _langfuse_client, _langfuse_init_attempted

    if _langfuse_init_attempted:
        return _langfuse_client
    _langfuse_init_attempted = True

    if not settings.langfuse.public_key or not settings.langfuse.secret_key:
        logger.info("Langfuse not configured - using local system prompt")
        return None

    try:
        _langfuse_client = Langfuse(
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
            host=settings.langfuse.base_url
        )
        logger.info("Langfuse client initialized for prompt management")
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse client, using local prompt: {e}")
        _langfuse_client = None

    return _langfuse_client

def get_langfuse_callback_handler():
    """Returns None if Langfuse isn't configured"""
    client = _get_langfuse_client()
    if client is None:
        return None

    try:
        return CallbackHandler()
    except Exception as e:
        logger.warning(f"Failed to create Langfuse handler: {e}")
        return None
