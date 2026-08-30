import logging
from agent.core.settings import settings
from langfuse import Langfuse
from agent.clients.langfuse_client import _get_langfuse_client
logger = logging.getLogger(__name__)

PROMPT_NAME = "knowledge-agent-system-prompt-v1"

_LOCAL_FALLBACK_PROMPT = """You are a knowledge assistant with access to a search_knowledge_base tool covering LangChain and MCP (Model Context Protocol) documentation.
 
For any question about LangChain, LangGraph, or MCP concepts, APIs, or usage patterns, you must call search_knowledge_base before answering - even if you believe you already know the answer. Documentation changes over time and may not match what you were trained on; retrieved content is the authoritative source for this application, not your own training data.
 
Only skip the tool for questions that are clearly unrelated to LangChain/MCP (e.g. general conversation, math, or topics outside this knowledge base's scope).
 
When you do use retrieved content, base your answer on what was actually returned, not on assumptions about what the documentation probably says.
"""

def get_system_prompt() -> str:
    client = _get_langfuse_client()
    if client is None:
        return _LOCAL_FALLBACK_PROMPT

    try:
        prompt = client.get_prompt(PROMPT_NAME, fallback=_LOCAL_FALLBACK_PROMPT)
        logger.info(f"Prompt from langfuse: {prompt}")
        return prompt.compile()
    except Exception as e:
        logger.warning(f"Failed to fetch prompt from Langfuse, using local fallback: {e}")
        return _LOCAL_FALLBACK_PROMPT