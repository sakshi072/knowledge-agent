"""
Isolated from graph logic on purpose - this module owns the connection to
knowledge-mcp-server, nothing else. The graph layer depends on this
returning a list of LangChain tools; it doesn't know or care how those
tools were obtained.
"""
import logging
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent.core.settings import settings

from typing import List

logger = logging.getLogger(__name__)

async def load_mcp_tools() -> List[BaseTool]:
    """
    Connects to knowledge-mcp-server and loads its tools
    as native LangChain tools.
    currently only knowledge_base
    """
    client = MultiServerMCPClient(
        {
            "knowledge_base": {
                "url": settings.mcp.server_url,
                "transport": "streamable_http"
            }
        }
    )
    tools = await client.get_tools()
    logger.info(f"Loaded {len(tools)} tool(s) from MCP server: {[t.name for t in tools]}")
    return tools