from typing import List
from langchain_core.messages import AIMessage, ToolMessage
from agent.schemas.chat import Citation
import json
import logging

logger = logging.getLogger(__name__)

def _get_text_content(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts = []
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(str)
        return "".join(parts)
    return ""

def extract_tools_used(messages:List) -> List[str]:
    seen = []

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name and name not in seen:
                    seen.append(name)

    return seen

def extract_citations(messages: List) -> List[Citation]:
    citations_by_url = {}

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data = json.loads(_get_text_content(msg))
        except (json.JSONDecodeError, TypeError):
            continue

        sources = data.get("sources") if isinstance(data, dict) else None
        if not sources:
            continue

        for source in sources:
            file_url = source.get("file_url")
            if not file_url:
                continue
            if file_url not in citations_by_url:
                citations_by_url[file_url] = Citation(
                    file_url=file_url, similarity=source.get("similarity"), text=source.get("parent_text") or source.get("text"),
                )
    return list(citations_by_url.values())