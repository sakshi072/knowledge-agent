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
                parts.append(block)
        return "".join(parts)
    return ""

def _iter_sources(messages: List):
    """Yield every search source from every ToolMessage, in order."""
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data = json.loads(_get_text_content(msg))
        except (json.JSONDecodeError, TypeError):
            continue
        sources = data.get("sources") if isinstance(data, dict) else None
        for source in sources or []:
            yield source

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
    """One citation per file_url. When several chunks come from the same page,
    keep the one with the highest similarity (not simply the first seen)."""
    best_by_url = {}  # file_url -> source dict

    for source in _iter_sources(messages):
        file_url = source.get("file_url")
        if not file_url:
            continue
        current = best_by_url.get(file_url)
        if current is None or (source.get("similarity") or 0) > (current.get("similarity") or 0):
            best_by_url[file_url] = source

    return [
        Citation(
            file_url=url,
            similarity=src.get("similarity"),
            text=src.get("parent_text") or src.get("text"),
        )
        for url, src in best_by_url.items()
    ]

def extract_context(messages: List) -> List[str]:
    seen,context = set(), []

    for source in _iter_sources(messages):
        text = source.get("parent_text") or source.get("text")
        if text and text not in seen:
            seen.add(text)
            context.append(text)

    return context