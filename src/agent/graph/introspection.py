from typing import List
from langchain_core.messages import AIMessage

def extract_tools_used(messages:List) -> List[str]:
    seen = []

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name and name not in seen:
                    seen.append(name)

    return seen