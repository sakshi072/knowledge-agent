from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, description="The user's query message")
    thread_id: Optional[str] = Field(
        None,
        description="Conversation/session identifier. Omit on the first "
        "message of a new conversation - the server generates one and "
        "returns it. Send the same value on every follow-up to continue "
        "that conversation; this IS the session identifier, LangGraph has "
        "no separate concept.",
    )

class ChatResponse(BaseModel):
    message:str = Field(..., description="The agent's final answer")
    tools_used: List[str] = Field(
        default_factory=list,
        description="Names of tools the agent actually called this turn - "
        "empty if it answered from its own knowledge without retrieving "
        "anything. Extracted from the graph's real execution, not inferred "
        "from the answer text.",
    )
    thread_id:str = Field(
        ..., description="Always echoed back - the caller's own value if provided, "
        "or a newly generated one if this was a fresh conversation."
    )