from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


MAX_MESSAGE_LENGTH = 4000

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH, description="The user's query message")
    thread_id: Optional[str] = Field(
        None,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        description="Conversation/session identifier. Omit on the first "
        "message of a new conversation - the server generates one and "
        "returns it. Send the same value on every follow-up to continue "
        "that conversation; this IS the session identifier, LangGraph has "
        "no separate concept.",
    )

class Citation(BaseModel):
    """One source document the answer was (at least partly) grounded in.
    Extracted mechanically from tool output, not from the model's prose -
    the model might use a source without mentioning it, or phrase a
    citation in a way that's not machine-parseable. Ground truth from the
    actual retrieval call, same reasoning as ChatResponse.tools_used."""

    file_url: Optional[str] = Field(None, description="Source document URL")
    similarity: Optional[float] = Field(None, description="Retrieval similarity score, if available")
    text: Optional[str] = Field(
        None,
        description="Content the LLM actually read for this source - parent_text "
        "when available, otherwise the child chunk text.",
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
    citations: List[Citation] = Field(
        default_factory=list,
        description="Source documents retrieved this turn, deduplicated by "
        "file_url. Empty if no retrieval tool was called, or a called "
        "tool's output wasn't parseable as a search response.",
    )
    thread_id:str = Field(
        ..., description="Always echoed back - the caller's own value if provided, "
        "or a newly generated one if this was a fresh conversation."
    )
    retrieved_contexts: List[str] = Field(
        default_factory=list,
        description="Text of the retrieved passages the model read this turn "
        "(parent text when available).",
    )