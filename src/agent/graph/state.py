from langgraph.graph import MessagesState

class KnowledgeAgentState(MessagesState):
    """Extends the minimal MessagesState with one field: how many times
    call_model has run WITHIN the current user turn's processing. Plain
    int field, no reducer (unlike `messages`, which uses add_messages to
    APPEND across calls) - this is deliberate: we want "last write wins"
    behavior so api/routes/chat.py can explicitly reset it to 0 on every
    fresh .ainvoke() call, while call_model still correctly increments it
    across MULTIPLE internal call_model<->tools loop iterations within
    that single invocation.
 
    Without the explicit per-turn reset, this would otherwise accumulate
    across the ENTIRE checkpointed conversation (since checkpointing
    persists the whole state dict, not just messages) - eventually
    capping out and refusing to call tools even for a fresh, simple
    question on turn 20 of a long conversation. That would be wrong;
    this field measures "how deep into a single reasoning loop are we
    right now", not "how many messages has this conversation had ever".
    """

    iteration_count:int
    guardrail_rejected:bool
    turn_start_index: int