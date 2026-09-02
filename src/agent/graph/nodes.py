"""
Node logic lives here, separated from graph WIRING (graph/builder.py) -
this file doesn't know about StateGraph, edges, or ToolNode at all, only
what a single step of the conversation does. Exported as a factory
(make_call_model_node) rather than a bare function, since the node needs
a model already bound to tools - that binding happens in builder.py,
where the tools are actually loaded, not here.
"""
from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage

from agent.graph.prompts import get_system_prompt
from agent.graph.state import AgentState
from agent.graph.introspection import extract_tools_used, extract_citations
import logging

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 4000

def make_call_model_node(model_with_tools: BaseChatModel) -> Callable:
    """Returns the actual node function"""
    async def call_model(state: AgentState):
        messages_with_system = [SystemMessage(content=get_system_prompt())] + state["messages"]
        response = await model_with_tools.ainvoke(messages_with_system)

        current_count = state.get("iteration_count", 0)
        return {"messages": [response], "iteration_count": current_count+1}

    return call_model

async def max_iteration_node(state:AgentState):
    """Graceful termination when the loop guard trips - without this,
    forcing END directly after call_model could leave the last message
    as a tool-call-only AIMessage with EMPTY content (the model was mid-
    request-a-tool when cut off), which would be a confusing, blank
    response to return to a user."""
    return {
        "messages":[
            AIMessage(
                content=(
                    "I wasn't able to complete this within a reasonable number of "
                    "steps. Could you try rephrasing or simplifying your question?"
                )
            )
        ]
    }

async def input_guardrail_node(state:AgentState):
    """Deterministic, no LLM call needed - runs BEFORE call_model, as the
    graph's entry point. Two checks:
    - empty/whitespace-only content: Pydantic's min_length=1 at the API
      schema layer only checks raw string LENGTH, not whether it's
      meaningful - a single space passes that check but shouldn't reach
      the LLM.
    - excessive length: nothing currently stops an oversized input from
      reaching (and costing) an LLM call - a real, previously unaddressed
      abuse/cost vector, not a hypothetical one.
    """
    user_message = state["messages"][-1]
    turn_start_index = len(state["messages"])
    content = user_message.content if isinstance(user_message.content, str) else ""

    if not content.strip():
        return {
            "guardrail_rejected": True,
            "turn_start_index": turn_start_index,
            "messages": [AIMessage(content="Your message appears to be empty. Could you rephrase your question?")],
        }

    if len(content) > MAX_INPUT_LENGTH:
        return {
            "guardrail_rejected": True,
             "turn_start_index": turn_start_index,
            "messages": [AIMessage(content=f"Your message is too long ({len(content)} characters, limit is "
                        f"{MAX_INPUT_LENGTH}). Could you shorten your question?")]
        }

    return {"guardrail_reject":False,  "turn_start_index": turn_start_index,}

_INTERNAL_IDENTIFIER_BLOCKLIST = [
    "AgentState",
    "input_guardrail_node",
    "max_iterations_node",
    "route_after_model",
    "route_after_guardrail",
]

async def output_guardrail_node(state:AgentState):
    latest_message = state["messages"][-1]
    content = latest_message.content if isinstance(latest_message.content, str) else ""

    leaked = [name for name in _INTERNAL_IDENTIFIER_BLOCKLIST if name in content]
    if leaked:
        logger.warning(
            f"Output guardrail caught internal identifier leakage: {leaked} - "
            f"replacing fabricated answer with a safe fallback"
        )
        return {
            "messages":
            [
                AIMessage(
                    content="I wasn't able to generate a reliable answer to that. Could you rephrase your question?",
                     id=latest_message.id,
                )
            ]
        }

    turn_start_index = state.get("turn_start_index", 0)
    this_turns_messages = state["messages"][turn_start_index:]
    tools_used_this_turn = extract_tools_used(this_turns_messages)

    if tools_used_this_turn:
        citations_this_turn = extract_citations(this_turns_messages)
        if not citations_this_turn:
            logger.warning(
                f"Citation gate triggered: tool(s) {tools_used_this_turn} called this turn "
                f"but no citations found - replacing answer with an honest non-answer"
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I don't have information about that in my knowledge base. "
                            "Could you rephrase your question or ask about a different "
                            "LangChain/MCP topic?"
                        ),
                        id=latest_message.id,
                    )
                ]
            }
 
    return {}
