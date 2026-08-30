"""
Graph WIRING only - node logic lives in graph/nodes.py, the external MCP
connection lives in clients/mcp_client.py, the checkpointer lives in
graph/checkpointer.py. This file's only job is assembling those pieces:
2 nodes (call_model, tools) + a conditional edge + a loop-back, the
standard ReAct tool-calling shape using LangGraph's own prebuilt helpers
(ToolNode, tools_condition)

Flow: user question -> input_guardrail (deterministic checks: empty/
      whitespace-only, excessive length) -> if rejected, straight to END
      with a clear rejection message already appended; otherwise ->
      call_model (LLM decides: answer, or call a tool?) ->
      route_after_model: if iteration_count is over the limit, force a
      graceful termination (max_iterations node) regardless of what the
      model wants to do next. Otherwise, delegate to LangGraph's own
      tools_condition: tool call requested -> tools node -> loop back to
      call_model; no tool call -> END.

Async on purpose: loading MCP tools requires an await, so the graph can't
be built at plain import time.

MEMORY: thread_id (passed at invocation time via
config={"configurable": {"thread_id": ...}}) IS the session identifier -
LangGraph has no separate "session" concept. Same thread_id across calls
accumulates history via the checkpointer; a different thread_id starts
fresh.
"""
from agent.clients.mcp_client import load_mcp_tools
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from agent.graph.nodes import make_call_model_node, max_iteration_node, input_guardrail_node
from agent.graph.checkpointer import get_checkpointer
from agent.core.settings import settings
from agent.graph.state import AgentState

MAX_ITERATIONS = settings.agent.max_tool_call_iterations

def route_after_input(state:AgentState):
    if state.get("guardrail_rejected", False):
        return END
    return "call_model"

def route_after_model(state:AgentState):
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "max_iterations"
    return tools_condition(state)

async def build_graph(checkpointer=None):
    tools = await load_mcp_tools()

    model = init_chat_model(settings.agent.agent_model)
    model_with_tools = model.bind_tools(tools)

    call_model = make_call_model_node(model_with_tools)

    graph = StateGraph(MessagesState)
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("max_iterations", max_iteration_node)

    graph.set_entry_point("input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_input,
        {"call_mode": "call_model", END:END}
    )
    graph.add_conditional_edges(
        "call_model", 
        route_after_model, 
        {"tools": "tools", "__end__": END, "max_iterations": "max_iterations"},
      )
    graph.add_edge("tools", "call_model")
    graph.add_edge("max_iterations", END)

    if checkpointer is None:
        checkpointer = get_checkpointer()

    return graph.compile(checkpointer=checkpointer)