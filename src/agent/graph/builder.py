"""
Graph WIRING only - node logic lives in graph/nodes.py, the external MCP
connection lives in clients/mcp_client.py, the checkpointer lives in
graph/checkpointer.py. This file's only job is assembling those pieces:
2 nodes (call_model, tools) + a conditional edge + a loop-back, the
standard ReAct tool-calling shape using LangGraph's own prebuilt helpers
(ToolNode, tools_condition)

Flow: user question -> call_model (LLM decides: answer, or call a tool?)
      -> if tool call: tools node executes it (via an MCP-loaded tool)
         -> back to call_model with the tool's result
      -> if no tool call: END, LLM's response is the final answer

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
from agent.graph.nodes import make_call_model_node
from agent.graph.checkpointer import get_checkpointer
from agent.core.settings import settings

async def build_graph():
    tools = await load_mcp_tools()

    model = init_chat_model(settings.agent.agent_model)
    model_with_tools = model.bind_tools(tools)

    call_model = make_call_model_node(model_with_tools)

    graph = StateGraph(MessagesState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("call_model")
    graph.add_conditional_edges("call_model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "call_model")

    return graph.compile(checkpointer=get_checkpointer())

