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
from langgraph.graph import MessagesState

def make_call_model_node(model_with_tools: BaseChatModel) -> Callable:
    """Returns the actual node function"""
    async def call_model(state:MessagesState):
        response = await model_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    return call_model