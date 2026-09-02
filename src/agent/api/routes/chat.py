from fastapi import APIRouter
import logging
import uuid

from agent.api.dependencies import get_graph
from agent.core.exception import GraphExecutionException, ServiceUnavailableException
from agent.graph.introspection import extract_tools_used, extract_citations
from agent.schemas.chat import ChatRequest, ChatResponse
from agent.clients.langfuse_client import get_langfuse_callback_handler

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request:ChatRequest
):
    graph = get_graph()
    if graph is None:
        raise ServiceUnavailableException(
            "Agent grap",
            "Not initialized"
        )

    # thread_id IS the session identifier - no separate concept. Generate
    # one for a fresh conversation; the caller must send it back on every
    # follow-up to continue that same conversation.
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id":thread_id}}

    handler = get_langfuse_callback_handler()
    if handler:
        config["callbacks"] = [handler]

    logger.info(f"Chat request, thread_id={thread_id}: {request.message[:50]!r}")

    try:
        resp = await graph.ainvoke(
            {"messages":[{"role":"user", "content":request.message}],
             "iteration_count": 0,
             "guardrail_rejected": False,},
            config=config
        )
    except Exception as e:
        raise GraphExecutionException(reason=str(e), thread_id=thread_id)

    messages = resp["messages"]
    final_message = messages[-1].content

    tools_used = extract_tools_used(messages)
    citations = extract_citations(messages)
    logger.info(
        f"Chat response, thread_id={thread_id}: tools_used={tools_used}, "
    )

    return ChatResponse(
        message=final_message, 
        tools_used=tools_used, 
        citations=citations,
        thread_id=thread_id
    )