from fastapi import APIRouter, Depends
import logging
import uuid
import asyncio

from agent.api.dependencies import get_graph
from agent.core.exception import GraphExecutionException, ServiceUnavailableException, RequestTimeoutException
from agent.core.security import rate_limit, require_api_key
from agent.core.settings import settings
from agent.graph.introspection import extract_tools_used, extract_citations, extract_context
from agent.schemas.chat import ChatRequest, ChatResponse
from agent.clients.langfuse_client import get_langfuse_callback_handler
from agent.core.request_context import request_id_var

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key), Depends(rate_limit)])
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
    config = {
        "configurable": {"thread_id":thread_id},
        "metadata": {
            "langfuse_session_id": thread_id,
            "request_id": request_id_var.get()
        },
    }

    handler = get_langfuse_callback_handler()
    if handler:
        config["callbacks"] = [handler]

    logger.info(f"Chat request, thread_id={thread_id}: {request.message[:50]!r}")

    timeout = settings.agent.request_timeout_seconds
    try:
        resp = await asyncio.wait_for(
            graph.ainvoke(
                {"messages":[{"role":"user", "content":request.message}],
                "iteration_count": 0,
                "guardrail_rejected": False,},
                config=config
            ),
            timeout=timeout
        )
    except asyncio.TimerHandle:
        raise RequestTimeoutException(timeout, thread_id=thread_id)
    except Exception as e:
        raise GraphExecutionException(reason=f"{type(e).__name__}: {e}", thread_id=thread_id)

    messages = resp["messages"]
    final_message = messages[-1].content

    this_turn = messages[resp.get("turn_start_index", 0):]

    tools_used = extract_tools_used(this_turn)
    citations = extract_citations(this_turn)
    retrieved_contexts = extract_context(this_turn)

    logger.info(
        f"Chat response, thread_id={thread_id}: tools_used={tools_used}, "
        f"{len(citations)} citation(s), {len(retrieved_contexts)} context chunk(s), "
        f"answer={final_message[:150]!r}"
    )

    return ChatResponse(
        message=final_message, 
        tools_used=tools_used, 
        citations=citations,
        retrieved_contexts=retrieved_contexts,
        thread_id=thread_id
    )