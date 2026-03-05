"""OpenAI-compatible proxy server for NotebookLM."""

import json
import secrets
import time
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from nlm_proxy.core import NotebookLMClient
from nlm_proxy.core.auth import load_cached_tokens
from nlm_proxy.core.auth_refresh import AuthRefreshService
from nlm_proxy.core.config import get_auth_settings, get_cache_settings, get_openai_settings, get_routing_settings, get_tracing_settings
from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.response_cache import ResponseCache
from nlm_proxy.core.tracing import init_tracing, shutdown_tracing, instrument_fastapi, instrument_httpx, get_tracer, add_span_attributes
from nlm_proxy.openai.notebook_cache import NotebookCache
from nlm_proxy.core.agent import AgentCore, RequestOptions, RoutingDecision
from nlm_proxy.core.llm_client import LangChainLLMClient, create_chat_model
from nlm_proxy.core.config import get_agent_settings

from nlm_proxy.openai.session import SessionStore
from nlm_proxy.openai.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    DeltaContent,
    ResponseChoice,
    ResponseMessage,
    Usage,
)

logger = get_logger(__name__)

app = FastAPI(
    title="NotebookLM OpenAI Proxy",
    description="OpenAI-compatible API for NotebookLM",
    version="0.1.0"
)

# Initialize session store (will be configured with TTL in main())
app.state.session_store = None
# Initialize notebook cache for smart routing (will be configured in main())
app.state.notebook_cache = None
# Initialize response cache (will be configured in main())
app.state.response_cache = None
# Initialize auth refresh service (will be configured in main())
app.state.auth_refresh_service = None
# Initialize agent core (will be configured in main())
app.state.agent_core = None

def verify_api_key(authorization: Annotated[str | None, Header()] = None) -> None:
    """Verify the API key from Authorization header."""
    settings = get_openai_settings()

    error_response = {
        "error": {
            "message": "Invalid API key",
            "type": "invalid_request_error",
            "code": "invalid_api_key"
        }
    }

    if not authorization:
        raise HTTPException(status_code=401, detail=error_response)

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail=error_response)

    token = authorization.removeprefix("Bearer ")
    if not secrets.compare_digest(token, settings.api_key):
        raise HTTPException(status_code=401, detail=error_response)


async def get_client() -> NotebookLMClient:
    """Get authenticated NotebookLM client."""
    tokens = load_cached_tokens()
    if not tokens or not tokens.cookies:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Run 'notebooklm-mcp-auth' first."
        )
    client = NotebookLMClient(
        cookies=tokens.cookies,
        csrf_token=tokens.csrf_token or "",
        session_id=tokens.session_id or ""
    )
    await client._ensure_initialized()
    return client


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """List notebooks as available models."""
    logger.debug("[PROXY] Received request: GET /v1/models")

    routing_settings = get_routing_settings()

    client = await get_client()
    try:
        logger.debug("[NOTEBOOKLM] Calling list_notebooks()")
        notebooks = await client.list_notebooks()
        logger.debug(f"[NOTEBOOKLM] Response: {len(notebooks)} notebooks found")

        # Smart router model
        smart_router_model = {
            "id": routing_settings.router_model_name,
            "object": "model",
            "created": 0,
            "owned_by": "nlm-proxy",
            "name": "Knowledge Finder",
            "description": "AI-powered routing to best notebook or external LLM",
        }

        notebook_models = [
            {
                "id": nb.id,
                "object": "model",
                "created": 0,
                "owned_by": "notebooklm",
                "name": nb.title,
                "source_count": nb.source_count,
            }
            for nb in notebooks
        ]

        response = {
            "object": "list",
            "data": [smart_router_model] + notebook_models
        }
        logger.debug(f"[PROXY] Returning {len(response['data'])} models")
        return response
    finally:
        await client.close()


async def _stream_cached_response(decision: RoutingDecision, request: ChatCompletionRequest,
                                   tracing_settings=None):
    """Stream a cached response as SSE (used by both Phase 0 and Phase 2 cache hits)."""
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("smart_router.handle_request") as span:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        cache_result = decision.cache_result
        hit_type = decision.cache_hit_type or "exact"

        # Record tracing attributes
        query = request.messages[-1].content if request.messages else ""
        if tracing_settings and tracing_settings.request_max_length:
            add_span_attributes(user_query=query[:tracing_settings.request_max_length])
        if tracing_settings and tracing_settings.response_max_length:
            add_span_attributes(response_content=cache_result.answer[:tracing_settings.response_max_length])
        add_span_attributes(
            response_source=f"cache_{hit_type}",
            cache_hit_type=hit_type,
            notebook_id=decision.notebook_id or "",
        )

        # Reasoning chunk
        reasoning = decision.reasoning or "Cache hit — returning cached response."
        reasoning_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(reasoning_content=reasoning + "\n\n"))],
        )
        yield f"data: {reasoning_chunk.model_dump_json()}\n\n"

        # Thinking chunk (if available and requested)
        if cache_result.thinking and request.include_thinking:
            thinking_chunk = ChatCompletionChunk(
                id=chunk_id, created=created, model=request.model,
                choices=[Choice(delta=DeltaContent(reasoning_content=cache_result.thinking))],
                system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
            )
            yield f"data: {thinking_chunk.model_dump_json()}\n\n"

        # Answer chunk
        answer_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(content=cache_result.answer))],
            system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
        )
        yield f"data: {answer_chunk.model_dump_json()}\n\n"

        # Final chunk
        final_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
            system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


def _json_cached_response(decision: RoutingDecision, request: ChatCompletionRequest,
                           tracing_settings=None):
    """Return a cached response as JSON (non-streaming cache hit)."""
    from fastapi.responses import JSONResponse
    cache_result = decision.cache_result
    hit_type = decision.cache_hit_type or "exact"

    # Record tracing attributes for cache hit
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("smart_router.handle_request"):
        query = request.messages[-1].content if request.messages else ""
        if tracing_settings and tracing_settings.request_max_length:
            add_span_attributes(user_query=query[:tracing_settings.request_max_length])
        if tracing_settings and tracing_settings.response_max_length:
            add_span_attributes(response_content=cache_result.answer[:tracing_settings.response_max_length])
        add_span_attributes(
            response_source=f"cache_{hit_type}",
            cache_hit_type=hit_type,
            notebook_id=decision.notebook_id or "",
        )

    resp = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[ResponseChoice(
            index=0,
            message=ResponseMessage(
                role="assistant",
                content=cache_result.answer,
                reasoning_content=decision.reasoning or "Cache hit — returning cached response.",
            ),
            finish_reason="stop",
        )],
        usage=Usage(
            prompt_tokens=len(request.messages[-1].content) if request.messages else 0,
            completion_tokens=len(cache_result.answer),
            total_tokens=(len(request.messages[-1].content) if request.messages else 0) + len(cache_result.answer),
        ),
        system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
    )
    return JSONResponse(
        content=resp.model_dump(),
        headers={"X-Cache-Status": f"HIT_{hit_type.upper()}"},
    )


async def stream_smart_response(agent_core: AgentCore, decision: RoutingDecision, query: str,
                                 request: ChatCompletionRequest, chat_id: str = None,
                                 tracing_settings=None):
    """Stream response — Phase 3a of the four-phase pipeline.

    Note: This function creates its own span because the span must live for the full
    streaming duration. The parent function does NOT create a span for streaming.
    """
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        # Add user_query to span
        if tracing_settings and tracing_settings.request_max_length > 0:
            span.set_attribute("user_query", query[:tracing_settings.request_max_length])

        response_source = "llm" if decision.request_type == "llm_task" else "notebooklm"

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        conversation_id = None
        accumulated_response = ""

        # Reasoning chunk
        reasoning_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(reasoning_content=decision.reasoning + "\n\n"))],
        )
        yield f"data: {reasoning_chunk.model_dump_json()}\n\n"

        if decision.request_type == "llm_task":
            # LLM_TASK: stream via LangChain ChatModel.astream()
            from nlm_proxy.core.llm_client import _convert_messages
            lc_messages = _convert_messages(
                [{"role": m.role, "content": m.content} for m in request.messages]
            )
            async for chunk in agent_core.chat_model.astream(lc_messages):
                delta_content = chunk.content if chunk.content else ""
                if delta_content:
                    accumulated_response += delta_content
                    openai_chunk = ChatCompletionChunk(
                        id=chunk_id, created=created, model=request.model,
                        choices=[Choice(delta=DeltaContent(content=delta_content))],
                    )
                    yield f"data: {openai_chunk.model_dump_json()}\n\n"
        else:
            # NOTEBOOKLM: stream via query_stream
            previous_thinking = ""
            previous_answer = ""

            async for chunk in agent_core.query_stream(
                decision.notebook_id, query,
                conversation_id=request.conversation_id,
            ):
                chunk_type = chunk.get("type")
                full_text = chunk.get("text", "")

                # Extract conversation_id from first chunk
                new_conv_id = chunk.get("conversation_id")
                if new_conv_id and not conversation_id:
                    conversation_id = new_conv_id
                    logger.info("conversation_id_from_nlm: conversation_id=%s, notebook_id=%s", conversation_id, decision.notebook_id)
                    agent_core.save_conversation_id(chat_id, conversation_id)

                # Filter thinking unless requested
                if chunk_type == "thinking" and not request.include_thinking:
                    previous_thinking = full_text
                    continue

                if chunk_type == "thinking":
                    delta_text = full_text[len(previous_thinking):]
                    previous_thinking = full_text
                    if delta_text:
                        delta = DeltaContent(reasoning_content=delta_text)
                        openai_chunk = ChatCompletionChunk(
                            id=chunk_id, created=created, model=request.model,
                            choices=[Choice(delta=delta)],
                            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None,
                        )
                        yield f"data: {openai_chunk.model_dump_json()}\n\n"
                else:
                    delta_text = full_text[len(previous_answer):]
                    previous_answer = full_text
                    if delta_text:
                        accumulated_response += delta_text
                        delta = DeltaContent(content=delta_text)
                        openai_chunk = ChatCompletionChunk(
                            id=chunk_id, created=created, model=request.model,
                            choices=[Choice(delta=delta)],
                            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None,
                        )
                        yield f"data: {openai_chunk.model_dump_json()}\n\n"

            # Store in response cache after stream completes
            if (
                agent_core.response_cache
                and accumulated_response
                and conversation_id
                and decision.notebook_id
            ):
                embedding = None
                if agent_core.response_cache._semantic_enabled:
                    emb = agent_core.response_cache._compute_embedding(query)
                    if emb is not None:
                        embedding = emb.tolist()
                agent_core.response_cache.store(
                    notebook_id=decision.notebook_id,
                    query=query,
                    answer=accumulated_response,
                    thinking=previous_thinking or None,
                    conversation_id=conversation_id,
                    embedding=embedding,
                )

        # Add response to trace BEFORE final yield (span still open)
        if tracing_settings and tracing_settings.response_max_length > 0:
            span.set_attribute("response_content", accumulated_response[:tracing_settings.response_max_length])
            span.set_attribute("response_source", response_source)

        # Final chunk
        final_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None,
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


async def _handle_non_streaming(agent_core: AgentCore, decision: RoutingDecision, query: str,
                                 request: ChatCompletionRequest, chat_id: str = None,
                                 tracing_settings=None):
    """Phase 3b: Non-streaming response (both NOTEBOOKLM and LLM_TASK)."""
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        if tracing_settings and tracing_settings.request_max_length > 0:
            span.set_attribute("user_query", query[:tracing_settings.request_max_length])

        if decision.request_type == "llm_task":
            # LLM_TASK: invoke via LangChain ChatModel
            from nlm_proxy.core.llm_client import _convert_messages
            lc_messages = _convert_messages(
                [{"role": m.role, "content": m.content} for m in request.messages]
            )
            result = await agent_core.chat_model.ainvoke(lc_messages)
            response_text = result.content
            response_source = "llm"
        else:
            # NOTEBOOKLM: query via agent_core
            result = await agent_core.query(
                notebook_id=decision.notebook_id,
                query=query,
                conversation_id=request.conversation_id,
            )
            response_text = result.get("answer", "") if result else ""
            response_source = "notebooklm"

            # Save conversation_id to session store
            conv_id = result.get("conversation_id", "") if result else ""
            agent_core.save_conversation_id(chat_id, conv_id)

            # Store in response cache
            if agent_core.response_cache and response_text and conv_id:
                embedding = None
                if agent_core.response_cache._semantic_enabled:
                    emb = agent_core.response_cache._compute_embedding(query)
                    if emb is not None:
                        embedding = emb.tolist()
                agent_core.response_cache.store(
                    notebook_id=decision.notebook_id,
                    query=query,
                    answer=response_text,
                    thinking=None,
                    conversation_id=conv_id,
                    embedding=embedding,
                )

        # Trace response
        if tracing_settings and tracing_settings.response_max_length > 0:
            add_span_attributes(
                response_content=response_text[:tracing_settings.response_max_length],
                response_source=response_source,
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[ResponseChoice(
                index=0,
                message=ResponseMessage(
                    role="assistant",
                    content=response_text,
                    reasoning_content=decision.reasoning,
                ),
                finish_reason="stop",
            )],
            usage=Usage(
                prompt_tokens=len(query),
                completion_tokens=len(response_text),
                total_tokens=len(query) + len(response_text),
            ),
            system_fingerprint=f"conv_{request.conversation_id}" if request.conversation_id else None,
        )


async def handle_smart_routing(request: ChatCompletionRequest, http_request: Request):
    """Handle requests to the smart router model — four-phase pipeline."""
    routing_settings = get_routing_settings()
    tracing_settings = get_tracing_settings()
    agent_core: AgentCore = app.state.agent_core

    if not agent_core:
        raise HTTPException(status_code=503, detail="Agent core not initialized.")

    # Extract chat_id from headers or request metadata
    chat_id = http_request.headers.get("X-OpenWebUI-Chat-Id")
    chat_id_source = "header" if chat_id else None
    if not chat_id and hasattr(request, 'metadata') and request.metadata:
        chat_id = request.metadata.get("chat_id")
        if chat_id:
            chat_id_source = "metadata"

    # Extract allowed_notebooks from request metadata (per-request ACL)
    request_allowed_notebooks = None
    if hasattr(request, 'metadata') and request.metadata:
        raw = request.metadata.get("allowed_notebooks")
        if raw is not None:
            if raw == ["*"]:
                request_allowed_notebooks = None
                logger.debug("[SMART-ROUTER] ACL wildcard ['*'] - all notebooks allowed")
            elif raw == []:
                request_allowed_notebooks = []
                logger.debug("[SMART-ROUTER] ACL empty list [] - no notebooks allowed")
            else:
                request_allowed_notebooks = raw
                logger.debug("[SMART-ROUTER] ACL filter: %d allowed notebooks", len(raw))
        else:
            logger.debug("[SMART-ROUTER] No ACL metadata - all notebooks allowed")
    else:
        logger.debug("[SMART-ROUTER] No metadata - all notebooks allowed")

    # Load conversation_id from session store (via agent_core or fallback)
    conversation_id = None
    stored_conv_id = agent_core.get_conversation_id(chat_id) if agent_core else None
    if stored_conv_id:
        logger.info("session_lookup: chat_id=%s, conversation_id=%s, source=%s", chat_id, stored_conv_id, chat_id_source)
        request.conversation_id = stored_conv_id
        conversation_id = stored_conv_id
    elif chat_id:
        logger.info("session_lookup: chat_id=%s, conversation_id=None, source=%s", chat_id, chat_id_source)

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    query = user_messages[-1].content

    # Build RequestOptions
    options = RequestOptions(
        bypass_cache=request.bypass_cache,
        include_thinking=request.include_thinking,
        allowed_notebooks=request_allowed_notebooks,
        conversation_id=conversation_id,
        chat_id=chat_id,
    )

    # Phase 0+1: Route (includes pre-routing cache check)
    decision = await agent_core.route(query, options)

    # Phase 0 hit → return cached response
    if decision.cache_result:
        if request.stream:
            return StreamingResponse(
                _stream_cached_response(decision, request, tracing_settings),
                media_type="text/event-stream",
                headers={"X-Cache-Status": f"HIT_{decision.cache_hit_type.upper()}"},
            )
        else:
            return _json_cached_response(decision, request, tracing_settings)

    logger.info("[SMART-ROUTER] Decision: %s, notebook=%s", decision.request_type, decision.notebook_id)

    # Phase 2: Post-routing cache check (notebook-scoped, NOTEBOOKLM only)
    if decision.request_type == "notebooklm" and not options.bypass_cache and agent_core.response_cache:
        cache_result, hit_type = await agent_core.response_cache.lookup_async(
            decision.notebook_id, query
        )
        if cache_result:
            decision.cache_result = cache_result
            decision.cache_hit_type = hit_type
            if request.stream:
                return StreamingResponse(
                    _stream_cached_response(decision, request, tracing_settings),
                    media_type="text/event-stream",
                    headers={"X-Cache-Status": f"HIT_{hit_type.upper()}"},
                )
            else:
                return _json_cached_response(decision, request, tracing_settings)

    # Phase 3: Execute query (streaming or non-streaming)
    if request.stream:
        return StreamingResponse(
            stream_smart_response(agent_core, decision, query, request, chat_id, tracing_settings),
            media_type="text/event-stream",
        )
    else:
        return await _handle_non_streaming(agent_core, decision, query, request, chat_id, tracing_settings)


async def stream_response(client, notebook_id: str, query_text: str, request: ChatCompletionRequest, chat_id: str = None, is_first_turn: bool = False):
    """Generate OpenAI-compatible SSE stream from NotebookLM query_stream."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    conversation_id = None
    chunk_count = 0

    # Track previous text to compute deltas (NotebookLM sends cumulative text)
    previous_thinking = ""
    previous_answer = ""

    logger.debug(f"[NOTEBOOKLM] Starting stream query: notebook_id={notebook_id}, query={query_text[:100]}..., conversation_id={request.conversation_id}, chat_id={chat_id}")

    try:
        async for chunk in client.query_stream(
            notebook_id=notebook_id,
            query_text=query_text,
            conversation_id=request.conversation_id
        ):
            chunk_count += 1
            chunk_type = chunk.get("type")
            full_text = chunk.get("text", "")

            logger.debug(f"[NOTEBOOKLM] Received chunk #{chunk_count}: type={chunk_type}, text_len={len(full_text)}")

            # Filter thinking unless requested
            if chunk_type == "thinking" and not request.include_thinking:
                logger.debug(f"[PROXY] Filtering thinking chunk (include_thinking={request.include_thinking})")
                previous_thinking = full_text  # Still track it for delta computation
                continue

            new_conv_id = chunk.get("conversation_id")
            if new_conv_id and not conversation_id:
                conversation_id = new_conv_id
                logger.info(f"conversation_id_from_nlm: conversation_id={conversation_id}, notebook_id={notebook_id}")
                # Save to session store if we have a chat_id
                if chat_id and app.state.agent_core:
                    app.state.agent_core.save_conversation_id(chat_id, conversation_id)

            # Compute delta: NotebookLM sends cumulative text, we need only the new part
            if chunk_type == "thinking":
                delta_text = full_text[len(previous_thinking):]
                previous_thinking = full_text
            else:  # answer
                delta_text = full_text[len(previous_answer):]
                previous_answer = full_text

            logger.debug(f"[PROXY] Delta text length: {len(delta_text)} chars (full={len(full_text)}, previous={len(previous_answer if chunk_type == 'answer' else previous_thinking)})")

            # Only yield if there's new content
            if delta_text:
                # Send thinking as reasoning_content, answers as content (OpenAI o1/o3 format)
                if chunk_type == "thinking":
                    delta = DeltaContent(reasoning_content=delta_text)
                else:  # answer
                    delta = DeltaContent(content=delta_text)

                openai_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=notebook_id,
                    choices=[Choice(delta=delta)],
                    system_fingerprint=f"conv_{conversation_id}" if conversation_id else None
                )
                logger.debug(f"[PROXY] Yielding OpenAI chunk with {len(delta_text)} chars (type={chunk_type})")
                yield f"data: {openai_chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        logger.debug(f"[NOTEBOOKLM] Stream complete: {chunk_count} total chunks, conversation_id={conversation_id}")
        final_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=notebook_id,
            choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        logger.debug("[PROXY] Stream finished")

        # Store in response cache after streaming (with embedding)
        # Note: all turns are cached because client rewrites follow-ups into standalone questions
        if app.state.response_cache and previous_answer and conversation_id:
            logger.debug(
                "[CACHE] Storing streamed response: query='%s', notebook=%s, is_first_turn=%s, answer_len=%d",
                query_text[:80], notebook_id[:12], is_first_turn, len(previous_answer),
            )
            embedding = None
            if app.state.response_cache._semantic_enabled:
                emb = app.state.response_cache._compute_embedding(query_text)
                if emb is not None:
                    embedding = emb.tolist()
            app.state.response_cache.store(
                notebook_id=notebook_id,
                query=query_text,
                answer=previous_answer,
                thinking=previous_thinking or None,
                conversation_id=conversation_id,
                embedding=embedding,
            )
    finally:
        await client.close()


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    # Check if using smart router
    routing_settings = get_routing_settings()
    if request.model == routing_settings.router_model_name:
        return await handle_smart_routing(request, http_request)

    # --- Direct notebook path (model == notebook_id) ---
    agent_core: AgentCore | None = app.state.agent_core

    # Extract chat_id from headers or request metadata
    chat_id = http_request.headers.get("X-OpenWebUI-Chat-Id")
    chat_id_source = "header" if chat_id else None
    if not chat_id and hasattr(request, 'metadata') and request.metadata:
        chat_id = request.metadata.get("chat_id")
        if chat_id:
            chat_id_source = "metadata"

    logger.info(
        f"request_received: model={request.model}, chat_id={chat_id}, "
        f"conversation_id={request.conversation_id}, stream={request.stream}, "
        f"has_metadata={request.metadata is not None}"
    )

    # Load existing conversation_id from session store
    if chat_id and agent_core:
        stored_conv_id = agent_core.get_conversation_id(chat_id)
        if stored_conv_id:
            logger.info(f"session_lookup: chat_id={chat_id}, conversation_id={stored_conv_id}, source={chat_id_source}")
            request.conversation_id = stored_conv_id
        else:
            logger.info(f"session_lookup: chat_id={chat_id}, conversation_id=None, source={chat_id_source}")
    elif chat_id and app.state.session_store:
        stored_conv_id = app.state.session_store.get(chat_id)
        if stored_conv_id:
            logger.info(f"session_lookup: chat_id={chat_id}, conversation_id={stored_conv_id}, source={chat_id_source}")
            request.conversation_id = stored_conv_id
        else:
            logger.info(f"session_lookup: chat_id={chat_id}, conversation_id=None, source={chat_id_source}")
    elif not chat_id:
        logger.debug("session_lookup: chat_id=None, using manual conversation_id mode")

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        logger.error("[PROXY] No user message found in request")
        raise HTTPException(status_code=400, detail="No user message found")

    query_text = user_messages[-1].content
    logger.debug(f"[PROXY] Extracted query: {query_text[:500]}{'...' if len(query_text) > 500 else ''}")

    # First-turn detection
    is_first_turn = request.conversation_id is None
    if chat_id and agent_core:
        stored_conv_id = agent_core.get_conversation_id(chat_id)
        if stored_conv_id:
            is_first_turn = False
    elif chat_id and app.state.session_store:
        stored_conv_id = app.state.session_store.get(chat_id)
        if stored_conv_id:
            is_first_turn = False

    # Cache check via AgentCore.handle_direct_query() (unified path)
    options = RequestOptions(
        bypass_cache=request.bypass_cache,
        conversation_id=request.conversation_id,
    )
    if agent_core:
        cache_result, hit_type = await agent_core.handle_direct_query(
            request.model, query_text, options
        )
    else:
        cache_result, hit_type = None, None

    if cache_result:
        if request.stream:
            # Streaming cache HIT: yield cached answer as SSE chunks
            async def stream_cached_response(cache_result, hit_type):
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())
                # Thinking chunk (if available)
                if cache_result.thinking and request.include_thinking:
                    thinking_chunk = ChatCompletionChunk(
                        id=chunk_id, created=created, model=request.model,
                        choices=[Choice(delta=DeltaContent(reasoning_content=cache_result.thinking))],
                        system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
                    )
                    yield f"data: {thinking_chunk.model_dump_json()}\n\n"
                # Answer chunk
                answer_chunk = ChatCompletionChunk(
                    id=chunk_id, created=created, model=request.model,
                    choices=[Choice(delta=DeltaContent(content=cache_result.answer))],
                    system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
                )
                yield f"data: {answer_chunk.model_dump_json()}\n\n"
                # Final chunk
                final_chunk = ChatCompletionChunk(
                    id=chunk_id, created=created, model=request.model,
                    choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
                    system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(
                stream_cached_response(cache_result, hit_type),
                media_type="text/event-stream",
                headers={"X-Cache-Status": f"HIT_{hit_type.upper()}"},
            )
        else:
            # Non-streaming cache HIT
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[ResponseChoice(
                    index=0,
                    message=ResponseMessage(role="assistant", content=cache_result.answer),
                    finish_reason="stop"
                )],
                usage=Usage(
                    prompt_tokens=len(query_text),
                    completion_tokens=len(cache_result.answer),
                    total_tokens=len(query_text) + len(cache_result.answer),
                ),
                system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
            )
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=response.model_dump(),
                headers={"X-Cache-Status": f"HIT_{hit_type.upper()}"},
            )

    # --- Cache miss: query NotebookLM ---

    if request.stream:
        logger.debug("[PROXY] Using streaming response")
        client = await get_client()
        return StreamingResponse(
            stream_response(client, request.model, query_text, request, chat_id, is_first_turn),
            media_type="text/event-stream"
        )

    # Non-streaming path — use AgentCore if available, else fallback to direct client
    logger.debug("[PROXY] Using non-streaming response")
    logger.debug(f"[NOTEBOOKLM] Calling query: notebook_id={request.model}, query={query_text[:100]}..., conversation_id={request.conversation_id}")

    if agent_core:
        result = await agent_core.query(
            notebook_id=request.model,
            query=query_text,
            conversation_id=request.conversation_id,
        )
    else:
        client = await get_client()
        try:
            result = await client.query(
                notebook_id=request.model,
                query_text=query_text,
                conversation_id=request.conversation_id,
            )
        finally:
            await client.close()

    answer = result.get("answer", "") if result else ""
    conv_id = result.get("conversation_id", "") if result else ""

    # Save conversation_id to session store
    if agent_core:
        agent_core.save_conversation_id(chat_id, conv_id)
    elif chat_id and conv_id and app.state.session_store:
        app.state.session_store.set(chat_id, conv_id)
    elif chat_id and not conv_id:
        logger.info(f"session_not_saved: chat_id={chat_id}, reason=no_conversation_id_from_nlm")

    logger.debug(f"[NOTEBOOKLM] Response received: answer_len={len(answer)}, conversation_id={conv_id}")
    logger.debug(f"[NOTEBOOKLM] Answer preview: {answer[:200]}{'...' if len(answer) > 200 else ''}")

    # Handle empty responses gracefully
    if not answer or not answer.strip():
        logger.warning(f"[NOTEBOOKLM] Empty answer received for query: {query_text[:100]}...")
        answer = "I apologize, but I couldn't generate a response for that query. This might happen when the question doesn't relate to the notebook content or uses unsupported formatting. Please try rephrasing your question."

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[ResponseChoice(
            index=0,
            message=ResponseMessage(role="assistant", content=answer),
            finish_reason="stop"
        )],
        usage=Usage(prompt_tokens=len(query_text), completion_tokens=len(answer), total_tokens=len(query_text) + len(answer)),
        system_fingerprint=f"conv_{conv_id}" if conv_id else None
    )

    # Store in response cache (with embedding)
    response_cache = agent_core.response_cache if agent_core else app.state.response_cache
    if response_cache and answer and conv_id:
        logger.debug(
            "[CACHE] Storing response: query='%s', notebook=%s, is_first_turn=%s",
            query_text[:80], request.model[:12], is_first_turn,
        )
        embedding = None
        if response_cache._semantic_enabled:
            emb = response_cache._compute_embedding(query_text)
            if emb is not None:
                embedding = emb.tolist()
        response_cache.store(
            notebook_id=request.model,
            query=query_text,
            answer=answer,
            thinking=None,
            conversation_id=conv_id,
            embedding=embedding,
        )

    logger.debug(f"[PROXY] Returning response with {len(answer)} characters")
    return response


@app.post("/v1/embeddings", dependencies=[Depends(verify_api_key)])
async def embeddings():
    """Embeddings endpoint - not supported by NotebookLM."""
    raise HTTPException(
        status_code=501,
        detail="Embeddings not supported. NotebookLM does not provide embedding generation."
    )


@app.get("/v1/sessions", dependencies=[Depends(verify_api_key)])
async def list_sessions():
    """List all active sessions (for debugging)."""
    if not app.state.session_store:
        raise HTTPException(status_code=503, detail="Session store not initialized")

    sessions = app.state.session_store.list_all()
    return {
        "sessions": sessions,
        "count": len(sessions)
    }


@app.delete("/v1/sessions/{chat_id}", dependencies=[Depends(verify_api_key)])
async def delete_session(chat_id: str):
    """Delete a specific session."""
    if not app.state.session_store:
        raise HTTPException(status_code=503, detail="Session store not initialized")

    deleted = app.state.session_store.delete(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session not found: {chat_id}")

    return {"status": "deleted", "chat_id": chat_id}


@app.get("/v1/sessions/stats", dependencies=[Depends(verify_api_key)])
async def session_stats():
    """Get session statistics."""
    if not app.state.session_store:
        raise HTTPException(status_code=503, detail="Session store not initialized")

    return app.state.session_store.get_stats()


# ── Response Cache Management ────────────────────────────────────────────


@app.get("/v1/cache/stats", dependencies=[Depends(verify_api_key)])
async def cache_stats():
    """Get response cache statistics."""
    if not app.state.response_cache:
        return {
            "enabled": False,
            "entry_count": 0,
            "notebook_count": 0,
        }

    cache = app.state.response_cache
    stats = cache.get_stats()
    stats.update({
        "enabled": True,
        "max_entries": cache._max_entries,
        "ttl_seconds": cache._ttl_seconds,
        "semantic_enabled": cache._semantic_enabled,
    })
    return stats


@app.delete("/v1/cache", dependencies=[Depends(verify_api_key)])
async def clear_cache():
    """Clear all response cache entries."""
    if not app.state.response_cache:
        raise HTTPException(status_code=503, detail="Response cache not initialized")

    count = app.state.response_cache.entry_count
    app.state.response_cache.clear()
    logger.info(f"[CACHE] Cleared all {count} entries via API")
    return {"status": "cleared", "entries_removed": count}


@app.delete("/v1/cache/{notebook_id}", dependencies=[Depends(verify_api_key)])
async def clear_cache_notebook(notebook_id: str):
    """Clear response cache entries for a specific notebook."""
    if not app.state.response_cache:
        raise HTTPException(status_code=503, detail="Response cache not initialized")

    app.state.response_cache.invalidate_notebook(notebook_id)
    logger.info(f"[CACHE] Cleared cache for notebook {notebook_id} via API")
    return {"status": "cleared", "notebook_id": notebook_id}

def main(host: str = "0.0.0.0", port: int = 8080, session_ttl: int = 86400):
    """Run the OpenAI-compatible proxy server."""
    # Initialize OpenTelemetry tracing
    init_tracing()
    instrument_fastapi(app)
    instrument_httpx()

    # Initialize session store with configured TTL
    app.state.session_store = SessionStore(ttl_seconds=session_ttl)
    logger.info(f"Session store initialized with TTL={session_ttl}s ({session_ttl/3600:.1f} hours)")

    # Initialize response cache
    routing_settings = get_routing_settings()
    cache_settings = get_cache_settings()
    if cache_settings.response_cache_enabled:
        try:
            llm_client = None
            if cache_settings.semantic_match_enabled and routing_settings.llm_api_key:
                chat_model = create_chat_model(
                    model=routing_settings.llm_model,
                    base_url=routing_settings.llm_base_url,
                    api_key=routing_settings.llm_api_key,
                )
                llm_client = LangChainLLMClient(chat_model=chat_model)

            app.state.response_cache = ResponseCache(
                max_entries=cache_settings.response_cache_max_entries,
                ttl_seconds=cache_settings.response_cache_ttl,
                semantic_enabled=cache_settings.semantic_match_enabled,
                llm_client=llm_client,
                embedding_model=cache_settings.embedding_model,
                similarity_threshold=cache_settings.similarity_threshold,
                similarity_exact_threshold=cache_settings.similarity_exact_threshold,
                top_k=cache_settings.semantic_match_top_k,
            )
            logger.info(
                "Response cache initialized: max_entries=%d, ttl=%ds, semantic=%s",
                cache_settings.response_cache_max_entries,
                cache_settings.response_cache_ttl,
                cache_settings.semantic_match_enabled,
            )
        except Exception as e:
            logger.error(f"Failed to initialize response cache: {e}")
    else:
        logger.debug("Response cache disabled (NLM_PROXY_CACHE_RESPONSE_CACHE_ENABLED=false)")

    # Initialize notebook cache with proactive refresh for smart routing
    # Smart routing is enabled if llm_api_key is configured
    if routing_settings.llm_api_key:
        try:
            tokens = load_cached_tokens()
            if tokens and tokens.cookies:
                nlm_client = NotebookLMClient(
                    cookies=tokens.cookies,
                    csrf_token=tokens.csrf_token or "",
                    session_id=tokens.session_id or "",
                    notebook_cache=None,  # Will be set after cache is created
                )
                # Wire on_sources_changed callback to invalidate response cache
                on_sources_changed = None
                if app.state.response_cache:
                    on_sources_changed = app.state.response_cache.invalidate_notebook

                app.state.notebook_cache = NotebookCache(
                    nlm_client=nlm_client,
                    ttl_seconds=routing_settings.summary_cache_ttl,
                    allowed_notebooks=routing_settings.allowed_notebooks,
                    on_sources_changed=on_sources_changed,
                )
                # Now wire notebook_cache back to the client
                nlm_client._notebook_cache = app.state.notebook_cache
                logger.info(f"Notebook cache initialized with TTL={routing_settings.summary_cache_ttl}s")

                # Create shared ChatModel for routing, L3 verification, and LLM_TASK
                agent_settings = get_agent_settings()
                chat_model = create_chat_model(
                    model=routing_settings.llm_model,
                    provider=agent_settings.llm_provider,
                    base_url=routing_settings.llm_base_url,
                    api_key=routing_settings.llm_api_key,
                )

                # Create AgentCore singleton (replaces per-request SmartRouter)
                app.state.agent_core = AgentCore(
                    nlm_client=nlm_client,
                    notebook_cache=app.state.notebook_cache,
                    response_cache=app.state.response_cache,
                    chat_model=chat_model,
                    session_store=app.state.session_store,
                    routing_settings=routing_settings,
                )
                logger.info("AgentCore initialized (singleton)")
            else:
                logger.warning("Smart routing configured but no auth tokens found - cache not initialized")
        except Exception as e:
            logger.error(f"Failed to initialize notebook cache / AgentCore: {e}")
    else:
        logger.debug("Smart routing not configured (no llm_api_key) - notebook cache not initialized")

    # Start background auth refresh service
    auth_settings = get_auth_settings()
    if auth_settings.auto_refresh_enabled:
        app.state.auth_refresh_service = AuthRefreshService(
            csrf_refresh_interval=auth_settings.csrf_refresh_interval,
            cookie_refresh_interval=auth_settings.cookie_refresh_interval,
            headless_port=auth_settings.headless_port,
        )
        app.state.auth_refresh_service.start()
    else:
        logger.info("[AUTH_REFRESH] Auto-refresh disabled (NLM_PROXY_AUTH_AUTO_REFRESH_ENABLED=false)")

    import uvicorn

    # Logging is now configured centrally via setup_logging() in cli.py

    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        # Shutdown tracing with timeout to prevent hanging on exit
        shutdown_tracing(timeout_seconds=3)
        # Stop auth refresh service
        if app.state.auth_refresh_service:
            app.state.auth_refresh_service.stop()
        # Cleanup notebook cache on shutdown
        if app.state.notebook_cache:
            app.state.notebook_cache.shutdown()
        # Cleanup session store on shutdown
        if app.state.session_store:
            app.state.session_store.shutdown()


if __name__ == "__main__":
    main()
