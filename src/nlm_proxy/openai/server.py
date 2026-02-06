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
from nlm_proxy.core.config import get_openai_settings, get_routing_settings, get_tracing_settings
from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.tracing import init_tracing, shutdown_tracing, instrument_fastapi, instrument_httpx, get_tracer, add_span_attributes
from nlm_proxy.openai.notebook_cache import NotebookCache
from nlm_proxy.openai.router import SmartRouter, RequestType
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


async def stream_smart_response(client, router: SmartRouter, decision, query: str, request: ChatCompletionRequest, chat_id: str = None, tracing_settings=None):
    """Stream response with routing reasoning as reasoning_content and response tracing.

    Note: This function creates its own span because the span must live for the full
    streaming duration. The parent function (handle_smart_routing) does NOT create
    a span for streaming requests to avoid duplicates.
    """
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        # Add user_query to span (using configurable max length)
        if tracing_settings and tracing_settings.request_max_length > 0:
            span.set_attribute("user_query", query[:tracing_settings.request_max_length])

        # Determine response source
        response_source = "llm" if decision.request_type == RequestType.LLM_TASK else "notebooklm"

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        conversation_id = None
        accumulated_response = ""

        # First, stream the routing decision as reasoning_content
        reasoning_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=request.model,
            choices=[Choice(delta=DeltaContent(reasoning_content=decision.reasoning + "\n\n"))]
        )
        yield f"data: {reasoning_chunk.model_dump_json()}\n\n"

        if decision.request_type == RequestType.LLM_TASK:
            # Stream from external LLM
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            stream = await router.llm_client.stream(messages)
            async for chunk in stream:
                # Safety check: some chunks may have empty choices array
                if not chunk.choices:
                    continue

                # Safety check: delta and content may be None
                delta = chunk.choices[0].delta
                delta_content = delta.content if delta and delta.content else ""

                if delta_content:
                    accumulated_response += delta_content  # Accumulate for tracing
                    openai_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=[Choice(delta=DeltaContent(content=delta_content))]
                    )
                    yield f"data: {openai_chunk.model_dump_json()}\n\n"
        else:
            # Stream from NotebookLM - reuse existing logic
            previous_thinking = ""
            previous_answer = ""

            async for chunk in client.query_stream(
                notebook_id=decision.notebook_id,
                query_text=query,
                conversation_id=request.conversation_id
            ):
                chunk_type = chunk.get("type")
                full_text = chunk.get("text", "")

                # Extract conversation_id from first chunk
                new_conv_id = chunk.get("conversation_id")
                if new_conv_id and not conversation_id:
                    conversation_id = new_conv_id
                    # Save to session store if we have a chat_id
                    if chat_id and app.state.session_store:
                        app.state.session_store.set(chat_id, conversation_id)
                        logger.debug(f"[SMART-ROUTER] Saved session: chat_id={chat_id}, conversation_id={conversation_id}")

                if chunk_type == "thinking" and not request.include_thinking:
                    previous_thinking = full_text
                    continue

                if chunk_type == "thinking":
                    delta_text = full_text[len(previous_thinking):]
                    previous_thinking = full_text
                    if delta_text:
                        delta = DeltaContent(reasoning_content=delta_text)
                        openai_chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=request.model,
                            choices=[Choice(delta=delta)]
                        )
                        yield f"data: {openai_chunk.model_dump_json()}\n\n"
                else:
                    delta_text = full_text[len(previous_answer):]
                    previous_answer = full_text
                    if delta_text:
                        accumulated_response += delta_text  # Accumulate for tracing
                        delta = DeltaContent(content=delta_text)
                        openai_chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=request.model,
                            choices=[Choice(delta=delta)]
                        )
                        yield f"data: {openai_chunk.model_dump_json()}\n\n"

        # Add response to trace BEFORE final yield (span still open)
        if tracing_settings and tracing_settings.response_max_length > 0:
            span.set_attribute("response_content", accumulated_response[:tracing_settings.response_max_length])
            span.set_attribute("response_source", response_source)

        # Final chunk
        final_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=request.model,
            choices=[Choice(delta=DeltaContent(), finish_reason="stop")]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


async def handle_smart_routing(request: ChatCompletionRequest, http_request: Request):
    """Handle requests to the smart router model."""
    routing_settings = get_routing_settings()
    tracing_settings = get_tracing_settings()
    tracer = get_tracer(__name__)

    client = await get_client()

    # Use shared notebook cache from app.state
    if not app.state.notebook_cache:
        await client.close()
        raise HTTPException(
            status_code=503,
            detail="Notebook cache not initialized. Smart routing is not available."
        )

    router = SmartRouter(
        nlm_client=client,
        notebook_cache=app.state.notebook_cache,
        llm_base_url=routing_settings.llm_base_url,
        llm_api_key=routing_settings.llm_api_key,
        llm_model=routing_settings.llm_model,
        allowed_notebooks=routing_settings.allowed_notebooks
    )

    # Extract chat_id from headers or request metadata
    chat_id = http_request.headers.get("X-OpenWebUI-Chat-Id")
    if not chat_id and hasattr(request, 'metadata') and request.metadata:
        chat_id = request.metadata.get("chat_id")

    logger.debug(f"[SMART-ROUTER] Extracted chat_id: {chat_id}")

    # Load existing conversation_id from session store if chat_id exists
    if chat_id and app.state.session_store:
        stored_conv_id = app.state.session_store.get(chat_id)
        if stored_conv_id:
            logger.info(f"[SMART-ROUTER] Reusing conversation: chat_id={chat_id}, conversation_id={stored_conv_id}")
            request.conversation_id = stored_conv_id
        else:
            logger.info(f"[SMART-ROUTER] New conversation for chat_id={chat_id}")

    # For streaming requests, the generator owns the span (so it lives for full duration)
    # For non-streaming, we create the span here
    if request.stream:
        try:
            user_messages = [m for m in request.messages if m.role == "user"]
            if not user_messages:
                await router.close()
                await client.close()
                raise HTTPException(status_code=400, detail="No user message found")

            query = user_messages[-1].content
            decision = await router.route(query)

            logger.info(f"[SMART-ROUTER] Decision: {decision.request_type.value}, notebook={decision.notebook_id}")

            # Streaming: generator creates its own span to capture response
            return StreamingResponse(
                stream_smart_response(client, router, decision, query, request, chat_id, tracing_settings),
                media_type="text/event-stream"
            )
        except Exception:
            await router.close()
            await client.close()
            raise

    # Non-streaming path: create span here
    with tracer.start_as_current_span("smart_router.handle_request") as span:
        try:
            user_messages = [m for m in request.messages if m.role == "user"]
            if not user_messages:
                raise HTTPException(status_code=400, detail="No user message found")

            query = user_messages[-1].content

            # Add user_query to span (using configurable max length)
            if tracing_settings.request_max_length > 0:
                add_span_attributes(user_query=query[:tracing_settings.request_max_length])

            decision = await router.route(query)

            logger.info(f"[SMART-ROUTER] Decision: {decision.request_type.value}, notebook={decision.notebook_id}")

            if decision.request_type == RequestType.LLM_TASK:
                # Call external LLM
                response_text = await router.llm_client.complete(
                    query,
                    max_tokens=4096
                )
                response_source = "llm"
            else:
                # Call NotebookLM with conversation_id for continuity
                result = await client.query(
                    notebook_id=decision.notebook_id,
                    query_text=query,
                    conversation_id=request.conversation_id
                )
                response_text = result.get("answer", "") if result else ""
                response_source = "notebooklm"

                # Save conversation_id to session store if we have a chat_id
                conv_id = result.get("conversation_id", "") if result else ""
                if chat_id and conv_id and app.state.session_store:
                    app.state.session_store.set(chat_id, conv_id)
                    logger.debug(f"[SMART-ROUTER] Saved session: chat_id={chat_id}, conversation_id={conv_id}")

            # Add response to trace
            if tracing_settings.response_max_length > 0:
                add_span_attributes(
                    response_content=response_text[:tracing_settings.response_max_length],
                    response_source=response_source
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
                        reasoning_content=decision.reasoning
                    ),
                    finish_reason="stop"
                )],
                usage=Usage(
                    prompt_tokens=len(query),
                    completion_tokens=len(response_text),
                    total_tokens=len(query) + len(response_text)
                )
            )
        finally:
            await router.close()
            await client.close()


async def stream_response(client, notebook_id: str, query_text: str, request: ChatCompletionRequest, chat_id: str = None):
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
                # Save to session store if we have a chat_id
                if chat_id and app.state.session_store:
                    app.state.session_store.set(chat_id, conversation_id)

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
    finally:
        await client.close()


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    logger.debug(f"[PROXY] Received request: POST /v1/chat/completions")
    logger.debug(f"[PROXY] Request params: model={request.model}, stream={request.stream}, messages={len(request.messages)}, conversation_id={request.conversation_id}, include_thinking={request.include_thinking}")

    # Check if using smart router
    routing_settings = get_routing_settings()
    if request.model == routing_settings.router_model_name:
        return await handle_smart_routing(request, http_request)

    # Log all headers for debugging
    logger.debug(f"[DEBUG] HTTP Headers: {dict(http_request.headers)}")

    # Log request metadata
    logger.debug(f"[DEBUG] Request has metadata attr: {hasattr(request, 'metadata')}")
    logger.debug(f"[DEBUG] Request.metadata value: {request.metadata}")
    logger.debug(f"[DEBUG] Request.metadata type: {type(request.metadata)}")

    # Extract chat_id from headers or request metadata
    chat_id = http_request.headers.get("X-OpenWebUI-Chat-Id")
    logger.debug(f"[DEBUG] chat_id from header: {chat_id}")

    if not chat_id and hasattr(request, 'metadata') and request.metadata:
        chat_id = request.metadata.get("chat_id")
        logger.debug(f"[DEBUG] chat_id from metadata: {chat_id}")

    logger.debug(f"[SESSION] Extracted chat_id: {chat_id}")

    # Load existing conversation_id from session store if chat_id exists
    if chat_id and app.state.session_store:
        stored_conv_id = app.state.session_store.get(chat_id)
        if stored_conv_id:
            logger.info(f"[SESSION] Reusing conversation: chat_id={chat_id}, conversation_id={stored_conv_id}")
            request.conversation_id = stored_conv_id
        else:
            logger.info(f"[SESSION] New conversation for chat_id={chat_id}")
    elif not chat_id:
        logger.debug("[SESSION] No chat_id found, using manual conversation_id mode")

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        logger.error("[PROXY] No user message found in request")
        raise HTTPException(status_code=400, detail="No user message found")

    query_text = user_messages[-1].content
    logger.debug(f"[PROXY] Extracted query: {query_text[:500]}{'...' if len(query_text) > 500 else ''}")

    client = await get_client()

    if request.stream:
        logger.debug("[PROXY] Using streaming response")
        return StreamingResponse(
            stream_response(client, request.model, query_text, request, chat_id),
            media_type="text/event-stream"
        )

    # Non-streaming path
    logger.debug("[PROXY] Using non-streaming response")
    try:
        logger.debug(f"[NOTEBOOKLM] Calling query: notebook_id={request.model}, query={query_text[:100]}..., conversation_id={request.conversation_id}")
        result = await client.query(
            notebook_id=request.model,
            query_text=query_text,
            conversation_id=request.conversation_id,
        )

        answer = result.get("answer", "") if result else ""
        conv_id = result.get("conversation_id", "") if result else ""

        # Save conversation_id to session store if we have a chat_id
        if chat_id and conv_id and app.state.session_store:
            app.state.session_store.set(chat_id, conv_id)

        logger.debug(f"[NOTEBOOKLM] Response received: answer_len={len(answer)}, conversation_id={conv_id}")
        logger.debug(f"[NOTEBOOKLM] Answer preview: {answer[:200]}{'...' if len(answer) > 200 else ''}")

        # Handle empty responses gracefully
        if not answer or not answer.strip():
            logger.warning(f"[NOTEBOOKLM] Empty answer received for query: {query_text[:100]}...")
            # Return a helpful error message instead of empty content
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
        logger.debug(f"[PROXY] Returning response with {len(answer)} characters")
        return response
    finally:
        await client.close()


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


def main(host: str = "0.0.0.0", port: int = 8080, session_ttl: int = 86400):
    """Run the OpenAI-compatible proxy server."""
    # Initialize OpenTelemetry tracing
    init_tracing()
    instrument_fastapi(app)
    instrument_httpx()

    # Initialize session store with configured TTL
    app.state.session_store = SessionStore(ttl_seconds=session_ttl)
    logger.info(f"Session store initialized with TTL={session_ttl}s ({session_ttl/3600:.1f} hours)")

    # Initialize notebook cache with proactive refresh for smart routing
    # Smart routing is enabled if llm_api_key is configured
    routing_settings = get_routing_settings()
    if routing_settings.llm_api_key:
        try:
            tokens = load_cached_tokens()
            if tokens and tokens.cookies:
                nlm_client = NotebookLMClient(
                    cookies=tokens.cookies,
                    csrf_token=tokens.csrf_token or "",
                    session_id=tokens.session_id or ""
                )
                app.state.notebook_cache = NotebookCache(
                    nlm_client=nlm_client,
                    ttl_seconds=routing_settings.summary_cache_ttl,
                    allowed_notebooks=routing_settings.allowed_notebooks
                )
                logger.info(f"Notebook cache initialized with TTL={routing_settings.summary_cache_ttl}s")
            else:
                logger.warning("Smart routing configured but no auth tokens found - cache not initialized")
        except Exception as e:
            logger.error(f"Failed to initialize notebook cache: {e}")
    else:
        logger.debug("Smart routing not configured (no llm_api_key) - notebook cache not initialized")

    import uvicorn

    # Logging is now configured centrally via setup_logging() in cli.py

    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        # Shutdown tracing with timeout to prevent hanging on exit
        shutdown_tracing(timeout_seconds=3)
        # Cleanup notebook cache on shutdown
        if app.state.notebook_cache:
            app.state.notebook_cache.shutdown()
        # Cleanup session store on shutdown
        if app.state.session_store:
            app.state.session_store.shutdown()


if __name__ == "__main__":
    main()
