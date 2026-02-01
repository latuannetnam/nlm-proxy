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
from nlm_proxy.core.config import get_openai_settings
from nlm_proxy.core.logging import get_logger
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
    client = await get_client()
    try:
        logger.debug("[NOTEBOOKLM] Calling list_notebooks()")
        notebooks = await client.list_notebooks()
        logger.debug(f"[NOTEBOOKLM] Response: {len(notebooks)} notebooks found")

        response = {
            "object": "list",
            "data": [
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
        }
        logger.debug(f"[PROXY] Returning {len(response['data'])} models")
        return response
    finally:
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
    logger.debug(f"[PROXY] Extracted query: {query_text[:200]}{'...' if len(query_text) > 200 else ''}")

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
    # Initialize session store with configured TTL
    app.state.session_store = SessionStore(ttl_seconds=session_ttl)
    logger.info(f"Session store initialized with TTL={session_ttl}s ({session_ttl/3600:.1f} hours)")

    import uvicorn

    # Logging is now configured centrally via setup_logging() in cli.py

    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        # Cleanup session store on shutdown
        if app.state.session_store:
            app.state.session_store.shutdown()


if __name__ == "__main__":
    main()
