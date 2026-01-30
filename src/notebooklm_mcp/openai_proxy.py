"""OpenAI-compatible proxy server for NotebookLM."""

import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .api_client import NotebookLMClient
from .auth import load_cached_tokens
from .openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    DeltaContent,
    ResponseChoice,
    ResponseMessage,
    Usage,
)

app = FastAPI(
    title="NotebookLM OpenAI Proxy",
    description="OpenAI-compatible API for NotebookLM",
    version="0.1.0"
)


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


@app.get("/v1/models")
async def list_models():
    """List notebooks as available models."""
    client = await get_client()
    try:
        notebooks = await client.list_notebooks()
        return {
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
    finally:
        await client.close()


async def stream_response(client, notebook_id: str, query_text: str, request: ChatCompletionRequest):
    """Generate OpenAI-compatible SSE stream from NotebookLM query_stream."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    conversation_id = None

    try:
        async for chunk in client.query_stream(
            notebook_id=notebook_id,
            query_text=query_text,
            conversation_id=request.conversation_id
        ):
            # Filter thinking unless requested
            if chunk["type"] == "thinking" and not request.include_thinking:
                continue

            conversation_id = chunk.get("conversation_id", conversation_id)

            openai_chunk = ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=notebook_id,
                choices=[Choice(delta=DeltaContent(content=chunk["text"]))],
                system_fingerprint=f"conv_{conversation_id}" if conversation_id else None
            )
            yield f"data: {openai_chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=notebook_id,
            choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await client.close()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    query_text = user_messages[-1].content

    client = await get_client()

    if request.stream:
        return StreamingResponse(
            stream_response(client, request.model, query_text, request),
            media_type="text/event-stream"
        )

    # Non-streaming path (existing code)
    try:
        result = await client.query(
            notebook_id=request.model,
            query_text=query_text,
            conversation_id=request.conversation_id,
        )

        answer = result.get("answer", "") if result else ""
        conv_id = result.get("conversation_id", "") if result else ""

        return ChatCompletionResponse(
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
    finally:
        await client.close()


@app.post("/v1/embeddings")
async def embeddings():
    """Embeddings endpoint - not supported by NotebookLM."""
    raise HTTPException(
        status_code=501,
        detail="Embeddings not supported. NotebookLM does not provide embedding generation."
    )
