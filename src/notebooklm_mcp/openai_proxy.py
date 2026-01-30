"""OpenAI-compatible proxy server for NotebookLM."""

from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="NotebookLM OpenAI Proxy",
    description="OpenAI-compatible API for NotebookLM",
    version="0.1.0"
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/v1/embeddings")
async def embeddings():
    """Embeddings endpoint - not supported by NotebookLM."""
    raise HTTPException(
        status_code=501,
        detail="Embeddings not supported. NotebookLM does not provide embedding generation."
    )
