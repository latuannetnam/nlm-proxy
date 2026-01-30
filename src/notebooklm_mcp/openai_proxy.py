"""OpenAI-compatible proxy server for NotebookLM."""

from fastapi import FastAPI

app = FastAPI(
    title="NotebookLM OpenAI Proxy",
    description="OpenAI-compatible API for NotebookLM",
    version="0.1.0"
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
