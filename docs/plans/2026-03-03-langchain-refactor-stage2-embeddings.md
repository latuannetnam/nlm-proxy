# Stage 2: Replace fastembed → LangChain Embeddings + L3 Adapter

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `fastembed.TextEmbedding` with LangChain `HuggingFaceEmbeddings` in the response cache, and wire the L3 LLM verification to use the new `LangChainLLMClient`.

**Architecture:** Swap the embedding provider in `_load_embedding_model()` and `_compute_embedding()`. The L3 verification's `.complete()` call works transparently because `LangChainLLMClient` exposes the same interface. Update the server `main()` init to create a `LangChainLLMClient` for cache L3.

**Inputs:** Stage 1 complete — `LangChainLLMClient` and `create_chat_model()` exist.

**Outputs:** Response cache uses `HuggingFaceEmbeddings` for L2. Server init creates `LangChainLLMClient` for L3 cache verification.

---

## Task 2.1: Replace fastembed with LangChain HuggingFaceEmbeddings

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py` (methods `_load_embedding_model`, `_compute_embedding`)
- Test: `tests/core/test_response_cache_semantic.py`, `tests/core/test_embedding_models.py`

**Step 1: Update `_load_embedding_model()` and `_compute_embedding()` in response_cache.py**

Replace the fastembed import and model loading with:

```python
# In __init__, change the fastembed loading block:
def _load_embedding_model(self):
    """Load the LangChain HuggingFace embedding model."""
    try:
        # langchain-huggingface >= 1.2 (NOT langchain-community)
        from langchain_huggingface import HuggingFaceEmbeddings
        import numpy as np
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self._embedding_model_name
        )
        self._np = np
        logger.info(
            "Loaded embedding model: %s", self._embedding_model_name
        )
    except Exception as e:
        logger.warning("Failed to load embedding model: %s", e)
        self._embedding_model = None

def _compute_embedding(self, query: str):
    """Compute query embedding using LangChain HuggingFace model."""
    if self._embedding_model is None:
        self._load_embedding_model()
    if self._embedding_model is None:
        return None
    try:
        embedding = self._embedding_model.embed_query(query)
        vec = self._np.array(embedding)
        norm = self._np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as e:
        logger.warning("Embedding computation failed: %s", e)
        return None
```

**Step 2: Update test_embedding_models.py**

Replace fastembed imports with LangChain equivalents:

```python
"""Test embedding model performance for Vietnamese and multilingual queries."""

import pytest

# Skip if sentence-transformers not installed
st = pytest.importorskip("sentence_transformers")

# langchain-huggingface >= 1.2 (NOT langchain-community)
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import time

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture(scope="module")
def model():
    """Load embedding model once for all tests."""
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


def cosine_sim(model: HuggingFaceEmbeddings, text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts."""
    embeddings = model.embed_documents([text_a, text_b])
    a, b = np.array(embeddings[0]), np.array(embeddings[1])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ... rest of test classes unchanged, just use new cosine_sim signature
```

**Step 3: Run tests**

Run: `uv run pytest tests/core/test_response_cache_semantic.py tests/core/test_embedding_models.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/nlm_proxy/core/response_cache.py tests/core/test_embedding_models.py
git commit -m "refactor: replace fastembed with LangChain HuggingFaceEmbeddings"
```

---

## Task 2.2: Update L3 verification to use LangChain ChatModel

**Files:**
- Modify: `src/nlm_proxy/openai/server.py` (the `main()` init section only)
- Test: `tests/core/test_response_cache_llm.py`

The L3 verification calls `self._llm_client.complete(prompt)`. The new `LangChainLLMClient` also exposes `.complete()` with the same signature, so **no changes needed to `_verify_semantic_match()`** — it works transparently.

Only update the server init in `main()` to pass a `LangChainLLMClient` instead of `ExternalLLMClient`:

```python
# In server.py main(), change:
# from nlm_proxy.core.llm_client import ExternalLLMClient
# llm_client = ExternalLLMClient(base_url=..., api_key=..., model=...)
# TO:
from nlm_proxy.core.llm_client import LangChainLLMClient, create_chat_model
chat_model = create_chat_model(
    model=routing_settings.llm_model,
    base_url=routing_settings.llm_base_url,
    api_key=routing_settings.llm_api_key,
)
llm_client = LangChainLLMClient(chat_model=chat_model)
```

No test changes needed — the mock `llm_client` in `test_response_cache_llm.py` already mocks `.complete()`.

**Step 1: Run tests**

Run: `uv run pytest tests/core/test_response_cache_llm.py -v`
Expected: PASS

**Step 2: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "refactor: use LangChainLLMClient for L3 cache verification"
```

---

> [!NOTE]
> #### GPU Embedding Support Migration
>
> The `cache-gpu` optional dependency has changed from `fastembed-gpu` to `torch>=2.0`. Users who previously used GPU-accelerated embeddings must:
> 1. Uninstall `fastembed-gpu`
> 2. Install `torch` with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
> 3. `HuggingFaceEmbeddings` automatically detects and uses GPU when `torch+CUDA` is available
>
> Verify GPU detection during Stage 8 documentation updates. No separate test is needed — `HuggingFaceEmbeddings` handles this transparently.

---

## 🔒 Stage 2 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS
