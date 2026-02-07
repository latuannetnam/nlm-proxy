# Plan: Enhance Notebook Selection with Source Descriptions

## Context

**Problem**: The OpenAI proxy's smart router struggles to disambiguate between notebooks when:
1. Multiple notebooks have sources with similar/identical titles
2. Source titles don't capture semantic meaning of content

**Current State**: Source descriptions (summaries + keywords) are already fetched and cached in `SourceInfo`, but deliberately excluded from the selection prompt to save tokens. The router only sees `source_titles`.

**Goal**: Add hybrid source metadata (keywords + truncated summaries) to the selection prompt, improving disambiguation accuracy.

## Implementation Plan

### Step 1: Add Configuration Fields
**File**: `src/nlm_proxy/core/config.py`

Add to `SmartRoutingSettings` class (after line 152):
```python
source_descriptions_enabled: bool = Field(
    default=True,
    description="Include source keywords and summaries in selection prompt"
)
source_max_keywords: int = Field(
    default=5,
    description="Max keywords per source to include"
)
source_summary_max_chars: int = Field(
    default=100,
    description="Max chars of source summary (first sentence or truncated)"
)
source_descriptions_max_sources: int = Field(
    default=10,
    description="Max sources with descriptions (others get title only)"
)
```

**Environment Variables**:
- `NLM_PROXY_ROUTING_SOURCE_DESCRIPTIONS_ENABLED` (default: true)
- `NLM_PROXY_ROUTING_SOURCE_MAX_KEYWORDS` (default: 5)
- `NLM_PROXY_ROUTING_SOURCE_SUMMARY_MAX_CHARS` (default: 100)
- `NLM_PROXY_ROUTING_SOURCE_DESCRIPTIONS_MAX_SOURCES` (default: 10)

### Step 2: Add Helper Function and Property
**File**: `src/nlm_proxy/openai/notebook_cache.py`

Add module-level helper (before line 23):
```python
import re

def _extract_first_sentence(text: str, max_chars: int = 100) -> str:
    """Extract first sentence or truncate to max_chars.

    Strategy:
    1. Find first sentence ending (. ! ?)
    2. If sentence <= max_chars, return it
    3. Otherwise truncate at word boundary with ellipsis
    """
    if not text:
        return ""

    # Clean up: remove markdown bold markers, normalize whitespace
    text = text.replace("**", "").strip()

    # Find first sentence boundary
    match = re.search(r'^[^.!?]+[.!?]', text)
    if match:
        sentence = match.group(0).strip()
        if len(sentence) <= max_chars:
            return sentence

    # Truncate at word boundary
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Find last space to avoid cutting words
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.6:  # Only if we keep at least 60%
        truncated = truncated[:last_space]

    return truncated.rstrip('.,;:') + "..."
```

Add property to `NotebookInfo` class (after line 63):
```python
def get_source_descriptions(
    self,
    max_sources: int = 10,
    max_keywords: int = 5,
    summary_max_chars: int = 100
) -> list[dict]:
    """Get source info with keywords and truncated summaries.

    Returns list of dicts with: title, keywords, summary_snippet
    First `max_sources` get full descriptions, rest get title only.
    """
    result = []
    for i, src in enumerate(self.sources):
        if i < max_sources:
            # Extract first sentence or truncate summary
            summary_snippet = _extract_first_sentence(
                src.summary, max_chars=summary_max_chars
            )
            result.append({
                "title": src.title[:MAX_SOURCE_TITLE_LENGTH],
                "keywords": src.keywords[:max_keywords] if src.keywords else [],
                "summary": summary_snippet
            })
        else:
            # Title only for remaining sources
            result.append({
                "title": src.title[:MAX_SOURCE_TITLE_LENGTH]
            })
    return result
```

### Step 3: Update Router Selection Logic
**File**: `src/nlm_proxy/openai/router.py`

Modify `select_notebook()` method (lines 113-130):

```python
# Get max source titles from env or use default
max_source_titles = int(
    os.environ.get("NLM_PROXY_ROUTING_MAX_SOURCE_TITLES", DEFAULT_MAX_SOURCE_TITLES)
)

# NEW: Source description settings
source_descriptions_enabled = os.environ.get(
    "NLM_PROXY_ROUTING_SOURCE_DESCRIPTIONS_ENABLED", "true"
).lower() in ("true", "1", "yes")
source_max_keywords = int(
    os.environ.get("NLM_PROXY_ROUTING_SOURCE_MAX_KEYWORDS", "5")
)
source_summary_max_chars = int(
    os.environ.get("NLM_PROXY_ROUTING_SOURCE_SUMMARY_MAX_CHARS", "80")
)
source_descriptions_max_sources = int(
    os.environ.get("NLM_PROXY_ROUTING_SOURCE_DESCRIPTIONS_MAX_SOURCES", "10")
)

# Build notebook info for LLM
notebooks_info = []
for nb in notebooks:
    info = {
        "id": nb.id,
        "title": nb.title,
        "summary": nb.summary[:500] if nb.summary else "",
        "topics": nb.topics[:5] if nb.topics else [],
        "source_count": nb.source_count,
        "source_types": nb.source_types,
    }

    if source_descriptions_enabled:
        # NEW: Include source descriptions with keywords and summaries
        info["sources"] = nb.get_source_descriptions(
            max_sources=source_descriptions_max_sources,
            max_keywords=source_max_keywords,
            summary_max_chars=source_summary_max_chars
        )
    else:
        # Fallback: title-only list (current behavior)
        info["source_titles"] = nb.source_titles[:max_source_titles]

    notebooks_info.append(info)
```

**New JSON structure per notebook** (when enabled):
```json
{
  "id": "abc-123-def",
  "title": "Machine Learning Research",
  "summary": "This notebook contains research papers on...",
  "topics": ["neural networks", "deep learning"],
  "source_count": 15,
  "source_types": {"pdf": 10, "url": 5},
  "sources": [
    {
      "title": "Attention Is All You Need.pdf",
      "keywords": ["transformer", "attention mechanism", "NLP"],
      "summary": "Introduces the Transformer architecture..."
    },
    {
      "title": "BERT Paper.pdf",
      "keywords": ["BERT", "pre-training", "language model"],
      "summary": "Presents BERT, a bidirectional encoder..."
    },
    {
      "title": "Some Other Document.pdf"
    }
  ]
}
```

### Step 4: Update Selection Prompt
**File**: `src/nlm_proxy/openai/prompts/select_notebook.txt`

Replace content with:
```text
You are a notebook selector. Given the user's query and available notebooks, select the most relevant notebook that can answer the query.

Available notebooks:
{notebooks_json}

User query:
{query}

Selection criteria (in order of importance):
1. **Source keywords** - If sources have keywords matching query terms, that notebook is highly relevant
2. **Source summaries** - Brief descriptions indicate what each source covers; match to query intent
3. **Source titles** - Specific document/paper/URL names that match the query
4. **Source types** - Match query intent to source types (e.g., "PDF paper" -> notebooks with PDF sources)
5. **Notebook summary** - Overall topic match
6. **Topics** - Suggested topics as additional context

JSON structure explanation:
- Each notebook has "sources" array with: title, keywords (list), summary (snippet)
- Sources beyond the first 10 may have title only (no keywords/summary)
- Empty keywords/summary means that data wasn't available

Respond with ONLY the notebook_id (UUID) of the most relevant notebook. If none seem relevant, respond with the first notebook's ID.
```

### Step 5: Update Documentation
**File**: `.claude/memory/smart-routing.md`

Add section documenting new configuration options.

## Critical Files

| File | Changes |
|------|---------|
| `src/nlm_proxy/core/config.py:118-159` | Add 4 new fields to `SmartRoutingSettings` |
| `src/nlm_proxy/openai/notebook_cache.py:1-63` | Add `_extract_first_sentence()` + `get_source_descriptions()` |
| `src/nlm_proxy/openai/router.py:100-163` | Update `select_notebook()` JSON structure |
| `src/nlm_proxy/openai/prompts/select_notebook.txt` | Update prompt with new criteria |

## Token Budget Estimate

| Component | Chars | Tokens (est.) |
|-----------|-------|---------------|
| Title | 100 | ~25 |
| Keywords (5) | 50 | ~15 |
| Summary snippet | 80 | ~20 |
| JSON overhead | 30 | ~10 |
| **Total per source** | 260 | **~70** |

| Scenario | Sources w/ desc | Title-only | Est. Tokens/Notebook |
|----------|-----------------|------------|---------------------|
| Small | 5 | 0 | ~350 |
| Medium | 10 | 5 | ~825 |
| Large | 10 | 20 | ~1,200 |

**Total prompt estimation** (5 notebooks): ~4,000 tokens vs ~1,500 current = +167%

## Graceful Degradation

The implementation handles missing data at multiple levels:
1. **Missing summary**: `_extract_first_sentence("")` returns `""`
2. **Missing keywords**: Returns `[]`
3. **Sources beyond limit**: Get `{"title": "..."}` only
4. **Feature disabled**: Falls back to current `source_titles` behavior

## Verification

### Unit Tests
Test `_extract_first_sentence()`:
```python
def test_extract_first_sentence_normal():
    assert _extract_first_sentence("This is a test. More text.", 80) == "This is a test."

def test_extract_first_sentence_long():
    text = "This is a very long sentence that exceeds the maximum character limit."
    result = _extract_first_sentence(text, 30)
    assert len(result) <= 33  # 30 + "..."
    assert result.endswith("...")

def test_extract_first_sentence_empty():
    assert _extract_first_sentence("", 80) == ""

def test_extract_first_sentence_markdown():
    assert _extract_first_sentence("**Bold text** here.", 80) == "Bold text here."
```

### Integration Test
1. Create 2 notebooks with overlapping topics but different source content
2. Query for specific source content
3. Verify router selects correct notebook

### Manual Test
```bash
# Enable debug logging to see selection JSON
NLM_PROXY_DEBUG=true nlm-proxy serve openai --port 8080

# Test with curl or OpenAI SDK
```
