"""Backward-compatible re-export. Actual implementation moved to core/."""
from nlm_proxy.core.notebook_cache import (  # noqa: F401
    NotebookCache, NotebookInfo, SourceInfo, _extract_first_sentence,
)
