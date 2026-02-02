"""Tests for prompt template loading."""

from pathlib import Path


def test_prompt_templates_exist():
    """Test that prompt template files exist."""
    prompts_dir = Path(__file__).parent.parent.parent / "src" / "nlm_proxy" / "openai" / "prompts"

    assert (prompts_dir / "classify_request.txt").exists()
    assert (prompts_dir / "select_notebook.txt").exists()


def test_classify_request_prompt_content():
    """Test classify_request prompt has required placeholders."""
    prompts_dir = Path(__file__).parent.parent.parent / "src" / "nlm_proxy" / "openai" / "prompts"
    content = (prompts_dir / "classify_request.txt").read_text()

    assert "{query}" in content
    assert "notebooklm" in content.lower()
    assert "llm_task" in content.lower()


def test_select_notebook_prompt_content():
    """Test select_notebook prompt has required placeholders."""
    prompts_dir = Path(__file__).parent.parent.parent / "src" / "nlm_proxy" / "openai" / "prompts"
    content = (prompts_dir / "select_notebook.txt").read_text()

    assert "{query}" in content
    assert "{notebooks_json}" in content
