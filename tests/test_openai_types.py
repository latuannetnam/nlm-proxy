# tests/test_openai_types.py
import pytest
from pydantic import ValidationError


def test_message_valid_user_role():
    from notebooklm_mcp.openai_types import Message
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_message_invalid_role_rejected():
    from notebooklm_mcp.openai_types import Message
    with pytest.raises(ValidationError):
        Message(role="invalid", content="Hello")


def test_chat_completion_request_minimal():
    from notebooklm_mcp.openai_types import ChatCompletionRequest, Message
    req = ChatCompletionRequest(
        model="notebook-uuid",
        messages=[Message(role="user", content="Hello")]
    )
    assert req.model == "notebook-uuid"
    assert req.stream is False  # Default
    assert req.conversation_id is None
    assert req.include_thinking is False


def test_chat_completion_request_with_extras():
    from notebooklm_mcp.openai_types import ChatCompletionRequest, Message
    req = ChatCompletionRequest(
        model="nb-123",
        messages=[Message(role="user", content="Hi")],
        stream=True,
        conversation_id="conv-456",
        include_thinking=True
    )
    assert req.stream is True
    assert req.conversation_id == "conv-456"
    assert req.include_thinking is True
