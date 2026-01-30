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
