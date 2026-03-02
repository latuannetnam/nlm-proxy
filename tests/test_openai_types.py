# tests/test_openai_types.py
import pytest
from pydantic import ValidationError
import json


@pytest.mark.openai
def test_message_valid_user_role():
    from nlm_proxy.openai.types import Message
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


@pytest.mark.openai
def test_message_invalid_role_rejected():
    from nlm_proxy.openai.types import Message
    with pytest.raises(ValidationError):
        Message(role="invalid", content="Hello")


@pytest.mark.openai
def test_chat_completion_request_minimal():
    from nlm_proxy.openai.types import ChatCompletionRequest, Message
    req = ChatCompletionRequest(
        model="notebook-uuid",
        messages=[Message(role="user", content="Hello")]
    )
    assert req.model == "notebook-uuid"
    assert req.stream is False  # Default
    assert req.conversation_id is None
    assert req.include_thinking is True  # Default changed to True


@pytest.mark.openai
def test_chat_completion_request_with_extras():
    from nlm_proxy.openai.types import ChatCompletionRequest, Message
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


@pytest.mark.openai
def test_chat_completion_chunk_serialization():
    from nlm_proxy.openai.types import ChatCompletionChunk, Choice, DeltaContent

    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        created=1700000000,
        model="nb-uuid",
        choices=[Choice(index=0, delta=DeltaContent(content="Hello"))],
        system_fingerprint="conv_abc123"
    )

    data = json.loads(chunk.model_dump_json())
    assert data["object"] == "chat.completion.chunk"
    assert data["choices"][0]["delta"]["content"] == "Hello"
    assert data["system_fingerprint"] == "conv_abc123"


@pytest.mark.openai
def test_chat_completion_chunk_final():
    from nlm_proxy.openai.types import ChatCompletionChunk, Choice, DeltaContent

    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        created=1700000000,
        model="nb-uuid",
        choices=[Choice(index=0, delta=DeltaContent(), finish_reason="stop")]
    )

    assert chunk.choices[0].finish_reason == "stop"


@pytest.mark.openai
def test_chat_completion_response_non_streaming():
    from nlm_proxy.openai.types import (
        ChatCompletionResponse, ResponseChoice, ResponseMessage, Usage
    )

    response = ChatCompletionResponse(
        id="chatcmpl-123",
        created=1700000000,
        model="nb-uuid",
        choices=[ResponseChoice(
            index=0,
            message=ResponseMessage(role="assistant", content="Hello!"),
            finish_reason="stop"
        )],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        system_fingerprint="conv_abc123"
    )

    data = response.model_dump()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello!"


@pytest.mark.openai
def test_bypass_cache_default_false():
    """bypass_cache should default to False."""
    from nlm_proxy.openai.types import ChatCompletionRequest
    req = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}]
    )
    assert req.bypass_cache is False


@pytest.mark.openai
def test_bypass_cache_from_extra_body():
    """bypass_cache should be settable via extra_body."""
    from nlm_proxy.openai.types import ChatCompletionRequest
    req = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        bypass_cache=True
    )
    assert req.bypass_cache is True
