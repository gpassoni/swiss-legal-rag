from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import openai
import pytest

from src.orchestration.query_engine import AnthropicAdapter, OpenAIAdapter

_REQUEST = httpx.Request("POST", "https://api.example.com/v1/messages")


def _anthropic_error(cls, status: int | None = None):
    if status is None:
        return cls(request=_REQUEST)
    response = httpx.Response(status, request=_REQUEST)
    return cls("error", response=response, body=None)


def _openai_error(cls, status: int | None = None):
    if status is None:
        return cls(message="error", request=_REQUEST)
    response = httpx.Response(status, request=_REQUEST)
    return cls("error", response=response, body=None)


def _anthropic_success_response(text="ok"):
    response = MagicMock()
    response.usage.input_tokens = 10
    response.usage.output_tokens = 5
    block = MagicMock()
    block.type = "text"
    block.text = text
    response.content = [block]
    return response


def _openai_success_response(text="ok"):
    response = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    choice = MagicMock()
    choice.message.content = text
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_anthropic_adapter_retries_transient_error_then_succeeds():
    conn_err = _anthropic_error(anthropic.APIConnectionError)
    success = _anthropic_success_response("ok")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[conn_err, success])

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        adapter = AnthropicAdapter(api_key="test-key")
        result = await adapter.complete("system", "user")

    assert result == "ok"
    assert mock_client.messages.create.await_count == 2


@pytest.mark.asyncio
async def test_anthropic_adapter_does_not_retry_non_transient_error():
    bad_request = _anthropic_error(anthropic.BadRequestError, status=400)
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=[bad_request])

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        adapter = AnthropicAdapter(api_key="test-key")
        with pytest.raises(anthropic.BadRequestError):
            await adapter.complete("system", "user")

    assert mock_client.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_anthropic_adapter_passes_temperature_and_timeout_to_client():
    success = _anthropic_success_response("ok")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=success)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client) as mock_ctor:
        adapter = AnthropicAdapter(
            api_key="test-key", model="claude-x", temperature=0.2, timeout=15.0
        )
        await adapter.complete("system", "user")

    mock_ctor.assert_called_once_with(api_key="test-key", timeout=15.0)
    _, kwargs = mock_client.messages.create.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["model"] == "claude-x"


@pytest.mark.asyncio
async def test_openai_adapter_retries_transient_error_then_succeeds():
    conn_err = _openai_error(openai.APIConnectionError)
    success = _openai_success_response("ok")

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=[conn_err, success])

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        adapter = OpenAIAdapter(api_key="test-key")
        result = await adapter.complete("system", "user")

    assert result == "ok"
    assert mock_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_openai_adapter_does_not_retry_non_transient_error():
    bad_request = _openai_error(openai.BadRequestError, status=400)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=[bad_request])

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        adapter = OpenAIAdapter(api_key="test-key")
        with pytest.raises(openai.BadRequestError):
            await adapter.complete("system", "user")

    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_openai_adapter_passes_temperature_and_timeout_to_client():
    success = _openai_success_response("ok")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=success)

    with patch("openai.AsyncOpenAI", return_value=mock_client) as mock_ctor:
        adapter = OpenAIAdapter(
            api_key="test-key", model="gpt-x", temperature=0.3, timeout=20.0
        )
        await adapter.complete("system", "user")

    mock_ctor.assert_called_once_with(api_key="test-key", timeout=20.0)
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.3
    assert kwargs["model"] == "gpt-x"
