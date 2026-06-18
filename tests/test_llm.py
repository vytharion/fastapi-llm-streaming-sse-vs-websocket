"""Tests for the pluggable LLM token-streamer layer."""

from __future__ import annotations

from typing import Any

import pytest

from app.llm import (
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_MODEL,
    ENV_PROVIDER,
    LLMStreamError,
    MockTokenStreamer,
    OPENAI_DEFAULT_MODEL,
    OpenAITokenStreamer,
    TokenStreamer,
    _extract_delta_content,
    build_default_streamer,
    get_token_streamer,
    reset_token_streamer,
)


@pytest.fixture(autouse=True)
def _clear_default_streamer() -> None:
    """Stop a streamer cached by one test from leaking into the next."""
    reset_token_streamer()
    yield
    reset_token_streamer()


class _FakeDelta:
    def __init__(self, content: Any) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: Any) -> None:
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: Any) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeEmptyChunk:
    """Mimics an SDK chunk whose ``choices`` list is empty (rare but legal)."""

    choices: list[_FakeChoice] = []


class _FakeStreamResponse:
    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self) -> "_FakeStreamResponse":
        return self

    async def __anext__(self) -> Any:
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class _FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeStreamResponse:
        self.calls.append(kwargs)
        return _FakeStreamResponse(list(self._chunks))


class _BoomCompletions:
    async def create(self, **kwargs: Any) -> _FakeStreamResponse:
        raise RuntimeError("api down")


class _FakeChat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: Any) -> None:
        self.chat = _FakeChat(completions)


def test_mock_streamer_advertises_provider_name() -> None:
    assert MockTokenStreamer.name == "mock"


def test_openai_streamer_advertises_provider_name() -> None:
    assert OpenAITokenStreamer.name == "openai"


def test_protocol_runtime_check_accepts_mock() -> None:
    assert isinstance(MockTokenStreamer(), TokenStreamer)


def test_protocol_runtime_check_accepts_openai() -> None:
    streamer = OpenAITokenStreamer(_FakeClient(_FakeCompletions([])))
    assert isinstance(streamer, TokenStreamer)


async def test_mock_streamer_yields_same_tokens_as_underlying_generator() -> None:
    streamer = MockTokenStreamer()

    tokens = [token async for token in streamer.stream("hi there", delay_seconds=0.0)]

    assert tokens[:2] == ["hi", "there"]
    assert tokens[-5:] == ["This", "is", "a", "mock", "stream."]


def test_extract_delta_content_pulls_string_payload() -> None:
    assert _extract_delta_content(_FakeChunk("hi")) == "hi"


def test_extract_delta_content_handles_empty_choices() -> None:
    assert _extract_delta_content(_FakeEmptyChunk()) is None


def test_extract_delta_content_handles_missing_choices_attr() -> None:
    assert _extract_delta_content(object()) is None


def test_extract_delta_content_rejects_non_string_content() -> None:
    assert _extract_delta_content(_FakeChunk(None)) is None
    assert _extract_delta_content(_FakeChunk(123)) is None


async def test_openai_streamer_yields_each_delta_content() -> None:
    completions = _FakeCompletions(
        [_FakeChunk("Hello"), _FakeChunk(" "), _FakeChunk("world")]
    )
    streamer = OpenAITokenStreamer(_FakeClient(completions), model="gpt-test")

    tokens = [token async for token in streamer.stream("ping")]

    assert tokens == ["Hello", " ", "world"]


async def test_openai_streamer_drops_empty_and_role_only_chunks() -> None:
    completions = _FakeCompletions(
        [
            _FakeChunk(None),       # role-only opening chunk
            _FakeChunk(""),         # whitespace placeholder
            _FakeChunk("token-a"),
            _FakeEmptyChunk(),      # zero choices, sometimes appears at end
            _FakeChunk("token-b"),
        ]
    )
    streamer = OpenAITokenStreamer(_FakeClient(completions))

    tokens = [token async for token in streamer.stream("ping")]

    assert tokens == ["token-a", "token-b"]


async def test_openai_streamer_forwards_model_and_messages() -> None:
    completions = _FakeCompletions([_FakeChunk("ok")])
    streamer = OpenAITokenStreamer(
        _FakeClient(completions), model="gpt-x", system_prompt="be brief"
    )

    [_ async for _ in streamer.stream("hello world")]

    assert completions.calls[0]["model"] == "gpt-x"
    assert completions.calls[0]["stream"] is True
    messages = completions.calls[0]["messages"]
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello world"},
    ]


async def test_openai_streamer_wraps_create_errors_in_llm_stream_error() -> None:
    streamer = OpenAITokenStreamer(_FakeClient(_BoomCompletions()))

    with pytest.raises(LLMStreamError, match="openai create"):
        async for _ in streamer.stream("ping"):
            pass


async def test_openai_streamer_ignores_delay_seconds() -> None:
    """Delay is part of the protocol for the mock; the real adapter must ignore it."""
    completions = _FakeCompletions([_FakeChunk("ok")])
    streamer = OpenAITokenStreamer(_FakeClient(completions))

    tokens = [
        token async for token in streamer.stream("ping", delay_seconds=10.0)
    ]

    assert tokens == ["ok"]


def test_build_default_streamer_returns_mock_when_provider_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv(ENV_OPENAI_API_KEY, raising=False)

    streamer = build_default_streamer()

    assert isinstance(streamer, MockTokenStreamer)


def test_build_default_streamer_returns_mock_for_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_PROVIDER, "anthropic-but-not-yet")

    streamer = build_default_streamer()

    assert isinstance(streamer, MockTokenStreamer)


def test_build_default_streamer_falls_back_to_mock_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_PROVIDER, "openai")
    monkeypatch.delenv(ENV_OPENAI_API_KEY, raising=False)

    streamer = build_default_streamer()

    assert isinstance(streamer, MockTokenStreamer)


def test_build_default_streamer_constructs_openai_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_PROVIDER, "openai")
    monkeypatch.setenv(ENV_OPENAI_API_KEY, "sk-fake-key")
    monkeypatch.delenv(ENV_OPENAI_MODEL, raising=False)

    streamer = build_default_streamer()

    assert isinstance(streamer, OpenAITokenStreamer)
    assert streamer.model == OPENAI_DEFAULT_MODEL


def test_build_default_streamer_honors_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_PROVIDER, "openai")
    monkeypatch.setenv(ENV_OPENAI_API_KEY, "sk-fake-key")
    monkeypatch.setenv(ENV_OPENAI_MODEL, "gpt-custom")

    streamer = build_default_streamer()

    assert isinstance(streamer, OpenAITokenStreamer)
    assert streamer.model == "gpt-custom"


def test_get_token_streamer_memoises_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_PROVIDER, raising=False)

    first = get_token_streamer()
    second = get_token_streamer()

    assert first is second


def test_reset_token_streamer_drops_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    first = get_token_streamer()

    reset_token_streamer()
    second = get_token_streamer()

    assert first is not second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
