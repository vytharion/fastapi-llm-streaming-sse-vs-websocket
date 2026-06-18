"""Tests for the mock async LLM token generator."""

from __future__ import annotations

import time

import pytest

from app.llm_mock import DEFAULT_TOKEN_DELAY_SECONDS, stream_tokens, tokenize


def test_tokenize_splits_on_whitespace() -> None:
    assert tokenize("hello world") == ["hello", "world"]
    assert tokenize("  spaced   out  ") == ["spaced", "out"]
    assert tokenize("") == []


async def test_stream_tokens_yields_prompt_then_suffix() -> None:
    received = [token async for token in stream_tokens("hi there", delay_seconds=0.0)]

    assert received[:2] == ["hi", "there"]
    assert received[-5:] == ["This", "is", "a", "mock", "stream."]


async def test_stream_tokens_is_deterministic() -> None:
    first = [token async for token in stream_tokens("ping", delay_seconds=0.0)]
    second = [token async for token in stream_tokens("ping", delay_seconds=0.0)]

    assert first == second


async def test_stream_tokens_default_delay_is_observable() -> None:
    start = time.perf_counter()
    tokens = [token async for token in stream_tokens("a b", delay_seconds=0.01)]
    elapsed = time.perf_counter() - start

    assert elapsed >= 0.01 * len(tokens) * 0.5
    assert DEFAULT_TOKEN_DELAY_SECONDS > 0


async def test_empty_prompt_falls_back_to_default() -> None:
    tokens = [token async for token in stream_tokens("   ", delay_seconds=0.0)]

    assert tokens[0] == "hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
