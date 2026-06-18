"""Deterministic mock async LLM token generator.

Both the SSE and WebSocket endpoints will consume this same generator so the
transport comparison stays apples-to-apples.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator


DEFAULT_TOKEN_DELAY_SECONDS: float = 0.02


def tokenize(prompt: str) -> list[str]:
    """Split a prompt into whitespace-delimited tokens.

    A real tokenizer would use BPE / SentencePiece; whitespace is enough to
    keep the streaming behaviour observable in tests.
    """
    return [token for token in prompt.split() if token]


def _canned_reply(prompt: str) -> list[str]:
    cleaned = prompt.strip() or "hello"
    echo = tokenize(cleaned)
    suffix = ["This", "is", "a", "mock", "stream."]
    return echo + suffix


async def stream_tokens(
    prompt: str,
    delay_seconds: float = DEFAULT_TOKEN_DELAY_SECONDS,
) -> AsyncIterator[str]:
    """Yield tokens one at a time with a short async delay between each.

    The delay simulates a real model's token cadence without burning credits
    or requiring network access.
    """
    for token in _canned_reply(prompt):
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        yield token
