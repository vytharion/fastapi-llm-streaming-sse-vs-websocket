"""Pluggable LLM token-streamer behind the SSE and WebSocket transports.

Earlier steps wired both endpoints directly to the deterministic mock generator
in ``app.llm_mock``. That kept the transport story honest — tokens really were
flushed one at a time — but pretended the upstream model was free, infallible,
and synchronous to start. A production app talks to a real SDK whose stream is
a network-backed async iterator that can fail, stall, or emit empty deltas.

This module hides that distinction behind a single ``TokenStreamer`` protocol.
The mock implementation forwards to the existing generator; the OpenAI
implementation calls ``AsyncOpenAI.chat.completions.create(stream=True)`` and
projects each chunk's ``delta.content`` onto the same string-token iterator
shape the transports already consume. Both endpoints depend on
``get_token_streamer`` via FastAPI dependency injection, so tests can swap in a
stub without monkey-patching either transport.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Protocol, runtime_checkable

from app.llm_mock import DEFAULT_TOKEN_DELAY_SECONDS, stream_tokens


OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
OPENAI_SYSTEM_PROMPT = (
    "You are a concise assistant. Answer in one short paragraph."
)

PROVIDER_MOCK = "mock"
PROVIDER_OPENAI = "openai"
ENV_PROVIDER = "LLM_PROVIDER"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_MODEL = "OPENAI_MODEL"


class LLMStreamError(RuntimeError):
    """Raised when the underlying SDK fails before or during a stream."""


@runtime_checkable
class TokenStreamer(Protocol):
    """Anything that yields LLM tokens as ``str`` one at a time."""

    name: str

    def stream(
        self, prompt: str, delay_seconds: float = DEFAULT_TOKEN_DELAY_SECONDS
    ) -> AsyncIterator[str]:
        ...


class MockTokenStreamer:
    """Adapter around the deterministic mock generator from ``llm_mock``."""

    name = PROVIDER_MOCK

    async def stream(
        self,
        prompt: str,
        delay_seconds: float = DEFAULT_TOKEN_DELAY_SECONDS,
    ) -> AsyncIterator[str]:
        async for token in stream_tokens(prompt, delay_seconds=delay_seconds):
            yield token


def _extract_delta_content(chunk: object) -> str | None:
    """Project an OpenAI streaming chunk down to its text delta, if any."""
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return None
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return None
    content = getattr(delta, "content", None)
    if not isinstance(content, str):
        return None
    return content


class OpenAITokenStreamer:
    """Adapter around ``AsyncOpenAI.chat.completions.create(stream=True)``.

    The transport contract is ``AsyncIterator[str]`` — one yielded value per
    visible piece of model output. The SDK delivers chunks whose ``delta``
    field may be empty (role-only opening chunk, tool-call only, etc.); we
    drop those silently so the transports only see real tokens.

    ``delay_seconds`` is accepted to keep the protocol uniform with
    ``MockTokenStreamer`` but is ignored — the live model controls its own
    cadence, and inserting a sleep would only add latency without smoothing
    bursty deltas.
    """

    name = PROVIDER_OPENAI

    def __init__(
        self,
        client: object,
        model: str = OPENAI_DEFAULT_MODEL,
        system_prompt: str = OPENAI_SYSTEM_PROMPT,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt

    @property
    def model(self) -> str:
        return self._model

    async def _open_stream(self, prompt: str) -> object:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            return await self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                messages=messages,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001 — re-raised as domain error
            raise LLMStreamError(f"openai create() failed: {exc}") from exc

    async def stream(
        self,
        prompt: str,
        delay_seconds: float = DEFAULT_TOKEN_DELAY_SECONDS,
    ) -> AsyncIterator[str]:
        del delay_seconds  # intentionally ignored; the model paces itself
        chunks = await self._open_stream(prompt)
        async for chunk in chunks:
            piece = _extract_delta_content(chunk)
            if piece:
                yield piece


def _build_openai_streamer() -> TokenStreamer:
    api_key = os.environ.get(ENV_OPENAI_API_KEY)
    if not api_key:
        return MockTokenStreamer()
    from openai import AsyncOpenAI

    model = os.environ.get(ENV_OPENAI_MODEL, OPENAI_DEFAULT_MODEL)
    return OpenAITokenStreamer(AsyncOpenAI(api_key=api_key), model=model)


def build_default_streamer() -> TokenStreamer:
    """Pick a streamer from the environment without caching."""
    provider = os.environ.get(ENV_PROVIDER, PROVIDER_MOCK).strip().lower()
    if provider == PROVIDER_OPENAI:
        return _build_openai_streamer()
    return MockTokenStreamer()


_default_streamer: TokenStreamer | None = None


def get_token_streamer() -> TokenStreamer:
    """FastAPI dependency. Lazily builds + memoises a process-wide streamer."""
    global _default_streamer
    if _default_streamer is None:
        _default_streamer = build_default_streamer()
    return _default_streamer


def reset_token_streamer() -> None:
    """Drop the cached streamer so the next ``get_token_streamer`` rebuilds."""
    global _default_streamer
    _default_streamer = None
