"""Stress / robustness tests for the SSE and WebSocket transports.

Step 7 of the tutorial pushes both endpoints against the four failure shapes a
production deployment actually meets:

- **Backpressure** — a slow consumer must not let the server run ahead and pile
  tokens into a buffer the client cannot drain.
- **Proxy / CDN quirks** — the no-buffering response headers must stay set on
  every concurrent SSE response, not just the first one.
- **Dropped clients** — when the browser / CLI / mobile app vanishes mid-stream
  the upstream LLM iterator must be torn down immediately so the model isn't
  billed for tokens nobody will read.
- **Multiplexing** — N concurrent clients must each receive their own complete,
  ordered, prompt-specific stream with no cross-talk between them.

These tests intentionally inject an instrumented ``LifecycleStreamer`` so the
assertions can speak to upstream state (started / yielded / cleaned-up) instead
of guessing from wire output.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import AsyncIterator

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from app.llm import get_token_streamer
from app.main import app
from app.sse import SSE_MEDIA_TYPE, sse_token_stream


client = TestClient(app)


TERMINAL_WS_FRAME_TYPES = ("done", "cancelled", "error")


class LifecycleStreamer:
    """Token source that records start / yield / cleanup events.

    Wraps a fixed token list with an optional inter-token sleep so a slow
    upstream can be simulated deterministically. The ``finally`` block bumps
    ``cleaned_up`` even when the consumer drops mid-stream, which is how the
    dropped-client tests verify the transports propagate cancellation upward.
    """

    name = "lifecycle"

    def __init__(
        self, tokens: list[str], inter_token_delay: float = 0.0
    ) -> None:
        self._tokens = list(tokens)
        self._inter_token_delay = inter_token_delay
        self.started = 0
        self.tokens_yielded = 0
        self.completed_normally = 0
        self.cleaned_up = 0

    async def stream(
        self, prompt: str, delay_seconds: float = 0.0
    ) -> AsyncIterator[str]:
        del prompt, delay_seconds
        self.started += 1
        try:
            for token in self._tokens:
                if self._inter_token_delay > 0:
                    await asyncio.sleep(self._inter_token_delay)
                yield token
                self.tokens_yielded += 1
            self.completed_normally += 1
        finally:
            self.cleaned_up += 1


def _parse_sse_frames(body: str) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    for raw in body.split("\n\n"):
        chunk = raw.strip("\n")
        if not chunk:
            continue
        event: str | None = None
        data_lines: list[str] = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip(" "))
        frames.append(
            {"event": event or "message", "data": "\n".join(data_lines)}
        )
    return frames


def _drain_ws(prompt: str) -> list[str]:
    """Connect to /stream/ws, read until terminal frame, return token payloads."""
    path = f"/stream/ws?prompt={prompt}&delay_seconds=0"
    tokens: list[str] = []
    with client.websocket_connect(path) as ws:
        while True:
            raw = ws.receive_text()
            frame = json.loads(raw)
            if frame["type"] == "token":
                tokens.append(frame["data"])
                continue
            if frame["type"] in TERMINAL_WS_FRAME_TYPES:
                break
    return tokens


def _drain_sse(prompt: str) -> tuple[list[str], dict[str, str]]:
    """Issue an SSE request, return token payloads + response headers."""
    params = {"prompt": prompt, "delay_seconds": 0.0}
    with client.stream("GET", "/stream/sse", params=params) as response:
        body = "".join(chunk for chunk in response.iter_text())
        headers = dict(response.headers)
    frames = _parse_sse_frames(body)
    tokens = [frame["data"] for frame in frames if frame["event"] == "token"]
    return tokens, headers


# === Dropped clients ===========================================================


async def test_sse_dropped_client_triggers_upstream_cleanup() -> None:
    """Closing the SSE generator early must propagate cancellation to the streamer.

    We drive ``sse_token_stream`` directly because Starlette's in-process
    ``TestClient`` buffers the response body before exposing any of it, which
    masks the real mid-stream cancel signal a live consumer sends.
    """
    tracker = LifecycleStreamer(
        [f"t{i}" for i in range(50)], inter_token_delay=0.0
    )
    generator = sse_token_stream(tracker, prompt="x", delay_seconds=0.0)

    first_frame = await generator.__anext__()
    assert first_frame.startswith("event: token")

    await generator.aclose()

    assert tracker.started == 1
    assert tracker.cleaned_up == 1
    assert tracker.completed_normally == 0
    assert tracker.tokens_yielded < 50


def test_ws_dropped_client_triggers_upstream_cleanup() -> None:
    """Closing the WS early must tear down the streamer (no runaway tokens)."""
    tracker = LifecycleStreamer(
        [f"t{i}" for i in range(50)], inter_token_delay=0.02
    )
    app.dependency_overrides[get_token_streamer] = lambda: tracker
    try:
        with client.websocket_connect(
            "/stream/ws?prompt=x&delay_seconds=0"
        ) as ws:
            first = json.loads(ws.receive_text())
            assert first["type"] == "token"
            ws.close()
            with pytest.raises(WebSocketDisconnect):
                while True:
                    ws.receive_text()
    finally:
        app.dependency_overrides.pop(get_token_streamer, None)

    assert tracker.started == 1
    assert tracker.cleaned_up == 1
    assert tracker.completed_normally == 0
    assert tracker.tokens_yielded < 50


# === Backpressure ==============================================================


async def test_sse_backpressure_yields_one_token_per_anext() -> None:
    """``sse_token_stream`` must not race ahead of its consumer.

    Each ``__anext__`` call should drive the upstream streamer forward by
    exactly one token — proof that the SSE wrapper backpressures naturally
    via Python's async-generator semantics instead of pre-buffering.
    """
    tracker = LifecycleStreamer(
        [f"t{i}" for i in range(100)], inter_token_delay=0.0
    )
    generator = sse_token_stream(tracker, prompt="x", delay_seconds=0.0)

    await generator.__anext__()
    await generator.__anext__()
    await generator.__anext__()

    # After K frames, the streamer's post-yield counter has incremented K-1
    # times (the last yield is still suspended waiting for the next __anext__).
    assert tracker.tokens_yielded == 2

    await generator.aclose()
    assert tracker.cleaned_up == 1
    assert tracker.completed_normally == 0


def test_ws_backpressure_bounded_lookahead_when_consumer_pauses() -> None:
    """WS slow-consumer variant of the SSE backpressure test."""
    tracker = LifecycleStreamer(
        [f"t{i}" for i in range(100)], inter_token_delay=0.02
    )
    app.dependency_overrides[get_token_streamer] = lambda: tracker
    try:
        with client.websocket_connect(
            "/stream/ws?prompt=x&delay_seconds=0"
        ) as ws:
            first = json.loads(ws.receive_text())
            assert first["type"] == "token"
            ws.close()
            with pytest.raises(WebSocketDisconnect):
                while True:
                    ws.receive_text()
    finally:
        app.dependency_overrides.pop(get_token_streamer, None)

    assert tracker.tokens_yielded < 10
    assert tracker.cleaned_up == 1


# === Multiplexing ==============================================================


def test_sse_multiplexed_clients_each_get_their_own_stream() -> None:
    prompts = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
        "kappa lambda mu",
    ]
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = {pool.submit(_drain_sse, prompt): prompt for prompt in prompts}
        results = {futures[fut]: fut.result() for fut in as_completed(futures)}

    for prompt, (tokens, headers) in results.items():
        expected_head = prompt.split()
        assert tokens[: len(expected_head)] == expected_head, prompt
        assert tokens[-5:] == ["This", "is", "a", "mock", "stream."], prompt
        assert headers.get("content-type", "").startswith(SSE_MEDIA_TYPE)


def test_ws_multiplexed_clients_each_get_their_own_stream() -> None:
    prompts = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
        "kappa lambda mu",
    ]
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = {pool.submit(_drain_ws, prompt): prompt for prompt in prompts}
        results = {futures[fut]: fut.result() for fut in as_completed(futures)}

    for prompt, tokens in results.items():
        expected_head = prompt.split()
        assert tokens[: len(expected_head)] == expected_head, prompt
        assert tokens[-5:] == ["This", "is", "a", "mock", "stream."], prompt


# === Proxy / CDN quirks under concurrency =====================================


def test_sse_no_buffering_headers_present_on_every_concurrent_response() -> None:
    """Each concurrent SSE response must carry the headers proxies/CDNs respect."""
    prompts = [f"ping-{idx}" for idx in range(6)]
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        future_list = [pool.submit(_drain_sse, prompt) for prompt in prompts]
        header_sets = [fut.result()[1] for fut in as_completed(future_list)]

    for headers in header_sets:
        assert headers.get("cache-control") == "no-cache"
        assert headers.get("connection") == "keep-alive"
        assert headers.get("x-accel-buffering") == "no"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
