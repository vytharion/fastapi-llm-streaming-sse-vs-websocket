"""Tests for the WebSocket streaming endpoint."""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from app.llm import LLMStreamError, get_token_streamer
from app.main import app
from app.ws import (
    CANCEL_ACTION,
    OUTCOME_CANCELLED,
    OUTCOME_DONE,
    OUTCOME_ERROR,
    encode_cancelled_frame,
    encode_done_frame,
    encode_error_frame,
    encode_token_frame,
    is_cancel_message,
)


client = TestClient(app)


TERMINAL_FRAME_TYPES = (OUTCOME_DONE, OUTCOME_CANCELLED, OUTCOME_ERROR)


def _drain_until_terminal(ws) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    while True:
        try:
            raw = ws.receive_text()
        except WebSocketDisconnect:
            break
        frame = json.loads(raw)
        frames.append(frame)
        if frame["type"] in TERMINAL_FRAME_TYPES:
            break
    return frames


def test_encode_token_frame_roundtrips_via_json() -> None:
    assert json.loads(encode_token_frame("hi")) == {"type": "token", "data": "hi"}


def test_encode_done_frame_payload() -> None:
    assert json.loads(encode_done_frame()) == {"type": "done", "data": "[DONE]"}


def test_encode_cancelled_frame_payload() -> None:
    payload = json.loads(encode_cancelled_frame())

    assert payload == {"type": "cancelled", "data": "[CANCELLED]"}


def test_is_cancel_message_accepts_bare_keyword() -> None:
    assert is_cancel_message(CANCEL_ACTION) is True
    assert is_cancel_message("CANCEL") is True
    assert is_cancel_message("  cancel  ") is True


def test_is_cancel_message_accepts_json_action() -> None:
    assert is_cancel_message(json.dumps({"action": "cancel"})) is True


def test_is_cancel_message_rejects_unrelated_strings() -> None:
    assert is_cancel_message("hello") is False
    assert is_cancel_message(json.dumps({"action": "ping"})) is False
    assert is_cancel_message(json.dumps([1, 2, 3])) is False
    assert is_cancel_message("not json {") is False


def test_ws_streams_tokens_then_done() -> None:
    with client.websocket_connect(
        "/stream/ws?prompt=hi+there&delay_seconds=0"
    ) as ws:
        frames = _drain_until_terminal(ws)

    token_payloads = [frame["data"] for frame in frames if frame["type"] == "token"]
    terminal = [frame for frame in frames if frame["type"] == "done"]

    assert token_payloads[:2] == ["hi", "there"]
    assert token_payloads[-5:] == ["This", "is", "a", "mock", "stream."]
    assert len(terminal) == 1
    assert terminal[0]["data"] == "[DONE]"
    assert frames[-1]["type"] == "done"


def test_ws_cancellation_stops_stream_and_emits_cancelled_frame() -> None:
    with client.websocket_connect(
        "/stream/ws?prompt=one+two+three+four+five&delay_seconds=0.05"
    ) as ws:
        first_raw = ws.receive_text()
        first = json.loads(first_raw)
        assert first["type"] == "token"

        ws.send_text(json.dumps({"action": "cancel"}))
        remaining = _drain_until_terminal(ws)

    terminal = [frame for frame in remaining if frame["type"] == "cancelled"]
    tokens_after_cancel = [frame for frame in remaining if frame["type"] == "token"]

    assert len(terminal) == 1
    assert terminal[0]["data"] == "[CANCELLED]"
    # We may have one already-queued token in flight, but never the whole stream.
    assert len(tokens_after_cancel) < 5


def test_ws_cancellation_also_accepts_bare_keyword() -> None:
    with client.websocket_connect(
        "/stream/ws?prompt=one+two+three+four+five&delay_seconds=0.05"
    ) as ws:
        ws.receive_text()
        ws.send_text("cancel")
        remaining = _drain_until_terminal(ws)

    terminal = [frame for frame in remaining if frame["type"] == "cancelled"]
    assert len(terminal) == 1


def test_ws_client_disconnect_does_not_crash_server() -> None:
    """Abrupt client disconnect must shut the handler down cleanly."""
    with client.websocket_connect(
        "/stream/ws?prompt=alpha+beta+gamma&delay_seconds=0.05"
    ) as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "token"
        ws.close()
        # Drain whatever the server has queued so the handler observes the
        # disconnect and finishes its cleanup before the test context exits.
        with pytest.raises(WebSocketDisconnect):
            while True:
                ws.receive_text()

    healthz = client.get("/healthz")
    assert healthz.status_code == 200


def test_ws_closes_socket_after_done_frame() -> None:
    with client.websocket_connect(
        "/stream/ws?prompt=hi&delay_seconds=0"
    ) as ws:
        _drain_until_terminal(ws)
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


def test_ws_endpoint_listed_in_openapi_schema() -> None:
    schema = client.get("/openapi.json").json()

    assert "/stream/ws" not in schema.get("paths", {})


def test_encode_error_frame_payload() -> None:
    payload = json.loads(encode_error_frame("boom"))

    assert payload == {"type": OUTCOME_ERROR, "data": "boom"}


class _ScriptedStreamer:
    name = "scripted"

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    async def stream(
        self, prompt: str, delay_seconds: float = 0.0
    ) -> AsyncIterator[str]:
        del prompt, delay_seconds
        for token in self._tokens:
            yield token


class _BrokenStreamer:
    name = "broken"

    async def stream(
        self, prompt: str, delay_seconds: float = 0.0
    ) -> AsyncIterator[str]:
        del prompt, delay_seconds
        if False:
            yield ""  # pragma: no cover — marks this as an async generator
        raise LLMStreamError("upstream unavailable")


def test_ws_endpoint_consumes_injected_streamer() -> None:
    app.dependency_overrides[get_token_streamer] = lambda: _ScriptedStreamer(
        ["alpha", "beta", "gamma"]
    )
    try:
        with client.websocket_connect(
            "/stream/ws?prompt=ignored&delay_seconds=0"
        ) as ws:
            frames = _drain_until_terminal(ws)
    finally:
        app.dependency_overrides.pop(get_token_streamer, None)

    token_payloads = [frame["data"] for frame in frames if frame["type"] == "token"]
    terminal = [frame for frame in frames if frame["type"] == OUTCOME_DONE]

    assert token_payloads == ["alpha", "beta", "gamma"]
    assert len(terminal) == 1


def test_ws_endpoint_emits_error_frame_on_llm_failure() -> None:
    app.dependency_overrides[get_token_streamer] = lambda: _BrokenStreamer()
    try:
        with client.websocket_connect(
            "/stream/ws?prompt=ignored&delay_seconds=0"
        ) as ws:
            frames = _drain_until_terminal(ws)
    finally:
        app.dependency_overrides.pop(get_token_streamer, None)

    error_frames = [frame for frame in frames if frame["type"] == OUTCOME_ERROR]
    done_frames = [frame for frame in frames if frame["type"] == OUTCOME_DONE]

    assert len(error_frames) == 1
    assert "upstream unavailable" in error_frames[0]["data"]
    assert done_frames == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
