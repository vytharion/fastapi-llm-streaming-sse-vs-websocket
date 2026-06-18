"""Tests for the SSE streaming endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.sse import SSE_MEDIA_TYPE, format_sse_event


client = TestClient(app)


def _parse_sse_frames(body: str) -> list[dict[str, str]]:
    """Parse a raw SSE response body into ``{event, data}`` dicts."""
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
        frames.append({"event": event or "message", "data": "\n".join(data_lines)})
    return frames


def test_format_sse_event_single_line() -> None:
    frame = format_sse_event("hello", event="token")

    assert frame == "event: token\ndata: hello\n\n"


def test_format_sse_event_splits_multiline_payload() -> None:
    frame = format_sse_event("line one\nline two")

    assert frame == "data: line one\ndata: line two\n\n"


def test_sse_endpoint_returns_event_stream_content_type() -> None:
    with client.stream("GET", "/stream/sse", params={"prompt": "hi", "delay_seconds": 0.0}) as response:
        assert response.status_code == 200
        content_type = response.headers["content-type"]
        assert content_type.startswith(SSE_MEDIA_TYPE)


def test_sse_endpoint_sets_no_buffering_headers() -> None:
    with client.stream("GET", "/stream/sse", params={"prompt": "hi", "delay_seconds": 0.0}) as response:
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("connection") == "keep-alive"
        assert response.headers.get("x-accel-buffering") == "no"


def test_sse_endpoint_streams_tokens_then_done() -> None:
    with client.stream(
        "GET",
        "/stream/sse",
        params={"prompt": "hi there", "delay_seconds": 0.0},
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())

    frames = _parse_sse_frames(body)

    token_frames = [frame for frame in frames if frame["event"] == "token"]
    done_frames = [frame for frame in frames if frame["event"] == "done"]

    payloads = [frame["data"] for frame in token_frames]
    assert payloads[:2] == ["hi", "there"]
    assert payloads[-5:] == ["This", "is", "a", "mock", "stream."]
    assert len(done_frames) == 1
    assert done_frames[0]["data"] == "[DONE]"
    assert frames[-1]["event"] == "done"


def test_sse_endpoint_rejects_out_of_range_delay() -> None:
    response = client.get("/stream/sse", params={"prompt": "hi", "delay_seconds": -1})

    assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
