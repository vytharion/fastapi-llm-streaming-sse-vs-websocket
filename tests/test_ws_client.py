"""Tests for the static WebSocket browser client.

The client lives at ``app/static/ws.html`` and is served at ``/ws``. It
must connect to the WS endpoint, reconnect on unexpected drops, run an
application-level heartbeat to detect dead sockets, and let the user
both cancel a stream and stop further reconnect attempts.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.client import (
    WS_CLIENT_HTML_PATH,
    load_client_html,
    load_ws_client_html,
)
from app.main import app


client = TestClient(app)


def test_ws_client_html_file_exists_on_disk() -> None:
    assert WS_CLIENT_HTML_PATH.is_file()


def test_load_ws_client_html_returns_doctype_html() -> None:
    html = load_ws_client_html()

    assert html.lstrip().lower().startswith("<!doctype html")


def test_ws_client_html_is_distinct_from_sse_client() -> None:
    assert load_ws_client_html() != load_client_html()


def test_ws_route_serves_ws_client_html() -> None:
    response = client.get("/ws")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == load_ws_client_html()


def test_client_ws_html_alias_serves_same_payload() -> None:
    canonical = client.get("/ws")
    alias = client.get("/client/ws.html")

    assert alias.status_code == 200
    assert alias.headers["content-type"].startswith("text/html")
    assert alias.text == canonical.text


def test_ws_client_opens_websocket_to_stream_ws() -> None:
    html = load_ws_client_html()

    assert "new WebSocket(" in html
    assert "/stream/ws" in html


def test_ws_client_uses_scheme_aware_url_builder() -> None:
    """wss:// must be used when the page itself is served over https."""
    html = load_ws_client_html()

    assert '"wss"' in html
    assert '"ws"' in html
    assert "window.location.protocol" in html


def test_ws_client_has_reconnect_logic_with_backoff() -> None:
    html = load_ws_client_html()

    assert "scheduleReconnect" in html
    assert "maxReconnectAttempts" in html
    assert "Math.pow(2," in html
    assert "maxBackoffMs" in html


def test_ws_client_caps_reconnect_attempts() -> None:
    html = load_ws_client_html()

    match = re.search(r"maxReconnectAttempts:\s*(\d+)", html)
    assert match is not None
    cap = int(match.group(1))
    assert 1 <= cap <= 20


def test_ws_client_runs_application_level_heartbeat() -> None:
    html = load_ws_client_html()

    assert "heartbeatIntervalMs" in html
    assert "heartbeatTimeoutMs" in html
    assert "startHeartbeat" in html
    assert "sendHeartbeat" in html
    assert '{ action: "ping" }' in html or "'action': 'ping'" in html


def test_ws_client_detects_idle_socket_via_watchdog() -> None:
    html = load_ws_client_html()

    assert "checkHeartbeatTimeout" in html
    assert "lastFrameAt" in html
    assert "forceReconnect" in html


def test_ws_client_clears_timers_on_close() -> None:
    """Reconnect / heartbeat timers MUST be cleared when the socket closes
    so a reload doesn't leak intervals into the next session."""
    html = load_ws_client_html()

    assert "clearInterval" in html
    assert "clearTimeout" in html
    assert "clearTimers" in html


def test_ws_client_handles_token_done_and_cancelled_frames() -> None:
    html = load_ws_client_html()

    assert '"token"' in html
    assert '"done"' in html
    assert '"cancelled"' in html


def test_ws_client_sends_cancel_action_frame() -> None:
    html = load_ws_client_html()

    assert '{ action: "cancel" }' in html or "'action': 'cancel'" in html


def test_ws_client_has_form_cancel_and_stop_controls() -> None:
    html = load_ws_client_html()

    assert 'id="prompt-form"' in html
    assert 'name="prompt"' in html
    assert 'id="output"' in html
    assert 'id="cancel-button"' in html
    assert 'id="stop-button"' in html


def test_ws_client_does_not_reconnect_after_clean_termination() -> None:
    """A `done` or `cancelled` frame must flip a flag that skips reconnect."""
    html = load_ws_client_html()

    assert "terminated" in html
    assert "stoppedByUser" in html


def test_ws_routes_excluded_from_openapi_schema() -> None:
    schema = client.get("/openapi.json").json()
    paths = schema.get("paths", {})

    assert "/ws" not in paths
    assert "/client/ws.html" not in paths


def test_existing_sse_index_still_served() -> None:
    """Regression: adding the WS client must not break the SSE landing page."""
    response = client.get("/")

    assert response.status_code == 200
    assert "EventSource" in response.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
