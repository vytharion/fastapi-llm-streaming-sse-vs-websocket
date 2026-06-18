"""Tests for the static EventSource browser client."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.client import CLIENT_HTML_PATH, load_client_html
from app.main import app


client = TestClient(app)


def test_client_html_file_exists_on_disk() -> None:
    assert CLIENT_HTML_PATH.is_file()


def test_load_client_html_returns_doctype_html() -> None:
    html = load_client_html()

    assert html.lstrip().lower().startswith("<!doctype html")


def test_index_route_serves_client_html() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == load_client_html()


def test_client_html_route_serves_same_payload_as_index() -> None:
    index_response = client.get("/")
    aliased_response = client.get("/client/index.html")

    assert aliased_response.status_code == 200
    assert aliased_response.headers["content-type"].startswith("text/html")
    assert aliased_response.text == index_response.text


def test_client_html_subscribes_to_sse_endpoint_with_eventsource() -> None:
    html = load_client_html()

    assert "new EventSource(" in html
    assert "/stream/sse" in html


def test_client_html_listens_for_token_and_done_events() -> None:
    html = load_client_html()

    listeners = re.findall(r"addEventListener\(\"(\w+)\"", html)
    assert "token" in listeners
    assert "done" in listeners


def test_client_html_closes_eventsource_on_done() -> None:
    html = load_client_html()

    assert ".close()" in html


def test_client_html_has_prompt_form_and_output_target() -> None:
    html = load_client_html()

    assert 'id="prompt-form"' in html
    assert 'name="prompt"' in html
    assert 'id="output"' in html


def test_index_route_excluded_from_openapi_schema() -> None:
    response = client.get("/openapi.json")
    schema = response.json()
    paths = schema.get("paths", {})

    assert "/" not in paths
    assert "/client/index.html" not in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
