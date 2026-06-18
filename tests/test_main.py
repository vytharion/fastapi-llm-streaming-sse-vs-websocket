"""Tests for the FastAPI entrypoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import __version__
from app.main import app


client = TestClient(app)


def test_healthz_returns_ok() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_openapi_schema_is_served() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "FastAPI LLM streaming demo"
