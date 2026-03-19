"""Tests for the FastAPI endpoints."""

import os

import pytest
from fastapi.testclient import TestClient

# Ensure mock mode
os.environ["USE_MOCK_LLM"] = "true"

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with the app fully initialized."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_response_fields(self, client):
        data = client.get("/api/health").json()
        assert data["status"] == "healthy"
        assert data["mode"] == "mock"
        assert data["documents_indexed"] == 5
        assert data["total_chunks"] > 0
        assert data["index_dimensions"] > 0


class TestQueryEndpoint:
    def test_in_scope_query(self, client):
        response = client.post(
            "/api/query",
            json={"query": "What is the IP rating of the ALR-SL-90W?", "language": "en"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_in_scope"] is True
        assert len(data["answer"]) > 0
        assert len(data["citations"]) > 0
        assert data["language"] == "en"
        assert data["latency_ms"] >= 0

    def test_out_of_scope_query(self, client):
        response = client.post(
            "/api/query",
            json={"query": "What is the weather in Riyadh?", "language": "en"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_in_scope"] is False
        assert "Alrouf" in data["answer"]
        assert data["citations"] == []

    def test_arabic_query(self, client):
        response = client.post(
            "/api/query",
            json={"query": "ما هي شهادات الجودة لشركة الروف؟", "language": "ar"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "ar"
        assert len(data["answer"]) > 0

    def test_warranty_query(self, client):
        response = client.post(
            "/api/query",
            json={"query": "What is the warranty period for LED components?", "language": "en"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_in_scope"] is True
        assert len(data["citations"]) > 0

    def test_invalid_language(self, client):
        response = client.post(
            "/api/query",
            json={"query": "test", "language": "fr"},
        )
        assert response.status_code == 422  # Validation error

    def test_empty_query(self, client):
        response = client.post(
            "/api/query",
            json={"query": "", "language": "en"},
        )
        assert response.status_code == 422


class TestUIEndpoint:
    def test_root_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Alrouf" in response.text
