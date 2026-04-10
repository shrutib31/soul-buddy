"""
Unit tests for Classify API: POST /classify with mocked get_classifications.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.classify import router as classify_router


@pytest.fixture
def app():
    from api.supabase_auth import verify_supabase_token
    app = FastAPI()
    app.include_router(classify_router, prefix="/api/v1", tags=["Classification"])
    # Override auth dependency so unit tests don't need a real Supabase token
    app.dependency_overrides[verify_supabase_token] = lambda: {"id": "test-user-id"}
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestClassifyEndpointUnit:
    """Unit tests for classify endpoint with mocked get_classifications."""

    def test_classify_returns_200_and_classifications(self, client):
        mock_result = {
            "intent": "venting",
            "situation": "GENERAL_OVERWHELM",
            "severity": "medium",
            "risk_score": 0.3,
            "risk_level": "medium",
        }
        with patch(
            "graph.nodes.agentic_nodes.classification_node.get_classifications",
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/v1/classify",
                json={"message": "I've been really stressed lately"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("success") is True
        assert data.get("message") == "I've been really stressed lately"
        assert "classifications" in data
        assert data["classifications"].get("intent") == "venting"

    def test_classify_validates_message_required(self, client):
        resp = client.post("/api/v1/classify", json={})
        assert resp.status_code == 422

    def test_classify_empty_message_rejected(self, client):
        resp = client.post("/api/v1/classify", json={"message": ""})
        assert resp.status_code == 422

    def test_classify_exception_returns_500(self, client):
        with patch(
            "graph.nodes.agentic_nodes.classification_node.get_classifications",
            side_effect=RuntimeError("Model not loaded"),
        ):
            resp = client.post(
                "/api/v1/classify",
                json={"message": "Hello"},
            )
        assert resp.status_code == 500
        data = resp.json()
        assert data.get("success") is False
        assert "error" in data
