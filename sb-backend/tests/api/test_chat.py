"""
Unit tests for Chat API: create_initial_state, invoke_graph, and chat endpoints.

Endpoints are tested with mocked graph so no DB or LLM is used.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.chat import (
    router as chat_router,
    create_initial_state,
    invoke_graph,
    get_flow,
)
from graph.state import ConversationState


# ============================================================================
# App for TestClient
# ============================================================================

@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ============================================================================
# create_initial_state
# ============================================================================

class TestCreateInitialState:
    """Unit tests for create_initial_state."""

    @pytest.mark.asyncio
    async def test_returns_conversation_state(self):
        state = await create_initial_state(
            message="Hello",
            mode="incognito",
            domain="student",
            conversation_id=None,
        )
        assert isinstance(state, ConversationState)
        assert state.user_message == "Hello"
        assert state.mode == "incognito"
        assert state.domain == "student"
        assert state.conversation_id == ""

    @pytest.mark.asyncio
    async def test_uses_provided_conversation_id(self):
        state = await create_initial_state(
            message="Hi",
            mode="cognito",
            domain="general",
            conversation_id="existing-conv-123",
        )
        assert state.conversation_id == "existing-conv-123"


# ============================================================================
# Chat endpoints (mocked flow)
# ============================================================================

class TestChatEndpointsUnit:
    """Unit tests for chat endpoints with mocked graph."""

    def test_incognito_chat_returns_200_with_mocked_flow(self, client):
        with patch("api.chat.get_flow", new_callable=AsyncMock) as mock_get_flow:
            mock_flow = AsyncMock()
            mock_flow.ainvoke = AsyncMock(return_value={
                "api_response": {
                    "success": True,
                    "conversation_id": "test-id",
                    "response": "I'm here for you.",
                    "metadata": {},
                }
            })
            mock_get_flow.return_value = mock_flow
            resp = client.post(
                "/api/v1/chat/incognito",
                json={"message": "I need support", "mode": "incognito", "domain": "student"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("success") is True
        assert "response" in data

    def test_incognito_chat_validates_message_required(self, client):
        resp = client.post(
            "/api/v1/chat/incognito",
            json={"mode": "incognito", "domain": "student"},
        )
        assert resp.status_code in (422, 400)

    def test_classify_endpoint_not_on_chat_router(self, client):
        # Classify is on classify_router; this client only mounts chat_router,
        # so /api/v1/classify must always be 404.
        resp = client.post("/api/v1/classify", json={"message": "Hello"})
        assert resp.status_code == 404
