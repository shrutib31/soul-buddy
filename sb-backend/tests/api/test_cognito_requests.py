"""
Unit tests for Cognito chat requests.

Covers token extraction quality, endpoint correctness, and short-circuit
behavior to avoid unnecessary work on invalid requests.
"""

import pytest
import sys
import types
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Prevent heavy model imports when loading api.chat in unit tests.
if "graph.graph_builder" not in sys.modules:
    graph_builder_stub = types.ModuleType("graph.graph_builder")

    def _stub_get_compiled_flow():
        class _StubFlow:
            async def ainvoke(self, _state):
                return {"api_response": {"success": True, "response": "stub", "conversation_id": "stub"}}

        return _StubFlow()

    graph_builder_stub.get_compiled_flow = _stub_get_compiled_flow
    sys.modules["graph.graph_builder"] = graph_builder_stub

from api.chat import (
    router as chat_router,
    verify_supabase_token,
    extract_token_from_headers,
)
from graph.state import ConversationState


# ============================================================================
# App for TestClient
# ============================================================================

@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])

    async def _fake_verified_user():
        return {"id": "dep-user-id"}

    app.dependency_overrides[verify_supabase_token] = _fake_verified_user
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def cognito_payload():
    return {
        "message": "I need help handling stress today.",
        "mode": "cognito",
        "domain": "student",
        "sb_conv_id": "conv-123",
    }


@pytest.fixture
def cognito_state():
    return ConversationState(
        conversation_id="conv-123",
        mode="cognito",
        domain="student",
        user_message="I need help handling stress today.",
        supabase_uid="sb-user-123",
        app_user_id=99,
    )


# ============================================================================
# extract_token_from_headers
# ============================================================================

class TestExtractTokenFromHeaders:
    """Unit tests for token extraction precedence and normalization."""

    # T: access_token is chosen over Authorization when extracting tokens
    # F: header precedence is broken and may use the wrong user token.
    def test_token_priority(self):
        """Prefers access_token over Authorization token """
        token = extract_token_from_headers("preferred-token", "Bearer fallback-token")
        assert token == "preferred-token"

    # T: "Bearer" prefix is stripped and raw token remains.
    # F: token parsing is broken and verification may fail.
    def test_bearer_strip(self):
        """Removes Bearer prefix from Authorization"""
        token = extract_token_from_headers(None, "Bearer abc123")
        assert token == "abc123"

    # T: blank/empty headers produce None.
    # F: empty values are treated as valid tokens.
    def test_blank_headers(self):
        """Returns None when headers are blank."""
        token = extract_token_from_headers("   ", "")
        assert token is None


# ============================================================================
# /chat/cognito
# ============================================================================

class TestCognitoChatRequestUnit:
    """Unit tests for /chat/cognito request handling."""

    # T: valid Cognito request returns 200 and identity/state/graph wiring is correct.
    # F: happy-path request handling is broken.
    def test_chat_success(self, client, cognito_payload, cognito_state):
        """Returns 200 and uses resolved Cognito identity."""
        with patch(
            "api.chat.resolve_cognito_identity_from_access_token",
            new_callable=AsyncMock,
            return_value=("sb-user-123", 99),
        ) as mock_resolve, patch(
            "api.chat.create_initial_state",
            new_callable=AsyncMock,
            return_value=cognito_state,
        ) as mock_create_state, patch(
            "api.chat.invoke_graph",
            new_callable=AsyncMock,
            return_value={
                "api_response": {
                    "success": True,
                    "conversation_id": "conv-123",
                    "response": "Thanks for sharing. Let's work through this together.",
                    "metadata": {"source": "unit-test"},
                }
            },
        ) as mock_invoke:
            resp = client.post(
                "/api/v1/chat/cognito",
                json=cognito_payload,
                headers={
                    "access_token": "preferred-token",
                    "Authorization": "Bearer fallback-token",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["conversation_id"] == "conv-123"
        assert "response" in data

        # access_token header must win over Authorization
        mock_resolve.assert_awaited_once_with("preferred-token")
        mock_create_state.assert_awaited_once()
        kwargs = mock_create_state.await_args.kwargs
        assert kwargs["mode"] == "cognito"
        assert kwargs["supabase_uid"] == "sb-user-123"
        assert kwargs["app_user_id"] == 99
        mock_invoke.assert_awaited_once_with(cognito_state)

    # T: missing token returns 400 and exits before expensive work.
    # F: missing-token guard failed; route may do unauthorized processing.
    def test_chat_missing_token(self, client, cognito_payload):
        """Short-circuits with 400 when token is missing."""
        with patch(
            "api.chat.resolve_cognito_identity_from_access_token",
            new_callable=AsyncMock,
        ) as mock_resolve, patch(
            "api.chat.create_initial_state",
            new_callable=AsyncMock,
        ) as mock_create_state, patch(
            "api.chat.invoke_graph",
            new_callable=AsyncMock,
        ) as mock_invoke:
            resp = client.post("/api/v1/chat/cognito", json=cognito_payload)

        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert "Missing access_token header for cognito mode" in data["error"]
        mock_resolve.assert_not_awaited()
        mock_create_state.assert_not_awaited()
        mock_invoke.assert_not_awaited()

    # T: invalid token returns 400 and skips state creation + graph invoke.
    # F: invalid token may leak into downstream execution.
    def test_chat_invalid_token(
        self, client, cognito_payload
    ):
        """Returns 400 and skips graph work on bad token."""
        with patch(
            "api.chat.resolve_cognito_identity_from_access_token",
            new_callable=AsyncMock,
            side_effect=ValueError("Failed to verify access_token: invalid signature"),
        ) as mock_resolve, patch(
            "api.chat.create_initial_state",
            new_callable=AsyncMock,
        ) as mock_create_state, patch(
            "api.chat.invoke_graph",
            new_callable=AsyncMock,
        ) as mock_invoke:
            resp = client.post(
                "/api/v1/chat/cognito",
                json=cognito_payload,
                headers={"access_token": "bad-token"},
            )

        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert "Failed to verify access_token" in data["error"]
        mock_resolve.assert_awaited_once_with("bad-token")
        mock_create_state.assert_not_awaited()
        mock_invoke.assert_not_awaited()


# ============================================================================
# /chat/cognito/stream
# ============================================================================

class TestCognitoStreamRequestUnit:
    """Unit tests for /chat/cognito/stream request handling."""

    # T: valid stream request returns SSE content and streams events.
    # F: streaming contract or identity flow is broken.
    def test_stream_success(self, client, cognito_payload, cognito_state):
        """Returns SSE stream for valid Cognito request."""
        async def fake_stream(_state):
            yield "data: {\"step\": \"ok\"}\n\n"

        with patch(
            "api.chat.resolve_cognito_identity_from_access_token",
            new_callable=AsyncMock,
            return_value=("sb-user-123", 99),
        ) as mock_resolve, patch(
            "api.chat.create_initial_state",
            new_callable=AsyncMock,
            return_value=cognito_state,
        ) as mock_create_state, patch(
            "api.chat.stream_as_sse",
            side_effect=fake_stream,
        ) as mock_stream:
            with client.stream(
                "POST",
                "/api/v1/chat/cognito/stream",
                json=cognito_payload,
                headers={"access_token": "stream-token"},
            ) as resp:
                body = "".join(resp.iter_text())

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "data:" in body
        mock_resolve.assert_awaited_once_with("stream-token")
        mock_create_state.assert_awaited_once()
        mock_stream.assert_called_once_with(cognito_state)

    # T: missing token returns 400 and does not start stream/state work.
    # F: stream route fails to short-circuit unauthorized requests.
    def test_stream_missing_token(self, client, cognito_payload):
        """Short-circuits stream route when token is missing."""
        with patch(
            "api.chat.resolve_cognito_identity_from_access_token",
            new_callable=AsyncMock,
        ) as mock_resolve, patch(
            "api.chat.create_initial_state",
            new_callable=AsyncMock,
        ) as mock_create_state, patch(
            "api.chat.stream_as_sse",
        ) as mock_stream:
            resp = client.post("/api/v1/chat/cognito/stream", json=cognito_payload)

        assert resp.status_code == 400
        data = resp.json()
        assert "Missing access_token header for cognito mode" in data["error"]
        mock_resolve.assert_not_awaited()
        mock_create_state.assert_not_awaited()
        mock_stream.assert_not_called()
