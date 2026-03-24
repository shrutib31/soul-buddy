"""
Unit tests for Chat API: create_initial_state, invoke_graph, and chat endpoints.

Endpoints are tested with mocked graph so no DB or LLM is used.
"""

import sys
import types
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

redis_module = types.ModuleType("redis")
redis_asyncio_module = types.ModuleType("redis.asyncio")
redis_exceptions_module = types.ModuleType("redis.exceptions")
redis_asyncio_module.Redis = object
redis_exceptions_module.ConnectionError = RuntimeError
redis_exceptions_module.TimeoutError = RuntimeError
redis_module.asyncio = redis_asyncio_module
redis_module.exceptions = redis_exceptions_module
sys.modules.setdefault("redis", redis_module)
sys.modules.setdefault("redis.asyncio", redis_asyncio_module)
sys.modules.setdefault("redis.exceptions", redis_exceptions_module)

from api.chat import (
    router as chat_router,
    create_initial_state,
    invoke_graph,
    get_flow,
)
from api.supabase_auth import optional_supabase_token, verify_supabase_token
from graph.state import ConversationState


FAKE_USER = {"id": "user-abc-123", "email": "test@example.com"}


# ============================================================================
# App for TestClient
# ============================================================================

@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
    # Override auth dependencies so tests don't need real Supabase tokens
    app.dependency_overrides[optional_supabase_token] = lambda: None
    app.dependency_overrides[verify_supabase_token] = lambda: FAKE_USER
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
            chat_preference="general",
            conversation_id=None,
        )
        assert isinstance(state, ConversationState)
        assert state.user_message == "Hello"
        assert state.mode == "incognito"
        assert state.domain == "student"
        assert state.conversation_id == ""

    @pytest.mark.asyncio
    async def test_uses_valid_uuid_conversation_id(self):
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        state = await create_initial_state(
            message="Hi",
            mode="cognito",
            domain="general",
            chat_preference="general",
            conversation_id=valid_uuid,
        )
        assert state.conversation_id == valid_uuid

    @pytest.mark.asyncio
    async def test_invalid_uuid_conversation_id_is_ignored(self):
        """Non-UUID strings (e.g. 'existing-conv-123') must be silently ignored."""
        state = await create_initial_state(
            message="Hi",
            mode="cognito",
            domain="general",
            chat_preference="general",
            conversation_id="existing-conv-123",
        )
        assert state.conversation_id == ""

    @pytest.mark.asyncio
    async def test_supabase_uid_is_stored_in_state(self):
        state = await create_initial_state(
            message="Hello",
            mode="cognito",
            domain="student",
            chat_preference="general",
            supabase_uid="user-abc-123",
        )
        assert state.supabase_uid == "user-abc-123"

    @pytest.mark.asyncio
    async def test_incognito_mode_supabase_uid_is_none(self):
        state = await create_initial_state(
            message="Hello",
            mode="incognito",
            domain="student",
            chat_preference="general",
        )
        assert state.supabase_uid is None


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
                "/api/v1/chat",
                json={
                    "message": "I need support",
                    "is_incognito": True,
                    "domain": "student",
                    "chat_preference": "general",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("success") is True
        assert "response" in data

    def test_incognito_chat_validates_message_required(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"is_incognito": True, "domain": "student", "chat_preference": "general"},
        )
        assert resp.status_code in (422, 400)

    def test_cognito_mode_without_auth_returns_401(self, client):
        """When is_incognito=False and no auth token is present, expect 401."""
        with patch("api.chat.get_flow", new_callable=AsyncMock):
            resp = client.post(
                "/api/v1/chat",
                json={
                    "message": "Hello",
                    "is_incognito": False,
                    "domain": "student",
                    "chat_preference": "general",
                },
            )
        assert resp.status_code == 401

    def test_nonsense_chat_fast_paths_through_graph(self, client):
        async def fake_conv_id_handler(_state):
            return {"conversation_id": "test-conv-123"}

        async def fake_load_user_context(_state):
            return {}

        async def fake_store_message(_state):
            return {}

        def unexpected_classification(_state):
            raise AssertionError("classification_node should not run")

        with patch("api.chat.flow", None), patch(
            "graph.graph_builder.conv_id_handler_node",
            new=fake_conv_id_handler,
        ), patch(
            "graph.graph_builder.load_user_context_node",
            new=fake_load_user_context,
        ), patch(
            "graph.graph_builder.store_message_node",
            new=fake_store_message,
        ), patch(
            "graph.graph_builder.classification_node",
            new=unexpected_classification,
        ):
            resp = client.post(
                "/api/v1/chat",
                json={
                    "message": "asdfghjkl",
                    "is_incognito": True,
                    "domain": "student",
                    "chat_preference": "general",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["metadata"]["intent"] == "out_of_scope"
        assert data["metadata"]["out_of_scope_reason"] == "nonsense"
        assert "SoulGym" in data["response"]

    def test_general_knowledge_chat_fast_paths_through_graph(self, client):
        async def fake_conv_id_handler(_state):
            return {"conversation_id": "test-conv-123"}

        async def fake_load_user_context(_state):
            return {}

        async def fake_store_message(_state):
            return {}

        def unexpected_classification(_state):
            raise AssertionError("classification_node should not run")

        with patch("api.chat.flow", None), patch(
            "graph.graph_builder.conv_id_handler_node",
            new=fake_conv_id_handler,
        ), patch(
            "graph.graph_builder.load_user_context_node",
            new=fake_load_user_context,
        ), patch(
            "graph.graph_builder.store_message_node",
            new=fake_store_message,
        ), patch(
            "graph.graph_builder.classification_node",
            new=unexpected_classification,
        ):
            resp = client.post(
                "/api/v1/chat",
                json={
                    "message": "What is the capital of France?",
                    "is_incognito": True,
                    "domain": "student",
                    "chat_preference": "general",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["metadata"]["intent"] == "out_of_scope"
        assert data["metadata"]["out_of_scope_reason"] == "general_knowledge"
        assert "SoulGym" in data["response"]

    def test_classify_endpoint_not_on_chat_router(self, client):
        # Classify is on classify_router; this client only mounts chat_router,
        # so /api/v1/classify must always be 404.
        resp = client.post("/api/v1/classify", json={"message": "Hello"})
        assert resp.status_code == 404

    def test_stream_endpoint_exists(self, client):
        """POST /chat/stream should be a valid route (not 404/405)."""
        with patch("api.chat.get_flow", new_callable=AsyncMock) as mock_get_flow:
            mock_flow = AsyncMock()

            async def empty_astream(state_dict):
                return
                yield  # make it an async generator

            mock_flow.astream = empty_astream
            mock_get_flow.return_value = mock_flow

            with patch("graph.streaming.get_compiled_flow", return_value=mock_flow):
                resp = client.post(
                    "/api/v1/chat/stream",
                    json={
                        "message": "Hello",
                        "is_incognito": True,
                        "domain": "student",
                        "chat_preference": "general",
                    },
                )
        # Streaming returns 200 with text/event-stream
        assert resp.status_code == 200

    def test_nonsense_stream_fast_paths_through_graph(self, client):
        async def fake_conv_id_handler(_state):
            return {"conversation_id": "test-conv-123"}

        async def fake_load_user_context(_state):
            return {}

        async def fake_store_message(_state):
            return {}

        def unexpected_classification(_state):
            raise AssertionError("classification_node should not run")

        with patch(
            "graph.graph_builder.conv_id_handler_node",
            new=fake_conv_id_handler,
        ), patch(
            "graph.graph_builder.load_user_context_node",
            new=fake_load_user_context,
        ), patch(
            "graph.graph_builder.store_message_node",
            new=fake_store_message,
        ), patch(
            "graph.graph_builder.classification_node",
            new=unexpected_classification,
        ):
            resp = client.post(
                "/api/v1/chat/stream",
                json={
                    "message": "a s d f g h",
                    "is_incognito": True,
                    "domain": "student",
                    "chat_preference": "general",
                },
            )

        assert resp.status_code == 200
        assert '"type": "complete"' in resp.text
        assert '"out_of_scope_reason": "nonsense"' in resp.text
        assert "SoulGym" in resp.text

    def test_general_knowledge_stream_fast_paths_through_graph(self, client):
        async def fake_conv_id_handler(_state):
            return {"conversation_id": "test-conv-123"}

        async def fake_load_user_context(_state):
            return {}

        async def fake_store_message(_state):
            return {}

        def unexpected_classification(_state):
            raise AssertionError("classification_node should not run")

        with patch(
            "graph.graph_builder.conv_id_handler_node",
            new=fake_conv_id_handler,
        ), patch(
            "graph.graph_builder.load_user_context_node",
            new=fake_load_user_context,
        ), patch(
            "graph.graph_builder.store_message_node",
            new=fake_store_message,
        ), patch(
            "graph.graph_builder.classification_node",
            new=unexpected_classification,
        ):
            resp = client.post(
                "/api/v1/chat/stream",
                json={
                    "message": "What is the capital of France?",
                    "is_incognito": True,
                    "domain": "student",
                    "chat_preference": "general",
                },
            )

        assert resp.status_code == 200
        assert '"type": "complete"' in resp.text
        assert '"out_of_scope_reason": "general_knowledge"' in resp.text
        assert "SoulGym" in resp.text

# ============================================================================
# GET /conversations/{conversation_id}/messages
# ============================================================================

class TestGetConversationMessagesEndpoint:
    """Unit tests for GET /api/v1/chat/conversations/{conversation_id}/messages."""

    VALID_UUID = "550e8400-e29b-41d4-a716-446655440000"

    def test_returns_messages_for_valid_conversation(self, client):
        messages = [
            {"id": "1", "turn_index": 0, "speaker": "user", "message": "Hello", "created_at": "2024-01-01T00:00:00"},
            {"id": "2", "turn_index": 1, "speaker": "bot", "message": "Hi there!", "created_at": "2024-01-01T00:00:01"},
        ]
        with patch(
            "graph.nodes.function_nodes.get_messages.get_conversation_messages",
            new_callable=AsyncMock,
            return_value=messages,
        ):
            resp = client.get(f"/api/v1/chat/conversations/{self.VALID_UUID}/messages")

        assert resp.status_code == 200
        data = resp.json()
        assert data["conversation_id"] == self.VALID_UUID
        assert data["messages"] == messages

    def test_invalid_uuid_returns_400(self, client):
        resp = client.get("/api/v1/chat/conversations/not-a-uuid/messages")
        assert resp.status_code == 400
        assert "Invalid conversation_id" in resp.json()["detail"]

    def test_empty_messages_list(self, client):
        with patch(
            "graph.nodes.function_nodes.get_messages.get_conversation_messages",
            new_callable=AsyncMock,
            return_value=[],
        ):
            resp = client.get(f"/api/v1/chat/conversations/{self.VALID_UUID}/messages")

        assert resp.status_code == 200
        assert resp.json()["messages"] == []

    def test_returns_404_when_conversation_not_owned(self, client):
        """Endpoint must return 404 when the conversation belongs to a different user."""
        with patch(
            "graph.nodes.function_nodes.get_messages.get_conversation_messages",
            new_callable=AsyncMock,
            side_effect=PermissionError("not owned"),
        ):
            resp = client.get(f"/api/v1/chat/conversations/{self.VALID_UUID}/messages")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_requires_auth(self, app, client):
        """Endpoint should return 401 when verify_supabase_token is not overridden."""
        fresh_app = FastAPI()
        fresh_app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
        # Do NOT override verify_supabase_token
        fresh_app.dependency_overrides[optional_supabase_token] = lambda: None
        unauthenticated_client = TestClient(fresh_app, raise_server_exceptions=False)
        resp = unauthenticated_client.get(f"/api/v1/chat/conversations/{self.VALID_UUID}/messages")
        # Without a valid token the dependency raises 401/403
        assert resp.status_code in (401, 403)


# ============================================================================
# GET /conversations/messages
# ============================================================================

class TestGetAllConversationsMessagesEndpoint:
    """Unit tests for GET /api/v1/chat/conversations/messages."""

    def test_returns_all_conversations(self, client):
        conversations = [
            {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "mode": "cognito",
                "started_at": "2024-01-01T00:00:00",
                "ended_at": None,
                "messages": [
                    {"id": "1", "turn_index": 0, "speaker": "user", "message": "Hello", "created_at": "2024-01-01T00:00:00"},
                ],
            }
        ]
        with patch(
            "graph.nodes.function_nodes.get_messages.get_all_user_conversations",
            new_callable=AsyncMock,
            return_value=conversations,
        ):
            resp = client.get("/api/v1/chat/conversations/messages")

        assert resp.status_code == 200
        data = resp.json()
        assert "conversations" in data
        assert data["conversations"] == conversations

    def test_returns_empty_list_when_no_conversations(self, client):
        with patch(
            "graph.nodes.function_nodes.get_messages.get_all_user_conversations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            resp = client.get("/api/v1/chat/conversations/messages")

        assert resp.status_code == 200
        assert resp.json()["conversations"] == []

    def test_uses_authenticated_user_id(self, client):
        """Verify that the endpoint passes the authenticated user's id to the service."""
        captured = {}

        async def capture_uid(supabase_uid):
            captured["uid"] = supabase_uid
            return []

        with patch(
            "graph.nodes.function_nodes.get_messages.get_all_user_conversations",
            side_effect=capture_uid,
        ):
            client.get("/api/v1/chat/conversations/messages")

        assert captured.get("uid") == FAKE_USER["id"]
