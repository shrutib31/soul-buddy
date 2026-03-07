"""
Unit tests for function nodes: conv_id_handler, store_message, store_bot_response, render.

DB-dependent nodes use mocked SQLAlchemy session/engine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from graph.state import ConversationState
from graph.nodes.function_nodes.conv_id_handler import conv_id_handler_node
from graph.nodes.function_nodes.store_message import store_message_node
from graph.nodes.function_nodes.store_bot_response import store_bot_response_node
from graph.nodes.function_nodes.render import render_node


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def state_no_conv_id():
    return ConversationState(
        conversation_id="",
        mode="incognito",
        domain="general",
        user_message="Hello",
    )


@pytest.fixture
def state_with_conv_id():
    return ConversationState(
        conversation_id="existing-conv-uuid",
        mode="incognito",
        domain="general",
        user_message="Hello",
    )


@pytest.fixture
def state_for_store_message():
    return ConversationState(
        conversation_id="conv-123",
        mode="incognito",
        domain="general",
        user_message="I need to talk",
    )


@pytest.fixture
def state_for_store_bot_response():
    return ConversationState(
        conversation_id="conv-123",
        mode="incognito",
        domain="general",
        user_message="Hi",
        response_draft="I'm here for you.",
    )


@pytest.fixture
def state_for_render():
    return ConversationState(
        conversation_id="conv-456",
        mode="incognito",
        domain="general",
        user_message="Hi",
        intent="greeting",
        situation="NO_SITUATION",
        severity="low",
        risk_level="low",
        response_draft="Hello! How can I support you today?",
    )


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


# ============================================================================
# conv_id_handler_node
# ============================================================================

class TestConvIdHandlerNodeUnit:
    """Unit tests for conv_id_handler_node with mocked DB."""

    @pytest.mark.asyncio
    async def test_empty_conv_id_creates_new(self, state_no_conv_id, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        with patch(
            "graph.nodes.function_nodes.conv_id_handler.data_db"
        ) as mock_db:
            mock_db.get_session.return_value = mock_session
            result = await conv_id_handler_node(state_no_conv_id)
        assert "error" not in result
        assert "conversation_id" in result
        assert len(result["conversation_id"]) == 36  # UUID string length

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_error(self, state_with_conv_id, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        state_with_conv_id.mode = "invalid"
        with patch(
            "graph.nodes.function_nodes.conv_id_handler.data_db"
        ) as mock_db:
            mock_db.get_session.return_value = mock_session
            result = await conv_id_handler_node(state_with_conv_id)
        assert "error" in result
        assert "Invalid mode" in result["error"]

    @pytest.mark.asyncio
    async def test_db_error_returns_error(self, state_no_conv_id, mock_session):
        from sqlalchemy.exc import SQLAlchemyError
        with patch(
            "graph.nodes.function_nodes.conv_id_handler.data_db"
        ) as mock_db:
            mock_db.get_session.return_value = mock_session
            mock_session.commit.side_effect = SQLAlchemyError("Connection failed")
            result = await conv_id_handler_node(state_no_conv_id)
        assert "error" in result
        assert "Database error" in result["error"] or "Error" in result["error"]


# ============================================================================
# store_message_node
# ============================================================================

class TestStoreMessageNodeUnit:
    """Unit tests for store_message_node with mocked DB."""

    @pytest.mark.asyncio
    async def test_missing_conv_id_returns_error(self):
        state = ConversationState(
            conversation_id="",
            mode="incognito",
            domain="general",
            user_message="Hi",
        )
        result = await store_message_node(state)
        assert "error" in result
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_user_message_returns_error(self):
        state = ConversationState(
            conversation_id="conv-123",
            mode="incognito",
            domain="general",
            user_message="",
        )
        result = await store_message_node(state)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_success_returns_empty_dict(self, state_for_store_message, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result
        with patch(
            "graph.nodes.function_nodes.store_message.data_db"
        ) as mock_db:
            mock_db.get_session.return_value = mock_session
            result = await store_message_node(state_for_store_message)
        assert "error" not in result
        assert result == {}


# ============================================================================
# store_bot_response_node
# ============================================================================

class TestStoreBotResponseNodeUnit:
    """Unit tests for store_bot_response_node with mocked DB."""

    @pytest.mark.asyncio
    async def test_missing_conv_id_returns_error(self):
        state = ConversationState(
            conversation_id="",
            mode="incognito",
            domain="general",
            user_message="Hi",
            response_draft="Reply",
        )
        result = await store_bot_response_node(state)
        assert "error" in result
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_response_draft_returns_error(self):
        state = ConversationState(
            conversation_id="conv-123",
            mode="incognito",
            domain="general",
            user_message="Hi",
            response_draft="",
        )
        result = await store_bot_response_node(state)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_success_returns_empty_dict(self, state_for_store_bot_response, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        with patch(
            "graph.nodes.function_nodes.store_bot_response.data_db"
        ) as mock_db:
            mock_db.get_session.return_value = mock_session
            result = await store_bot_response_node(state_for_store_bot_response)
        assert "error" not in result
        assert result == {}


# ============================================================================
# render_node
# ============================================================================

class TestRenderNodeUnit:
    """Unit tests for render_node (no DB)."""

    @pytest.mark.asyncio
    async def test_success_returns_api_response(self, state_for_render):
        result = await render_node(state_for_render)
        assert "api_response" in result
        ar = result["api_response"]
        assert ar["success"] is True
        assert ar["conversation_id"] == "conv-456"
        assert ar["response"] == "Hello! How can I support you today?"
        assert "metadata" in ar
        assert ar["metadata"]["intent"] == "greeting"
        assert "timestamp" in ar["metadata"]

    @pytest.mark.asyncio
    async def test_empty_response_draft_sets_success_false(self):
        state = ConversationState(
            conversation_id="conv-456",
            mode="incognito",
            domain="general",
            user_message="Hi",
            response_draft="",
        )
        result = await render_node(state)
        assert result["api_response"]["success"] is False
        assert result["api_response"]["response"] == ""

    @pytest.mark.asyncio
    async def test_error_in_state_included_in_metadata(self):
        state = ConversationState(
            conversation_id="conv-456",
            mode="incognito",
            domain="general",
            user_message="Hi",
            response_draft="",
            error="Something went wrong",
        )
        result = await render_node(state)
        assert "error" in result["api_response"]["metadata"]
        assert result["api_response"].get("error") == "Something went wrong"


