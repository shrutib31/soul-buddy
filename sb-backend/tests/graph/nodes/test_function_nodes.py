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
# KEY MANAGER MOCK HELPERS
# ============================================================================

def _make_km(encryption_enabled: bool = False):
    """Return a MagicMock KeyManager that passes through plaintext."""
    km = MagicMock()
    km.is_encryption_enabled.return_value = encryption_enabled
    km.encrypt = AsyncMock(side_effect=lambda conv_id, text: f"ENC:v1:{text}" if encryption_enabled else text)
    km.decrypt = AsyncMock(side_effect=lambda conv_id, text: text.replace("ENC:v1:", "") if text.startswith("ENC:v1:") else text)
    return km


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
        chat_preference="general",
    )


@pytest.fixture
def state_with_conv_id():
    return ConversationState(
        conversation_id="existing-conv-uuid",
        mode="incognito",
        domain="general",
        user_message="Hello",
        chat_preference="general",
    )


@pytest.fixture
def state_for_store_message():
    return ConversationState(
        conversation_id="conv-123",
        mode="incognito",
        domain="general",
        user_message="I need to talk",
        chat_preference="general",
    )


@pytest.fixture
def state_for_store_bot_response():
    return ConversationState(
        conversation_id="conv-123",
        mode="incognito",
        domain="general",
        user_message="Hi",
        response_draft="I'm here for you.",
        chat_preference="general",
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
        chat_preference="general",
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
            chat_preference="general",
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
            chat_preference="general",
        )
        result = await store_message_node(state)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_success_encryption_disabled(self, state_for_store_message, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result
        km = _make_km(encryption_enabled=False)
        with patch("graph.nodes.function_nodes.store_message.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_message.get_key_manager", return_value=km):
            mock_db.get_session.return_value = mock_session
            result = await store_message_node(state_for_store_message)
        assert "error" not in result
        assert result == {}
        # encrypt should NOT have been called when encryption is disabled
        km.encrypt.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_encryption_enabled_encrypts_message(self, state_for_store_message, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result
        km = _make_km(encryption_enabled=True)
        with patch("graph.nodes.function_nodes.store_message.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_message.get_key_manager", return_value=km):
            mock_db.get_session.return_value = mock_session
            result = await store_message_node(state_for_store_message)
        assert "error" not in result
        km.encrypt.assert_awaited_once_with(
            state_for_store_message.conversation_id,
            state_for_store_message.user_message,
        )


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
            chat_preference="general",
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
            chat_preference="general",
        )
        result = await store_bot_response_node(state)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_success_encryption_disabled(self, state_for_store_bot_response, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        km = _make_km(encryption_enabled=False)
        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.get_key_manager", return_value=km), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache:
            mock_cache.invalidate_conversation_history = AsyncMock()
            mock_db.get_session.return_value = mock_session
            result = await store_bot_response_node(state_for_store_bot_response)
        assert "error" not in result
        assert result == {}
        km.encrypt.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_encryption_enabled_encrypts_response(self, state_for_store_bot_response, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        km = _make_km(encryption_enabled=True)
        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.get_key_manager", return_value=km), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache:
            mock_cache.invalidate_conversation_history = AsyncMock()
            mock_db.get_session.return_value = mock_session
            result = await store_bot_response_node(state_for_store_bot_response)
        assert "error" not in result
        # The code encrypts the main message and, depending on the classifier, possibly romanised/canonical/mixed
        # For the default fixture, response_draft is English, so only main and canonical are encrypted
        expected_calls = [
            ((state_for_store_bot_response.conversation_id, state_for_store_bot_response.response_draft),),
            ((state_for_store_bot_response.conversation_id, state_for_store_bot_response.response_draft),)
        ]
        actual_calls = km.encrypt.await_args_list
        assert len(actual_calls) == len(expected_calls)
        for call, expected in zip(actual_calls, expected_calls):
            assert call.args == expected[0]


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
            chat_preference="general",
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
            chat_preference="general",
        )
        result = await render_node(state)
        assert "error" in result["api_response"]["metadata"]
        assert result["api_response"].get("error") == "Something went wrong"


# ============================================================================
# get_messages_node
# ============================================================================

class TestGetMessagesNodeUnit:
    """Unit tests for get_messages_node with mocked DB and KeyManager."""

    def _make_turn(self, turn_index, speaker, message):
        turn = MagicMock()
        turn.id = f"id-{turn_index}"
        turn.turn_index = turn_index
        turn.speaker = speaker
        turn.message = message
        turn.created_at = datetime(2026, 3, 8, 10, 0, turn_index)
        return turn

    @pytest.mark.asyncio
    async def test_incognito_mode_returns_empty_history(self):
        from graph.nodes.function_nodes.get_messages import get_messages_node
        state = ConversationState(
            conversation_id="conv-123",
            mode="incognito",
            domain="general",
            user_message="Hi",
            chat_preference="general",
        )
        result = await get_messages_node(state)
        assert result == {"conversation_history": []}

    @pytest.mark.asyncio
    async def test_missing_conversation_id_returns_error(self):
        from graph.nodes.function_nodes.get_messages import get_messages_node
        state = ConversationState(
            conversation_id="",
            mode="cognito",
            domain="general",
            user_message="Hi",
            chat_preference="general",
        )
        result = await get_messages_node(state)
        assert "error" in result
        assert result["conversation_history"] == []

    @pytest.mark.asyncio
    async def test_decrypts_messages_when_encryption_enabled(self, mock_session):
        from graph.nodes.function_nodes.get_messages import get_messages_node
        turns = [
            self._make_turn(0, "user", "ENC:v1:Hello"),
            self._make_turn(1, "bot", "ENC:v1:Hi there"),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = turns
        mock_session.execute.return_value = mock_result
        km = _make_km(encryption_enabled=True)

        state = ConversationState(
            conversation_id="conv-123",
            mode="cognito",
            domain="general",
            user_message="Hi",
            chat_preference="general",
        )
        with patch("graph.nodes.function_nodes.get_messages._data_db") as mock_db, \
             patch("graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km):
            mock_db.get_session.return_value = mock_session
            result = await get_messages_node(state)

        assert "error" not in result
        msgs = result["conversation_history"]
        assert len(msgs) == 2
        assert msgs[0]["message"] == "Hello"
        assert msgs[1]["message"] == "Hi there"

    @pytest.mark.asyncio
    async def test_plaintext_messages_returned_as_is(self, mock_session):
        from graph.nodes.function_nodes.get_messages import get_messages_node
        turns = [self._make_turn(0, "user", "Hello plain")]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = turns
        mock_session.execute.return_value = mock_result
        km = _make_km(encryption_enabled=False)

        state = ConversationState(
            conversation_id="conv-123",
            mode="cognito",
            domain="general",
            user_message="Hi",
            chat_preference="general",
        )
        with patch("graph.nodes.function_nodes.get_messages._data_db") as mock_db, \
             patch("services.key_manager.get_key_manager", return_value=km):
            mock_db.get_session.return_value = mock_session
            result = await get_messages_node(state)

        assert result["conversation_history"][0]["message"] == "Hello plain"


# ============================================================================
# get_conversation_messages (standalone utility)
# ============================================================================

class TestGetConversationMessagesUnit:
    """Unit tests for the standalone get_conversation_messages utility."""

    def _make_turn(self, turn_index, speaker, message):
        turn = MagicMock()
        turn.id = f"id-{turn_index}"
        turn.turn_index = turn_index
        turn.speaker = speaker
        turn.message = message
        turn.created_at = datetime(2026, 3, 8, 10, 0, turn_index)
        return turn

    @pytest.mark.asyncio
    async def test_returns_decrypted_messages(self, mock_session):
        from graph.nodes.function_nodes.get_messages import get_conversation_messages
        turns = [
            self._make_turn(0, "user", "ENC:v1:Hello"),
            self._make_turn(1, "bot", "ENC:v1:Hi"),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = turns
        mock_session.execute.return_value = mock_result
        km = _make_km(encryption_enabled=True)

        with patch("graph.nodes.function_nodes.get_messages._data_db") as mock_db, \
             patch("graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km):
            mock_db.get_session.return_value = mock_session
            messages = await get_conversation_messages("3fa85f64-5717-4562-b3fc-2c963f66afa6")

        assert len(messages) == 2
        assert messages[0]["speaker"] == "user"
        assert messages[0]["message"] == "Hello"
        assert messages[1]["message"] == "Hi"

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_db_error(self, mock_session):
        from graph.nodes.function_nodes.get_messages import get_conversation_messages
        mock_session.execute.side_effect = Exception("DB down")

        with patch("graph.nodes.function_nodes.get_messages._data_db") as mock_db, \
             patch("services.key_manager.get_key_manager", return_value=_make_km()):
            mock_db.get_session.return_value = mock_session
            messages = await get_conversation_messages("conv-123")

        assert messages == []


# ============================================================================
# get_all_user_conversations
# ============================================================================

class TestGetAllUserConversationsUnit:
    """Unit tests for get_all_user_conversations."""

    def _make_conv(self, conv_id, mode="cognito"):
        import uuid as _uuid
        conv = MagicMock()
        conv.id = _uuid.UUID(conv_id)
        conv.mode = mode
        conv.started_at = datetime(2026, 3, 8, 10, 0, 0)
        conv.ended_at = None
        return conv

    @pytest.mark.asyncio
    async def test_returns_conversations_with_messages(self, mock_session):
        import uuid as _uuid
        from graph.nodes.function_nodes.get_messages import get_all_user_conversations
        conv_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
        convs = [self._make_conv(conv_id)]

        # Fake turn that matches fake_messages
        fake_turn = MagicMock()
        fake_turn.id = "m1"
        fake_turn.session_id = _uuid.UUID(conv_id)
        fake_turn.turn_index = 0
        fake_turn.speaker = "user"
        fake_turn.message = "Hello"
        fake_turn.created_at = None

        # get_all_user_conversations makes two execute() calls in the same session:
        # 1st → conversations, 2nd → turns
        conv_mock_result = MagicMock()
        conv_mock_result.scalars.return_value.all.return_value = convs
        turns_mock_result = MagicMock()
        turns_mock_result.scalars.return_value.all.return_value = [fake_turn]
        mock_session.execute.side_effect = [conv_mock_result, turns_mock_result]

        km = _make_km(encryption_enabled=False)
        fake_messages = [{"id": "m1", "turn_index": 0, "speaker": "user", "message": "Hello", "created_at": None}]
        with patch("graph.nodes.function_nodes.get_messages._data_db") as mock_db, \
             patch("graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km):
            mock_db.get_session.return_value = mock_session
            result = await get_all_user_conversations("3fa85f64-5717-4562-b3fc-2c963f66afa7")

        assert len(result) == 1
        assert result[0]["conversation_id"] == conv_id
        assert result[0]["mode"] == "cognito"
        assert result[0]["messages"] == fake_messages

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self, mock_session):
        from graph.nodes.function_nodes.get_messages import get_all_user_conversations
        mock_session.execute.side_effect = Exception("DB error")

        with patch("graph.nodes.function_nodes.get_messages._data_db") as mock_db:
            mock_db.get_session.return_value = mock_session
            result = await get_all_user_conversations("bad-uid")

        assert result == []

