"""
Unit tests for get_messages.py:
  - get_messages_node     (LangGraph node)
  - get_conversation_messages  (standalone utility)
  - get_all_user_conversations (batch utility)

All DB and key-manager calls are mocked.
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graph.state import ConversationState
from graph.nodes.function_nodes.get_messages import (
    get_messages_node,
    get_conversation_messages,
    get_all_user_conversations,
)


# ============================================================================
# Helpers
# ============================================================================

CONV_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
USER_ID = "11111111-2222-3333-4444-555555555555"


def _make_turn(speaker: str, message: str, turn_index: int = 1):
    turn = MagicMock()
    turn.id = uuid.uuid4()
    turn.session_id = uuid.UUID(CONV_ID)
    turn.speaker = speaker
    turn.message = message
    turn.turn_index = turn_index
    turn.created_at = datetime(2024, 1, 1, 12, 0, 0)
    return turn


def _make_conv(conv_id: str = CONV_ID, mode: str = "cognito"):
    conv = MagicMock()
    conv.id = uuid.UUID(conv_id)
    conv.mode = mode
    conv.started_at = datetime(2024, 1, 1, 10, 0, 0)
    conv.ended_at = None
    conv.supabase_user_id = uuid.UUID(USER_ID)
    return conv


def _make_km(plaintext_passthrough: bool = True):
    km = MagicMock()
    km.decrypt = AsyncMock(
        side_effect=lambda conv_id, text: text.replace("ENC:v1:", "") if text.startswith("ENC:v1:") else text
    )
    return km


def _make_db_session(turns=None, conversations=None, ownership_result=None):
    """Build a mock session that returns given turns/conversations."""
    session = AsyncMock()

    async def execute_side_effect(stmt):
        result = MagicMock()
        if turns is not None:
            scalars = MagicMock()
            scalars.all.return_value = turns
            result.scalars.return_value = scalars
        if conversations is not None:
            scalars = MagicMock()
            scalars.all.return_value = conversations
            result.scalars.return_value = scalars
        if ownership_result is not None:
            result.scalar_one_or_none.return_value = ownership_result
        return result

    session.execute = AsyncMock(side_effect=execute_side_effect)
    return session


# ============================================================================
# get_messages_node
# ============================================================================

class TestGetMessagesNode:
    def _make_state(self, mode="cognito", conv_id=CONV_ID):
        return ConversationState(
            conversation_id=conv_id,
            mode=mode,
            domain="general",
            user_message="hello",
            chat_preference="general",
        )

    @pytest.mark.asyncio
    async def test_incognito_returns_empty_history(self):
        state = self._make_state(mode="incognito")
        result = await get_messages_node(state)
        assert result == {"conversation_history": []}

    @pytest.mark.asyncio
    async def test_missing_conv_id_returns_error(self):
        state = self._make_state(conv_id="")
        result = await get_messages_node(state)
        assert result["conversation_history"] == []
        assert "error" in result

    @pytest.mark.asyncio
    async def test_returns_decrypted_history(self):
        turn = _make_turn("user", "ENC:v1:hello world", turn_index=1)
        km = _make_km()

        session = AsyncMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = [turn]
        session.execute = AsyncMock(return_value=exec_result)

        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            state = self._make_state()
            result = await get_messages_node(state)

        assert len(result["conversation_history"]) == 1
        msg = result["conversation_history"][0]
        assert msg["message"] == "hello world"
        assert msg["speaker"] == "user"
        assert msg["turn_index"] == 1

    @pytest.mark.asyncio
    async def test_plaintext_message_returned_as_is(self):
        turn = _make_turn("bot", "I am here to help", turn_index=2)
        km = _make_km()

        session = AsyncMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = [turn]
        session.execute = AsyncMock(return_value=exec_result)

        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            result = await get_messages_node(self._make_state())

        assert result["conversation_history"][0]["message"] == "I am here to help"

    @pytest.mark.asyncio
    async def test_decryption_failure_returns_placeholder(self):
        turn = _make_turn("user", "ENC:v1:corrupted", turn_index=1)
        km = MagicMock()
        km.decrypt = AsyncMock(side_effect=Exception("bad key"))

        session = AsyncMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = [turn]
        session.execute = AsyncMock(return_value=exec_result)

        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            result = await get_messages_node(self._make_state())

        assert result["conversation_history"][0]["message"] == "[Decryption failed]"

    @pytest.mark.asyncio
    async def test_db_exception_returns_error(self):
        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager"
        ):
            mock_db.get_session.side_effect = Exception("db down")
            result = await get_messages_node(self._make_state())

        assert result["conversation_history"] == []
        assert "Error retrieving messages" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_turns_returns_empty_history(self):
        session = AsyncMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=exec_result)

        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=_make_km()
        ):
            mock_db.get_session.return_value = ctx_manager
            result = await get_messages_node(self._make_state())

        assert result["conversation_history"] == []

    @pytest.mark.asyncio
    async def test_created_at_none_is_handled(self):
        turn = _make_turn("user", "hi", turn_index=1)
        turn.created_at = None
        km = _make_km()

        session = AsyncMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = [turn]
        session.execute = AsyncMock(return_value=exec_result)

        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            result = await get_messages_node(self._make_state())

        assert result["conversation_history"][0]["created_at"] is None


# ============================================================================
# get_conversation_messages
# ============================================================================

class TestGetConversationMessages:
    @pytest.mark.asyncio
    async def test_returns_decrypted_messages(self):
        turn = _make_turn("user", "ENC:v1:test message", turn_index=1)
        km = _make_km()

        session = AsyncMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = [turn]
        session.execute = AsyncMock(return_value=exec_result)

        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            messages = await get_conversation_messages(CONV_ID)

        assert len(messages) == 1
        assert messages[0]["message"] == "test message"

    @pytest.mark.asyncio
    async def test_ownership_check_passes(self):
        turn = _make_turn("user", "hi", turn_index=1)
        conv = _make_conv()
        km = _make_km()

        session = AsyncMock()
        call_count = 0

        async def execute_side_effect(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                # Ownership check
                result.scalar_one_or_none.return_value = conv
            else:
                # Turns query
                result.scalars.return_value.all.return_value = [turn]
            return result

        session.execute = AsyncMock(side_effect=execute_side_effect)
        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            messages = await get_conversation_messages(CONV_ID, supabase_uid=USER_ID)

        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_ownership_check_fails_raises_permission_error(self):
        session = AsyncMock()

        async def execute_side_effect(stmt):
            result = MagicMock()
            result.scalar_one_or_none.return_value = None  # not found
            return result

        session.execute = AsyncMock(side_effect=execute_side_effect)
        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=_make_km()
        ):
            mock_db.get_session.return_value = ctx_manager
            with pytest.raises(PermissionError):
                await get_conversation_messages(CONV_ID, supabase_uid=USER_ID)

    @pytest.mark.asyncio
    async def test_decryption_failure_returns_placeholder(self):
        turn = _make_turn("user", "ENC:v1:data", turn_index=1)
        km = MagicMock()
        km.decrypt = AsyncMock(side_effect=Exception("key error"))

        session = AsyncMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = [turn]
        session.execute = AsyncMock(return_value=exec_result)

        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            messages = await get_conversation_messages(CONV_ID)

        assert messages[0]["message"] == "[Decryption failed]"

    @pytest.mark.asyncio
    async def test_db_exception_returns_empty_list(self):
        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager"
        ):
            mock_db.get_session.side_effect = Exception("db error")
            result = await get_conversation_messages(CONV_ID)

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_conversation_returns_empty_list(self):
        session = AsyncMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=exec_result)

        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=_make_km()
        ):
            mock_db.get_session.return_value = ctx_manager
            messages = await get_conversation_messages(CONV_ID)

        assert messages == []


# ============================================================================
# get_all_user_conversations
# ============================================================================

class TestGetAllUserConversations:
    @pytest.mark.asyncio
    async def test_no_conversations_returns_empty(self):
        session = AsyncMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=exec_result)

        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch("graph.nodes.function_nodes.get_messages._data_db") as mock_db:
            mock_db.get_session.return_value = ctx_manager
            result = await get_all_user_conversations(USER_ID)

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_conversations_with_messages(self):
        conv = _make_conv()
        turn = _make_turn("user", "ENC:v1:hello", turn_index=1)
        turn.session_id = conv.id
        km = _make_km()

        session = AsyncMock()
        call_count = 0

        async def execute_side_effect(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalars.return_value.all.return_value = [conv]
            else:
                result.scalars.return_value.all.return_value = [turn]
            return result

        session.execute = AsyncMock(side_effect=execute_side_effect)
        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            result = await get_all_user_conversations(USER_ID)

        assert len(result) == 1
        assert result[0]["conversation_id"] == str(conv.id)
        assert result[0]["mode"] == "cognito"
        assert len(result[0]["messages"]) == 1
        assert result[0]["messages"][0]["message"] == "hello"

    @pytest.mark.asyncio
    async def test_decryption_failure_uses_placeholder(self):
        conv = _make_conv()
        turn = _make_turn("user", "ENC:v1:data", turn_index=1)
        turn.session_id = conv.id

        km = MagicMock()
        km.decrypt = AsyncMock(side_effect=Exception("key error"))

        session = AsyncMock()
        call_count = 0

        async def execute_side_effect(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalars.return_value.all.return_value = [conv]
            else:
                result.scalars.return_value.all.return_value = [turn]
            return result

        session.execute = AsyncMock(side_effect=execute_side_effect)
        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            result = await get_all_user_conversations(USER_ID)

        assert result[0]["messages"][0]["message"] == "[Decryption failed]"

    @pytest.mark.asyncio
    async def test_db_exception_returns_empty_list(self):
        with patch("graph.nodes.function_nodes.get_messages._data_db") as mock_db:
            mock_db.get_session.side_effect = Exception("db error")
            result = await get_all_user_conversations(USER_ID)

        assert result == []

    @pytest.mark.asyncio
    async def test_started_at_none_is_handled(self):
        conv = _make_conv()
        conv.started_at = None
        conv.ended_at = None
        km = _make_km()

        session = AsyncMock()
        call_count = 0

        async def execute_side_effect(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalars.return_value.all.return_value = [conv]
            else:
                result.scalars.return_value.all.return_value = []
            return result

        session.execute = AsyncMock(side_effect=execute_side_effect)
        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            result = await get_all_user_conversations(USER_ID)

        assert result[0]["started_at"] is None
        assert result[0]["ended_at"] is None

    @pytest.mark.asyncio
    async def test_multiple_conversations(self):
        conv1 = _make_conv(conv_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        conv2_id = "ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"
        conv2 = _make_conv(conv_id=conv2_id)

        turn1 = _make_turn("user", "hi", turn_index=1)
        turn1.session_id = conv1.id
        turn2 = _make_turn("bot", "hello", turn_index=1)
        turn2.session_id = conv2.id

        km = _make_km()
        session = AsyncMock()
        call_count = 0

        async def execute_side_effect(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalars.return_value.all.return_value = [conv1, conv2]
            else:
                result.scalars.return_value.all.return_value = [turn1, turn2]
            return result

        session.execute = AsyncMock(side_effect=execute_side_effect)
        ctx_manager = MagicMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=session)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "graph.nodes.function_nodes.get_messages._data_db"
        ) as mock_db, patch(
            "graph.nodes.function_nodes.get_messages.get_key_manager", return_value=km
        ):
            mock_db.get_session.return_value = ctx_manager
            result = await get_all_user_conversations(USER_ID)

        assert len(result) == 2
