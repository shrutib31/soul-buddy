"""
Unit tests for the intelligence-layer background tasks in store_bot_response.

Covers:
  - _get_previous_unfinalised_session helper
  - score_session task is created for cognito users
  - summarize_session_incremental task is created every 5 bot turns
  - Finalisation task is created when is_new_session=True and a prior session exists
  - Incognito users skip all background tasks
"""

import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graph.state import ConversationState
from graph.nodes.function_nodes.store_bot_response import (
    _get_previous_unfinalised_session,
    store_bot_response_node,
)


# ============================================================================
# Constants
# ============================================================================

_VALID_CONV_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_PREV_CONV_UUID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
_VALID_USER_UUID = "550e8400-e29b-41d4-a716-446655440000"


# ============================================================================
# Helpers
# ============================================================================

def _cognito_state(**kwargs) -> ConversationState:
    defaults = dict(
        conversation_id=_VALID_CONV_UUID,
        mode="cognito",
        domain="student",
        user_message="I'm overwhelmed",
        response_draft="I hear you.",
        supabase_uid=_VALID_USER_UUID,
        chat_preference="general",
    )
    defaults.update(kwargs)
    return ConversationState(**defaults)


def _incognito_state(**kwargs) -> ConversationState:
    defaults = dict(
        conversation_id=_VALID_CONV_UUID,
        mode="incognito",
        domain="student",
        user_message="Hello",
        response_draft="Hi there.",
        supabase_uid=None,
        chat_preference="general",
    )
    defaults.update(kwargs)
    return ConversationState(**defaults)


def _make_km(encryption_enabled: bool = False):
    km = MagicMock()
    km.is_encryption_enabled.return_value = encryption_enabled
    km.encrypt = AsyncMock(side_effect=lambda cid, text: f"ENC:{text}")
    return km


def _make_session(turn_count: int = 4):
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = turn_count
    session.execute = AsyncMock(return_value=mock_result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _make_task_collector():
    """
    Returns (collector_list, fake_create_task).
    fake_create_task records coroutines and closes them to suppress warnings.
    """
    collected = []

    def fake_create_task(coro):
        collected.append(coro)
        coro.close()  # prevent "coroutine never awaited" warning
        return MagicMock()

    return collected, fake_create_task


# ============================================================================
# _get_previous_unfinalised_session
# ============================================================================

class TestGetPreviousUnfinalisedSession:

    def _make_session(self, scalar_result):
        session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = scalar_result
        session.execute = AsyncMock(return_value=mock_result)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        return session

    @pytest.mark.asyncio
    async def test_returns_session_id_when_found(self):
        prev_uuid = uuid.UUID(_PREV_CONV_UUID)
        session = self._make_session(scalar_result=prev_uuid)

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db:
            mock_db.get_session.return_value = session
            result = await _get_previous_unfinalised_session(_VALID_USER_UUID, _VALID_CONV_UUID)

        assert result == _PREV_CONV_UUID

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        session = self._make_session(scalar_result=None)

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db:
            mock_db.get_session.return_value = session
            result = await _get_previous_unfinalised_session(_VALID_USER_UUID, _VALID_CONV_UUID)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_db_error(self):
        session = MagicMock()
        session.execute = AsyncMock(side_effect=Exception("DB down"))
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db:
            mock_db.get_session.return_value = session
            result = await _get_previous_unfinalised_session(_VALID_USER_UUID, _VALID_CONV_UUID)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_uuid(self):
        result = await _get_previous_unfinalised_session("not-a-uuid", _VALID_CONV_UUID)
        assert result is None


# ============================================================================
# store_bot_response_node — intelligence-layer background tasks
# ============================================================================

class TestStoreBotResponseNodeIntelligenceLayer:

    @pytest.mark.asyncio
    async def test_cognito_creates_score_session_task(self):
        """cognito user → score_session background task is created every turn."""
        state = _cognito_state()
        session = _make_session(turn_count=4)
        tasks, fake_create_task = _make_task_collector()

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.get_key_manager", return_value=_make_km()), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache, \
             patch("graph.nodes.function_nodes.store_bot_response.score_session") as mock_score, \
             patch("graph.nodes.function_nodes.store_bot_response._get_previous_unfinalised_session", new_callable=AsyncMock, return_value=None), \
             patch("asyncio.create_task", side_effect=fake_create_task):
            mock_db.get_session.return_value = session
            mock_cache.invalidate_conversation_history = AsyncMock()
            mock_score.return_value = AsyncMock()()

            result = await store_bot_response_node(state)

        assert "error" not in result
        # At least one task (score_session) should have been queued
        assert len(tasks) >= 1

    @pytest.mark.asyncio
    async def test_incognito_skips_all_background_tasks(self):
        """incognito user (no supabase_uid) → no background tasks created."""
        state = _incognito_state()
        session = _make_session(turn_count=2)
        tasks, fake_create_task = _make_task_collector()

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.get_key_manager", return_value=_make_km()), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache, \
             patch("asyncio.create_task", side_effect=fake_create_task):
            mock_db.get_session.return_value = session
            mock_cache.invalidate_conversation_history = AsyncMock()

            result = await store_bot_response_node(state)

        assert "error" not in result
        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_every_5_bot_turns_triggers_incremental_summary(self):
        """
        turn_count=9 → total_turns=10, bot_turn_number=5, 5%5==0
        → summarize_session_incremental task is created in addition to score_session.
        """
        state = _cognito_state()
        # 9 existing turns → total becomes 10 → bot_turn_number = 5
        session = _make_session(turn_count=9)
        tasks, fake_create_task = _make_task_collector()

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.get_key_manager", return_value=_make_km()), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache, \
             patch("graph.nodes.function_nodes.store_bot_response.score_session") as mock_score, \
             patch("graph.nodes.function_nodes.store_bot_response.summarize_session_incremental") as mock_summary, \
             patch("graph.nodes.function_nodes.store_bot_response._get_previous_unfinalised_session", new_callable=AsyncMock, return_value=None), \
             patch("asyncio.create_task", side_effect=fake_create_task):
            mock_db.get_session.return_value = session
            mock_cache.invalidate_conversation_history = AsyncMock()
            mock_score.return_value = AsyncMock()()
            mock_summary.return_value = AsyncMock()()

            result = await store_bot_response_node(state)

        assert "error" not in result
        # score_session + summarize_session_incremental = 2 tasks
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_non_milestone_turn_only_creates_score_task(self):
        """turn_count=4 → total_turns=5, bot_turn_number=2, not a multiple of 5 → 1 task."""
        state = _cognito_state()
        session = _make_session(turn_count=4)
        tasks, fake_create_task = _make_task_collector()

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.get_key_manager", return_value=_make_km()), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache, \
             patch("graph.nodes.function_nodes.store_bot_response.score_session") as mock_score, \
             patch("graph.nodes.function_nodes.store_bot_response._get_previous_unfinalised_session", new_callable=AsyncMock, return_value=None), \
             patch("asyncio.create_task", side_effect=fake_create_task):
            mock_db.get_session.return_value = session
            mock_cache.invalidate_conversation_history = AsyncMock()
            mock_score.return_value = AsyncMock()()

            result = await store_bot_response_node(state)

        assert "error" not in result
        assert len(tasks) == 1  # only score_session

    @pytest.mark.asyncio
    async def test_is_new_session_triggers_finalisation_task(self):
        """is_new_session=True + prior unfinalised session → finalisation task queued."""
        state = _cognito_state(is_new_session=True)
        session = _make_session(turn_count=2)
        tasks, fake_create_task = _make_task_collector()

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.get_key_manager", return_value=_make_km()), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache, \
             patch("graph.nodes.function_nodes.store_bot_response.score_session") as mock_score, \
             patch("graph.nodes.function_nodes.store_bot_response._get_previous_unfinalised_session", new_callable=AsyncMock, return_value=_PREV_CONV_UUID), \
             patch("asyncio.create_task", side_effect=fake_create_task):
            mock_db.get_session.return_value = session
            mock_cache.invalidate_conversation_history = AsyncMock()
            mock_score.return_value = AsyncMock()()

            result = await store_bot_response_node(state)

        assert "error" not in result
        # score_session + finalisation closure = 2 tasks
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_is_new_session_no_prev_session_skips_finalisation(self):
        """is_new_session=True but no prior unfinalised session → only score_session task."""
        state = _cognito_state(is_new_session=True)
        session = _make_session(turn_count=2)
        tasks, fake_create_task = _make_task_collector()

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.get_key_manager", return_value=_make_km()), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache, \
             patch("graph.nodes.function_nodes.store_bot_response.score_session") as mock_score, \
             patch("graph.nodes.function_nodes.store_bot_response._get_previous_unfinalised_session", new_callable=AsyncMock, return_value=None), \
             patch("asyncio.create_task", side_effect=fake_create_task):
            mock_db.get_session.return_value = session
            mock_cache.invalidate_conversation_history = AsyncMock()
            mock_score.return_value = AsyncMock()()

            result = await store_bot_response_node(state)

        assert "error" not in result
        # Only score_session — no finalisation task since prev_session_id is None
        assert len(tasks) == 1

    @pytest.mark.asyncio
    async def test_missing_response_draft_returns_error(self):
        state = _cognito_state(response_draft="")
        result = await store_bot_response_node(state)
        assert "error" in result
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_conversation_id_returns_error(self):
        state = _cognito_state(conversation_id="")
        result = await store_bot_response_node(state)
        assert "error" in result
        assert "Missing" in result["error"]
