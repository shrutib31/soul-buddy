"""
Unit tests for _build_summary and _upsert_conversation_summary in store_bot_response.

These cover the summary-building logic and upsert path that are not tested
in the existing test_function_nodes.py tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graph.state import ConversationState
from graph.nodes.function_nodes.store_bot_response import (
    _build_summary,
    _upsert_conversation_summary,
    store_bot_response_node,
)


# ============================================================================
# Helpers
# ============================================================================

_VALID_CONV_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_VALID_USER_UUID = "550e8400-e29b-41d4-a716-446655440000"


def _state(**kwargs) -> ConversationState:
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


def _make_mock_session():
    session = MagicMock()
    session.add = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


# ============================================================================
# _build_summary
# ============================================================================

class TestBuildSummary:

    def test_all_fields_present(self):
        state = _state(
            situation="EXAM_ANXIETY",
            severity="medium",
            intent="VENTING",
            risk_level="low",
            is_crisis_detected=False,
            flow_id="FLOW_GENERAL_OVERWHELM",
        )
        summary = _build_summary(state, turn_count=6)

        assert "Domain: student." in summary
        assert "Situation: EXAM_ANXIETY (medium severity)." in summary
        assert "Intent: VENTING." in summary
        assert "Risk: low." in summary
        assert "Flow: FLOW_GENERAL_OVERWHELM." in summary
        assert "Turns: 6." in summary
        assert "Crisis detected." not in summary

    def test_missing_optional_fields_omitted(self):
        state = _state(
            situation=None,
            severity=None,
            intent=None,
            risk_level="",
            is_crisis_detected=False,
            flow_id=None,
        )
        summary = _build_summary(state, turn_count=1)

        assert "Domain: student." in summary
        assert "Situation" not in summary
        assert "Intent" not in summary
        assert "Risk" not in summary
        assert "Flow" not in summary
        assert "Turns: 1." in summary

    def test_crisis_detected_included(self):
        state = _state(is_crisis_detected=True)
        summary = _build_summary(state, turn_count=2)
        assert "Crisis detected." in summary

    def test_situation_without_severity(self):
        state = _state(situation="WORK_STRESS", severity=None)
        summary = _build_summary(state, turn_count=3)
        assert "Situation: WORK_STRESS." in summary
        assert "severity" not in summary

    def test_date_prefix_format(self):
        """Summary must start with a [YYYY-MM-DD] date prefix."""
        import re
        state = _state()
        summary = _build_summary(state, turn_count=1)
        assert re.match(r"^\[\d{4}-\d{2}-\d{2}\]", summary)

    def test_turn_count_zero(self):
        state = _state()
        summary = _build_summary(state, turn_count=0)
        assert "Turns: 0." in summary

    def test_domain_included(self):
        state = _state(domain="employee")
        summary = _build_summary(state, turn_count=4)
        assert "Domain: employee." in summary


# ============================================================================
# _upsert_conversation_summary
# ============================================================================

class TestUpsertConversationSummary:

    @pytest.mark.asyncio
    async def test_success_upserts_db_and_refreshes_cache(self):
        state = _state()
        session = _make_mock_session()

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache, \
             patch("graph.nodes.function_nodes.store_bot_response.pg_insert") as mock_pg_insert:
            mock_db.get_session.return_value = session
            mock_cache.set_conversation_summary = AsyncMock()
            # pg_insert(...).values(...).on_conflict_do_update(...) returns a stmt object
            mock_stmt = MagicMock()
            mock_pg_insert.return_value.values.return_value.on_conflict_do_update.return_value = mock_stmt

            await _upsert_conversation_summary(state, turn_count=3)

        session.execute.assert_awaited_once_with(mock_stmt)
        session.commit.assert_awaited_once()
        mock_cache.set_conversation_summary.assert_awaited_once()
        # Verify the cached summary text contains key fields
        cached_text = mock_cache.set_conversation_summary.call_args[0][1]
        assert "Domain: student." in cached_text
        assert "Turns: 3." in cached_text

    @pytest.mark.asyncio
    async def test_db_exception_fails_silently(self):
        """Summary upsert failure must not propagate — it's non-fatal."""
        state = _state()
        session = _make_mock_session()
        session.execute.side_effect = Exception("DB error")

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service"):
            mock_db.get_session.return_value = session
            # Must not raise
            await _upsert_conversation_summary(state, turn_count=2)

    @pytest.mark.asyncio
    async def test_invalid_supabase_uid_fails_silently(self):
        """Non-UUID supabase_uid must not crash the caller."""
        state = _state(supabase_uid="not-a-uuid")

        with patch("graph.nodes.function_nodes.store_bot_response.data_db"), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service"):
            await _upsert_conversation_summary(state, turn_count=1)

    @pytest.mark.asyncio
    async def test_invalid_conversation_id_fails_silently(self):
        """Non-UUID conversation_id must not crash the caller."""
        state = _state(conversation_id="not-a-uuid")

        with patch("graph.nodes.function_nodes.store_bot_response.data_db"), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service"):
            await _upsert_conversation_summary(state, turn_count=1)


# ============================================================================
# store_bot_response_node — cognito summary path
# ============================================================================

class TestStoreBotResponseNodeSummaryPath:

    def _make_km(self):
        km = MagicMock()
        km.is_encryption_enabled.return_value = False
        km.encrypt = AsyncMock(side_effect=lambda cid, text: text)
        return km

    @pytest.mark.asyncio
    async def test_cognito_user_triggers_upsert_summary(self):
        """supabase_uid present → _upsert_conversation_summary must be called."""
        state = _state()
        session = _make_mock_session()
        turn_count_result = MagicMock()
        turn_count_result.scalar.return_value = 4
        session.execute.return_value = turn_count_result

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.get_key_manager", return_value=self._make_km()), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache, \
             patch("graph.nodes.function_nodes.store_bot_response._upsert_conversation_summary") as mock_upsert:
            mock_db.get_session.return_value = session
            mock_cache.invalidate_conversation_history = AsyncMock()
            mock_upsert.return_value = None  # make it a regular coroutine
            mock_upsert.return_value = AsyncMock()()

            # Patch _upsert properly as async
            async def fake_upsert(s, tc):
                pass
            mock_upsert.side_effect = fake_upsert

            result = await store_bot_response_node(state)

        mock_upsert.assert_called_once()
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_incognito_user_skips_upsert_summary(self):
        """No supabase_uid → summary upsert must NOT be called."""
        state = _state(mode="incognito", supabase_uid=None)
        session = _make_mock_session()
        turn_count_result = MagicMock()
        turn_count_result.scalar.return_value = 2
        session.execute.return_value = turn_count_result

        with patch("graph.nodes.function_nodes.store_bot_response.data_db") as mock_db, \
             patch("graph.nodes.function_nodes.store_bot_response.get_key_manager", return_value=self._make_km()), \
             patch("graph.nodes.function_nodes.store_bot_response.cache_service") as mock_cache, \
             patch("graph.nodes.function_nodes.store_bot_response._upsert_conversation_summary") as mock_upsert:
            mock_db.get_session.return_value = session
            mock_cache.invalidate_conversation_history = AsyncMock()

            result = await store_bot_response_node(state)

        mock_upsert.assert_not_called()
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_missing_response_draft_returns_error(self):
        state = _state(response_draft="")

        result = await store_bot_response_node(state)

        assert "error" in result
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_conversation_id_returns_error(self):
        state = _state(conversation_id="")

        result = await store_bot_response_node(state)

        assert "error" in result
