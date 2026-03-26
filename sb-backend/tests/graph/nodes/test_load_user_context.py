"""
Unit tests for load_user_context_node and its DB fetch helpers.

All DB and cache interactions are fully mocked — no real DB or Redis required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graph.state import ConversationState
from graph.nodes.function_nodes.load_user_context import (
    load_user_context_node,
    _fetch_personality_profile_from_db,
    _fetch_user_profile_from_db,
    _fetch_conversation_summary_from_db,
    _fetch_conversation_history_from_db,
    _fetch_domain_config_from_db,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_state(**kwargs) -> ConversationState:
    defaults = dict(
        conversation_id="conv-abc",
        mode="cognito",
        domain="student",
        user_message="I'm struggling with exams",
        supabase_uid="user-uid-123",
        chat_preference="general",
    )
    defaults.update(kwargs)
    return ConversationState(**defaults)


def _make_mock_session():
    session = MagicMock()
    session.execute = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _scalar_result(value):
    """Return a mock execute result that returns `value` via scalar_one_or_none."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _scalars_all_result(rows):
    """Return a mock execute result that returns `rows` via scalars().all()."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    return result


def _one_or_none_result(row):
    """Return a mock execute result that returns `row` via one_or_none()."""
    result = MagicMock()
    result.one_or_none.return_value = row
    return result


# ============================================================================
# load_user_context_node — mode routing
# ============================================================================

class TestLoadUserContextNodeModeRouting:

    @pytest.mark.asyncio
    async def test_incognito_mode_skips_cognito_sections(self):
        """Non-cognito mode must not call any personality/profile/summary fetches."""
        state = _make_state(mode="incognito", supabase_uid=None)

        with patch("graph.nodes.function_nodes.load_user_context.cache_service") as mock_cache:
            mock_cache.get_conversation_history = AsyncMock(return_value=None)
            mock_cache.set_conversation_history = AsyncMock()
            mock_cache.get_config = AsyncMock(return_value=None)
            mock_cache.set_config = AsyncMock()

            with patch(
                "graph.nodes.function_nodes.load_user_context._fetch_conversation_history_from_db",
                return_value=None,
            ) as mock_hist:
                result = await load_user_context_node(state)

            mock_cache.get_personality_profile.assert_not_called() if hasattr(
                mock_cache, "get_personality_profile"
            ) else None
            # The main assertion: result has no personality/profile/summary keys
            assert "user_personality_profile" not in result
            assert "user_preferences" not in result
            assert "conversation_summary" not in result

    @pytest.mark.asyncio
    async def test_cognito_mode_with_no_supabase_uid_skips_cognito_sections(self):
        """mode=cognito but supabase_uid=None — is_cognito is False, skip user-specific sections."""
        state = _make_state(mode="cognito", supabase_uid=None)

        with patch("graph.nodes.function_nodes.load_user_context.cache_service") as mock_cache:
            mock_cache.get_conversation_history = AsyncMock(return_value=None)
            mock_cache.set_conversation_history = AsyncMock()
            mock_cache.get_config = AsyncMock(return_value=None)
            mock_cache.set_config = AsyncMock()

            with patch(
                "graph.nodes.function_nodes.load_user_context._fetch_conversation_history_from_db",
                return_value=None,
            ):
                result = await load_user_context_node(state)

        assert "user_personality_profile" not in result
        assert "user_preferences" not in result
        assert "conversation_summary" not in result


# ============================================================================
# load_user_context_node — cache hit paths
# ============================================================================

class TestLoadUserContextNodeCacheHits:

    @pytest.mark.asyncio
    async def test_all_cache_hits_returns_all_fields_no_db_calls(self):
        """When every cache_service.get_* returns a value, no DB helpers are invoked."""
        state = _make_state()

        cached_personality = {"openness": 0.8}
        cached_profile = {"full_name": "Alice"}
        cached_summary = "Prior context summary."
        cached_history = [{"speaker": "user", "message": "hi", "turn_index": 0}]
        cached_ui = {"page": "dashboard"}

        with patch("graph.nodes.function_nodes.load_user_context.cache_service") as cs, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_personality_profile_from_db") as fp, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_user_profile_from_db") as fup, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_summary_from_db") as fs, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_history_from_db") as fh, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_domain_config_from_db") as fd:

            cs.get_personality_profile = AsyncMock(return_value=cached_personality)
            cs.set_personality_profile = AsyncMock()
            cs.get_user_profile = AsyncMock(return_value=cached_profile)
            cs.set_user_profile = AsyncMock()
            cs.get_conversation_summary = AsyncMock(return_value=cached_summary)
            cs.set_conversation_summary = AsyncMock()
            cs.get_conversation_history = AsyncMock(return_value=cached_history)
            cs.set_conversation_history = AsyncMock()
            cs.get_ui_state = AsyncMock(return_value=cached_ui)
            cs.set_ui_state = AsyncMock()
            cs.get_config = AsyncMock(return_value=None)
            cs.set_config = AsyncMock()
            fd.return_value = None

            result = await load_user_context_node(state)

        fp.assert_not_called()
        fup.assert_not_called()
        fs.assert_not_called()
        fh.assert_not_called()

        assert result["user_personality_profile"] == cached_personality
        assert result["user_preferences"] == cached_profile
        assert result["conversation_summary"] == cached_summary
        assert result["conversation_history"] == cached_history


# ============================================================================
# load_user_context_node — cache miss → DB fallback
# ============================================================================

class TestLoadUserContextNodeCacheMissDB:

    @pytest.mark.asyncio
    async def test_cache_miss_personality_calls_db_and_caches(self):
        state = _make_state()
        db_profile = {"openness": 0.9}

        with patch("graph.nodes.function_nodes.load_user_context.cache_service") as cs, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_personality_profile_from_db",
                   return_value=db_profile) as fp, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_user_profile_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_summary_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_history_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_domain_config_from_db", return_value=None):

            cs.get_personality_profile = AsyncMock(return_value=None)
            cs.set_personality_profile = AsyncMock()
            cs.get_user_profile = AsyncMock(return_value=None)
            cs.set_user_profile = AsyncMock()
            cs.get_conversation_summary = AsyncMock(return_value=None)
            cs.set_conversation_summary = AsyncMock()
            cs.get_conversation_history = AsyncMock(return_value=None)
            cs.set_conversation_history = AsyncMock()
            cs.get_ui_state = AsyncMock(return_value=None)
            cs.set_ui_state = AsyncMock()
            cs.get_config = AsyncMock(return_value=None)
            cs.set_config = AsyncMock()

            result = await load_user_context_node(state)

        fp.assert_called_once_with(state.supabase_uid)
        cs.set_personality_profile.assert_awaited_once_with(state.supabase_uid, db_profile)
        assert result["user_personality_profile"] == db_profile

    @pytest.mark.asyncio
    async def test_cache_miss_history_calls_db_and_caches(self):
        state = _make_state()
        db_history = [{"speaker": "user", "message": "help", "turn_index": 0}]

        with patch("graph.nodes.function_nodes.load_user_context.cache_service") as cs, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_personality_profile_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_user_profile_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_summary_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_history_from_db",
                   return_value=db_history) as fh, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_domain_config_from_db", return_value=None):

            cs.get_personality_profile = AsyncMock(return_value=None)
            cs.set_personality_profile = AsyncMock()
            cs.get_user_profile = AsyncMock(return_value=None)
            cs.set_user_profile = AsyncMock()
            cs.get_conversation_summary = AsyncMock(return_value=None)
            cs.set_conversation_summary = AsyncMock()
            cs.get_conversation_history = AsyncMock(return_value=None)
            cs.set_conversation_history = AsyncMock()
            cs.get_ui_state = AsyncMock(return_value=None)
            cs.set_ui_state = AsyncMock()
            cs.get_config = AsyncMock(return_value=None)
            cs.set_config = AsyncMock()

            result = await load_user_context_node(state)

        fh.assert_called_once_with(state.conversation_id)
        cs.set_conversation_history.assert_awaited_once_with(state.conversation_id, db_history)
        assert result["conversation_history"] == db_history

    @pytest.mark.asyncio
    async def test_no_conversation_id_skips_history(self):
        state = _make_state(conversation_id="")

        with patch("graph.nodes.function_nodes.load_user_context.cache_service") as cs, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_personality_profile_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_user_profile_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_summary_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_history_from_db") as fh, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_domain_config_from_db", return_value=None):

            cs.get_personality_profile = AsyncMock(return_value=None)
            cs.set_personality_profile = AsyncMock()
            cs.get_user_profile = AsyncMock(return_value=None)
            cs.set_user_profile = AsyncMock()
            cs.get_conversation_summary = AsyncMock(return_value=None)
            cs.set_conversation_summary = AsyncMock()
            cs.get_ui_state = AsyncMock(return_value=None)
            cs.set_ui_state = AsyncMock()
            cs.get_config = AsyncMock(return_value=None)
            cs.set_config = AsyncMock()

            await load_user_context_node(state)

        fh.assert_not_called()


# ============================================================================
# load_user_context_node — UI state logic
# ============================================================================

class TestLoadUserContextNodeUIState:

    @pytest.mark.asyncio
    async def test_page_context_in_request_persists_to_cache(self):
        """When state has page_context, it should be written to cache."""
        page = {"page": "profile"}
        state = _make_state(page_context=page)

        with patch("graph.nodes.function_nodes.load_user_context.cache_service") as cs, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_personality_profile_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_user_profile_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_summary_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_history_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_domain_config_from_db", return_value=None):

            cs.get_personality_profile = AsyncMock(return_value=None)
            cs.set_personality_profile = AsyncMock()
            cs.get_user_profile = AsyncMock(return_value=None)
            cs.set_user_profile = AsyncMock()
            cs.get_conversation_summary = AsyncMock(return_value=None)
            cs.set_conversation_summary = AsyncMock()
            cs.get_conversation_history = AsyncMock(return_value=None)
            cs.set_conversation_history = AsyncMock()
            cs.get_ui_state = AsyncMock(return_value=None)
            cs.set_ui_state = AsyncMock()
            cs.get_config = AsyncMock(return_value=None)
            cs.set_config = AsyncMock()

            result = await load_user_context_node(state)

        cs.set_ui_state.assert_awaited_once_with(state.supabase_uid, page)
        # page_context was in request, not restored from cache — should not be overwritten
        assert "page_context" not in result

    @pytest.mark.asyncio
    async def test_no_page_context_restores_from_cache(self):
        """No page_context in request and cached ui_state → restored into state."""
        state = _make_state(page_context={})
        cached_ui = {"page": "dashboard"}

        with patch("graph.nodes.function_nodes.load_user_context.cache_service") as cs, \
             patch("graph.nodes.function_nodes.load_user_context._fetch_personality_profile_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_user_profile_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_summary_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_conversation_history_from_db", return_value=None), \
             patch("graph.nodes.function_nodes.load_user_context._fetch_domain_config_from_db", return_value=None):

            cs.get_personality_profile = AsyncMock(return_value=None)
            cs.set_personality_profile = AsyncMock()
            cs.get_user_profile = AsyncMock(return_value=None)
            cs.set_user_profile = AsyncMock()
            cs.get_conversation_summary = AsyncMock(return_value=None)
            cs.set_conversation_summary = AsyncMock()
            cs.get_conversation_history = AsyncMock(return_value=None)
            cs.set_conversation_history = AsyncMock()
            cs.get_ui_state = AsyncMock(return_value=cached_ui)
            cs.set_ui_state = AsyncMock()
            cs.get_config = AsyncMock(return_value=None)
            cs.set_config = AsyncMock()

            result = await load_user_context_node(state)

        assert result.get("page_context") == cached_ui


# ============================================================================
# _fetch_personality_profile_from_db
# ============================================================================

class TestFetchPersonalityProfileFromDb:

    @pytest.mark.asyncio
    async def test_row_exists_returns_profile_data(self):
        session = _make_mock_session()
        row = MagicMock()
        row.personality_profile_data = {"openness": 0.7}
        session.execute.return_value = _scalar_result(row)

        with patch("graph.nodes.function_nodes.load_user_context._auth_db") as db:
            db.get_session.return_value = session
            result = await _fetch_personality_profile_from_db("user-uid-1")

        assert result == {"openness": 0.7}

    @pytest.mark.asyncio
    async def test_no_row_returns_none(self):
        session = _make_mock_session()
        session.execute.return_value = _scalar_result(None)

        with patch("graph.nodes.function_nodes.load_user_context._auth_db") as db:
            db.get_session.return_value = session
            result = await _fetch_personality_profile_from_db("user-uid-1")

        assert result is None

    @pytest.mark.asyncio
    async def test_db_exception_returns_none(self):
        session = _make_mock_session()
        session.execute.side_effect = Exception("DB down")

        with patch("graph.nodes.function_nodes.load_user_context._auth_db") as db:
            db.get_session.return_value = session
            result = await _fetch_personality_profile_from_db("user-uid-1")

        assert result is None


# ============================================================================
# _fetch_user_profile_from_db
# ============================================================================

class TestFetchUserProfileFromDb:

    @pytest.mark.asyncio
    async def test_user_with_detail_returns_full_profile(self):
        session = _make_mock_session()

        auth_user = MagicMock()
        auth_user.full_name = "Alice"
        auth_user.email = "alice@example.com"
        auth_user.role = "student"

        detail = MagicMock()
        detail.first_name = "Alice"
        detail.last_name = "Smith"
        detail.age = 22
        detail.age_group = "young_adult"
        detail.gender = "female"
        detail.pronouns = "she/her"
        detail.country = "US"
        detail.timezone = "UTC"
        detail.languages = ["en"]
        detail.communication_language = "en"
        detail.education_level = "undergraduate"
        detail.occupation = "student"
        detail.marital_status = "single"
        detail.hobbies = ["reading"]
        detail.interests = ["tech"]
        detail.mental_health_history = {}
        detail.physical_health_history = {}

        session.execute.return_value = _one_or_none_result((auth_user, detail))

        with patch("graph.nodes.function_nodes.load_user_context._auth_db") as db:
            db.get_session.return_value = session
            result = await _fetch_user_profile_from_db("user-uid-1")

        assert result["full_name"] == "Alice"
        assert result["age"] == 22
        assert result["email"] == "alice@example.com"
        assert result["languages"] == ["en"]

    @pytest.mark.asyncio
    async def test_user_without_detail_returns_base_profile(self):
        session = _make_mock_session()

        auth_user = MagicMock()
        auth_user.full_name = "Bob"
        auth_user.email = "bob@example.com"
        auth_user.role = "employee"

        session.execute.return_value = _one_or_none_result((auth_user, None))

        with patch("graph.nodes.function_nodes.load_user_context._auth_db") as db:
            db.get_session.return_value = session
            result = await _fetch_user_profile_from_db("user-uid-2")

        assert result["full_name"] == "Bob"
        assert "age" not in result
        assert "languages" not in result

    @pytest.mark.asyncio
    async def test_user_not_found_returns_none(self):
        session = _make_mock_session()
        session.execute.return_value = _one_or_none_result(None)

        with patch("graph.nodes.function_nodes.load_user_context._auth_db") as db:
            db.get_session.return_value = session
            result = await _fetch_user_profile_from_db("unknown-uid")

        assert result is None

    @pytest.mark.asyncio
    async def test_db_exception_returns_none(self):
        session = _make_mock_session()
        session.execute.side_effect = Exception("connection error")

        with patch("graph.nodes.function_nodes.load_user_context._auth_db") as db:
            db.get_session.return_value = session
            result = await _fetch_user_profile_from_db("user-uid-1")

        assert result is None


# ============================================================================
# _fetch_conversation_summary_from_db
# ============================================================================

class TestFetchConversationSummaryFromDb:

    @pytest.mark.asyncio
    async def test_row_exists_returns_summary(self):
        session = _make_mock_session()
        row = MagicMock()
        row.summary = "Prior context: exam anxiety, 3 turns."
        session.execute.return_value = _scalar_result(row)

        with patch("graph.nodes.function_nodes.load_user_context._data_db") as db:
            db.get_session.return_value = session
            result = await _fetch_conversation_summary_from_db(
                "550e8400-e29b-41d4-a716-446655440000"
            )

        assert result == "Prior context: exam anxiety, 3 turns."

    @pytest.mark.asyncio
    async def test_no_row_returns_none(self):
        session = _make_mock_session()
        session.execute.return_value = _scalar_result(None)

        with patch("graph.nodes.function_nodes.load_user_context._data_db") as db:
            db.get_session.return_value = session
            result = await _fetch_conversation_summary_from_db(
                "550e8400-e29b-41d4-a716-446655440000"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_none(self):
        """Passing a non-UUID string raises ValueError which is caught and returns None."""
        result = await _fetch_conversation_summary_from_db("not-a-valid-uuid")
        assert result is None


# ============================================================================
# _fetch_conversation_history_from_db
# ============================================================================

class TestFetchConversationHistoryFromDb:

    @pytest.mark.asyncio
    async def test_rows_returned_oldest_first_with_decryption(self):
        session = _make_mock_session()

        turn1 = MagicMock()
        turn1.speaker = "user"
        turn1.message = "hello"
        turn1.turn_index = 0

        turn2 = MagicMock()
        turn2.speaker = "bot"
        turn2.message = "hi there"
        turn2.turn_index = 1

        # Returned desc from DB (turn2 first), reversed to chronological by code
        session.execute.return_value = _scalars_all_result([turn2, turn1])

        km = MagicMock()
        km.decrypt = AsyncMock(side_effect=lambda cid, msg: msg)

        with patch("graph.nodes.function_nodes.load_user_context._data_db") as db, \
             patch("services.key_manager.get_key_manager", return_value=km):
            db.get_session.return_value = session
            result = await _fetch_conversation_history_from_db("conv-123")

        assert result is not None
        assert len(result) == 2
        # reversed([turn2, turn1]) gives turn1 first (index 0), turn2 second (index 1)
        assert result[0]["speaker"] == "user"   # turn1
        assert result[1]["speaker"] == "bot"    # turn2

    @pytest.mark.asyncio
    async def test_rows_ordered_oldest_first(self):
        session = _make_mock_session()

        turn_old = MagicMock(speaker="user", message="first msg", turn_index=0)
        turn_new = MagicMock(speaker="bot", message="reply", turn_index=1)

        # DB returns desc (newest first), code reverses to chronological
        session.execute.return_value = _scalars_all_result([turn_new, turn_old])

        km = MagicMock()
        km.decrypt = AsyncMock(side_effect=lambda cid, msg: msg)

        with patch("graph.nodes.function_nodes.load_user_context._data_db") as db, \
             patch("services.key_manager.get_key_manager", return_value=km):
            db.get_session.return_value = session
            result = await _fetch_conversation_history_from_db("conv-123")

        assert result[0]["turn_index"] == 0  # oldest first
        assert result[1]["turn_index"] == 1

    @pytest.mark.asyncio
    async def test_no_rows_returns_none(self):
        session = _make_mock_session()
        session.execute.return_value = _scalars_all_result([])

        km = MagicMock()
        km.decrypt = AsyncMock()

        with patch("graph.nodes.function_nodes.load_user_context._data_db") as db, \
             patch("services.key_manager.get_key_manager", return_value=km):
            db.get_session.return_value = session
            result = await _fetch_conversation_history_from_db("conv-empty")

        assert result is None

    @pytest.mark.asyncio
    async def test_decryption_failure_falls_back_to_raw_message(self):
        """If km.decrypt raises, the raw message is used instead."""
        session = _make_mock_session()
        turn = MagicMock(speaker="user", message="ENC:v1:opaque", turn_index=0)
        session.execute.return_value = _scalars_all_result([turn])

        km = MagicMock()
        km.decrypt = AsyncMock(side_effect=Exception("decryption failed"))

        with patch("graph.nodes.function_nodes.load_user_context._data_db") as db, \
             patch("services.key_manager.get_key_manager", return_value=km):
            db.get_session.return_value = session
            result = await _fetch_conversation_history_from_db("conv-123")

        assert result is not None
        assert result[0]["message"] == "ENC:v1:opaque"

    @pytest.mark.asyncio
    async def test_db_exception_returns_none(self):
        session = _make_mock_session()
        session.execute.side_effect = Exception("DB error")

        km = MagicMock()

        with patch("graph.nodes.function_nodes.load_user_context._data_db") as db, \
             patch("services.key_manager.get_key_manager", return_value=km):
            db.get_session.return_value = session
            result = await _fetch_conversation_history_from_db("conv-123")

        assert result is None


# ============================================================================
# _fetch_domain_config_from_db — stub
# ============================================================================

class TestFetchDomainConfigFromDb:

    @pytest.mark.asyncio
    async def test_stub_always_returns_none(self):
        """Domain config table doesn't exist yet — stub must always return None."""
        result = await _fetch_domain_config_from_db("student")
        assert result is None

    @pytest.mark.asyncio
    async def test_stub_returns_none_for_any_domain(self):
        for domain in ("student", "employee", "general", "unknown"):
            assert await _fetch_domain_config_from_db(domain) is None
