"""
Load User Context Node for LangGraph

Cache-aside pattern: read from Redis first; on miss, fall back to DB and
populate the cache.  Redis unavailability is fully transparent — every
fetch degrades gracefully to a DB read.

What this node loads
--------------------
  personality_profile   user:<userId>:personality_profile  (2 h TTL)
  user_profile          user:<userId>:profile               (2 h TTL)
  session_summary       conv:<conversationId>:session_summary (2 h TTL)
  user_memory           user:<userId>:user_memory           (24 h TTL)
  conversation_history  conv:<conversationId>:history       (30 min TTL)
  ui_state              user:<userId>:ui_state              (30 min TTL)
  domain_config         config:domain:<domain>              (24 h TTL)

New-session detection
---------------------
  If conversation_history is empty (first turn of this conversation_id),
  is_new_session=True is set in state.  store_bot_response_node reads this
  flag to trigger lazy finalisation of the previous session.

Node placement in the graph
----------------------------
  conv_id_handler → load_user_context → store_message ┐
                                      → out_of_scope → classification_node / render
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from graph.state import ConversationState
from orm.models import ConversationTurn, SessionSummary, UserMemory
from orm.auth_models import AuthUser, UserPersonalityProfile, UserDetailedProfile
from config.sqlalchemy_db import SQLAlchemyDataDB, SQLAlchemyAuthDB
from services.cache_service import cache_service

logger = logging.getLogger(__name__)

# Data DB: conversations, conversation turns, conversation summaries
_data_db = SQLAlchemyDataDB()
# Auth DB: user profiles, personality profiles, user details
_auth_db = SQLAlchemyAuthDB()


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

async def load_user_context_node(state: ConversationState) -> Dict[str, Any]:
    """
    Load all per-user and per-conversation context into the graph state.

    Always returns a dict with the fields that changed; unchanged fields are
    left at their defaults so the graph merge is additive.
    """
    updates: Dict[str, Any] = {}

    mode = state.mode
    supabase_uid = state.supabase_uid
    conversation_id = state.conversation_id
    domain = state.domain

    is_cognito = mode == "cognito" and bool(supabase_uid)

    # ------------------------------------------------------------------ #
    # 1. Personality profile  (cognito only)                              #
    # ------------------------------------------------------------------ #
    if is_cognito:
        profile = await cache_service.get_personality_profile(supabase_uid)
        if profile is None:
            profile = await _fetch_personality_profile_from_db(supabase_uid)
            if profile:
                await cache_service.set_personality_profile(supabase_uid, profile)
        if profile:
            updates["user_personality_profile"] = profile

    # ------------------------------------------------------------------ #
    # 2. User profile / preferences  (cognito only)                       #
    # ------------------------------------------------------------------ #
    if is_cognito:
        user_profile = await cache_service.get_user_profile(supabase_uid)
        if user_profile is None:
            user_profile = await _fetch_user_profile_from_db(supabase_uid)
            if user_profile:
                await cache_service.set_user_profile(supabase_uid, user_profile)
        if user_profile:
            updates["user_preferences"] = user_profile

    # ------------------------------------------------------------------ #
    # 3. Session summary for current conversation (both modes)            #
    # ------------------------------------------------------------------ #
    if conversation_id:
        session_summary = await cache_service.get_session_summary(conversation_id)
        if session_summary is None:
            session_summary = await _fetch_session_summary_from_db(conversation_id)
            if session_summary:
                await cache_service.set_session_summary(conversation_id, session_summary)
        if session_summary:
            updates["session_summary"] = session_summary

    # ------------------------------------------------------------------ #
    # 4. User memory  (cognito only — evolving cross-session context)     #
    # ------------------------------------------------------------------ #
    if is_cognito:
        user_memory = await cache_service.get_user_memory(supabase_uid)
        if user_memory is None:
            user_memory = await _fetch_user_memory_from_db(supabase_uid)
            if user_memory:
                await cache_service.set_user_memory(supabase_uid, user_memory)
        if user_memory:
            updates["user_memory"] = user_memory

    # ------------------------------------------------------------------ #
    # 5. Conversation history  (both modes, keyed by conversation_id)     #
    # ------------------------------------------------------------------ #
    if conversation_id:
        history = await cache_service.get_conversation_history(conversation_id)
        if history is None:
            history = await _fetch_conversation_history_from_db(conversation_id)
            if history:
                await cache_service.set_conversation_history(conversation_id, history)
        if history:
            updates["conversation_history"] = history

    # New-session detection: no existing turns means this is the first message
    if conversation_id and not updates.get("conversation_history"):
        updates["is_new_session"] = True

    # ------------------------------------------------------------------ #
    # 6. UI / navigation state  (cognito only — derived from page_context)#
    # ------------------------------------------------------------------ #
    if is_cognito:
        cached_ui_state = await cache_service.get_ui_state(supabase_uid)
        current_page_context = state.page_context

        if current_page_context:
            # Persist latest page_context for the next request
            await cache_service.set_ui_state(supabase_uid, current_page_context)
        elif cached_ui_state:
            # No page_context in this request — restore last known state
            updates["page_context"] = cached_ui_state

    # ------------------------------------------------------------------ #
    # 6. Domain config  (global, long TTL — no dedicated table yet)       #
    # ------------------------------------------------------------------ #
    config_key = f"domain:{domain}"
    domain_config = await cache_service.get_config(config_key)
    if domain_config is None:
        domain_config = await _fetch_domain_config_from_db(domain)
        if domain_config:
            await cache_service.set_config(config_key, domain_config)
    if domain_config:
        updates["domain_config"] = domain_config

    return updates


# ---------------------------------------------------------------------------
# DB fetch helpers  (called on cache miss)
# ---------------------------------------------------------------------------

async def _fetch_conversation_history_from_db(
    conversation_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch the last CONV_HISTORY_MAX_TURNS turns for a conversation from the DB.

    Returns a list of {speaker, message, turn_index} dicts, oldest-first,
    or None on error / empty result.
    """
    from services.cache_service import CONV_HISTORY_MAX_TURNS

    try:
        async with _data_db.get_session() as session:
            # Fetch the most recent N turns, then reverse to chronological order
            stmt = (
                select(ConversationTurn)
                .where(ConversationTurn.session_id == conversation_id)
                .order_by(ConversationTurn.turn_index.desc())
                .limit(CONV_HISTORY_MAX_TURNS)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

            if not rows:
                return None

            from services.key_manager import get_key_manager
            km = get_key_manager()

            turns = []
            for row in reversed(rows):  # oldest first
                # Prefer plaintext content columns (set when encryption is disabled).
                # canonical_content → romanised_content → mixed_content → decrypt message.
                if row.canonical_content:
                    message = row.canonical_content
                elif row.romanised_content:
                    message = row.romanised_content
                elif row.mixed_content:
                    message = row.mixed_content
                else:
                    try:
                        message = await km.decrypt(conversation_id, row.message)
                    except Exception:
                        message = row.message or ""
                turns.append({
                    "speaker": row.speaker,
                    "message": message,
                    "turn_index": row.turn_index,
                })
            return turns

    except Exception as exc:
        logger.warning("load_user_context: DB history fetch failed: %s", exc)
        return None


async def _fetch_session_summary_from_db(
    conversation_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch the latest session summary for a conversation from session_summaries.

    Returns the incremental_summary JSONB dict, or None if no row or on error.
    """
    import uuid as _uuid
    try:
        async with _data_db.get_session() as session:
            stmt = select(SessionSummary).where(
                SessionSummary.session_id == _uuid.UUID(conversation_id)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            # Prefer final_summary if available, fall back to incremental
            return row.final_summary or row.incremental_summary
    except Exception as exc:
        logger.warning("load_user_context: DB session_summary fetch failed: %s", exc)
        return None


async def _fetch_user_memory_from_db(supabase_uid: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the user_memory row for a cognito user.

    Returns a dict with growth_summary, recurring_themes, behavioral_patterns,
    risk_signals, or None if no row exists or on error.
    """
    import uuid as _uuid
    try:
        async with _data_db.get_session() as session:
            stmt = select(UserMemory).where(
                UserMemory.user_id == _uuid.UUID(supabase_uid)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return {
                "growth_summary": row.growth_summary,
                "recurring_themes": row.recurring_themes,
                "behavioral_patterns": row.behavioral_patterns,
                "risk_signals": row.risk_signals,
                "emotional_baseline": row.emotional_baseline,
                "last_updated": str(row.last_updated) if row.last_updated else None,
            }
    except Exception as exc:
        logger.warning("load_user_context: DB user_memory fetch failed: %s", exc)
        return None


async def _fetch_personality_profile_from_db(
    supabase_uid: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch personality profile from auth DB (user_personality_profiles table).

    Returns the personality_profile_data JSONB dict, or None if no row exists or on error.
    """
    try:
        async with _auth_db.get_session() as session:
            stmt = select(UserPersonalityProfile).where(
                UserPersonalityProfile.supabase_uid == supabase_uid
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return row.personality_profile_data if row else None
    except Exception as exc:
        logger.warning("load_user_context: DB personality profile fetch failed: %s", exc)
        return None


async def _fetch_user_profile_from_db(
    supabase_uid: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch user profile from auth DB (users + user_detailed_profiles tables).

    Joins users → user_detailed_profiles and returns a flat dict of fields
    useful for conversation context.
    Returns None if the user is not found or on error.
    """
    try:
        async with _auth_db.get_session() as session:
            stmt = (
                select(AuthUser, UserDetailedProfile)
                .outerjoin(
                    UserDetailedProfile,
                    UserDetailedProfile.user_id == AuthUser.id,
                )
                .where(AuthUser.supabase_uid == supabase_uid)
            )
            result = await session.execute(stmt)
            row = result.one_or_none()
            if row is None:
                return None

            auth_user, detail = row

            profile: Dict[str, Any] = {
                "full_name": auth_user.full_name,
                "email": auth_user.email,
                "role": auth_user.role,
            }

            if detail:
                profile.update({
                    "first_name": detail.first_name,
                    "last_name": detail.last_name,
                    "age": detail.age,
                    "age_group": detail.age_group,
                    "gender": detail.gender,
                    "pronouns": detail.pronouns,
                    "country": detail.country,
                    "timezone": detail.timezone,
                    "languages": detail.languages,
                    "communication_language": detail.communication_language,
                    "education_level": detail.education_level,
                    "occupation": detail.occupation,
                    "marital_status": detail.marital_status,
                    "hobbies": detail.hobbies,
                    "interests": detail.interests,
                    "mental_health_history": detail.mental_health_history,
                    "physical_health_history": detail.physical_health_history,
                })

            return profile
    except Exception as exc:
        logger.warning("load_user_context: DB user profile fetch failed: %s", exc)
        return None


async def _fetch_domain_config_from_db(
    domain: str,  # noqa: ARG001
) -> Optional[Dict[str, Any]]:
    """
    Stub — domain_config table does not exist yet.
    Returns None until the schema is implemented.
    """
    return None
