"""
Store Bot Response Node for LangGraph

Persists the bot's generated response as a ConversationTurn row (speaker="bot"),
then fires off async background tasks (non-blocking):

  Every turn:
    - Invalidate conversation-history cache
    - InsightScoringService: compute rule-based metrics (emotional stability,
      engagement, progress, mode preference, risk) → user_insights table

  Every 5 turns (cognito only):
    - SummarizationService.summarize_session_incremental() → lightweight
      mode-aware JSON snapshot upserted into session_summaries

  On new-session detection (cognito only):
    - SummarizationService.summarize_session_final() on the PREVIOUS session
      (if it exists and is not yet finalised)
    - SummarizationService.update_user_memory() chained after final summary

All background tasks use asyncio.create_task() — zero latency impact on the
critical path. Failures are fully contained and logged by each service.

Graph position: runs after response_generator, before render.
"""

import asyncio
import uuid
import logging
from typing import Dict, Any, Optional
from sqlalchemy import select, func, or_

from graph.state import ConversationState
from sqlalchemy import update as sa_update
from orm.models import ConversationTurn, ConversationContext, SessionSummary, SbConversation
from config.sqlalchemy_db import SQLAlchemyDataDB
from services.cache_service import cache_service
from services.key_manager import get_key_manager
from services.summarization_service import (
    summarize_session_incremental,
    summarize_session_final,
    update_user_memory,
)
from services.insight_scoring import score_session
from utils.lang_classifier import classify_language_format, ROMANISED, CANONICAL, MIXED

logger = logging.getLogger(__name__)

data_db = SQLAlchemyDataDB()

# How often to generate an incremental summary (every N bot turns)
_INCREMENTAL_SUMMARY_EVERY_N_TURNS = 5


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def store_bot_response_node(state: ConversationState) -> Dict[str, Any]:
    """
    Store bot response and fire background intelligence tasks.
    """
    try:
        conversation_id = state.conversation_id
        bot_response = state.response_draft

        if not conversation_id or not bot_response:
            logger.error(
                "store_bot_response: missing required fields | conversation_id=%r has_response=%s",
                conversation_id, bool(bot_response),
            )
            return {"error": "Missing conversation_id or bot response"}

        # ----------------------------------------------------------------
        # Language format classification (mirrors store_message_node)
        # ----------------------------------------------------------------
        format_type = classify_language_format(bot_response, state.language or "en-IN")
        logger.info(
            "store_bot_response: language format=%s lang=%s | conversation_id=%s",
            format_type, state.language, conversation_id,
        )

        km = get_key_manager()
        if km.is_encryption_enabled():
            message_to_store = await km.encrypt(conversation_id, bot_response)
            romanised = None
            canonical = None
            mixed = None
        else:
            message_to_store = bot_response
            romanised = bot_response if format_type == ROMANISED else None
            canonical = bot_response if format_type == CANONICAL else None
            mixed = bot_response if format_type == MIXED else None

        async with data_db.get_session() as session:
            # ----------------------------------------------------------------
            # 1. Count existing turns → determine turn_index
            # ----------------------------------------------------------------
            turn_count_stmt = select(func.count(ConversationTurn.id)).where(
                ConversationTurn.session_id == conversation_id
            )
            result = await session.execute(turn_count_stmt)
            turn_count = result.scalar() or 0

            # ----------------------------------------------------------------
            # 2. Persist bot response
            # ----------------------------------------------------------------
            turn = ConversationTurn(
                session_id=conversation_id,
                turn_index=turn_count,
                speaker="bot",
                message=message_to_store,
                language=state.language,
                romanised_content=romanised,
                canonical_content=canonical,
                mixed_content=mixed,
            )
            session.add(turn)

            # ----------------------------------------------------------------
            # 3. Back-fill intensity on the most recent USER turn's context.
            #    classification_node has run by now and state.severity is set.
            # ----------------------------------------------------------------
            _SEVERITY_INTENSITY = {"low": 0.2, "medium": 0.5, "high": 0.8}
            computed_intensity = _SEVERITY_INTENSITY.get(state.severity or "", None)
            if computed_intensity is not None:
                # Find the latest user turn id for this conversation
                latest_user_turn_stmt = (
                    select(ConversationTurn.id)
                    .where(
                        ConversationTurn.session_id == conversation_id,
                        ConversationTurn.speaker == "user",
                    )
                    .order_by(ConversationTurn.turn_index.desc())
                    .limit(1)
                )
                r2 = await session.execute(latest_user_turn_stmt)
                user_turn_id = r2.scalar_one_or_none()
                if user_turn_id:
                    await session.execute(
                        sa_update(ConversationContext)
                        .where(ConversationContext.turn_id == user_turn_id)
                        .values(
                            intensity=computed_intensity,
                            detected_emotion=state.intent or None,
                        )
                    )

            await session.commit()

        # turn_count after the bot turn is now turn_count + 1
        total_turns = turn_count + 1

        # ----------------------------------------------------------------
        # 3. Invalidate conversation-history cache
        # ----------------------------------------------------------------
        await cache_service.invalidate_conversation_history(conversation_id)

        # ----------------------------------------------------------------
        # 4. Background tasks (cognito users only — need user_id)
        # ----------------------------------------------------------------
        if state.supabase_uid:
            user_id = state.supabase_uid
            dominant_mode = state.chat_mode or "default"
            risk_level = state.risk_level or "low"

            # Session start time — derive from sb_conversations.started_at if available.
            # For now we pass None; insight_scoring handles None gracefully.
            session_duration: Optional[float] = None

            # 4a. Rule-based insight scoring (every turn, no LLM)
            asyncio.create_task(
                score_session(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    turn_count=total_turns,
                    session_duration_seconds=session_duration,
                    risk_level=risk_level,
                    dominant_mode=dominant_mode,
                )
            )

            # 4b. Incremental summarization every N bot turns (LLM, fire-and-forget)
            # total_turns counts both user and bot turns; bot turns are ~half.
            # We trigger when the bot-turn estimate crosses a multiple of N.
            bot_turn_estimate = total_turns // 2
            if bot_turn_estimate > 0 and bot_turn_estimate % _INCREMENTAL_SUMMARY_EVERY_N_TURNS == 0:
                asyncio.create_task(
                    summarize_session_incremental(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        dominant_mode=dominant_mode,
                        turn_count=total_turns,
                    )
                )

            # 4c. Lazy finalisation of the PREVIOUS session (if is_new_session flag is set).
            # is_new_session is set by load_user_context_node when no prior history exists.
            if state.is_new_session:
                prev_session_id = await _get_previous_unfinalised_session(user_id, conversation_id)
                if prev_session_id:
                    existing_memory = state.user_memory  # already loaded into state

                    async def _finalise_prev_session(prev_id: str) -> None:
                        final = await summarize_session_final(
                            conversation_id=prev_id,
                            user_id=user_id,
                            dominant_mode=dominant_mode,
                        )
                        if final:
                            await update_user_memory(
                                user_id=user_id,
                                new_session_summary=final,
                                existing_memory=existing_memory,
                            )

                    asyncio.create_task(_finalise_prev_session(prev_session_id))

        return {}

    except asyncio.CancelledError:
        logger.debug("store_bot_response: cancelled by parallel path | conversation_id=%r", state.conversation_id)
        return {}
    except Exception as e:
        logger.error(
            "store_bot_response: failed | conversation_id=%r error=%s",
            state.conversation_id, e, exc_info=True,
        )
        return {"error": f"Error storing bot response: {str(e)}"}


# ============================================================================
# HELPERS
# ============================================================================

async def _get_previous_unfinalised_session(
    user_id: str, current_conversation_id: str
) -> Optional[str]:
    """
    Find the most recent cognito session for this user (from sb_conversations) that:
    - is NOT the current conversation
    - is NOT yet finalised (no row in session_summaries, OR is_finalised=False)

    Querying sb_conversations as the primary source ensures sessions that never
    reached the incremental-summary threshold (< 5 turns) are also considered.

    Returns the conversation_id string, or None if no such session exists.
    """
    try:
        user_uuid = uuid.UUID(user_id)
        current_uuid = uuid.UUID(current_conversation_id)

        async with data_db.get_session() as session:
            stmt = (
                select(SbConversation.id)
                .outerjoin(
                    SessionSummary,
                    SessionSummary.session_id == SbConversation.id,
                )
                .where(
                    SbConversation.supabase_user_id == user_uuid,
                    SbConversation.id != current_uuid,
                    SbConversation.mode == "cognito",
                    or_(
                        SessionSummary.session_id == None,  # noqa: E711 – no summary row yet
                        SessionSummary.is_finalised == False,  # noqa: E712
                    ),
                )
                .order_by(SbConversation.started_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return str(row) if row else None

    except Exception as exc:
        logger.warning(
            "store_bot_response: prev session lookup failed | user_id=%s error=%s", user_id, exc
        )
        return None
