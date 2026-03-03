"""
Store Bot Response Node for LangGraph

Persists the bot's generated response as a ConversationTurn row (speaker="bot"),
then:
  1. Invalidates the Redis conversation-history cache so the next
     load_user_context call fetches fresh data from the database.
  2. Builds a structured conversation summary from the current state and
     upserts it into user_conversation_summaries (cognito users only),
     then refreshes the Redis summary cache.

The summary is built entirely from fields already in the state — no extra
LLM call — so it adds no latency to the critical path.

Graph position: runs after response_generator, before render.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from graph.state import ConversationState
from orm.models import ConversationTurn, UserConversationSummary
from config.sqlalchemy_db import SQLAlchemyDataDB
from services.cache_service import cache_service
import logging

# Logger setup
logger = logging.getLogger(__name__)

# Initialize database connection
data_db = SQLAlchemyDataDB()


# ============================================================================
# SUMMARY BUILDER
# ============================================================================

def _build_summary(state: ConversationState, turn_count: int) -> str:
    """
    Build a compact structured summary string from the current conversation state.

    Format (example):
        [2026-03-02] Domain: student. Situation: EXAM_ANXIETY (medium severity).
        Intent: VENTING. Risk: low. Flow: FLOW_GENERAL_OVERWHELM. Turns: 6.

    All fields are optional — missing classification data is simply omitted.
    The output is designed to be injected as prior-context for the response
    generator in future conversations.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [f"[{date_str}]", f"Domain: {state.domain}."]

    if state.situation:
        severity_str = f" ({state.severity} severity)" if state.severity else ""
        parts.append(f"Situation: {state.situation}{severity_str}.")

    if state.intent:
        parts.append(f"Intent: {state.intent}.")

    if state.risk_level:
        parts.append(f"Risk: {state.risk_level}.")

    if state.is_crisis_detected:
        parts.append("Crisis detected.")

    if state.flow_id:
        parts.append(f"Flow: {state.flow_id}.")

    parts.append(f"Turns: {turn_count}.")

    return " ".join(parts)


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def store_bot_response_node(state: ConversationState) -> Dict[str, Any]:
    """
    Store the bot's response and update the user's conversation summary.

    Steps:
      1. Count existing turns to determine turn_index.
      2. Insert the bot ConversationTurn row.
      3. Invalidate conversation-history cache.
      4. If cognito (supabase_uid present): build summary, upsert to DB, refresh cache.

    Args:
        state: Current conversation state

    Returns:
        Dict with any updates or error
    """
    try:
        conversation_id = state.conversation_id
        bot_response = state.response_draft
        logger.info(f"Storing bot response for conversation_id: {conversation_id}")

        if not conversation_id or not bot_response:
            return {
                "error": "Missing conversation_id or bot response",
                "response_draft": state.response_draft or ""
            }

        async with data_db.get_session() as session:
            # ----------------------------------------------------------------
            # 1. Get current turn count to assign turn_index
            # ----------------------------------------------------------------
            turn_count_stmt = select(func.count(ConversationTurn.id)).where(
                ConversationTurn.session_id == conversation_id
            )
            result = await session.execute(turn_count_stmt)
            turn_count = result.scalar() or 0

            # ----------------------------------------------------------------
            # 2. Persist bot response as a ConversationTurn row
            # ----------------------------------------------------------------
            turn = ConversationTurn(
                session_id=conversation_id,
                turn_index=turn_count,
                speaker="bot",
                message=bot_response
            )
            session.add(turn)
            await session.commit()

        # ----------------------------------------------------------------
        # 3. Invalidate conversation-history cache
        # ----------------------------------------------------------------
        await cache_service.invalidate_conversation_history(conversation_id)

        # ----------------------------------------------------------------
        # 4. Build + persist conversation summary  (cognito users only)
        # ----------------------------------------------------------------
        if state.supabase_uid:
            await _upsert_conversation_summary(state, turn_count + 1)

        return {"api_response": None}

    except Exception as e:
        return {
            "error": f"Error storing bot response: {str(e)}"
        }


# ============================================================================
# SUMMARY UPSERT HELPER
# ============================================================================

async def _upsert_conversation_summary(
    state: ConversationState,
    turn_count: int,
) -> None:
    """
    Build the summary string from state, upsert it into user_conversation_summaries,
    and refresh the Redis cache entry.

    Fails silently — a summary write failure must never break the main flow.
    """
    try:
        summary_text = _build_summary(state, turn_count)
        user_uuid = uuid.UUID(state.supabase_uid)
        conv_uuid = uuid.UUID(state.conversation_id)
        now = datetime.now(timezone.utc)

        logger.debug(
            "summary upsert start | supabase_uid=%s conversation_id=%s turns=%d",
            state.supabase_uid,
            state.conversation_id,
            turn_count,
        )

        async with data_db.get_session() as session:
            stmt = (
                pg_insert(UserConversationSummary)
                .values(
                    user_id=user_uuid,
                    summary=summary_text,
                    last_conversation_id=conv_uuid,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["user_id"],
                    set_={
                        "summary": summary_text,
                        "last_conversation_id": conv_uuid,
                        "updated_at": now,
                    },
                )
            )
            await session.execute(stmt)
            await session.commit()

        logger.debug(
            "summary DB upsert ok | supabase_uid=%s conversation_id=%s",
            state.supabase_uid,
            state.conversation_id,
        )

        # Keep Redis in sync so the next load_user_context hits cache
        await cache_service.set_conversation_summary(state.supabase_uid, summary_text)

        logger.debug(
            "summary cache refresh ok | supabase_uid=%s turns=%d summary=%r",
            state.supabase_uid,
            turn_count,
            summary_text,
        )

    except Exception as exc:
        logger.warning(
            "summary upsert failed (non-fatal) | supabase_uid=%s error=%s",
            state.supabase_uid,
            exc,
        )

