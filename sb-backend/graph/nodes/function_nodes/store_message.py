"""
Store Message Node for LangGraph

Persists the current user message as a ConversationTurn row (speaker="user"),
then:
  1. Classifies the message language format (CANONICAL / ROMANISED / MIXED) and
     populates the appropriate content column:
       canonical_content  — pure native script (Devanagari etc.) or English
       romanised_content  — Hinglish / romanised Indian language in latin script
       mixed_content      — combination of native + latin scripts
     When encryption is enabled only the encrypted message column is written;
     the format-specific columns are left NULL to avoid storing plaintext.

  2. Writes a conversation_context row capturing mode, style, detected emotion
     and intensity for this turn — enables mode-aware summarization later.

  3. Detects mode/style transitions within the session and maintains the
     session_mode_segments timeline (closes the previous segment, opens a new one).

  4. Invalidates the Redis conversation-history cache so the next
     load_user_context call fetches fresh data.

Graph position: runs after load_user_context in parallel with out_of_scope
and stays off the classification/render critical path.
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy import select, func

from graph.state import ConversationState
from orm.models import ConversationTurn, ConversationContext, SessionModeSegment
from config.sqlalchemy_db import SQLAlchemyDataDB
from services.cache_service import cache_service
from services.key_manager import get_key_manager
from utils.lang_classifier import classify_language_format, ROMANISED, CANONICAL, MIXED

logger = logging.getLogger(__name__)

data_db = SQLAlchemyDataDB()


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def store_message_node(state: ConversationState) -> Dict[str, Any]:
    """
    Store user message, language content columns, mode/emotion context,
    and mode-transition segments.
    """
    try:
        conversation_id = state.conversation_id
        user_message = state.user_message

        if not conversation_id or not user_message:
            return {"error": "Missing conversation_id or user_message"}

        # ----------------------------------------------------------------
        # Language format classification
        # ----------------------------------------------------------------
        format_type = classify_language_format(user_message, state.language)
        logger.info(
            "store_message: language format=%s lang=%s | conversation_id=%s",
            format_type, state.language, conversation_id,
        )

        km = get_key_manager()
        if km.is_encryption_enabled():
            # Encrypt raw message; leave format columns NULL to avoid plaintext storage.
            message_to_store = await km.encrypt(conversation_id, user_message)
            romanised = None
            canonical = None
            mixed = None
        else:
            message_to_store = user_message
            romanised = user_message if format_type == ROMANISED else None
            canonical = user_message if format_type == CANONICAL else None
            mixed = user_message if format_type == MIXED else None

        async with data_db.get_session() as session:
            # ----------------------------------------------------------------
            # 1. Get current turn count → determines turn_index
            # ----------------------------------------------------------------
            turn_count_stmt = select(func.count(ConversationTurn.id)).where(
                ConversationTurn.session_id == conversation_id
            )
            result = await session.execute(turn_count_stmt)
            turn_index = result.scalar() or 0

            # ----------------------------------------------------------------
            # 2. Persist the user message turn with language columns
            # ----------------------------------------------------------------
            turn = ConversationTurn(
                session_id=conversation_id,
                turn_index=turn_index,
                speaker="user",
                message=message_to_store,
                language=state.language,
                romanised_content=romanised,
                canonical_content=canonical,
                mixed_content=mixed,
            )
            session.add(turn)
            await session.flush()  # get turn.id before referencing it below

            # ----------------------------------------------------------------
            # 3. Write conversation_context for this turn
            #    (mode, style, detected_emotion, intensity)
            # ----------------------------------------------------------------
            chat_mode = state.chat_mode or "default"
            chat_preference = state.chat_preference or "general"
            style = _preference_to_style(chat_preference)
            emotion_intensity = state.emotion_intensity

            ctx = ConversationContext(
                turn_id=turn.id,
                mode=chat_mode,
                style=style,
                detected_emotion=None,   # classifier hasn't run yet
                intensity=emotion_intensity,
            )
            session.add(ctx)

            # ----------------------------------------------------------------
            # 4. Mode segment tracking
            # ----------------------------------------------------------------
            await _update_mode_segments(session, conversation_id, chat_mode, style, turn_index)

            await session.commit()

        # ----------------------------------------------------------------
        # 5. Invalidate cached history
        # ----------------------------------------------------------------
        await cache_service.invalidate_conversation_history(conversation_id)

        return {}

    except Exception as e:
        logger.error(
            "store_message: failed | conversation_id=%r error=%s",
            state.conversation_id, e, exc_info=True,
        )
        return {"error": f"Error storing message: {str(e)}"}


# ============================================================================
# HELPERS
# ============================================================================

def _preference_to_style(chat_preference: str) -> str:
    """Map chat_preference value to the style taxonomy (gentle/balanced/practical)."""
    mapping = {
        "gentle_reflective": "gentle",
        "direct_practical": "practical",
        "general": "balanced",
    }
    return mapping.get(chat_preference, "balanced")


async def _update_mode_segments(
    session,
    conversation_id: str,
    current_mode: str,
    current_style: str,
    current_turn_index: int,
) -> None:
    """
    Maintain the session_mode_segments timeline.

    - Fetch the latest open segment (end_turn IS NULL) for this session.
    - If none exists → create the first segment starting at turn 0.
    - If mode or style changed vs the open segment → close it and open a new one.
    - If no change → do nothing.
    """
    try:
        open_seg_stmt = (
            select(SessionModeSegment)
            .where(
                SessionModeSegment.session_id == conversation_id,
                SessionModeSegment.end_turn.is_(None),
            )
            .order_by(SessionModeSegment.start_turn.desc())
            .limit(1)
        )
        result = await session.execute(open_seg_stmt)
        open_segment: Optional[SessionModeSegment] = result.scalar_one_or_none()

        if open_segment is None:
            session.add(SessionModeSegment(
                session_id=conversation_id,
                mode=current_mode,
                style=current_style,
                start_turn=current_turn_index,
                end_turn=None,
            ))
        elif open_segment.mode != current_mode or open_segment.style != current_style:
            open_segment.end_turn = current_turn_index - 1
            session.add(SessionModeSegment(
                session_id=conversation_id,
                mode=current_mode,
                style=current_style,
                start_turn=current_turn_index,
                end_turn=None,
            ))

    except Exception as exc:
        logger.warning(
            "store_message: mode segment update failed | conversation_id=%r error=%s",
            conversation_id, exc,
        )
