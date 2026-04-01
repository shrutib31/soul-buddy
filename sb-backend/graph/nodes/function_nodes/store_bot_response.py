"""
Store Bot Response Node for LangGraph
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from graph.state import ConversationState
from orm.models import ConversationTurn, UserConversationSummary
from config.sqlalchemy_db import SQLAlchemyDataDB
from services.cache_service import cache_service
from services.key_manager import get_key_manager
from utils.lang_classifier import classify_language_format, ROMANISED, CANONICAL, MIXED

logger = logging.getLogger(__name__)
data_db = SQLAlchemyDataDB()

def _build_summary(state: ConversationState, turn_count: int) -> str:
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

async def store_bot_response_node(state: ConversationState) -> Dict[str, Any]:
    try:
        conversation_id = state.conversation_id
        bot_response = state.response_draft
        if not conversation_id or not bot_response:
            return {"error": "Missing conversation_id or bot response"}

        # Classify language format
        format_type = classify_language_format(bot_response, state.language or 'en-IN')
        logger.info("[LanguageClassifier] Bot response format: %s (lang: %s)", format_type, state.language)
        romanised = bot_response if format_type == ROMANISED else None
        canonical = bot_response if format_type == CANONICAL else None
        mixed = bot_response if format_type == MIXED else None

        km = get_key_manager()
        message_to_store = await km.encrypt(conversation_id, bot_response) if km.is_encryption_enabled() else bot_response

        async with data_db.get_session() as session:
            turn_count_stmt = select(func.count(ConversationTurn.id)).where(
                ConversationTurn.session_id == conversation_id
            )
            result = await session.execute(turn_count_stmt)
            turn_count = result.scalar() or 0

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
            await session.commit()

        await cache_service.invalidate_conversation_history(conversation_id)
        if state.supabase_uid:
            await _upsert_conversation_summary(state, turn_count + 1)

        return {}
    except Exception as e:
        logger.error("store_bot_response: failed | conversation_id=%r error=%s", state.conversation_id, e)
        return {"error": f"Error storing bot response: {str(e)}"}

async def _upsert_conversation_summary(state: ConversationState, turn_count: int) -> None:
    try:
        summary_text = _build_summary(state, turn_count)
        user_uuid = uuid.UUID(state.supabase_uid)
        conv_uuid = uuid.UUID(state.conversation_id)
        now = datetime.now(timezone.utc)

        async with data_db.get_session() as session:
            stmt = (
                pg_insert(UserConversationSummary)
                .values(user_id=user_uuid, summary=summary_text, last_conversation_id=conv_uuid, updated_at=now)
                .on_conflict_do_update(
                    index_elements=["user_id"],
                    set_={"summary": summary_text, "last_conversation_id": conv_uuid, "updated_at": now}
                )
            )
            await session.execute(stmt)
            await session.commit()
        await cache_service.set_conversation_summary(state.supabase_uid, summary_text)
    except Exception as exc:
        logger.warning("summary upsert failed (non-fatal) | supabase_uid=%s error=%s", state.supabase_uid, exc)
