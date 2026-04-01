"""
Store Message Node for LangGraph

Persists the current user message as a ConversationTurn row (speaker="user"),
then invalidates the Redis conversation-history cache for this conversation so
the next load_user_context call fetches fresh data from the database.

Graph position: runs after load_user_context in parallel with out_of_scope
and stays off the classification/render critical path.
"""

import logging
from typing import Dict, Any
from sqlalchemy import select, func

from graph.state import ConversationState
from orm.models import ConversationTurn
from config.sqlalchemy_db import SQLAlchemyDataDB
from services.cache_service import cache_service
from services.key_manager import get_key_manager
from utils.lang_classifier import classify_language_format, ROMANISED, CANONICAL, MIXED

logger = logging.getLogger(__name__)
data_db = SQLAlchemyDataDB()

async def store_message_node(state: ConversationState) -> Dict[str, Any]:
    """
    Store user message in the database.

    This node saves the current user message to the conversation history
    in the ConversationTurn table while out_of_scope runs in parallel.

    Args:
        state: Current conversation state

    Returns:
        Dict with any updates (typically empty unless error)
    """
    try:
        conversation_id = state.conversation_id
        user_message = state.user_message
        
        if not conversation_id or not user_message:
            return {"error": "Missing conversation_id or user_message"}
        
        # Classify language format
        format_type = classify_language_format(user_message, state.language or 'en-IN')
        logger.info("[LanguageClassifier] User message format: %s (lang: %s)", format_type, state.language)
        romanised = user_message if format_type == ROMANISED else None
        canonical = user_message if format_type == CANONICAL else None
        mixed = user_message if format_type == MIXED else None

        km = get_key_manager()
        message_to_store = await km.encrypt(conversation_id, user_message) if km.is_encryption_enabled() else user_message

        async with data_db.get_session() as session:
            turn_count_stmt = select(func.count(ConversationTurn.id)).where(
                ConversationTurn.session_id == conversation_id
            )
            result = await session.execute(turn_count_stmt)
            turn_count = result.scalar() or 0

            turn = ConversationTurn(
                session_id=conversation_id,
                turn_index=turn_count,
                speaker="user",
                message=message_to_store,
                language=state.language,
                romanised_content=romanised,
                canonical_content=canonical,
                mixed_content=mixed,
            )
            session.add(turn)
            await session.commit()
            await cache_service.invalidate_conversation_history(conversation_id)

            return {}

    except Exception as e:
        logger.error("store_message: failed | conversation_id=%r error=%s", state.conversation_id, e)
        return {"error": f"Error storing message: {str(e)}"}
