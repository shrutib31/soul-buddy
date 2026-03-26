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

logger = logging.getLogger(__name__)

# Initialize database connection
data_db = SQLAlchemyDataDB()


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

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
        
        km = get_key_manager()
        message_to_store = await km.encrypt(conversation_id, user_message) if km.is_encryption_enabled() else user_message

        async with data_db.get_session() as session:
            # Get the current turn count for this conversation to set turn_index
            turn_count_stmt = select(func.count(ConversationTurn.id)).where(
                ConversationTurn.session_id == conversation_id
            )
            result = await session.execute(turn_count_stmt)
            turn_count = result.scalar() or 0

            # Create a new conversation turn record
            # Note: id is auto-generated UUID, created_at is auto-set by DB
            turn = ConversationTurn(
                session_id=conversation_id,
                turn_index=turn_count,  # Sequential turn number (0-indexed)
                speaker="user",
                message=message_to_store
            )
            session.add(turn)
            await session.commit()

            # Invalidate cached history so the next load picks up the new turn
            await cache_service.invalidate_conversation_history(conversation_id)

            return {}

    except Exception as e:
        logger.error("store_message: failed | conversation_id=%r error=%s", state.conversation_id, e, exc_info=True)
        return {
            "error": f"Error storing message: {str(e)}"
        }
