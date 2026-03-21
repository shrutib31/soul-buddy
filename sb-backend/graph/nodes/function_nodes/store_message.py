"""
Store Message Node for LangGraph
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
data_db = SQLAlchemyDataDB()

async def store_message_node(state: ConversationState) -> Dict[str, Any]:
    try:
        conversation_id = state.conversation_id
        user_message = state.user_message
        
        if not conversation_id or not user_message:
            return {"error": "Missing conversation_id or user_message"}
        
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
                # Classification logic commented out per user request
            )
            session.add(turn)
            await session.commit()
            await cache_service.invalidate_conversation_history(conversation_id)

            return {}

    except Exception as e:
        logger.error("store_message: failed | conversation_id=%r error=%s", state.conversation_id, e)
        return {"error": f"Error storing message: {str(e)}"}
