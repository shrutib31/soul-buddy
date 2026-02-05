"""
Store Message Node for LangGraph

This node stores the user message in the database for conversation history.
It runs in parallel with intent detection.
"""

from typing import Dict, Any
from datetime import datetime
from sqlalchemy import select, func

from graph.state import ConversationState
from orm.models import ConversationTurn
from config.sqlalchemy_db import SQLAlchemyDataDB

# Initialize database connection
data_db = SQLAlchemyDataDB()


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def store_message_node(state: ConversationState) -> Dict[str, Any]:
    """
    Store user message in the database.
    
    This node saves the current user message to the conversation history
    in the ConversationTurn table. Runs in parallel with intent detection.
    
    The turn_id is auto-generated (UUID), so we don't need to set it.
    
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
        
        async with data_db.get_session() as session:
            # Get the current turn count for this conversation to set turn_index
            from sqlalchemy import func
            
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
                message=user_message
            )
            session.add(turn)
            await session.commit()
            
            return {}
            
    except Exception as e:
        return {
            "error": f"Error storing message: {str(e)}"
        }
