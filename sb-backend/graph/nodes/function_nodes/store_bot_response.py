"""
Store Bot Response Node for LangGraph

This node stores the bot's generated response in the database.
It creates a ConversationTurn record with the bot's message.
"""

from typing import Dict, Any
from sqlalchemy import select, func


from graph.state import ConversationState
from orm.models import ConversationTurn
from config.sqlalchemy_db import SQLAlchemyDataDB
import logging

# Logger setup
logger = logging.getLogger(__name__)

# Initialize database connection
data_db = SQLAlchemyDataDB()


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def store_bot_response_node(state: ConversationState) -> Dict[str, Any]:
    """
    Store the bot's response message in the database.
    
    This node saves the generated response to the conversation history
    in the ConversationTurn table with speaker="bot".
    
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
            # Get the current turn count for this conversation to set turn_index
            turn_count_stmt = select(func.count(ConversationTurn.id)).where(
                ConversationTurn.session_id == conversation_id
            )
            result = await session.execute(turn_count_stmt)
            turn_count = result.scalar() or 0
            
            # Create a new conversation turn record for bot response
            turn = ConversationTurn(
                session_id=conversation_id,
                turn_index=turn_count,  # Sequential turn number
                speaker="bot",
                message=bot_response
            )
            session.add(turn)
            await session.commit()
            
            return {
                "api_response": None
            }
            
    except Exception as e:
        return {
            "error": f"Error storing bot response: {str(e)}"
        }

