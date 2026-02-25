"""
Store Bot Response Node for LangGraph

This node stores the bot's generated response in the database.
It creates a ConversationTurn record with the bot's message.
"""

from typing import Dict, Any
from datetime import datetime
from sqlalchemy import select, func

from graph.state import ConversationState
from orm.models import ConversationTurn
from config.sqlalchemy_db import get_data_db


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def store_bot_response_node(state: ConversationState) -> Dict[str, Any]:
    """
    Store the bot's response message in the database.
    
    This node saves the generated response to the conversation history
    in the ConversationTurn table with speaker="bot".
    
    - cognito mode: response is encrypted before storing using AES-256-GCM
    - incognito mode: no DB storage (privacy preserved)
    
    Args:
        state: Current conversation state
    
    Returns:
        Dict with any updates or error
    """
    try:
        conversation_id = state.conversation_id
        bot_response = state.response_draft
        mode = state.mode

        # Incognito = no persistence, return immediately
        if mode == "incognito":
            return {}

        if not conversation_id or not bot_response:
            return {
                "error": "Missing conversation_id or bot response",
                "response_draft": state.response_draft or ""
            }

        # Encrypt the response for cognito mode before storing
        from services.key_manager import get_key_manager
        km = get_key_manager()
        message_to_store = await km.encrypt(conversation_id, bot_response)
        # message_to_store is now "ENC:v1:<base64>" format

        data_db = get_data_db()
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
                message=message_to_store  # stored encrypted as "ENC:v1:..."
            )
            session.add(turn)
            await session.commit()
            
            return {}
            
    except Exception as e:
        return {
            "error": f"Error storing bot response: {str(e)}"
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def get_conversation_history(conversation_id: str) -> list:
    """
    Retrieve and decrypt the full conversation history for a conversation.
    
    Args:
        conversation_id: The conversation ID
    
    Returns:
        List of decrypted conversation turns
    """
    try:
        from services.key_manager import get_key_manager
        km = get_key_manager()

        data_db = get_data_db()
        async with data_db.get_session() as session:
            stmt = select(ConversationTurn).where(
                ConversationTurn.session_id == conversation_id
            ).order_by(ConversationTurn.turn_index)
            result = await session.execute(stmt)
            turns = result.scalars().all()
            
            history = []
            for turn in turns:
                try:
                    # km.decrypt handles ENC:v1: check — returns plaintext as-is if not encrypted
                    plaintext = await km.decrypt(conversation_id, turn.message)
                except Exception:
                    plaintext = "[Decryption failed]"

                history.append({
                    "turn_index": turn.turn_index,
                    "speaker": turn.speaker,
                    "message": plaintext,
                    "created_at": turn.created_at.isoformat() if turn.created_at else None
                })

            return history

    except Exception:
        return []