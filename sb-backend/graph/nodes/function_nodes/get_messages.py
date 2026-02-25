"""
Get Messages Node for LangGraph

This node retrieves and decrypts the full conversation history for a given conversation.
If messages are plaintext (incognito legacy), they are returned as-is.
"""

from typing import Dict, Any, List
from sqlalchemy import select

from graph.state import ConversationState
from orm.models import ConversationTurn
from config.sqlalchemy_db import get_data_db


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def get_messages_node(state: ConversationState) -> Dict[str, Any]:
    """
    Retrieve and decrypt all conversation turns for a conversation.

    - cognito mode: decrypts ENC:v1: prefixed messages using AES-256-GCM
    - incognito mode: no DB storage was done, returns empty history
    - plaintext messages (legacy): returned as-is

    Args:
        state: Current conversation state

    Returns:
        Dict with conversation_history list
    """
    try:
        conversation_id = state.conversation_id
        mode = state.mode

        # Incognito = nothing was stored, return empty
        if mode == "incognito":
            return {"conversation_history": []}

        if not conversation_id:
            return {"conversation_history": [], "error": "Missing conversation_id"}

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
                    # km.decrypt returns plaintext as-is if not ENC:v1: prefixed
                    plaintext = await km.decrypt(conversation_id, turn.message)
                except Exception:
                    plaintext = "[Decryption failed]"

                history.append({
                    "id": str(turn.id),
                    "turn_index": turn.turn_index,
                    "speaker": turn.speaker,
                    "message": plaintext,
                    "created_at": turn.created_at.isoformat() if turn.created_at else None
                })

        return {"conversation_history": history}

    except Exception as e:
        return {
            "conversation_history": [],
            "error": f"Error retrieving messages: {str(e)}"
        }


# ============================================================================
# STANDALONE UTILITY (used directly by chat.py endpoint)
# ============================================================================

async def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """
    Standalone function to retrieve and decrypt all messages for a conversation.
    Used directly by the chat API endpoint without going through the graph.

    Args:
        conversation_id: UUID of the conversation

    Returns:
        List of decrypted message dicts
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

            messages = []
            for turn in turns:
                try:
                    # Returns plaintext as-is if not encrypted (ENC:v1: check inside decrypt)
                    plaintext = await km.decrypt(conversation_id, turn.message)
                except Exception:
                    plaintext = "[Decryption failed]"

                messages.append({
                    "id": str(turn.id),
                    "turn_index": turn.turn_index,
                    "speaker": turn.speaker,
                    "message": plaintext,
                    "created_at": turn.created_at.isoformat() if turn.created_at else None
                })

        return messages

    except Exception as e:
        return []