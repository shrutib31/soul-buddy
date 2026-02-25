"""
Get Bot Response Node for LangGraph

This node retrieves and decrypts the latest bot response for a given conversation.
If the message is plaintext (legacy), it is returned as-is.
"""

from typing import Dict, Any, Optional
from sqlalchemy import select

from graph.state import ConversationState
from orm.models import ConversationTurn
from config.sqlalchemy_db import get_data_db


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def get_bot_response_node(state: ConversationState) -> Dict[str, Any]:
    """
    Retrieve and decrypt the latest bot response for a conversation.

    - cognito mode: decrypts ENC:v1: prefixed message using AES-256-GCM
    - incognito mode: nothing was stored, returns empty
    - plaintext messages (legacy): returned as-is

    Args:
        state: Current conversation state

    Returns:
        Dict with latest_bot_response string
    """
    try:
        conversation_id = state.conversation_id
        mode = state.mode

        # Incognito = nothing was stored, return empty
        if mode == "incognito":
            return {"latest_bot_response": ""}

        if not conversation_id:
            return {"latest_bot_response": "", "error": "Missing conversation_id"}

        from services.key_manager import get_key_manager
        km = get_key_manager()

        data_db = get_data_db()
        async with data_db.get_session() as session:
            # Get the latest bot turn only
            stmt = select(ConversationTurn).where(
                ConversationTurn.session_id == conversation_id,
                ConversationTurn.speaker == "bot"
            ).order_by(ConversationTurn.turn_index.desc()).limit(1)

            result = await session.execute(stmt)
            turn = result.scalars().first()

            if not turn:
                return {"latest_bot_response": ""}

            try:
                # km.decrypt returns plaintext as-is if not ENC:v1: prefixed
                plaintext = await km.decrypt(conversation_id, turn.message)
            except Exception:
                plaintext = "[Decryption failed]"

        return {"latest_bot_response": plaintext}

    except Exception as e:
        return {
            "latest_bot_response": "",
            "error": f"Error retrieving bot response: {str(e)}"
        }


# ============================================================================
# STANDALONE UTILITY (used directly by chat.py endpoint)
# ============================================================================

async def get_latest_bot_response(conversation_id: str) -> Optional[str]:
    """
    Standalone function to retrieve and decrypt the latest bot response.
    Used directly by the chat API endpoint without going through the graph.

    Args:
        conversation_id: UUID of the conversation

    Returns:
        Decrypted bot response string or None
    """
    try:
        from services.key_manager import get_key_manager
        km = get_key_manager()

        data_db = get_data_db()
        async with data_db.get_session() as session:
            stmt = select(ConversationTurn).where(
                ConversationTurn.session_id == conversation_id,
                ConversationTurn.speaker == "bot"
            ).order_by(ConversationTurn.turn_index.desc()).limit(1)

            result = await session.execute(stmt)
            turn = result.scalars().first()

            if not turn:
                return None

            # Returns plaintext as-is if not encrypted (ENC:v1: check inside decrypt)
            return await km.decrypt(conversation_id, turn.message)

    except Exception:
        return None