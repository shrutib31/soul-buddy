"""
Get Messages Node for LangGraph

This node retrieves and decrypts the full conversation history for a given conversation.
If messages are plaintext (incognito legacy), they are returned as-is.
"""

import uuid
from typing import Dict, Any, List
from sqlalchemy import select

from graph.state import ConversationState
from orm.models import ConversationTurn, SbConversation
from config.sqlalchemy_db import SQLAlchemyDataDB

_data_db = SQLAlchemyDataDB()


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

        data_db = _data_db
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

        data_db = _data_db
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


async def get_all_user_conversations(supabase_uid: str) -> List[Dict[str, Any]]:
    """
    Retrieve all conversations and their decrypted messages for a user.
    Used directly by the chat API endpoint.

    Args:
        supabase_uid: Supabase user UUID string

    Returns:
        List of conversation dicts, each containing its decrypted messages.
    """
    try:
        async with _data_db.get_session() as session:
            stmt = (
                select(SbConversation)
                .where(SbConversation.supabase_user_id == uuid.UUID(supabase_uid))
                .order_by(SbConversation.started_at.desc())
            )
            result = await session.execute(stmt)
            conversations = result.scalars().all()

        all_conversations = []
        for conv in conversations:
            conv_id = str(conv.id)
            messages = await get_conversation_messages(conv_id)
            all_conversations.append({
                "conversation_id": conv_id,
                "mode": conv.mode,
                "started_at": conv.started_at.isoformat() if conv.started_at else None,
                "ended_at": conv.ended_at.isoformat() if conv.ended_at else None,
                "messages": messages,
            })

        return all_conversations

    except Exception as e:
        return []