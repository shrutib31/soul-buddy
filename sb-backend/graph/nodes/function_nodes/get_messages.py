"""
Get Messages Node for LangGraph

This node retrieves and decrypts the full conversation history for a given conversation.
If messages are plaintext (incognito legacy), they are returned as-is.
"""

import uuid
from collections import defaultdict
from typing import Dict, Any, List, Optional
from sqlalchemy import select

from graph.state import ConversationState
from orm.models import ConversationTurn, SbConversation
from config.sqlalchemy_db import SQLAlchemyDataDB
from services.key_manager import get_key_manager

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

async def get_conversation_messages(
    conversation_id: str,
    supabase_uid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Standalone function to retrieve and decrypt all messages for a conversation.
    Used directly by the chat API endpoint without going through the graph.

    Args:
        conversation_id: UUID of the conversation
        supabase_uid: When provided, ownership is verified against this Supabase user ID.
                      Raises PermissionError if the conversation does not belong to the user.

    Returns:
        List of decrypted message dicts
    """
    try:
        km = get_key_manager()
        conv_uuid = uuid.UUID(conversation_id)

        data_db = _data_db
        async with data_db.get_session() as session:
            if supabase_uid is not None:
                ownership_stmt = select(SbConversation).where(
                    SbConversation.id == conv_uuid,
                    SbConversation.supabase_user_id == uuid.UUID(supabase_uid),
                )
                ownership_result = await session.execute(ownership_stmt)
                if ownership_result.scalar_one_or_none() is None:
                    raise PermissionError(
                        f"Conversation {conversation_id} not found or not owned by user"
                    )

            stmt = select(ConversationTurn).where(
                ConversationTurn.session_id == conv_uuid
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

    except PermissionError:
        raise
    except Exception:
        return []


async def get_all_user_conversations(supabase_uid: str) -> List[Dict[str, Any]]:
    """
    Retrieve all conversations and their decrypted messages for a user.
    Uses a single batched query for conversation turns to avoid N+1 DB calls.
    Used directly by the chat API endpoint.

    Args:
        supabase_uid: Supabase user UUID string

    Returns:
        List of conversation dicts, each containing its decrypted messages.
    """
    try:
        async with _data_db.get_session() as session:
            conv_stmt = (
                select(SbConversation)
                .where(SbConversation.supabase_user_id == uuid.UUID(supabase_uid))
                .order_by(SbConversation.started_at.desc())
            )
            conv_result = await session.execute(conv_stmt)
            conversations = conv_result.scalars().all()

            if not conversations:
                return []

            # Batch-fetch all turns for all conversations in a single query
            conv_ids = [conv.id for conv in conversations]
            turns_stmt = (
                select(ConversationTurn)
                .where(ConversationTurn.session_id.in_(conv_ids))
                .order_by(ConversationTurn.session_id, ConversationTurn.turn_index)
            )
            turns_result = await session.execute(turns_stmt)
            all_turns = turns_result.scalars().all()

        # Group turns by conversation ID in Python
        turns_by_conv: Dict[str, list] = defaultdict(list)
        for turn in all_turns:
            turns_by_conv[str(turn.session_id)].append(turn)

        km = get_key_manager()

        all_conversations = []
        for conv in conversations:
            conv_id = str(conv.id)
            messages = []
            for turn in turns_by_conv[conv_id]:
                try:
                    plaintext = await km.decrypt(conv_id, turn.message)
                except Exception:
                    plaintext = "[Decryption failed]"

                messages.append({
                    "id": str(turn.id),
                    "turn_index": turn.turn_index,
                    "speaker": turn.speaker,
                    "message": plaintext,
                    "created_at": turn.created_at.isoformat() if turn.created_at else None,
                })

            all_conversations.append({
                "conversation_id": conv_id,
                "mode": conv.mode,
                "started_at": conv.started_at.isoformat() if conv.started_at else None,
                "ended_at": conv.ended_at.isoformat() if conv.ended_at else None,
                "messages": messages,
            })

        return all_conversations

    except Exception:
        return []