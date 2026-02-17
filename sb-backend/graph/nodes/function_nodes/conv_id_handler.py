"""
Conversation ID Handler Node for LangGraph

This node handles conversation ID management based on mode (cognito/incognito).
- If no conversation_id: generates new one and creates DB record
- If incognito mode: checks expiration (24 hours) and regenerates if needed
- If cognito mode: validates and uses provided conversation_id

Standard LangGraph Node Pattern:
    - Input: ConversationState (Pydantic model)
    - Output: Dict[str, Any] with state updates
    - Always handle errors gracefully
    - Return only fields that need updating
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from graph.state import ConversationState
from orm.models import SbConversation
from config.sqlalchemy_db import SQLAlchemyDataDB

# Initialize database connection
data_db = SQLAlchemyDataDB()

# Expiration time for incognito conversations (24 hours)
INCOGNITO_EXPIRATION_HOURS = 24


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def conv_id_handler_node(state: ConversationState) -> Dict[str, Any]:
    """
    LangGraph node for conversation ID management.
    
    This is the main node function that will be registered in the graph.
    It processes the conversation ID based on the mode and returns state updates.
    
    Logic:
    1. If no conversation_id provided: Generate new one and create DB record
    2. If conversation_id provided and mode is 'incognito': 
       - Check if conversation exists and is not expired
       - If expired or doesn't exist, generate new one
    3. If conversation_id provided and mode is 'cognito':
       - Validate that conversation exists in DB
       - If doesn't exist, create new record
    
    Args:
        state: Current conversation state (ConversationState)
    
    Returns:
        Dict with updated fields: conversation_id, error (if any)
    """
    conversation_id = state.conversation_id
    mode = state.mode
    user_id = state.user_id
    
    try:
        async with data_db.get_session() as session:
            # Case 1: No conversation ID provided - generate new one
            if not conversation_id or conversation_id.strip() == "":
                new_conv_id = str(uuid.uuid4())
                
                # Create new conversation record
                new_conversation = SbConversation(
                    id=new_conv_id,
                    mode=mode,
                    started_at=datetime.utcnow()
                )
                session.add(new_conversation)
                await session.commit()
                
                return {
                    "conversation_id": new_conv_id,
                }
            
            # Case 2: Conversation ID provided - validate and check expiration
            # Query the conversation from database
            stmt = select(SbConversation).where(SbConversation.id == conversation_id)
            result = await session.execute(stmt)
            existing_conversation = result.scalar_one_or_none()
            
            # Handle incognito mode
            if mode == "incognito":
                # Check if conversation exists and is not expired
                if existing_conversation:
                    # Check if conversation has expired
                    expiration_time = existing_conversation.started_at + timedelta(hours=INCOGNITO_EXPIRATION_HOURS)
                    
                    if datetime.utcnow() > expiration_time:
                        # Conversation expired - generate new one
                        new_conv_id = str(uuid.uuid4())
                        new_conversation = SbConversation(
                            id=new_conv_id,
                            mode=mode,
                            started_at=datetime.utcnow()
                        )
                        session.add(new_conversation)
                        await session.commit()
                        
                        return {
                            "conversation_id": new_conv_id,
                        }
                    else:
                        # Conversation still valid - use existing one
                        return {
                            "conversation_id": conversation_id,
                        }
                else:
                    # Conversation doesn't exist - create new one
                    new_conv_id = str(uuid.uuid4())
                    new_conversation = SbConversation(
                        id=new_conv_id,
                        mode=mode,
                        started_at=datetime.utcnow()
                    )
                    session.add(new_conversation)
                    await session.commit()
                    
                    return {
                        "conversation_id": new_conv_id,
                    }
            
            # Handle cognito mode
            elif mode == "cognito":
                if not user_id:
                    return {
                        "error": "Missing user_id for cognito mode"
                    }
                # For cognito mode, conversation must exist or we create it
                if not existing_conversation:
                    # Create new conversation record with provided ID
                    new_conversation = SbConversation(
                        id=conversation_id,
                        supabase_user_id=user_id,
                        mode=mode,
                        started_at=datetime.utcnow()
                    )
                    session.add(new_conversation)
                    await session.commit()
                else:
                    # Ensure conversation belongs to the authenticated user
                    if existing_conversation.supabase_user_id and str(existing_conversation.supabase_user_id) != str(user_id):
                        return {
                            "error": "Conversation does not belong to user"
                        }
                    # Backfill supabase_user_id if missing
                    if existing_conversation.supabase_user_id is None:
                        existing_conversation.supabase_user_id = user_id
                        await session.commit()
                
                # Use the provided conversation ID
                return {
                    "conversation_id": conversation_id,
                }
            
            else:
                # Invalid mode
                return {
                    "error": f"Invalid mode: {mode}. Must be 'cognito' or 'incognito'"
                }
                
    except SQLAlchemyError as e:
        # Database error - return error state
        return {
            "error": f"Database error in conversation ID handler: {str(e)}"
        }
    except Exception as e:
        # General error
        return {
            "error": f"Error in conversation ID handler: {str(e)}"
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def create_conversation_id() -> str:
    """
    Simple utility function to generate a new UUID for conversation ID.
    
    Returns:
        New UUID string
    """
    return str(uuid.uuid4())


async def validate_conversation_exists(conversation_id: str) -> bool:
    """
    Check if a conversation exists in the database.
    
    Args:
        conversation_id: The conversation ID to check
    
    Returns:
        True if conversation exists, False otherwise
    """
    try:
        async with data_db.get_session() as session:
            stmt = select(SbConversation).where(SbConversation.id == conversation_id)
            result = await session.execute(stmt)
            conversation = result.scalar_one_or_none()
            return conversation is not None
    except Exception:
        return False


async def end_conversation(conversation_id: str) -> bool:
    """
    Mark a conversation as ended by setting the ended_at timestamp.
    
    Args:
        conversation_id: The conversation ID to end
    
    Returns:
        True if successful, False otherwise
    """
    try:
        async with data_db.get_session() as session:
            stmt = select(SbConversation).where(SbConversation.id == conversation_id)
            result = await session.execute(stmt)
            conversation = result.scalar_one_or_none()
            
            if conversation:
                conversation.ended_at = datetime.utcnow()
                await session.commit()
                return True
            return False
    except Exception:
        return False
