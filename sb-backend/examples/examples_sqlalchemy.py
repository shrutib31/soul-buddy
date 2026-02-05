"""
SQLAlchemy Usage Examples

This file demonstrates various ways to use SQLAlchemy for database operations
with the SoulBuddy backend.
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from config.sqlalchemy_db import data_db_sqlalchemy, auth_db_sqlalchemy
from models import Conversation, Message, User, Role

router = APIRouter()


# ============================================================================
# Example 1: Simple Query (Read)
# ============================================================================

@router.get("/conversations/{user_id}")
async def get_user_conversations(user_id: str):
    """Get all conversations for a user"""
    async with data_db_sqlalchemy.get_session() as session:
        # Build query
        stmt = select(Conversation).where(Conversation.user_id == user_id)
        
        # Execute query
        result = await session.execute(stmt)
        conversations = result.scalars().all()
        
        return {"conversations": [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat()
            }
            for conv in conversations
        ]}


# ============================================================================
# Example 2: Create with Automatic Transaction
# ============================================================================

@router.post("/conversations")
async def create_conversation(user_id: str, title: str):
    """Create a new conversation"""
    async with data_db_sqlalchemy.get_transaction() as session:
        # Create new conversation
        new_conversation = Conversation(
            user_id=user_id,
            title=title
        )
        session.add(new_conversation)
        
        # Auto-commit happens when context exits successfully
        # Access the ID after flush
        await session.flush()
        
        return {
            "id": new_conversation.id,
            "user_id": new_conversation.user_id,
            "title": new_conversation.title
        }


# ============================================================================
# Example 3: Update Operation
# ============================================================================

@router.put("/conversations/{conversation_id}")
async def update_conversation(conversation_id: int, title: str):
    """Update conversation title"""
    async with data_db_sqlalchemy.get_transaction() as session:
        # Build update statement
        stmt = (
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(title=title)
        )
        
        # Execute update
        result = await session.execute(stmt)
        
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"message": "Updated successfully"}


# ============================================================================
# Example 4: Delete Operation
# ============================================================================

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int):
    """Delete a conversation"""
    async with data_db_sqlalchemy.get_transaction() as session:
        # Build delete statement
        stmt = delete(Conversation).where(Conversation.id == conversation_id)
        
        # Execute delete
        result = await session.execute(stmt)
        
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"message": "Deleted successfully"}


# ============================================================================
# Example 5: Query with Relationships (Eager Loading)
# ============================================================================

@router.get("/conversations/{conversation_id}/full")
async def get_conversation_with_messages(conversation_id: int):
    """Get conversation with all messages"""
    async with data_db_sqlalchemy.get_session() as session:
        # Build query with eager loading
        stmt = (
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages))
        )
        
        # Execute query
        result = await session.execute(stmt)
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "id": conversation.id,
            "title": conversation.title,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat()
                }
                for msg in conversation.messages
            ]
        }


# ============================================================================
# Example 6: Complex Transaction with Multiple Operations
# ============================================================================

@router.post("/conversations/{conversation_id}/messages")
async def add_message_to_conversation(conversation_id: int, role: str, content: str):
    """Add a message to a conversation"""
    async with data_db_sqlalchemy.get_transaction() as session:
        # First, verify conversation exists
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await session.execute(stmt)
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Create new message
        new_message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content
        )
        session.add(new_message)
        
        # Flush to get the message ID
        await session.flush()
        
        return {
            "id": new_message.id,
            "conversation_id": new_message.conversation_id,
            "role": new_message.role,
            "content": new_message.content
        }


# ============================================================================
# Example 7: Manual Transaction Control
# ============================================================================

@router.post("/conversations/bulk")
async def create_bulk_conversations(user_id: str, titles: list[str]):
    """Create multiple conversations at once"""
    async with data_db_sqlalchemy.get_session() as session:
        try:
            # Start transaction
            async with session.begin():
                conversations = []
                for title in titles:
                    conv = Conversation(user_id=user_id, title=title)
                    session.add(conv)
                    conversations.append(conv)
                
                # Flush to get IDs
                await session.flush()
                
                # Manual commit (happens automatically on context exit)
                
                return {
                    "created": len(conversations),
                    "ids": [conv.id for conv in conversations]
                }
        except Exception as e:
            # Rollback happens automatically
            raise HTTPException(status_code=500, detail=f"Error creating conversations: {str(e)}")


# ============================================================================
# Example 8: Working with Auth Database
# ============================================================================

@router.get("/users/{email}")
async def get_user_by_email(email: str):
    """Get user from auth database"""
    async with auth_db_sqlalchemy.get_session() as session:
        # Build query with role eager loading
        stmt = (
            select(User)
            .where(User.email == email)
            .options(selectinload(User.roles).selectinload(Role))
        )
        
        # Execute query
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "is_active": user.is_active,
            "roles": [user_role.role.name for user_role in user.roles]
        }


# ============================================================================
# Example 9: Raw SQL Queries (when needed)
# ============================================================================

@router.get("/stats/conversations")
async def get_conversation_stats():
    """Get conversation statistics using raw SQL"""
    async with data_db_sqlalchemy.get_session() as session:
        from sqlalchemy import text
        
        # Execute raw SQL
        stmt = text("""
            SELECT 
                user_id,
                COUNT(*) as conversation_count,
                MAX(created_at) as last_activity
            FROM conversations
            GROUP BY user_id
            ORDER BY conversation_count DESC
            LIMIT 10
        """)
        
        result = await session.execute(stmt)
        rows = result.fetchall()
        
        return {
            "stats": [
                {
                    "user_id": row[0],
                    "conversation_count": row[1],
                    "last_activity": row[2].isoformat() if row[2] else None
                }
                for row in rows
            ]
        }


# ============================================================================
# Example 10: Pagination
# ============================================================================

@router.get("/conversations")
async def get_conversations_paginated(skip: int = 0, limit: int = 10):
    """Get conversations with pagination"""
    async with data_db_sqlalchemy.get_session() as session:
        # Build paginated query
        stmt = (
            select(Conversation)
            .offset(skip)
            .limit(limit)
            .order_by(Conversation.created_at.desc())
        )
        
        # Execute query
        result = await session.execute(stmt)
        conversations = result.scalars().all()
        
        return {
            "skip": skip,
            "limit": limit,
            "conversations": [
                {
                    "id": conv.id,
                    "user_id": conv.user_id,
                    "title": conv.title
                }
                for conv in conversations
            ]
        }
