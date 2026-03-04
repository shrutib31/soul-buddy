from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import asyncio
import uuid
import json

from graph.state import ConversationState
from graph.graph_builder import get_compiled_flow
from graph.streaming import stream_response_words, stream_as_sse
from graph.nodes.function_nodes.get_messages import get_conversation_messages
from graph.nodes.function_nodes.get_bot_response import get_latest_bot_response

router = APIRouter(prefix="/chat")

# Initialize the compiled LangGraph
flow = None


async def get_flow():
    """Get or initialize the compiled graph flow."""
    global flow
    if flow is None:
        flow = get_compiled_flow()
    return flow

async def ensure_conversation_exists(conversation_id: str, mode: str) -> None:
    """
    Create a row in sb_conversations if it doesn't already exist.
    Required before any inserts into conversation_turns due to FK constraint.
    Uses SQLAlchemy so no separate psycopg2 connection is needed.
    """
    from config.sqlalchemy_db import get_data_db
    from orm.models import SbConversation
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    data_db = get_data_db()
    async with data_db.get_session() as session:
        stmt = pg_insert(SbConversation).values(
            id=conversation_id,
            mode=mode,
        ).on_conflict_do_nothing(index_elements=["id"])
        await session.execute(stmt)
        await session.commit()


# ============================================================================
# REQUEST MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User input message")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata (client info, tags, etc.)"
    )
    sb_conv_id: str = None #conversation id for cognito mode
    domain: str = "student"  # "student", "employee", "corporate"
    

class IncognitoChatRequest(ChatRequest):
    mode: str = "incognito"  # "incognito" or "cognito"
    sb_conv_id: str = None #conversation id for cognito mode
    domain: str = "student"  # "student", "employee", "corporate"
    upgrade_suggested: bool = False

class CognitoChatRequest(ChatRequest):
    mode: str = "cognito"  # "incognito" or "cognito"
    sb_conv_id: str = None #conversation id for cognito mode
    user_id: str = None  # user identifier for cognito mode. This is a supabase user ID in string format
    domain: str = "student"  # "student", "employee", "corporate"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def invoke_graph(state: ConversationState) -> Dict[str, Any]:
    """
    Invoke the LangGraph with the given conversation state.
    
    Args:
        state: ConversationState with user message and metadata
    
    Returns:
        Final state from the graph
    """
    try:
        flow = await get_flow()
        result = await flow.ainvoke(state.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")


async def create_initial_state(
    message: str,
    mode: str,
    domain: str,
    conversation_id: Optional[str] = None
) -> ConversationState:
    """
    Create the initial conversation state for the graph.
    
    Args:
        message: User message
        mode: "incognito" or "cognito"
        domain: "student", "employee", or "corporate"
        conversation_id: Optional existing conversation ID
    
    Returns:
        ConversationState ready for graph invocation
    """
    print(f"\nReturned Conversation State:\n conv_id: {conversation_id},\n mode: {mode},\n domain: {domain},\n message: {message}\n")
    return ConversationState(
        conversation_id=conversation_id or "",  # Empty string triggers ID generation
        mode=mode,
        domain=domain,
        user_message=message,
    )


# ============================================================================
# INCOGNITO ENDPOINTS (no DB storage)
# ============================================================================

@router.post("/incognito/stream")
async def incognito_chat_stream(req: IncognitoChatRequest):
    """
    Anonymous chat with streaming response using LangGraph.
    Returns server-sent events (SSE) with real-time graph updates.
    """
    try:
        # Create initial state
        state = await create_initial_state(
            message=req.message,
            mode="incognito",
            domain=req.domain,
            conversation_id=req.sb_conv_id
        )
        
        # Stream from graph
        async def event_stream():
            async for event_data in stream_as_sse(state):
                yield event_data
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Chat stream failed: {str(e)}"}
        )


@router.post("/incognito")
async def incognito_chat(req: IncognitoChatRequest):
    """
    Anonymous chat with complete response.
    Returns the full response at once.
    """
    try:
        # Create initial state
        state = await create_initial_state(
            message=req.message,
            mode="incognito",
            domain=req.domain,
            conversation_id=req.sb_conv_id
        )
        
        # Invoke the graph
        result = await invoke_graph(state)
        
        # Return the API response from render node
        return result.get("api_response", {
            "success": False,
            "error": "No response generated"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Chat failed: {str(e)}"}
        )


# ============================================================================
# COGNITO ENDPOINTS 
# ============================================================================

@router.post("/cognito/stream")
async def cognito_chat_stream(req: CognitoChatRequest):
    """
    Authenticated chat with streaming response using LangGraph.
    Returns server-sent events (SSE) with real-time graph updates.
    """
    try:
        if not req.sb_conv_id:
            return JSONResponse(
                status_code=400,
                content={"error": "sb_conv_id is required for cognito mode"}
            )

        # Ensure parent row exists before graph runs 
        await ensure_conversation_exists(req.sb_conv_id, "cognito")

        # Create initial state
        state = await create_initial_state(
            message=req.message,
            mode="cognito",
            domain=req.domain,
            conversation_id=req.sb_conv_id
        )
        
        # Stream from graph
        # Graph handles: store_message (encrypted) → classify → generate → store_bot_response (encrypted) → render
        async def event_stream():
            async for event_data in stream_as_sse(state):
                yield event_data
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Chat stream failed: {str(e)}"}
        )


@router.post("/cognito")
async def cognito_chat(req: CognitoChatRequest):
    """
    Authenticated chat with complete response.
    Returns the full response at once.
    """
    try:
        if not req.sb_conv_id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "sb_conv_id is required for cognito mode"}
            )

        # Ensure parent row exists before graph runs (FK constraint on conversation_turns)
        await ensure_conversation_exists(req.sb_conv_id, "cognito")

        # Create initial state
        state = await create_initial_state(
            message=req.message,
            mode="cognito",
            domain=req.domain,
            conversation_id=req.sb_conv_id
        )
        
        # Invoke the graph
        # Graph handles: store_message (encrypted) → classify → generate → store_bot_response (encrypted) → render
        result = await invoke_graph(state)
        
        # Return the API response from render node
        return result.get("api_response", {
            "success": False,
            "error": "No response generated"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Chat failed: {str(e)}"}
        )


# ============================================================================
# GET MESSAGE ENDPOINTS
# ============================================================================

@router.get("/conversation/{conversation_id}/messages")
async def get_messages(conversation_id: str):
    """
    Retrieve and decrypt all messages for a cognito conversation.
    Plaintext messages (legacy/incognito) are returned as-is.
    """
    try:
        messages = await get_conversation_messages(conversation_id)
        return {
            "success": True,
            "conversation_id": conversation_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@router.get("/conversation/{conversation_id}/bot-response/latest")
async def get_latest_bot_response_endpoint(conversation_id: str):
    """
    Retrieve and decrypt the latest bot response for a cognito conversation.
    Plaintext messages (legacy/incognito) are returned as-is.
    """
    try:
        response = await get_latest_bot_response(conversation_id)
        if response is None:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "No bot response found for this conversation"}
            )
        return {
            "success": True,
            "conversation_id": conversation_id,
            "bot_response": response
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )