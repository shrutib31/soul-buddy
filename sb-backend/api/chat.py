from fastapi import APIRouter, Depends, HTTPException
from api.supabase_auth import verify_supabase_token, optional_supabase_token
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any
import asyncio
import uuid
import json
import logging

from graph.state import ConversationState
from graph.graph_builder import get_compiled_flow
from graph.streaming import stream_response_words, stream_as_sse

router = APIRouter(prefix="/chat")
logger = logging.getLogger(__name__)

# Initialize the compiled LangGraph
flow = None


async def get_flow():
    """Get or initialize the compiled graph flow."""
    global flow
    if flow is None:
        flow = get_compiled_flow()
    return flow


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
    supabase_uid: str = None  # supabase user ID (cognito mode only)
    domain: str = "student"  # "student", "employee", "corporate"


class UnifiedChatRequest(BaseModel):
    """
    Single request model for both incognito and cognito chat.

    Set is_incognito=True (default) for anonymous sessions — no token or
    supabase_uid required.  Set is_incognito=False for authenticated sessions —
    an Authorization: Bearer <token> header and supabase_uid must be provided.
    """
    message: str = Field(..., min_length=1, description="User input message")
    is_incognito: bool = Field(True, description="True for anonymous session, False for authenticated session")
    sb_conv_id: Optional[str] = None
    supabase_uid: Optional[str] = None  # required when is_incognito=False
    domain: str = "student"
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def supabase_uid_required_for_cognito(self) -> "UnifiedChatRequest":
        if not self.is_incognito and not self.supabase_uid:
            raise ValueError("supabase_uid is required when is_incognito is False")
        return self


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
    conversation_id: Optional[str] = None,
    supabase_uid: Optional[str] = None,
) -> ConversationState:
    """
    Create the initial conversation state for the graph.

    Args:
        message: User message
        mode: "incognito" or "cognito"
        domain: "student", "employee", or "corporate"
        conversation_id: Optional existing conversation ID
        supabase_uid: Optional supabase user ID (cognito mode only)

    Returns:
        ConversationState ready for graph invocation
    """
    logger.debug(
        "Returned Conversation State | conv_id=%s mode=%s domain=%s message=%s",
        conversation_id,
        mode,
        domain,
        message,
    )
    return ConversationState(
        conversation_id=conversation_id or "",  # Empty string triggers ID generation
        mode=mode,
        domain=domain,
        user_message=message,
        supabase_uid=supabase_uid,
    )


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


@router.post("/cognito/stream")
async def cognito_chat_stream(req: CognitoChatRequest, user=Depends(verify_supabase_token)):
    """
    Authenticated chat with streaming response using LangGraph.
    Returns server-sent events (SSE) with real-time graph updates.
    """
    try:
        # Create initial state
        state = await create_initial_state(
            message=req.message,
            mode="cognito",
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            supabase_uid=req.supabase_uid,
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


@router.post("/cognito")
async def cognito_chat(req: CognitoChatRequest, user=Depends(verify_supabase_token)):
    """
    Authenticated chat with complete response.
    Returns the full response at once.
    """
    try:
        # Create initial state
        state = await create_initial_state(
            message=req.message,
            mode="cognito",
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            supabase_uid=req.supabase_uid,
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
# UNIFIED ENDPOINTS  (incognito + cognito in one)
# ============================================================================

@router.post("")
async def chat(req: UnifiedChatRequest, user=Depends(optional_supabase_token)):
    """
    Unified chat endpoint — handles both incognito and cognito in a single route.

    Incognito  (is_incognito=True):  no Authorization header needed.
    Cognito    (is_incognito=False): Authorization: Bearer <token> + supabase_uid required.
    """
    if not req.is_incognito and user is None:
        raise HTTPException(status_code=401, detail="Authorization header required for cognito mode")

    mode = "incognito" if req.is_incognito else "cognito"
    try:
        state = await create_initial_state(
            message=req.message,
            mode=mode,
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            supabase_uid=None if req.is_incognito else req.supabase_uid,
        )
        result = await invoke_graph(state)
        return result.get("api_response", {"success": False, "error": "No response generated"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Chat failed: {str(e)}"})


@router.post("/stream")
async def chat_stream(req: UnifiedChatRequest, user=Depends(optional_supabase_token)):
    """
    Unified streaming chat endpoint — handles both incognito and cognito.

    Returns server-sent events (SSE).
    Incognito  (is_incognito=True):  no Authorization header needed.
    Cognito    (is_incognito=False): Authorization: Bearer <token> + supabase_uid required.
    """
    if not req.is_incognito and user is None:
        raise HTTPException(status_code=401, detail="Authorization header required for cognito mode")

    mode = "incognito" if req.is_incognito else "cognito"
    try:
        state = await create_initial_state(
            message=req.message,
            mode=mode,
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            supabase_uid=None if req.is_incognito else req.supabase_uid,
        )

        async def event_stream():
            async for event_data in stream_as_sse(state):
                yield event_data

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat stream failed: {str(e)}"})
