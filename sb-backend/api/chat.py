from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from graph.state import ConversationState
from config.supabase import get_user_by_id
from graph.graph_builder import get_compiled_flow
from graph.streaming import stream_response_words, stream_as_sse

router = APIRouter(prefix="/chat")

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
    user_id: str  # user identifier for cognito mode. This is a supabase user ID in string format
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
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None
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
        user_id=user_id,
    )


async def require_cognito_user_id(user_id: str) -> str:
    """
    Validate that the provided user_id exists in Supabase.
    """
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="Missing user_id for cognito mode")

    try:
        user = await get_user_by_id(user_id)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"User verification failed: {str(e)}")

    verified_user_id = getattr(user, "id", None)
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="User verification failed: missing id")

    return verified_user_id


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
async def cognito_chat_stream(req: CognitoChatRequest):
    """
    Authenticated chat with streaming response using LangGraph.
    Returns server-sent events (SSE) with real-time graph updates.
    """
    try:
        verified_user_id = await require_cognito_user_id(req.user_id)

        # Create initial state
        state = await create_initial_state(
            message=req.message,
            mode="cognito",
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            user_id=verified_user_id
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
async def cognito_chat(req: CognitoChatRequest):
    """
    Authenticated chat with complete response.
    Returns the full response at once.
    """
    try:
        verified_user_id = await require_cognito_user_id(req.user_id)

        # Create initial state
        state = await create_initial_state(
            message=req.message,
            mode="cognito",
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            user_id=verified_user_id
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

