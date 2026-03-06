from fastapi import APIRouter, Header, HTTPException, Depends
from api.supabase_auth import verify_supabase_token, optional_supabase_token
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from graph.state import ConversationState
from graph.graph_builder import get_compiled_flow
from graph.streaming import stream_as_sse
from graph.nodes.function_nodes.user_context_helpers import resolve_cognito_identity_from_access_token

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
    sb_conv_id: Optional[str] = None  # conversation id for cognito mode
    domain: Optional[str] = None  # "student", "employee", "corporate"
    domain_config: Optional[Dict[str, Any]] = None
    user_profile: Optional[Dict[str, Any]] = None
    user_personality_profiles: Optional[Dict[str, Any]] = None
    

class IncognitoChatRequest(ChatRequest):
    mode: str = "incognito"  # "incognito" or "cognito"
    sb_conv_id: Optional[str] = None  # conversation id for cognito mode
    domain: str = "student"  # "student", "employee", "corporate"
    upgrade_suggested: bool = False

class CognitoChatRequest(ChatRequest):
    mode: str = "cognito"  # "incognito" or "cognito"
    sb_conv_id: str = None #conversation id for cognito mode
    domain: str = "student"  # "student", "employee", "corporate"
    # supabase_uid is NOT accepted from the client — always derived from the verified JWT.


class UnifiedChatRequest(BaseModel):
    """
    Single request model for both incognito and cognito chat.

    Set is_incognito=True (default) for anonymous sessions — no Authorization
    header required.  Set is_incognito=False for authenticated sessions — an
    Authorization: Bearer <token> header must be present; the supabase_uid is
    always derived from the verified token, never accepted from the client.
    """
    message: str = Field(..., min_length=1, description="User input message")
    is_incognito: bool = Field(True, description="True for anonymous session, False for authenticated session")
    sb_conv_id: Optional[str] = None
    domain: str = "student"
    metadata: Optional[Dict[str, Any]] = None


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
    app_user_id: Optional[int] = None,
    domain_config: Optional[Dict[str, Any]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    user_personality_profiles: Optional[Dict[str, Any]] = None,
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
        app_user_id=app_user_id,
        domain_config=domain_config or {},
        user_profile=user_profile or {},
        user_personality_profile=user_personality_profiles or {},
    )


def extract_token_from_headers(access_token: Optional[str], authorization: Optional[str]) -> Optional[str]:
    """
    Resolve the auth token from headers.
    Priority: access_token header, then Authorization: Bearer <token>.
    """
    for raw_value in (access_token, authorization):
        value = (raw_value or "").strip()
        if not value:
            continue
        if value.lower().startswith("bearer "):
            value = value.split(" ", 1)[1].strip()
        if value:
            return value
    return None


def _resolve_supabase_uid(user: Any) -> Optional[str]:
    if isinstance(user, dict):
        return user.get("id")
    return getattr(user, "id", None)


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
            conversation_id=req.sb_conv_id,
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
            conversation_id=req.sb_conv_id,
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
async def cognito_chat_stream(
    req: CognitoChatRequest, user=Depends(verify_supabase_token),
    access_token: Optional[str] = Header(default=None, alias="access_token", convert_underscores=False),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    """
    Authenticated chat with streaming response using LangGraph.
    Returns server-sent events (SSE) with real-time graph updates.
    """
    supabase_uid: str = _resolve_supabase_uid(user)
    try:
        token = extract_token_from_headers(access_token, authorization)
        if not token:
            return JSONResponse(status_code=400, content={"error": "Missing access_token header for cognito mode"})

        try:
            supabase_uid, app_user_id = await resolve_cognito_identity_from_access_token(token)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

        # Create initial state
        state = await create_initial_state(
            message=req.message,
            mode="cognito",
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            supabase_uid=supabase_uid,
            app_user_id=app_user_id,
            domain_config=req.domain_config,
            user_profile=req.user_profile,
            user_personality_profiles=req.user_personality_profiles,
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
async def cognito_chat(
    req: CognitoChatRequest, user=Depends(verify_supabase_token),
    access_token: Optional[str] = Header(default=None, alias="access_token", convert_underscores=False),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    """
    Authenticated chat with complete response.
    Returns the full response at once.
    """
    supabase_uid: str = _resolve_supabase_uid(user)
    try:
        token = extract_token_from_headers(access_token, authorization)
        if not token:
            return JSONResponse(status_code=400, content={"success": False, "error": "Missing access_token header for cognito mode"})

        try:
            supabase_uid, app_user_id = await resolve_cognito_identity_from_access_token(token)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"success": False, "error": str(e)})

        # Create initial state
        state = await create_initial_state(
            message=req.message,
            mode="cognito",
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            supabase_uid=supabase_uid,
            app_user_id=app_user_id,
            domain_config=req.domain_config,
            user_profile=req.user_profile,
            user_personality_profiles=req.user_personality_profiles,
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

    supabase_uid: Optional[str] = None if req.is_incognito else _resolve_supabase_uid(user)
    mode = "incognito" if req.is_incognito else "cognito"
    try:
        state = await create_initial_state(
            message=req.message,
            mode=mode,
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            supabase_uid=supabase_uid,
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

    supabase_uid: Optional[str] = None if req.is_incognito else _resolve_supabase_uid(user)
    mode = "incognito" if req.is_incognito else "cognito"
    try:
        state = await create_initial_state(
            message=req.message,
            mode=mode,
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            supabase_uid=supabase_uid,
        )

        async def event_stream():
            async for event_data in stream_as_sse(state):
                yield event_data

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat stream failed: {str(e)}"})
