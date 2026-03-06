from fastapi import APIRouter, Depends, HTTPException
from api.supabase_auth import optional_supabase_token
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uuid
import logging

from graph.state import ConversationState
from graph.graph_builder import get_compiled_flow
from graph.streaming import stream_as_sse

router = APIRouter(prefix="/chat")
logger = logging.getLogger(__name__)

flow = None


async def get_flow():
    global flow
    if flow is None:
        flow = get_compiled_flow()
    return flow


class ChatRequest(BaseModel):
    """
    Request model for both incognito and cognito chat.

    Set is_incognito=True (default) for anonymous sessions — no Authorization
    header required.  Set is_incognito=False for authenticated sessions — an
    Authorization: Bearer <token> header must be present; supabase_uid is
    always derived from the verified token, never accepted from the client.
    """
    message: str = Field(..., min_length=1, description="User input message")
    is_incognito: bool = Field(True, description="True for anonymous session, False for authenticated session")
    sb_conv_id: Optional[str] = None
    domain: str = "student"
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# HELPERS
# ============================================================================

async def invoke_graph(state: ConversationState) -> Dict[str, Any]:
    try:
        flow = await get_flow()
        return await flow.ainvoke(state.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


async def create_initial_state(
    message: str,
    mode: str,
    domain: str,
    conversation_id: Optional[str] = None,
    supabase_uid: Optional[str] = None,
) -> ConversationState:
    # Reject non-UUID values (e.g. Swagger default "string") — treat as new session
    valid_conv_id = conversation_id if (conversation_id and _is_valid_uuid(conversation_id)) else None
    if conversation_id and not valid_conv_id:
        logger.warning("Invalid conversation_id ignored (not a UUID): %r", conversation_id)

    logger.debug(
        "Returned Conversation State | conv_id=%s mode=%s domain=%s message=%s",
        valid_conv_id, mode, domain, message,
    )
    return ConversationState(
        conversation_id=valid_conv_id or "",  # Empty string triggers ID generation
        mode=mode,
        domain=domain,
        user_message=message,
        supabase_uid=supabase_uid,
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("")
async def chat(req: ChatRequest, user=Depends(optional_supabase_token)):
    """
    Chat endpoint — handles both incognito and cognito in a single route.

    Incognito  (is_incognito=True):  no Authorization header needed.
    Cognito    (is_incognito=False): Authorization: Bearer <token> required.
    """
    if not req.is_incognito and user is None:
        raise HTTPException(status_code=401, detail="Authorization header required for cognito mode")

    mode = "incognito" if req.is_incognito else "cognito"
    supabase_uid: Optional[str] = None if req.is_incognito else user["id"]
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
async def chat_stream(req: ChatRequest, user=Depends(optional_supabase_token)):
    """
    Streaming chat endpoint — handles both incognito and cognito.

    Returns server-sent events (SSE).
    Incognito  (is_incognito=True):  no Authorization header needed.
    Cognito    (is_incognito=False): Authorization: Bearer <token> required.
    """
    if not req.is_incognito and user is None:
        raise HTTPException(status_code=401, detail="Authorization header required for cognito mode")

    mode = "incognito" if req.is_incognito else "cognito"
    supabase_uid: Optional[str] = None if req.is_incognito else user["id"]
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
