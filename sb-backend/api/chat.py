from fastapi import APIRouter, Depends, HTTPException
from api.supabase_auth import optional_supabase_token, verify_supabase_token
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
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
    domain: str
    metadata: Optional[Dict[str, Any]] = None
    chat_preference: str


class VeraMessage(BaseModel):
    role: str
    content: str = Field(..., min_length=1)


class VeraChatRequest(BaseModel):
    """
    Compatibility request shape for VERA-MH style chat clients.

    The wrapper defaults SoulBuddy-specific fields so VERA only needs to send
    its model/messages/stream/conversation_id payload.
    """
    model: str = Field("app", description="Client/model identifier from VERA-MH")
    messages: List[VeraMessage] = Field(..., min_length=1)
    stream: bool = Field(False, description="This wrapper currently supports non-streaming calls")
    conversation_id: Optional[str] = None
    domain: str = Field("general", description="Optional SoulBuddy domain override")
    chat_preference: str = Field("general", description="Optional SoulBuddy chat preference override")


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


def _latest_user_message(messages: List[VeraMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    raise HTTPException(status_code=400, detail="At least one user message is required")


async def create_initial_state(
    message: str,
    mode: str,
    domain: str,
    chat_preference: str,
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
        chat_preference=chat_preference,
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
            chat_preference=req.chat_preference,
        )
        logging.debug(
            "*****  Initial Conversation State | conv_id=%s mode=%s domain=%s message=%s chat_preference=%s *****",
            state.conversation_id, state.mode, state.domain, state.user_message, state.chat_preference,
        )
        result = await invoke_graph(state)
        return result.get("api_response", {"success": False, "error": "No response generated"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Chat failed: {str(e)}"})


@router.post("/vera")
async def vera_chat(req: VeraChatRequest):
    """
    VERA-MH compatibility wrapper around the normal chat graph.

    Accepts a messages-style request, forwards the latest user message through
    SoulBuddy, and returns only the reply plus the conversation id to reuse on
    the next turn.
    """
    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="VERA chat wrapper only supports stream=false. Use /api/v1/chat/stream for SSE.",
        )

    try:
        user_message = _latest_user_message(req.messages)
        state = await create_initial_state(
            message=user_message,
            mode="incognito",
            domain=req.domain,
            conversation_id=req.conversation_id,
            supabase_uid=None,
            chat_preference=req.chat_preference,
        )
        result = await invoke_graph(state)
        api_response = result.get("api_response") or {}
        reply = api_response.get("response", "")
        conversation_id = api_response.get("conversation_id") or state.conversation_id or req.conversation_id

        if not api_response.get("success") and not reply:
            raise HTTPException(
                status_code=500,
                detail=api_response.get("error", "No response generated"),
            )

        message = {"content": reply}
        message_id = api_response.get("message_id")
        if message_id:
            message["id"] = message_id

        return {
            "message": message,
            "conversation_id": conversation_id,
            "model": req.model,
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": {"content": ""},
                "conversation_id": req.conversation_id,
                "model": req.model,
                "error": f"VERA chat failed: {str(e)}",
            },
        )


@router.get("/conversations/messages")
async def get_all_conversations_messages(
    user=Depends(verify_supabase_token),
):
    """
    Retrieve all conversations and their decrypted messages for the authenticated user.

    Requires a valid Authorization: Bearer <token> header.
    Returns conversations ordered by start time (newest first), each with its full message list.
    """
    from graph.nodes.function_nodes.get_messages import get_all_user_conversations

    supabase_uid = user["id"]
    try:
        conversations = await get_all_user_conversations(supabase_uid)
        return {"conversations": conversations}
    except Exception as e:
        logger.error("get_all_conversations_messages failed | supabase_uid=%r error=%s", supabase_uid, e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to retrieve conversations: {str(e)}"})


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    _user=Depends(verify_supabase_token),
):
    """
    Retrieve all decrypted messages for a conversation.

    Requires a valid Authorization: Bearer <token> header.
    Returns 404 if the conversation does not exist or belongs to another user.
    Messages stored with encryption are transparently decrypted before returning.
    """
    if not _is_valid_uuid(conversation_id):
        raise HTTPException(status_code=400, detail="Invalid conversation_id — must be a UUID")

    supabase_uid = _user["id"]
    try:
        from graph.nodes.function_nodes.get_messages import get_conversation_messages as fetch_messages
        messages = await fetch_messages(conversation_id, supabase_uid=supabase_uid)
        return {"conversation_id": conversation_id, "messages": messages}
    except PermissionError:
        raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        logger.error(
            "get_conversation_messages failed | conversation_id=%r supabase_uid=%r error=%s",
            conversation_id,
            supabase_uid,
            e,
            exc_info=True,
        )
        return JSONResponse(status_code=500, content={"error": f"Failed to retrieve messages: {str(e)}"})


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
            chat_preference=req.chat_preference,
        )

        async def event_stream():
            async for event_data in stream_as_sse(state):
                yield event_data

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat stream failed: {str(e)}"})
