from fastapi import APIRouter, Depends, HTTPException
from api.supabase_auth import optional_supabase_token, verify_supabase_token
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
    domain: str = Field("student", description="Conversation domain, defaults to 'student' for backward compatibility")
    metadata: Optional[Dict[str, Any]] = None
    chat_preference: str = Field("general", description="Chat preference, defaults to 'general' for backward compatibility")
    chat_mode: str = Field("default", description="Interaction mode: default | reflection | venting | therapist")
    language: str = Field("en-IN", description="BCP-47 language tag from Sarvam STT (e.g. en-IN, hi-IN, ta-IN)")


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
    chat_preference: str,
    chat_mode: str = "default",
    conversation_id: Optional[str] = None,
    supabase_uid: Optional[str] = None,
    language: str = "en-IN",
) -> ConversationState:
    # Reject non-UUID values (e.g. Swagger default "string") — treat as new session
    valid_conv_id = conversation_id if (conversation_id and _is_valid_uuid(conversation_id)) else None
    if conversation_id and not valid_conv_id:
        logger.warning("Invalid conversation_id ignored (not a UUID): %r", conversation_id)

    logger.debug(
        "Returned Conversation State | conv_id=%s mode=%s domain=%s message=%s chat_mode=%s",
        valid_conv_id, mode, domain, message, chat_mode,
    )
    return ConversationState(
        conversation_id=valid_conv_id or "",  # Empty string triggers ID generation
        mode=mode,
        domain=domain,
        user_message=message,
        supabase_uid=supabase_uid,
        chat_preference=chat_preference,
        chat_mode=chat_mode,
        language=language,
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
    logging.debug("***  supabase_uid: %s", supabase_uid) 
    try:
        state = await create_initial_state(
            message=req.message,
            mode=mode,
            domain=req.domain,
            conversation_id=req.sb_conv_id,
            supabase_uid=supabase_uid,
            chat_preference=req.chat_preference,
            chat_mode=req.chat_mode,
            language=req.language,
        )
        logging.debug(
            "*****  Initial Conversation State | conv_id=%s mode=%s domain=%s message=%s chat_preference=%s chat_mode=%s *****",
            state.conversation_id, state.mode, state.domain, state.user_message, state.chat_preference, state.chat_mode,
        )
        result = await invoke_graph(state)
        return result.get("api_response", {"success": False, "error": "No response generated"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Chat failed: {str(e)}"})


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
            chat_mode=req.chat_mode,
            language=req.language,
        )

        async def event_stream():
            async for event_data in stream_as_sse(state):
                yield event_data

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat stream failed: {str(e)}"})
