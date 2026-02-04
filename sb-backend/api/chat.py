from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import asyncio

router = APIRouter(prefix="/chat")

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

@router.post("/incognito/stream")
async def incognito_chat_stream(req: IncognitoChatRequest):
    """
    Anonymous chat with streaming response.
    Returns server-sent events (SSE) stream.
    """
    async def event_stream():
        dummy_response = f"Thank you for your message: '{req.message}'. This is a dummy streaming response in incognito mode."
        for word in dummy_response.split():
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/incognito")
async def incognito_chat(req: IncognitoChatRequest):
    """
    Anonymous chat with complete response.
    Returns the full response at once.
    """
    response = f"Thank you for your message: '{req.message}'. This is a dummy response in incognito mode."
    return {"message": response}


@router.post("/cognito/stream")
async def cognito_chat_stream(req: CognitoChatRequest):
    """
    Authenticated chat with streaming response.
    Returns server-sent events (SSE) stream.
    """
    # auth / identity resolution would go here
    async def event_stream():
        dummy_response = f"Hello user {req.user_id}! You said: '{req.message}'. This is a dummy streaming response in cognito mode."
        for word in dummy_response.split():
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/cognito")
async def cognito_chat(req: CognitoChatRequest):
    """
    Authenticated chat with complete response.
    Returns the full response at once.
    """
    # auth / identity resolution would go here
    response = f"Hello user {req.user_id}! You said: '{req.message}'. This is a dummy response in cognito mode."
    return {"message": response}
