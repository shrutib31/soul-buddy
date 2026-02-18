from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import psycopg2
import os

from services.key_manager import get_key_manager
from graph.state import ConversationState
from graph.graph_builder import get_compiled_flow
from graph.streaming import stream_as_sse

router = APIRouter(prefix="/chat")


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None
    sb_conv_id: Optional[str] = None
    domain: str = "student"

class IncognitoChatRequest(ChatRequest):
    mode: str = "incognito"

class CognitoChatRequest(ChatRequest):
    mode: str = "cognito"
    user_id: Optional[str] = None


# ============================================================
# GRAPH
# ============================================================

_flow = None

async def get_flow():
    global _flow
    if _flow is None:
        _flow = get_compiled_flow()
    return _flow


def ensure_conversation_exists(conversation_id: str, mode: str) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO sb_conversations (id, mode, started_at)
            VALUES (%s::uuid, %s, NOW())
            ON CONFLICT (id) DO NOTHING
            """,
            (conversation_id, mode),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()
        
# ============================================================
# ENCRYPTION HELPERS
# ============================================================

def get_current_turn_index(conversation_id: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT COALESCE(MAX(turn_index), 0) FROM conversation_turns WHERE session_id = %s",
            (conversation_id,),
        )
        return cursor.fetchone()[0] + 1
    finally:
        cursor.close()
        conn.close()


async def encrypt_and_save_message(
    conversation_id: str,
    speaker: str,
    plaintext: str,
    turn_index: int
) -> str:
    km = get_key_manager()
    encrypted_content = await km.encrypt(conversation_id, plaintext)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO conversation_turns
                (session_id, turn_index, speaker, message, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING id
            """,
            (conversation_id, turn_index, speaker, encrypted_content),
        )
        turn_id = str(cursor.fetchone()[0])

        cursor.execute(
            """
            INSERT INTO encryption_audit_log
                (entity_type, operation, accessed_by_type, accessed_by_id, vault_key_name)
            VALUES (%s, %s, %s, %s, %s)
            """,
            ('message', 'encrypt', 'system', 'soulbuddy_bot', f'conversation_{conversation_id}'),
        )
        conn.commit()
        return turn_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


async def load_and_decrypt_messages(conversation_id: str) -> list:
    km = get_key_manager()

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, turn_index, speaker, message, created_at
            FROM conversation_turns
            WHERE session_id = %s
            ORDER BY turn_index ASC
            """,
            (conversation_id,),
        )

        messages = []
        for row in cursor.fetchall():
            turn_id, turn_index, speaker, encrypted_content, created_at = row
            try:
                plaintext = await km.decrypt(conversation_id, encrypted_content)
                messages.append({
                    'id': str(turn_id),
                    'turn_index': turn_index,
                    'speaker': speaker,
                    'message': plaintext,
                    'created_at': created_at.isoformat(),
                })
            except Exception:
                messages.append({
                    'id': str(turn_id),
                    'turn_index': turn_index,
                    'speaker': speaker,
                    'message': '[Decryption failed]',
                    'created_at': created_at.isoformat(),
                    'error': True,
                })

        cursor.execute(
            """
            INSERT INTO encryption_audit_log
                (entity_type, operation, accessed_by_type, accessed_by_id, vault_key_name)
            VALUES (%s, %s, %s, %s, %s)
            """,
            ('conversation', 'decrypt', 'system', 'soulbuddy_bot', f'conversation_{conversation_id}'),
        )
        conn.commit()
        return messages
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


# ============================================================
# GRAPH HELPERS
# ============================================================

async def invoke_graph(state: ConversationState) -> Dict[str, Any]:
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
) -> ConversationState:
    return ConversationState(
        conversation_id=conversation_id or "",
        mode=mode,
        domain=domain,
        user_message=message,
    )


# ============================================================
# INCOGNITO ENDPOINTS
# ============================================================

@router.post("/incognito/stream")
async def incognito_chat_stream(req: IncognitoChatRequest):
    try:
        state = await create_initial_state(req.message, "incognito", req.domain, req.sb_conv_id)

        async def event_stream():
            async for event_data in stream_as_sse(state):
                yield event_data

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Stream failed: {str(e)}"})


@router.post("/incognito")
async def incognito_chat(req: IncognitoChatRequest):
    try:
        state = await create_initial_state(req.message, "incognito", req.domain, req.sb_conv_id)
        result = await invoke_graph(state)
        return result.get("api_response", {"success": False, "error": "No response generated"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ============================================================
# COGNITO ENDPOINTS
# ============================================================

@router.post("/cognito/stream")
async def cognito_chat_stream(req: CognitoChatRequest):
    try:
        state = await create_initial_state(req.message, "cognito", req.domain, req.sb_conv_id)

        async def event_stream():
            async for event_data in stream_as_sse(state):
                yield event_data

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Stream failed: {str(e)}"})


@router.post("/cognito")
async def cognito_chat(req: CognitoChatRequest):
    try:
        if not req.sb_conv_id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "sb_conv_id is required for cognito mode"},
            )

        ensure_conversation_exists(req.sb_conv_id, "cognito")  # ← add this line

        turn_index = get_current_turn_index(req.sb_conv_id)

        await encrypt_and_save_message(req.sb_conv_id, "user", req.message, turn_index)

        state = await create_initial_state(req.message, "cognito", req.domain, req.sb_conv_id)
        result = await invoke_graph(state)
        bot_message = result.get("api_response", {}).get("message", "")

        await encrypt_and_save_message(req.sb_conv_id, "bot", bot_message, turn_index + 1)

        return result.get("api_response", {"success": False, "error": "No response generated"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ============================================================
# HISTORY ENDPOINT
# ============================================================

@router.get("/conversation/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    try:
        messages = await load_and_decrypt_messages(conversation_id)
        return {"success": True, "messages": messages}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})