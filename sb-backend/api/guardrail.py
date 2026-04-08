from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import asyncio

from api.supabase_auth import verify_supabase_token

router = APIRouter(prefix="/guardrail")


class GuardrailRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User input message to evaluate")
    domain: str = Field("general", description="Conversation domain")


@router.post("")
async def guardrail_message(req: GuardrailRequest, _user=Depends(verify_supabase_token)):
    try:
        from graph.nodes.function_nodes.out_of_scope import detect_out_of_scope

        guardrail = await asyncio.to_thread(
            detect_out_of_scope,
            req.message,
            req.domain,
        )

        return {
            "success": True,
            "message": req.message,
            "guardrail": guardrail,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": req.message,
                "guardrail": {},
                "error": f"Guardrail failed: {str(e)}",
            },
        )
