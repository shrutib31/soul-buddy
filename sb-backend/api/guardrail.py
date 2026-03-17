from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import asyncio

router = APIRouter(prefix="/guardrail")


class GuardrailRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User input message to evaluate")
    domain: str = Field("general", description="Conversation domain")


@router.post("")
async def guardrail_message(req: GuardrailRequest):
    try:
        from graph.nodes.agentic_nodes.guardrail import detect_out_of_scope

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
