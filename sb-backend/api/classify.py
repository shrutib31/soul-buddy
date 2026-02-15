from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import asyncio

router = APIRouter(prefix="/classify")


class ClassificationRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User input message to classify")


@router.post("")
async def classify_message(req: ClassificationRequest):
    """
    Classify a user message into intent, situation, severity, and risk categories.
    
    This endpoint uses the SoulBuddyClassifier model to analyze the message and
    return classification results without generating a response.
    
    Args:
        req: ClassificationRequest with the message to classify
    
    Returns:
        JSON response with classification results
    
    Example:
        POST /api/v1/classify
        {
            "message": "I'm feeling really stressed about my exams"
        }
        
        Response:
        {
            "success": true,
            "message": "I'm feeling really stressed about my exams",
            "classifications": {
                "intent": "venting",
                "situation": "academic_stress",
                "severity": "medium",
                "risk_score": 0.15
            }
        }
    """
    try:
        from graph.nodes.agentic_nodes.classification_node import get_classifications
        
        # Get classifications from the model (run in thread to avoid blocking)
        classifications = await asyncio.to_thread(
            get_classifications,
            req.message
        )
        
        return {
            "success": True,
            "message": req.message,
            "classifications": classifications
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": req.message,
                "classifications": {},
                "error": f"Classification failed: {error_msg}",
                "details": tb
            }
        )
