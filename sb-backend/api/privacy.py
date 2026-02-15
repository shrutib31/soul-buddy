# API endpoint for PII and PHI Data

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Dict, Any, Optional
import asyncio
import traceback
import sys
import os

# Add project root to path to ensure imports work if running standalone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the actual privacy node logic
# We wrap this in a try/except to prevent the whole API from crashing 
# if the path to the graph node is slightly different in your setup.
try:
    from graph.nodes.function_nodes.privacy_masking import privacy_masking_node
except ImportError:
    print("⚠️ WARNING: Could not import privacy_masking_node. Check your file paths.")
    privacy_masking_node = None

router = APIRouter(prefix="/privacy")


# --- DATA MODELS ---

class PrivacyMaskingRequest(BaseModel):
    """
    Simple request model ensuring we get a non-empty string.
    """
    message: str = Field(
        ..., 
        min_length=1, 
        example="My name is Sarah and my MRN is 99281.", 
        description="The raw user input containing potential PII/PHI."
    )


# --- MOCK STATE ---

@dataclass
class MockState:
    """
    A lightweight fake State object.
    
    Why do we need this? 
    The 'privacy_masking_node' in LangGraph expects a specific State object 
    (ConversationState) with a '.user_message' attribute. 
    
    Instead of instantiating the entire DB-connected State, we just pass 
    this simple object that 'looks' like the real state (Duck Typing).
    """
    user_message: str


# --- ENDPOINT ---

@router.post("/")
async def mask_sensitive_data(req: PrivacyMaskingRequest):
    """
    **Test the PII/PHI Privacy Masking Logic.**
    
    This endpoint isolates the `privacy_masking_node` to verify exactly what data 
    is being stripped before it reaches the LLM. It uses Microsoft Presidio 
    under the hood.

    ---
    ### Example Request
    ```json
    {
      "message": "Call me at 555-0199. My medical ID is MRN:8812."
    }
    ```

    ### Example Response
    ```json
    {
      "success": true,
      "original_message": "Call me at 555-0199. My medical ID is MRN:8812.",
      "masked_message": "Call me at <PHONE>. My medical ID is <MRN_ID>.",
      "changes_made": true
    }
    ```
    """
    
    # 1. Safety Check: Ensure the node logic was imported correctly
    if privacy_masking_node is None:
        return JSONResponse(
            status_code=500,
            content={"error": "The privacy_masking_node could not be loaded on startup."}
        )

    try:
        # 2. Setup the Input
        # We wrap the string in our MockState to mimic the LangGraph flow
        input_state = MockState(user_message=req.message)

        # 3. Run the Node (Async)
        # We run this in a thread because text analysis (Presidio/Spacy) is CPU-heavy.
        # This prevents blocking the main FastAPI event loop.
        node_output = await asyncio.to_thread(
            privacy_masking_node, 
            input_state
        )
        
        # 4. Extract Result
        # The node returns a dict update: {'user_message': '<MASKED_TEXT>'}
        masked_text = node_output.get("user_message", req.message)
        
        # 5. Return JSON
        return {
            "success": True,
            "original_message": req.message,
            "masked_message": masked_text,
            "changes_made": req.message != masked_text, # Boolean flag to easily spot if PII was found
            "node_used": "privacy_masking_node"
        }

    except Exception as e:
        # 6. Error Handling
        # If Presidio crashes, we catch it here and return a 500
        error_msg = str(e)
        tb = traceback.format_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": req.message,
                "error": f"Privacy masking failed: {error_msg}",
                "debug_info": tb
            }
        )