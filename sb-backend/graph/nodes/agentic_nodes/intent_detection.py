"""
Intent Detection Node for LangGraph

This node detects the user's intent from their message using Ollama (small LLM).
It runs in parallel with message storage.
"""

from typing import Dict, Any
import json
import logging
import os
from config.logging_config import setup_logging, get_logger
import time
from graph.state import ConversationState

# Note: Configure Ollama connection details as needed
OLLAMA_BASE_URL = "http://194.164.151.158:11434"  # Default Ollama URL
OLLAMA_MODEL = "phi3:latest"  # Change to your preferred small model (e.g., "neural-chat", "orca-mini")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # Timeout in seconds (default 120s for inference)

logger = logging.getLogger(__name__)


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def intent_detection_node(state: ConversationState) -> Dict[str, Any]:
    """
    Detect intent from user message using Ollama.
    
    This node uses a small LLM (via Ollama) to classify the user's intent.
    Runs in parallel with message storage.
    
    Intent categories might include: greeting, complaint, praise, question, request, etc.
    
    Args:
        state: Current conversation state
    
    Returns:
        Dict with intent field and any error if occurred
    """
    try:
        user_message = state.user_message
        
        if not user_message or user_message.strip() == "":
            return {"error": "Empty user message for intent detection"}

        logger.info(
            "intent_detection: starting",
            extra={"ollama_base_url": OLLAMA_BASE_URL, "ollama_model": OLLAMA_MODEL}
        )

        intent = await detect_intent_with_ollama(user_message)
        
        return {
            "intent": intent
        }
        
    except Exception as e:
        logger.exception("intent_detection: failed")
        return {
            "error": f"Error detecting intent: {str(e)}"
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def detect_intent_with_ollama(message: str) -> str:
    """
    Call Ollama to detect intent from message.
    
    Args:
        message: User message to analyze
    
    Returns:
        Detected intent as string
    """
    try:
        import aiohttp
        
        prompt = f"""Analyze the following message and classify its intent. 
        
        Message: "{message}"
        
        Respond with ONLY the intent category (one word or short phrase) from these options:
        Intent Categories:
        GREETING, VENTING, SEEK_INFORMATION, SEEK_UNDERSTANDING, OPEN_TO_SOLUTION, TRY_TOOL, SEEK_SUPPORT, UNCLEAR
       
        based on these descriptions:
        - greeting: User is greeting or saying hello
        - venting: User wants to express and be heard
        - seek_information: User is asking for information or clarification
        - seek_understanding: User wants explanation or clarity
        - open_to_solution: User is open to suggestions or solutions
        - try_tool: User is requesting to use a tool or resource
        - seek_support: User is seeking emotional support or empathy
        - unclear: Other or unclear intent
    .  
        Response:"""
        
        timeout = aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)
        request_payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,  # Lower temperature for more focused responses
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info("intent_detection: calling ollama", extra={"url": f"{OLLAMA_BASE_URL}/api/generate", "timeout": OLLAMA_TIMEOUT})
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=request_payload,
            ) as resp:
                status = resp.status
                text_body = await resp.text()
                logger.info(
                    "intent_detection: ollama response",
                    extra={"status": status, "body_preview": text_body[:500]}
                )

                if status == 200:
                    data = json.loads(text_body)
                    response_text = data.get("response", "other").strip().lower()
                    
                    # Extract just the intent word
                    intent = response_text.split('\n')[0].strip()
                    
                    # Validate against known intents
                    valid_intents = [
                        "greeting", "venting", "seek_information", "seek_understanding",
                        "open_to_solution", "try_tool", "seek_support", "unclear"
                    ]
                    
                    if intent not in valid_intents:
                        logger.warning("intent_detection: invalid intent", extra={"intent": intent})
                        intent = "unclear"
                    
                    return intent

                logger.warning(
                    "intent_detection: non-200 response",
                    extra={"status": status}
                )
                return "unclear"

    except Exception:
        logger.exception("intent_detection: ollama call failed")
        return "unclear"


async def get_intent_description(intent: str) -> str:
    """
    Get a human-readable description of the intent.
    
    Args:
        intent: Intent category
    
    Returns:
        Description of the intent
    """
    descriptions = {
        "greeting": "User is greeting or saying hello",
        "venting": "User wants to express and be heard",
        "seek_information": "User is asking for information or clarification",
        "seek_understanding": "User wants explanation or clarity",
        "open_to_solution": "User is open to suggestions or solutions",
        "try_tool": "User is requesting to use a tool or resource",
        "seek_support": "User is seeking emotional support or empathy",
        "unclear": "Other or unclear intent"
    }
    
    return descriptions.get(intent, "Unknown intent")
