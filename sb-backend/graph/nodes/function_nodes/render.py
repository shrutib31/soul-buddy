"""
Render Node for LangGraph

This node takes the final response_draft and formats it into an API-ready response.
It prepares all the necessary data to be returned to the client.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

from graph.state import ConversationState

logger = logging.getLogger(__name__)


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def render_node(state: ConversationState) -> Dict[str, Any]:
    """
    Format the conversation state into an API response.
    
    Prepares the final response to be sent back to the client with all
    relevant conversation metadata and the generated response.
    
    Args:
        state: Current conversation state
    
    Returns:
        Dict with formatted API response data
    """
    try:
        # Determine success based on whether we have a response
        has_response = bool(state.response_draft and state.response_draft.strip())
        has_error = bool(state.error)
        
        api_response = {
            "success": has_response,  # Only success if we have a response
            "conversation_id": state.conversation_id,
            "mode": state.mode,
            "domain": state.domain,
            "response": state.response_draft or "",
            "metadata": {
                "intent": state.intent,
                "situation": state.situation,
                "severity": state.severity,
                "risk_level": state.risk_level,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }
        
        # Add optional fields if they exist
        if hasattr(state, 'ollama_response') and state.ollama_response:
            api_response["metadata"]["ollama_response"] = state.ollama_response
        
        if hasattr(state, 'gpt_response') and state.gpt_response:
            api_response["metadata"]["gpt_response"] = state.gpt_response
        
        # Add error to metadata if present
        if has_error:
            api_response["metadata"]["error"] = state.error
            api_response["error"] = state.error
        
        logger.info(
            "render: response",
            extra={
                "success": api_response["success"],
                "has_response": has_response,
                "has_error": has_error,
                "response_length": len(state.response_draft) if state.response_draft else 0
            }
        )
        
        return {
            "api_response": api_response
        }
        
    except Exception as e:
        logger.exception("render: failed to render response")
        error_response = {
            "success": False,
            "error": f"Error rendering response: {str(e)}",
            "conversation_id": state.conversation_id,
        }
        return {
            "api_response": error_response
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_api_response(
    conversation_id: str,
    response_text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format a standard API response structure.
    
    Args:
        conversation_id: The conversation ID
        response_text: The response to send
        metadata: Optional metadata to include
    
    Returns:
        Formatted API response
    """
    return {
        "success": True,
        "conversation_id": conversation_id,
        "response": response_text,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat(),
    }


def format_error_response(
    error_message: str,
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format an error API response.
    
    Args:
        error_message: The error message
        conversation_id: Optional conversation ID
    
    Returns:
        Formatted error response
    """
    response = {
        "success": False,
        "error": error_message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if conversation_id:
        response["conversation_id"] = conversation_id
    
    return response
