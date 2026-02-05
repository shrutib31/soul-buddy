"""
Render Node for LangGraph

This node takes the final response_draft and formats it into an API-ready response.
It prepares all the necessary data to be returned to the client.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from graph.state import ConversationState


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
        api_response = {
            "success": True,
            "conversation_id": state.conversation_id,
            "mode": state.mode,
            "domain": state.domain,
            "response": state.response_draft,
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
        
        # Check for errors
        if state.error:
            api_response["success"] = False
            api_response["error"] = state.error
        
        return {
            "api_response": api_response
        }
        
    except Exception as e:
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
