"""
Tool Recommendation Trigger Gate Node for LangGraph

This node acts as a decision gate that determines whether to call the tool
recommendation service based on:
1. Global enable flag
2. User intent (only trigger on solution-seeking intents, not venting)
3. Safety checks (no crisis, not out of scope)

If all conditions pass, it calls run_tool_recommendation() and stores the result.
If conditions fail, it returns tool=None silently (flag = False).

Standard LangGraph Node Pattern:
    - Input: ConversationState (Pydantic model)
    - Output: Dict[str, Any] with state updates (only 'tool' field)
    - Always handle errors gracefully
    - Return only fields that need updating
"""

import logging
from typing import Dict, Any, Optional

from graph.state import ConversationState
from graph.nodes.function_nodes.tool_recommendation import run_tool_recommendation
from config.settings import settings

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION FLAGS
# ============================================================================

# Global enable/disable flag for tool recommendations
ENABLE_TOOL_RECOMMENDATION: bool = True

# Intent types that should trigger tool recommendation
# Only include intents where user is solution-seeking, NOT venting/ranting
TRIGGER_INTENTS: set = {
    "open_to_solution",  # User is open to taking action
    "try_tool",          # User explicitly wants to try a tool/worksheet
    "seek_help",         # User is asking for help/support
    "seek_support",      # User seeks emotional or practical support
}

# Intent types that should explicitly NOT trigger (venting, crisis, etc.)
BLOCK_INTENTS: set = {
    "venting",                # Just ranting, not seeking solution
    "greeting",               # Greeting message, no issue yet
    "unclear",                # Can't determine intent
    "crisis_disclosure",      # Crisis handled separately by guardrails
    "self_harm_disclosure",   # Crisis handled separately by guardrails
}


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def tool_trigger_gate_node(state: ConversationState) -> Dict[str, Any]:
    """
    LangGraph node that acts as a decision gate for tool recommendations.
    
    This node checks multiple conditions to decide whether the user would benefit
    from a tool recommendation. It only triggers when the user is actively
    seeking a solution (not just venting), and when no safety concerns exist.
    
    Conditions checked:
    1. ENABLE_TOOL_RECOMMENDATION flag is True
    2. Intent is in TRIGGER_INTENTS (solution-seeking)
    3. Intent is NOT in BLOCK_INTENTS (venting, crisis, etc.)
    4. is_crisis_detected is False (crisis handled separately)
    5. is_out_of_scope is False (guardrails handled separately)
    
    If all conditions pass:
    - Calls run_tool_recommendation() to fetch personalized recommendation
    - Stores tool object in state.tool
    - Logs the recommendation for audit
    
    If any condition fails:
    - Returns tool=None silently
    - Does NOT raise errors (graceful degradation)
    
    Args:
        state: Current conversation state (ConversationState)
    
    Returns:
        Dict with updated 'tool' field (either tool object from API or None)
    """
    
    # Early exit conditions (all log at DEBUG level, not errors)
    
    if not ENABLE_TOOL_RECOMMENDATION:
        logger.debug(
            "tool_trigger_gate: Global recommendation flag disabled",
            extra={"conversation_id": state.conversation_id}
        )
        return {"tool": None}
    
    # Read intent (default to "unclear" if not set yet)
    intent = state.intent or "unclear"
    
    if intent in BLOCK_INTENTS:
        logger.debug(
            "tool_trigger_gate: Intent in BLOCK_INTENTS, skipping recommendation",
            extra={
                "conversation_id": state.conversation_id,
                "intent": intent,
                "blocked_intents": list(BLOCK_INTENTS),
            }
        )
        return {"tool": None}
    
    if intent not in TRIGGER_INTENTS:
        logger.debug(
            "tool_trigger_gate: Intent not in TRIGGER_INTENTS, skipping recommendation",
            extra={
                "conversation_id": state.conversation_id,
                "intent": intent,
                "trigger_intents": list(TRIGGER_INTENTS),
            }
        )
        return {"tool": None}
    
    # Safety checks
    if state.is_crisis_detected:
        logger.debug(
            "tool_trigger_gate: Crisis detected, skipping recommendation",
            extra={"conversation_id": state.conversation_id}
        )
        return {"tool": None}
    
    is_out_of_scope = getattr(state, "is_out_of_scope", False)
    if is_out_of_scope:
        logger.debug(
            "tool_trigger_gate: Out of scope detected, skipping recommendation",
            extra={"conversation_id": state.conversation_id}
        )
        return {"tool": None}
    
    # All conditions passed — call tool recommendation service
    logger.info(
        "tool_trigger_gate: All conditions passed, calling tool_recommendation service",
        extra={
            "conversation_id": state.conversation_id,
            "intent": intent,
            "user_message": state.user_message[:100] if state.user_message else None,
        }
    )
    
    try:
        # Prepare inputs for tool_recommendation
        message = state.user_message
        personality = state.user_personality_profile or {}
        
        if not message:
            logger.warning(
                "tool_trigger_gate: Missing user_message, cannot generate recommendation",
                extra={"conversation_id": state.conversation_id}
            )
            return {"tool": None}
        
        # Call the tool recommendation service (async)
        recommendation = await run_tool_recommendation(
            message=message,
            personality=personality,
        )
        
        logger.info(
            "tool_trigger_gate: Recommendation generated successfully",
            extra={
                "conversation_id": state.conversation_id,
                "recommendation_length": len(recommendation) if recommendation else 0,
            }
        )
        
        # Return the recommendation (stored in state.tool as a string or structured object)
        # Note: If your tool_recommendation returns structured data, parse it accordingly
        return {"tool": recommendation}
    
    except Exception as e:
        logger.error(
            f"tool_trigger_gate: Failed to generate recommendation: {str(e)}",
            extra={
                "conversation_id": state.conversation_id,
                "error": str(e),
            },
            exc_info=True,
        )
        # Graceful degradation: if recommendation fails, just return None
        # (don't break the conversation flow)
        return {"tool": None}
