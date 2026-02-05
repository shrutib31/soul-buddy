"""
Response Generator Node for LangGraph

This node generates responses using both Ollama and GPT-4-mini to compare
which provides better responses. It creates parallel response generation
and returns both for comparison/evaluation.
"""

from typing import Dict, Any, Optional
import os
import asyncio
import logging

from graph.state import ConversationState

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://194.164.151.158:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # Timeout in seconds (default 120s for inference)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

logger = logging.getLogger(__name__)


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def response_generator_node(state: ConversationState) -> Dict[str, Any]:
    """
    Generate responses using both Ollama and GPT-4-mini in parallel.
    
    This node generates compassionate responses using two different LLM sources
    to compare their quality. Both responses are generated concurrently to minimize
    total latency. Both responses are stored in the state for evaluation/selection.
    
    Args:
        state: Current conversation state
    
    Returns:
        Dict with ollama_response, gpt_response, and selected_response
    """
    try:
        user_message = state.user_message
        situation = state.situation
        severity = state.severity
        intent = state.intent
        response_draft = state.response_draft
        
        if not user_message:
            return {"error": "Missing user message for response generation"}
        
        logger.info(
            "response_generator: starting",
            extra={"intent": intent, "situation": situation, "severity": severity}
        )
        
        # Generate responses from both sources IN PARALLEL using asyncio.gather
        # This reduces total execution time from ~5-7s (sequential) to ~2-5s (parallel)
        ollama_response, gpt_response = await asyncio.gather(
            generate_response_ollama(
                user_message, situation, severity, intent, response_draft
            ),
            generate_response_gpt(
                user_message, situation, severity, intent, response_draft
            ),
            return_exceptions=False  # If either fails, exception will be raised
        )
        
        # For now, default to GPT response (can be changed to Ollama or use a selector node)
        selected_response = gpt_response if gpt_response else ollama_response
        
        logger.info(
            "response_generator: completed",
            extra={
                "used_gpt": bool(gpt_response),
                "gpt_length": len(gpt_response) if gpt_response else 0,
                "ollama_length": len(ollama_response) if ollama_response else 0
            }
        )
        
        return {
            "response_draft": selected_response,
            "ollama_response": ollama_response,
            "gpt_response": gpt_response,
        }
        
    except Exception:
        logger.exception("response_generator: failed")
        return {
            "error": "Error generating response"
        }


# ============================================================================
# OLLAMA RESPONSE GENERATION
# ============================================================================

async def generate_response_ollama(
    user_message: str,
    situation: Optional[str] = None,
    severity: Optional[str] = None,
    intent: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Generate a compassionate response using Ollama.
    
    Args:
        user_message: The user's message
        situation: The identified situation/issue
        severity: Severity level (low, medium, high)
        intent: User's intent
        context: Additional context or draft
    
    Returns:
        Generated response string
    """
    try:
        import aiohttp
        
        # Build context-aware prompt
        context_info = ""
        if situation:
            context_info += f"\nSituation identified: {situation}"
        if severity:
            context_info += f"\nSeverity: {severity}"
        if intent:
            context_info += f"\nUser intent: {intent}"
        
        prompt = f"""You are a compassionate mental health support chatbot. 
Your role is to provide empathetic, supportive responses that validate the user's feelings.

User message: "{user_message}"{context_info}

Guidelines:
- Be warm, empathetic, and non-judgmental
- Validate their feelings and experiences
- Ask clarifying questions if needed
- Offer practical support or resources when appropriate
- Keep response concise (2-3 sentences)
- Avoid being prescriptive or dismissive

Compassionate response:"""
        
        async with aiohttp.ClientSession() as session:
            logger.info("generate_response_ollama: calling ollama", extra={"model": OLLAMA_MODEL})
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                },
                timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = data.get("response", "").strip()
                    logger.info("generate_response_ollama: success", extra={"length": len(response)})
                    return response if response else ""
                else:
                    logger.warning("generate_response_ollama: non-200 response", extra={"status": resp.status})
                    return ""
                    
    except Exception:
        logger.exception("generate_response_ollama: failed")
        return ""


# ============================================================================
# GPT-4-MINI RESPONSE GENERATION
# ============================================================================

async def generate_response_gpt(
    user_message: str,
    situation: Optional[str] = None,
    severity: Optional[str] = None,
    intent: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Generate a compassionate response using GPT-4-mini.
    
    Args:
        user_message: The user's message
        situation: The identified situation/issue
        severity: Severity level (low, medium, high)
        intent: User's intent
        context: Additional context or draft
    
    Returns:
        Generated response string
    """
    try:
        if not OPENAI_API_KEY:
            print("OpenAI API key not configured")
            return ""
        
        import aiohttp
        import json
        
        # Build context-aware prompt
        context_info = ""
        if situation:
            context_info += f"\nSituation identified: {situation}"
        if severity:
            context_info += f"\nSeverity: {severity}"
        if intent:
            context_info += f"\nUser intent: {intent}"
        
        messages = [
            {
                "role": "system",
                "content": """You are a compassionate mental health support chatbot.
Your role is to provide empathetic, supportive responses that validate the user's feelings.

Guidelines:
- Be warm, empathetic, and non-judgmental
- Validate their feelings and experiences
- Ask clarifying questions if needed
- Offer practical support or resources when appropriate
- Keep response concise (2-3 sentences)
- Avoid being prescriptive or dismissive"""
            },
            {
                "role": "user",
                "content": f"User message: \"{user_message}\"{context_info}\n\nProvide a compassionate response:"
            }
        ]
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 200,
        }
        
        if not OPENAI_API_KEY:
            logger.warning("generate_response_gpt: OPENAI_API_KEY not configured")
            return ""

        async with aiohttp.ClientSession() as session:
            logger.info("generate_response_gpt: calling openai", extra={"model": "gpt-4-mini"})
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    logger.info("generate_response_gpt: success", extra={"length": len(response)})
                    return response if response else ""
                else:
                    error_text = await resp.text()
                    logger.error(
                        "generate_response_gpt: api error",
                        extra={"status": resp.status, "error": error_text[:300]}
                    )
                    return ""
                    
    except Exception:
        logger.exception("generate_response_gpt: failed")


# ============================================================================
# RESPONSE COMPARISON UTILITIES
# ============================================================================

async def compare_responses(
    ollama_response: str,
    gpt_response: str,
    user_message: str
) -> Dict[str, Any]:
    """
    Compare two responses for quality metrics.
    
    Could be extended to use LLM-based evaluation.
    
    Args:
        ollama_response: Response from Ollama
        gpt_response: Response from GPT
        user_message: Original user message for context
    
    Returns:
        Comparison metrics
    """
    return {
        "ollama_length": len(ollama_response),
        "gpt_length": len(gpt_response),
        "ollama_available": bool(ollama_response),
        "gpt_available": bool(gpt_response),
    }


async def select_best_response(
    ollama_response: str,
    gpt_response: str,
    preference: str = "gpt"  # "gpt", "ollama", or "longer"
) -> str:
    """
    Select the best response based on preference.
    
    Args:
        ollama_response: Response from Ollama
        gpt_response: Response from GPT
        preference: Selection strategy
    
    Returns:
        Selected response
    """
    if preference == "gpt":
        return gpt_response if gpt_response else ollama_response
    elif preference == "ollama":
        return ollama_response if ollama_response else gpt_response
    elif preference == "longer":
        return gpt_response if len(gpt_response) >= len(ollama_response) else ollama_response
    else:
        return gpt_response if gpt_response else ollama_response
