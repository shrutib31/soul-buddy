"""
Response Generator Node for LangGraph

This node generates responses using both Ollama and GPT-4o-mini to compare
which provides better responses. It creates parallel response generation
and returns both for comparison/evaluation.
"""

from typing import Dict, Any, Optional
import asyncio
import logging

from graph.state import ConversationState
from graph.nodes.agentic_nodes.response_templates import get_template_response, get_chat_preference_style
from graph.nodes.agentic_nodes.response_evaluator import select_best_response
from config.settings import settings

logger = logging.getLogger(__name__)

# Configuration — read once from centralised settings
OLLAMA_BASE_URL = settings.ollama.base_url
OLLAMA_MODEL = settings.ollama.model
OLLAMA_TIMEOUT = settings.ollama.timeout
OPENAI_API_KEY = settings.openai.api_key

COMPARE_RESULTS = settings.llm.compare_results
OLLAMA_FLAG = settings.llm.ollama_flag
OPENAI_FLAG = settings.llm.openai_flag

# Startup validation
if COMPARE_RESULTS:
    if not OLLAMA_BASE_URL:
        logger.warning("COMPARE_RESULTS=true but OLLAMA_BASE_URL is not set")
    if not OPENAI_API_KEY:
        logger.warning("COMPARE_RESULTS=true but OPENAI_API_KEY is not set")
elif not OLLAMA_FLAG and not OPENAI_FLAG:
    logger.warning("COMPARE_RESULTS=false and neither OLLAMA_FLAG nor OPENAI_FLAG is true — no LLM will be called")


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def response_generator_node(state: ConversationState) -> Dict[str, Any]:
    """
    Generate responses using both Ollama and GPT-4o-mini in parallel.
    
    This node generates compassionate responses using two different LLM sources
    to compare their quality. Both responses are generated concurrently to minimize
    total latency. Both responses are stored in the state for evaluation/selection.
    
    Tailor your response style according to chat_preference.
    
    Args:
        state: Current conversation state
    
    Returns:
        Dict with ollama_response, gpt_response, and selected_response
    """
    selected_chat_preference = state.chat_preference
    preference_style = get_chat_preference_style(selected_chat_preference)
    try:
        user_message = state.user_message
        situation = state.situation
        severity = state.severity
        intent = state.intent
        domain = state.domain
        response_draft = state.response_draft
        is_crisis_detected = state.is_crisis_detected
        is_greeting = state.is_greeting
        is_out_of_scope = getattr(state, "is_out_of_scope", False)
        chat_preference = preference_style

        if not user_message:
            return {"error": "Missing user message for response generation"}

        logger.info(
            "response_generator: starting",
            extra={
                "intent": intent,
                "situation": situation,
                "severity": severity,
                "is_crisis_detected": is_crisis_detected,
                "is_out_of_scope": is_out_of_scope,
                "is_greeting": is_greeting,
                "chat_preference": selected_chat_preference,
                "chat_preference_style": chat_preference,
            }
        )

        # Use a readymade template when crisis, out-of-scope, or greeting is explicitly detected.
        template = get_template_response(
            is_crisis_detected,
            is_greeting,
            domain,
            is_out_of_scope=is_out_of_scope,
        )
        if template:
            logger.info(
                "response_generator: using template response",
                extra={
                    "is_crisis_detected": is_crisis_detected,
                    "is_out_of_scope": is_out_of_scope,
                    "is_greeting": is_greeting,
                    "domain": domain,
                }
            )
            return {"response_draft": template}

        # No template — route to LLM(s) based on provider flags.
        args = (user_message, chat_preference, situation, severity, intent, response_draft)

        if COMPARE_RESULTS:
            # Call both in parallel and pick the best response.
            ollama_response, gpt_response = await asyncio.gather(
                generate_response_ollama(*args),
                generate_response_gpt(*args),
            )
            selected_response, source, ollama_score, gpt_score = select_best_response(
                ollama_response, gpt_response
            )
            logger.info(
                "response_generator: completed via comparison",
                extra={
                    "selected_source": source,
                    "ollama_score": round(ollama_score, 2),
                    "gpt_score": round(gpt_score, 2),
                    "intent": intent,
                }
            )
            return {
                "response_draft": selected_response,
                "api_response": {
                    "ollama": ollama_response,
                    "gpt": gpt_response,
                    "selected_source": source,
                    "ollama_score": round(ollama_score, 2),
                    "gpt_score": round(gpt_score, 2),
                }
            }

        # Single-provider or first-available mode.
        if OLLAMA_FLAG and OPENAI_FLAG:
            # Both enabled but no comparison — use first successful response.
            ollama_response, gpt_response = await asyncio.gather(
                generate_response_ollama(*args),
                generate_response_gpt(*args),
            )
            selected_response = ollama_response or gpt_response
            source = "ollama" if ollama_response else "openai"
        elif OLLAMA_FLAG:
            selected_response = await generate_response_ollama(*args)
            source = "ollama"
        elif OPENAI_FLAG:
            selected_response = await generate_response_gpt(*args)
            source = "openai"
        else:
            logger.error("response_generator: no LLM provider enabled")
            return {"error": "No LLM provider enabled — set COMPARE_RESULTS, OLLAMA_FLAG, or OPENAI_FLAG"}

        logger.info(
            "response_generator: completed via single provider",
            extra={"source": source, "length": len(selected_response) if selected_response else 0, "intent": intent}
        )
        return {"response_draft": selected_response or ""}
        
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
    chat_preference: str,
    situation: Optional[str] = None,
    severity: Optional[str] = None,
    intent: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    """
    Generate a compassionate response using Ollama.
    
    Args:
        user_message: The user's message
        chat_preference: Selected style of response
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
        if chat_preference:
            context_info += f"\nChat Preference: {chat_preference}"
        
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
-Tailor your response according to Chat Preference 

Compassionate response:"""
        
        async with aiohttp.ClientSession() as session:
            logger.info("generate_response_ollama: calling ollama", extra={"model": OLLAMA_MODEL, "url": OLLAMA_BASE_URL})
            try:
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
                        response = data.get("response", "").strip().strip('"').strip("'")
                        logger.info("generate_response_ollama: success", extra={"length": len(response)})
                        logger.debug(f"generate_response_ollama: full response: {response}")
                        return response if response else ""
                    else:
                        error_text = await resp.text()
                        logger.error(
                            "generate_response_ollama: non-200 response",
                            extra={"status": resp.status, "error": error_text[:200]}
                        )
                        return ""
            except asyncio.TimeoutError:
                logger.error("generate_response_ollama: timeout", extra={"timeout": OLLAMA_TIMEOUT})
                return ""
            except aiohttp.ClientConnectionError as e:
                logger.error("generate_response_ollama: connection error", extra={"error": str(e)[:200]})
                return ""
                    
    except Exception:
        logger.exception("generate_response_ollama: failed")
        return ""


# ============================================================================
# GPT-4o-MINI RESPONSE GENERATION
# ============================================================================

async def generate_response_gpt(
    user_message: str,
    chat_preference: str,
    situation: Optional[str] = None,
    severity: Optional[str] = None,
    intent: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Generate a compassionate response using GPT-4o-mini.
    
    Args:
        user_message: The user's message
        chat_preference: Selected style of response
        situation: The identified situation/issue
        severity: Severity level (low, medium, high)
        intent: User's intent
        context: Additional context or draft
    
    Returns:
        Generated response string
    """
    try:
        if not OPENAI_API_KEY:
            logger.debug("generate_response_gpt: OpenAI API key not configured")
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
        if chat_preference:
            context_info += f"\nChat Preference: {chat_preference}"
        
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
- Avoid being prescriptive or dismissive
-Tailor your response according to Chat Preference """
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
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 200,
        }
        
        if not OPENAI_API_KEY:
            logger.warning("generate_response_gpt: OPENAI_API_KEY not configured")
            return ""

        async with aiohttp.ClientSession() as session:
            logger.info("generate_response_gpt: calling openai", extra={"model": "gpt-4o-mini"})
            try:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().strip('"').strip("'")
                        logger.info("generate_response_gpt: success", extra={"length": len(response)})
                        logger.debug(f"generate_response_gpt: full response: {response}")
                        return response if response else ""
                    else:
                        error_text = await resp.text()
                        logger.error(
                            "generate_response_gpt: api error",
                            extra={"status": resp.status, "error": error_text[:500]}
                        )
                        return ""
            except asyncio.TimeoutError:
                logger.error("generate_response_gpt: timeout", extra={"timeout": 30})
                return ""
            except aiohttp.ClientConnectionError as e:
                logger.error("generate_response_gpt: connection error", extra={"error": str(e)[:200]})
                return ""
                    
    except Exception:
        logger.exception("generate_response_gpt: failed")
        return ""
