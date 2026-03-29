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
from graph.nodes.agentic_nodes.response_templates import get_template_response
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
        domain = state.domain
        response_draft = state.response_draft
        is_crisis_detected = state.is_crisis_detected
        is_greeting = state.is_greeting
        is_out_of_scope = getattr(state, "is_out_of_scope", False)
        crisis_category = getattr(state, "crisis_category", None)

        if not user_message:
            return {"error": "Missing user message for response generation"}

        logger.info(
            "response_generator: starting",
            extra={"intent": intent, "situation": situation, "severity": severity,
                   "is_crisis_detected": is_crisis_detected, "is_out_of_scope": is_out_of_scope,
                   "is_greeting": is_greeting}
        )

        # Use a readymade template when crisis, out-of-scope, or greeting is explicitly detected.
        template = get_template_response(
            is_crisis_detected,
            is_greeting,
            domain,
            is_out_of_scope=is_out_of_scope,
            crisis_category=crisis_category,
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
        history = state.conversation_history or []
        args = (user_message, situation, severity, intent, history)

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
# INTENT-AWARE PROMPT BUILDER
# ============================================================================

_INTENT_GUIDELINES = {
    "venting": (
        "The user is venting and needs emotional validation above all else.\n"
        "- Lead with empathy: acknowledge and name the emotion they expressed\n"
        "- DO NOT jump to advice or solutions — they haven't asked for any\n"
        "- Reflect back what you heard to show you understand\n"
        "- Gently invite them to share more if they want to\n"
        "- Keep it warm and short (2-4 sentences)"
    ),
    "seek_support": (
        "The user is seeking emotional support and connection.\n"
        "- Validate their feelings and make them feel heard\n"
        "- Reassure them that reaching out was the right thing to do\n"
        "- Be present and warm — avoid clinical language\n"
        "- Ask a gentle follow-up to show you care\n"
        "- Keep response supportive (3-5 sentences)"
    ),
    "open_to_solution": (
        "The user is open to practical advice or solutions.\n"
        "- Briefly validate their feelings first (1 sentence)\n"
        "- Then offer 1-2 concrete, actionable suggestions\n"
        "- Frame suggestions gently (\"you might try\" not \"you should\")\n"
        "- Keep it practical and encouraging (3-5 sentences)"
    ),
    "try_tool": (
        "The user wants a specific exercise, technique, or coping tool.\n"
        "- Suggest ONE specific, easy-to-follow exercise (breathing, grounding, journaling, etc.)\n"
        "- Give clear step-by-step instructions they can do right now\n"
        "- Keep the tone warm and encouraging\n"
        "- End by asking how it went or if they'd like another option (3-6 sentences)"
    ),
    "seek_information": (
        "The user wants to understand something about their mental health.\n"
        "- Provide clear, accurate psychoeducation in simple language\n"
        "- Normalise their experience where appropriate\n"
        "- Avoid being overly clinical or textbook-like\n"
        "- Keep it conversational and accessible (3-5 sentences)"
    ),
    "seek_understanding": (
        "The user is trying to make sense of their own feelings or behaviour.\n"
        "- Help them explore and understand their emotions without judging\n"
        "- Normalise their experience (\"that's a really common reaction\")\n"
        "- Ask reflective questions to deepen their self-understanding\n"
        "- Avoid over-intellectualising — keep it human (3-5 sentences)"
    ),
}

_SITUATION_CONTEXT = {
    "EXAM_ANXIETY": "They are dealing with exam-related stress or anxiety. Acknowledge the pressure without minimising it.",
    "ACADEMIC_COMPARISON": "They are comparing themselves to peers academically. Validate the feeling, remind them that everyone's path is different.",
    "RELATIONSHIP_ISSUES": "They are going through relationship difficulties. Be sensitive and avoid taking sides.",
    "FINANCIAL_STRESS": "They are stressed about money or finances. Acknowledge the real weight of financial pressure.",
    "HEALTH_CONCERNS": "They are worried about health issues. Be empathetic and encourage professional medical consultation if needed.",
    "BELONGING_DOUBT": "They feel like they don't belong or are isolated. Validate the pain of loneliness and remind them they're not alone.",
    "LOW_MOTIVATION": "They are struggling with low motivation or feeling stuck. Avoid pushing productivity — meet them where they are.",
    "FUTURE_UNCERTAINTY": "They are anxious about their future, career, or life direction. Normalise the uncertainty and help ground them.",
    "GENERAL_OVERWHELM": "They feel overwhelmed by everything. Acknowledge the weight without rushing to fix it.",
    "ANXIETY": "They are experiencing anxiety or panic. Be calming and grounding in your response.",
    "SLEEP_ISSUES": "They are struggling with sleep. Acknowledge how exhausting that is.",
    "BURNOUT": "They are experiencing burnout. Validate the exhaustion without pushing them to do more.",
    "GRIEF_LOSS": "They are grieving a loss. Be especially gentle — avoid silver linings or rushing through grief.",
}

_SEVERITY_GUIDANCE = {
    "high": (
        "This is a high-severity message — the user is in significant distress.\n"
        "- Prioritise emotional safety and validation\n"
        "- Take extra care with your words — avoid anything that could feel dismissive\n"
        "- If appropriate, gently encourage professional support\n"
        "- Respond with 4-6 thoughtful sentences"
    ),
    "medium": (
        "This is a moderate-severity message — the user is struggling but not in crisis.\n"
        "- Balance validation with gentle exploration\n"
        "- Respond with 3-5 sentences"
    ),
    "low": (
        "This is a lower-severity message — the user may be checking in or exploring.\n"
        "- Be warm and conversational\n"
        "- Respond with 2-3 sentences"
    ),
}


def _format_history(conversation_history) -> str:
    """Format conversation history into a readable block for the prompt."""
    if not conversation_history:
        return ""
    lines = []
    for turn in conversation_history:
        speaker = turn.get("speaker", "user")
        message = turn.get("message", "")
        if speaker == "user":
            lines.append(f"User: {message}")
        else:
            lines.append(f"SoulBuddy: {message}")
    return "\n".join(lines)


def _build_prompt(
    user_message: str,
    situation: Optional[str] = None,
    severity: Optional[str] = None,
    intent: Optional[str] = None,
    conversation_history: list = None,
) -> str:
    """Build an intent-aware, situation-specific prompt for the LLM."""
    intent_guide = _INTENT_GUIDELINES.get(intent, _INTENT_GUIDELINES["venting"])
    severity_guide = _SEVERITY_GUIDANCE.get(severity, _SEVERITY_GUIDANCE["low"])

    situation_line = ""
    if situation and situation not in ("NO_SITUATION", "unclear", "OTHER", None):
        situation_line = _SITUATION_CONTEXT.get(
            situation,
            f"Situation: {situation}. Respond with empathy appropriate to this context."
        )
        situation_line = f"\nSituation context: {situation_line}"

    history_block = ""
    if conversation_history:
        formatted = _format_history(conversation_history)
        history_block = (
            f"\nConversation so far:\n{formatted}\n\n"
            f"Continue the conversation naturally. Reference earlier messages "
            f"when relevant — don't repeat yourself or re-introduce topics "
            f"already discussed. Build on what has been shared.\n"
        )

    return (
        f"You are SoulBuddy, a compassionate mental wellness companion. "
        f"You are NOT a therapist — you are a warm, caring friend who listens without judgment.\n\n"
        f"{history_block}"
        f"User message: \"{user_message}\"\n"
        f"{situation_line}\n"
        f"Severity: {severity or 'low'}\n\n"
        f"Response approach (based on user's intent — {intent or 'unclear'}):\n"
        f"{intent_guide}\n\n"
        f"Severity guidance:\n"
        f"{severity_guide}\n\n"
        f"Core rules:\n"
        f"- NEVER say you are an AI, a chatbot, or a language model\n"
        f"- NEVER use phrases like \"as an AI\" or \"I'm not a therapist\"\n"
        f"- DO NOT force optimism or use toxic positivity\n"
        f"- DO NOT lecture, moralise, or compare them to others\n"
        f"- DO NOT give medical diagnoses or prescribe medication\n"
        f"- DO NOT repeat advice or exercises you already suggested in the conversation\n"
        f"- DO NOT assume or insert exam or academic stress context unless the user's message explicitly mentions exams, tests, assignments, or academic topics\n"
        f"- Always end with a gentle question or invitation to share more\n\n"
        f"Compassionate response:"
    )


# ============================================================================
# OLLAMA RESPONSE GENERATION
# ============================================================================

async def generate_response_ollama(
    user_message: str,
    situation: Optional[str] = None,
    severity: Optional[str] = None,
    intent: Optional[str] = None,
    conversation_history: list = None,
) -> str:
    """
    Generate a compassionate response using Ollama.
    
    Args:
        user_message: The user's message
        situation: The identified situation/issue
        severity: Severity level (low, medium, high)
        intent: User's intent
        conversation_history: Previous turns [{speaker, message}]
    
    Returns:
        Generated response string
    """
    try:
        import aiohttp
        
        # Build intent-aware, situation-specific prompt with conversation history
        prompt = _build_prompt(user_message, situation, severity, intent, conversation_history)
        
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
    situation: Optional[str] = None,
    severity: Optional[str] = None,
    intent: Optional[str] = None,
    conversation_history: list = None,
) -> str:
    """
    Generate a compassionate response using GPT-4o-mini.
    
    Args:
        user_message: The user's message
        situation: The identified situation/issue
        severity: Severity level (low, medium, high)
        intent: User's intent
        conversation_history: Previous turns [{speaker, message}]
    
    Returns:
        Generated response string
    """
    try:
        if not OPENAI_API_KEY:
            logger.debug("generate_response_gpt: OpenAI API key not configured")
            return ""
        import aiohttp
        import json
        
        prompt = _build_prompt(user_message, situation, severity, intent, conversation_history)
        
        messages = [
            {
                "role": "system",
                "content": prompt.split("User message:")[0].strip()
            },
            {
                "role": "user",
                "content": "User message:" + prompt.split("User message:", 1)[1]
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

