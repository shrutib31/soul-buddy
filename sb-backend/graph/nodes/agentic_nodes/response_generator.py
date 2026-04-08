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
from graph.nodes.agentic_nodes.response_templates import get_template_response, get_chat_preference_style, get_chat_mode_instructions
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
# CROSS-SESSION CONTEXT BUILDER
# ============================================================================

def _build_cross_session_context(
    user_memory: Optional[Dict[str, Any]],
    session_summary: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Build a compact cross-session context string injected into the LLM system
    prompt only on the first turn of a new session.

    Token budget: ~250 tokens total
      - growth_summary  (~100 tokens): evolving user narrative from user_memory
      - last session    (~150 tokens): key highlights from the previous session

    Returns None if neither source has useful content.
    """
    parts = []

    if user_memory:
        growth_summary = user_memory.get("growth_summary")
        if growth_summary:
            parts.append(f"[Your journey so far]\n{growth_summary}")

    if session_summary:
        # Prefer the richer holistic final_summary fields if present
        key_takeaways = session_summary.get("key_takeaways")
        session_story = session_summary.get("session_story")
        emotional_arc = session_summary.get("emotional_arc")

        last_session_lines = []
        if session_story:
            last_session_lines.append(session_story)
        if emotional_arc:
            last_session_lines.append(f"Emotional arc: {emotional_arc}")
        if key_takeaways:
            if isinstance(key_takeaways, list):
                last_session_lines.append("Key takeaways: " + "; ".join(key_takeaways))
            else:
                last_session_lines.append(f"Key takeaways: {key_takeaways}")

        if last_session_lines:
            parts.append("[Last session]\n" + "\n".join(last_session_lines))

    return "\n\n".join(parts) if parts else None


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
    chat_mode_instructions = get_chat_mode_instructions(state.chat_mode)
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
        out_of_scope_reason = getattr(state, "out_of_scope_reason", None)
        chat_preference = preference_style
        conversation_history = state.conversation_history or []

        # ── Cross-session context (token-optimised, first turn only) ─────────
        # Injected only when a new conversation has just started.
        # Injects at most ~250 tokens: growth_summary (~100) + last session (~150).
        # Mid-session turns use conversation_history exclusively — zero extra tokens.
        cross_session_context: Optional[str] = None
        if state.is_new_session:
            cross_session_context = _build_cross_session_context(
                user_memory=state.user_memory,
                session_summary=state.session_summary,
            )

        # Keep backward-compat variable name for passing into LLM helpers
        conversation_summary = cross_session_context

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
                "out_of_scope_reason": out_of_scope_reason,
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
            out_of_scope_reason=out_of_scope_reason,
        )
        if template:
            logger.info(
                "response_generator: using template response",
                extra={
                    "is_crisis_detected": is_crisis_detected,
                    "is_out_of_scope": is_out_of_scope,
                    "out_of_scope_reason": out_of_scope_reason,
                    "is_greeting": is_greeting,
                    "domain": domain,
                }
            )
            return {"response_draft": template}

        # No template — route to LLM(s) based on provider flags.
        # Therapist and reflection modes need deeper history; lighter modes cap at 6 turns.
        _deep_modes = {"therapist", "reflection"}
        history_limit = 16 if state.chat_mode in _deep_modes else 6
        trimmed_history = conversation_history[-history_limit:] if conversation_history else []

        args = (user_message, chat_preference, situation, severity, intent, response_draft,
                chat_mode_instructions, trimmed_history, conversation_summary)

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
    chat_preference: Optional[str] = None,
    situation: Optional[str] = None,
    severity: Optional[str] = None,
    intent: Optional[str] = None,
    context: Optional[str] = None,
    chat_mode_instructions: Optional[str] = None,
    conversation_history: Optional[list] = None,
    conversation_summary: Optional[str] = None,
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
        conversation_history: Prior turns [{speaker, message, turn_index}]
        conversation_summary: Summarised context from previous sessions

    Returns:
        Generated response string
    """
    try:
        import ssl
        import certifi
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

        mode_instruction = f"\nInteraction mode instructions: {chat_mode_instructions}" if chat_mode_instructions else ""

        summary_section = ""
        if conversation_summary:
            summary_section = f"\nSummary of previous sessions: {conversation_summary}\n"

        history_section = ""
        if conversation_history:
            lines = []
            for turn in conversation_history:
                speaker_label = "User" if turn.get("speaker") == "user" else "SoulBuddy"
                lines.append(f"{speaker_label}: {turn.get('message', '')}")
            history_section = "\n[Conversation so far]\n" + "\n".join(lines) + "\n"

        prompt = f"""You are SoulBuddy — a caring personal companion for emotional support.
{mode_instruction}{summary_section}{history_section}
[Current turn]
User message: "{user_message}"{context_info}

Rules:
- Follow the Interaction mode instructions above exactly — they define your persona and tone
- Keep response short (1–2 sentences unless the mode requires more depth)
- Use natural, everyday language — no clinical or counselor-speak
- Tailor tone to Chat Preference if provided
- Never use phrases like "It's completely normal to feel", "I understand that", "That sounds really hard", or "Would you like to tell me more about..."

Response:"""

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as session:
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
    chat_preference: Optional[str] = None,
    situation: Optional[str] = None,
    severity: Optional[str] = None,
    intent: Optional[str] = None,
    context: Optional[str] = None,
    chat_mode_instructions: Optional[str] = None,
    conversation_history: Optional[list] = None,
    conversation_summary: Optional[str] = None,
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
        conversation_history: Prior turns [{speaker, message, turn_index}]
        conversation_summary: Summarised context from previous sessions

    Returns:
        Generated response string
    """
    try:
        if not OPENAI_API_KEY:
            logger.debug("generate_response_gpt: OpenAI API key not configured")
            return ""
        import ssl
        import certifi
        import aiohttp

        # Build context-aware metadata appended to the current user turn
        context_info = ""
        if situation:
            context_info += f"\nSituation identified: {situation}"
        if severity:
            context_info += f"\nSeverity: {severity}"
        if intent:
            context_info += f"\nUser intent: {intent}"
        if chat_preference:
            context_info += f"\nChat Preference: {chat_preference}"

        mode_section = f"\nInteraction mode instructions: {chat_mode_instructions}" if chat_mode_instructions else ""
        summary_section = f"\nSummary of previous sessions: {conversation_summary}" if conversation_summary else ""

        system_content = (
            f"You are SoulBuddy — a caring personal companion for emotional support.{mode_section}{summary_section}\n\n"
            "Rules:\n"
            "- Follow the Interaction mode instructions above exactly — they define your persona and tone\n"
            "- Keep response short (1–2 sentences unless the mode requires more depth)\n"
            "- Use natural, everyday language — no clinical or counselor-speak\n"
            "- Tailor tone to Chat Preference if provided\n"
            "- Never use phrases like 'It's completely normal to feel', 'I understand that', "
            "'That sounds really hard', or 'Would you like to tell me more about...'"
        )

        messages = [{"role": "system", "content": system_content}]

        # Inject prior turns as real multi-turn messages so the model has genuine conversation memory
        for turn in (conversation_history or []):
            role = "user" if turn.get("speaker") == "user" else "assistant"
            messages.append({"role": role, "content": turn.get("message", "")})

        # Current user turn with classification metadata appended
        current_content = f"{user_message}{context_info}" if context_info else user_message
        messages.append({"role": "user", "content": current_content})
        
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

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as session:
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
