"""
Situation & Severity Detection Node for LangGraph

This node detects the user's situation and severity using Ollama (small LLM).
It runs in parallel with message storage and intent detection.
"""

from typing import Dict, Any, Optional
import json
import os
import logging

from graph.state import ConversationState

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://194.164.151.158:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # Timeout in seconds (default 120s for inference)

VALID_SEVERITIES = {"low", "medium", "high"}

logger = logging.getLogger(__name__)


# ============================================================================
# LANGRAPH NODE FUNCTION
# ============================================================================

async def situation_severity_detection_node(state: ConversationState) -> Dict[str, Any]:
    """
    Detect situation and severity from user message using Ollama.

    Args:
        state: Current conversation state

    Returns:
        Dict with situation and severity fields (and error if any)
    """
    try:
        user_message = state.user_message

        if not user_message or user_message.strip() == "":
            return {"error": "Empty user message for situation/severity detection"}

        logger.info(
            "situation_severity_detection: starting",
            extra={"ollama_base_url": OLLAMA_BASE_URL, "ollama_model": OLLAMA_MODEL}
        )

        result = await detect_situation_severity_with_ollama(user_message)
        logger.info(
            "situation_severity_detection: completed",
            extra={"situation": result.get("situation"), "severity": result.get("severity")}
        )
        return result

    except Exception:
        logger.exception("situation_severity_detection: failed")
        return {
            "error": f"Error detecting situation/severity"
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def detect_situation_severity_with_ollama(message: str) -> Dict[str, Any]:
    """
    Call Ollama to detect situation and severity.

    Returns:
        Dict with keys: situation (str|None), severity (str|None)
    """
    try:
        import aiohttp

        prompt = f"""Analyze the following message and return a JSON object with:
- situation: a short label describing the user's situation (lowercase, snake_case)
- severity: one of [low, medium, high]

Message: "{message}"

Return ONLY valid JSON like:
{{"situation": "work_stress", "severity": "medium"}}
"""

        timeout = aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)
        request_payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info("situation_severity_detection: calling ollama", extra={"url": f"{OLLAMA_BASE_URL}/api/generate"})
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=request_payload,
            ) as resp:
                status = resp.status
                text_body = await resp.text()
                logger.info(
                    "situation_severity_detection: ollama response",
                    extra={"status": status, "body_preview": text_body[:500]}
                )

                if status != 200:
                    logger.warning(
                        "situation_severity_detection: non-200 response",
                        extra={"status": status}
                    )
                    return {"situation": None, "severity": None}

                data = json.loads(text_body)
                response_text = data.get("response", "").strip()

                parsed = _extract_json(response_text)
                if not parsed:
                    logger.warning("situation_severity_detection: json extraction failed", extra={"response": response_text[:200]})
                    return {"situation": None, "severity": None}

                situation = parsed.get("situation")
                severity = parsed.get("severity")
                if isinstance(severity, str):
                    severity = severity.strip().lower()

                if severity not in VALID_SEVERITIES:
                    logger.warning("situation_severity_detection: invalid severity", extra={"severity": severity})
                    severity = None

                return {
                    "situation": situation,
                    "severity": severity,
                }

    except Exception:
        logger.exception("situation_severity_detection: ollama call failed")
        return {"situation": None, "severity": None}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from model output."""
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None
