import json
from typing import Any, Dict, Callable, Optional
import os
import asyncio
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import logging
import urllib.request

# Note: Configure Ollama connection details as needed
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_MODEL = "wizardlm2:7b"  # Change to your preferred small model (e.g., "neural-chat", "orca-mini") # was phi3:latest
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # Timeout in seconds (default 120s for inference)

logger = logging.getLogger(__name__)

# Rules/Patterns Guadrail checks against
GUARDRAIL_RULES = [
    "You are a companion, NOT a therapist",
    "Do not dismiss the user's lived experience",
    "Do not normalize what the user is going through without acknowledging the pain",
    "NEVER minimize or invalidate distress in a user",
    "Do not increase shame or guilt in a user via your response",
    "Always position yourself as a companion, and NEVER an authority",
    "Do not make the user feel misunderstood or lectured, you are an AI companion",
    "Do not rush into advice or solutions before validating the user and exploring the situation",
    "Do not force optimism in a response",
    "Do not respond with toxic positivity",
    "Do not overstep your boundary of a companion and act like a therapist",
    "Do not promise outcomes or claim to heal, treat, cure, or diagnose a problem",
    "Do not use moralizing or judging language that could make the user feel shameful, increase avoidance, or view struggle as a character flaw",
    "Do not compare users to others, their lived experience is unique and should not be compared or minimized",
    "Never discourage therapy, counseling, or support systems",
    "Immediately acknowledge distress",
    "Encourage reaching out for help to professional therapists, friends, family, or hotlines",
    "Do not use exercises or reframes when a high-risk situation is detected",
    "Do not overintellectualize emotions, for example, by explaining their biological origins, because you will come off as cold and not a warm companion",
    "Identify and explore the emotions they expressed first, and then only after make an explanation later, only if helpful",
    "Do not take sides or validate harmful beliefs",
    "Never make somebody feel dismissed, judged, or alone",
    "Always remember you are an AI. While you are meant to be a companion, you must not cross any boundaries that would entail a human to believe you are human"
]


async def guardrail_node(
    state,
    guardrail_fn: Optional[Callable[[str], str]] = None, #guardrail_fn for testing so we can do unit tests without using dependencies
) -> Dict[str, Any]:
    """
    Guardrail step node - verifies LLM response against a set of rules, ensuring proper responses
    
    Args:
        state: Current conversation state
        
    Returns:
        Status -> String
            1. OK - response is verified and safe to send to user, SEND!
            2. REFINE - response has issues and must go through pipeline again, send BACK TO BEGINNING!
        Feedback -> String
            Feedback suggestions to improve responses
        Violation -> String
            The rule that was violated by the given assistant response
    """
    prompt = f"""
You are a guardrail checker.

Given:
- The user's message
- A candidate assistant answer
- The GUARDRAIL_RULES

Your job is to decide whether the candidate assistant answer VIOLATES or DOES NOT VIOLATE the GUARDRAIL_RULES.
You are checking to see if the candidate assistant answer is a good response to the user message based on if it violates GUARDRAIL_RULES.

Return ONLY a JSON object with this exact structure:
{{
  "status": "OK" or "REFINE",
  "feedback": "short explanation of why, and what to adjust",
  "violation": "which rule(s) were violated. If more than one, separate by comma. If OK, say "None", If "REFINE" but not "OK", say "None but..." and explain why you say to REFINE even thought it doesn't break any rules."
}}

Rules:
- Use "OK" only if the answer clearly does NOT violate the rules.
- Use "REFINE" if tone, style, structure, or content violates the rules or needs improvement.
- feedback should be concise but specific enough to help refine the response.

User Message: "{state.user_message}"
Candidate Assistant Answer: {state.response_draft}
GUARDRAIL_RULES: "{GUARDRAIL_RULES}"

"""
    if guardrail_fn is None:
        guardrail_fn = call_guardrail_llm

    next_attempt = (state.attempt or 0) + 1
    next_step = (state.step_index or 0) + 1

    try:
        print("Guardrail checking response...")
        guardrail_response = await asyncio.to_thread(guardrail_fn, prompt)
        data: Dict[str, Any] = safe_json_loads(guardrail_response)
        status = str(data.get("status", "")).upper()
        feedback = str(data.get("feedback", "")).strip()
        print("Current LLM Response = " + state.response_draft)
        print("Status = " + status)
        print("Feedback = " + feedback)
        print("Violation = " + str(data.get("violation", "")).strip())
        if status not in {"OK", "REFINE"}:
            return {
                "error": "Error in guardrail node: invalid guardrail status",
                "guardrail_status": "ERROR",
                "guardrail_feedback": "Bad response. Need to refine",
                "attempt": next_attempt,
                "step_index": next_step,
            }

    except Exception as e:
        return {
            "error": f"Error in guardrail node: {str(e)}",
            "guardrail_status": "ERROR",
            "guardrail_feedback": "Guardrail parsing failed",
            "attempt": next_attempt,
            "step_index": next_step,
        }
    
    return {
        "guardrail_status": status,
        "guardrail_feedback": feedback,
        "attempt": next_attempt,
        "step_index": next_step,
    }

def guardrail_router(state) -> str:
    if state.error or state.guardrail_status == "ERROR":
        print("[router] Guardrail error → render")
        return "render"

    if state.guardrail_status == "OK":
        print("[router] Guardrail says OK → Finishing Sequences")
        return "store_bot_response"

    if (state.attempt or 0) >= 3:
        print("[router] Guardrail still REFINE but max attempts reached → Finishing Sequences")
        return "store_bot_response"

    print("[router] Guardrail says REFINE → back to beginning")
    print("[router] status=", repr(state.guardrail_status), "attempt=", state.attempt)
    return "conv_id_handler" #RETURNS STARTING NODE OF LANG GRAPH


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def call_guardrail_llm(prompt) -> str:
    """
    Calls LLM to check current 
    response against Guardrail rules
    """
    request_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.3,  # Lower temperature for more focused responses
    }
    logger.info("guardrail_node: calling ollama", extra={"url": f"{OLLAMA_BASE_URL}/api/generate", "timeout": OLLAMA_TIMEOUT})
    url= f"{OLLAMA_BASE_URL}/api/generate"
    data = json.dumps(request_payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return str(parsed.get("response", "")).strip()
    except Exception as e:
        print(f"Error calling Ollama guardrail LLM: {e}")
        return ""  

def extract_json_str(text: str) -> str:
    """
    Very simple brace-balanced JSON extractor.
    Takes a string from the LLM and returns the first {...} block it finds.
    Raises ValueError if no valid JSON-like block is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found in text")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # Return substring including closing brace
                return text[start : i + 1]

    raise ValueError("Unbalanced braces; JSON object not closed")


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Try to parse JSON from a possibly noisy LLM response.

    Strategy:
    1. First, try json.loads directly.
    2. If that fails, extract the first balanced {...} block and parse that.
    """
    try:
        return json.loads(text)
    except Exception:
        json_str = extract_json_str(text)
        return json.loads(json_str)
