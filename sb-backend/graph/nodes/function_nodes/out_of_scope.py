import logging
import re
import uuid
from typing import Any, Callable, Dict, Optional

from graph.state import ConversationState
from graph.nodes.agentic_nodes.response_templates import get_out_of_scope_response

logger = logging.getLogger(__name__)

VOWELS = set("aeiouy")

KEYBOARD_ROWS = (
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
)

GENERAL_KNOWLEDGE_PATTERNS = (
    re.compile(r"\bcapital of\b"),
    re.compile(r"\bpopulation of\b"),
    re.compile(r"\bcurrency of\b"),
    re.compile(r"\bpresident of\b"),
    re.compile(r"\bprime minister of\b"),
    re.compile(r"\bweather in\b"),
    re.compile(r"\bsolve\b.+[0-9x+y=z]"),
    re.compile(r"\bcalculate\b"),
    re.compile(r"\btranslate\b"),
    re.compile(r"\bdefinition of\b"),
    re.compile(r"\bwhat(?:'s| is) the\b.+\b(country|capital|currency|population|formula|weather)\b"),
)

IN_SCOPE_SUPPORT_KEYWORDS = {
    "anxious", "anxiety", "burnout", "depressed", "emotion", "exams", "feeling",
    "feelings", "friendship", "journal", "lonely", "motivation", "overwhelmed",
    "sad", "self", "soulgym", "stressed", "stress", "support", "therapy",
    "tired", "wellbeing", "wellness", "work", "worry",
}

IN_SCOPE_KNOWLEDGE_KEYWORDS = {
    "attachment", "boundaries", "cbt", "chakra", "compassion", "coping",
    "dissociation", "ego", "energy", "grounding", "healing", "inner child",
    "meditation", "mindfulness", "nervous system", "psychology", "self compassion",
    "self-compassion", "self worth", "self-worth", "shadow work", "somatic",
    "spirit", "spiritual", "spirituality", "therapy", "trauma", "values",
}

KNOWN_SUPPORT_SHORT_FORMS = {
    "adhd", "cbt", "cptsd", "dbt", "emdr", "ocd", "ptsd",
}

IN_SCOPE_KNOWLEDGE_PATTERNS = (
    re.compile(r"\bdefinition of\b"),
    re.compile(r"\bmeaning of\b"),
    re.compile(r"\bwhat(?:'s| is)\b"),
    re.compile(r"\bwhat does\b"),
    re.compile(r"\bhow does\b"),
    re.compile(r"\bcan you explain\b"),
    re.compile(r"\bexplain\b"),
    re.compile(r"\btell me about\b"),
    re.compile(r"\bhelp me understand\b"),
)

TRIVIA_STYLE_PATTERNS = (
    re.compile(r"\bwho\b"),
    re.compile(r"\bwhen\b"),
    re.compile(r"\bwhere\b"),
    re.compile(r"\bcapital of\b"),
    re.compile(r"\bpopulation of\b"),
    re.compile(r"\bcurrency of\b"),
    re.compile(r"\bpresident of\b"),
    re.compile(r"\bprime minister of\b"),
    re.compile(r"\bweather in\b"),
)

SUPPORT_TERM_TOKENS = {
    re.sub(r"[^a-z]", "", term.lower())
    for term in (IN_SCOPE_SUPPORT_KEYWORDS | IN_SCOPE_KNOWLEDGE_KEYWORDS)
    if len(re.sub(r"[^a-z]", "", term.lower())) >= 5
}
SUPPORT_TERM_TOKENS.update(KNOWN_SUPPORT_SHORT_FORMS)

CODE_HINTS = (
    "```", "def ", "class ", "function ", "const ", "let ", "var ",
    "return ", "import ", "from ", "=>",
)

COMMON_WORD_BITS = {
    "and", "care", "feel", "ground", "have", "hello", "help", "ing",
    "ion", "just", "love", "ment", "mind", "ness", "please", "self",
    "stress", "thank", "ther", "thing", "tion", "well", "what", "with",
    "work", "you", "your",
}

MESSAGE_GIBBERISH_THRESHOLD = 5


async def out_of_scope_node(state: ConversationState) -> Dict[str, Any]:
    """
    Detect cheap, high-confidence out-of-scope cases before expensive graph work.

    Returns:
        Empty dict when the message should continue through the normal graph.
        Otherwise returns a render-ready out-of-scope state update.
    """
    out_of_scope_reason = detect_pattern_reason(state.user_message)
    if out_of_scope_reason is None:
        return {}

    logger.info(
        "out_of_scope: matched pattern",
        extra={"reason": out_of_scope_reason, "domain": state.domain},
    )
    return {
        "conversation_id": state.conversation_id or str(uuid.uuid4()),
        "intent": "out_of_scope",
        "situation": "NO_SITUATION",
        "severity": "low",
        "risk_level": "low",
        "is_out_of_scope": True,
        "out_of_scope_reason": out_of_scope_reason,
        "response_draft": get_out_of_scope_response(state.domain),
    }


def out_of_scope_router(state: ConversationState) -> str:
    if getattr(state, "is_out_of_scope", False):
        return "render"
    return "classification_node"


def detect_out_of_scope(
    message: str,
    domain: str = "general",
    llm_fn: Optional[Callable[[str], str]] = None,
    allow_llm_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Detect whether a user message is outside SoulBuddy's scope.
    """
    if not isinstance(message, str) or not message.strip():
        return get_out_of_scope_result(True, "other_out_of_scope", domain)

    pattern_reason = detect_pattern_reason(message)
    if pattern_reason:
        return get_out_of_scope_result(True, pattern_reason, domain)

    if looks_like_in_scope_support(message):
        return get_out_of_scope_result(False, "in_scope", domain)

    if not allow_llm_fallback:
        return get_out_of_scope_result(False, "in_scope", domain)

    if llm_fn is None:
        from graph.nodes.agentic_nodes.guardrail import call_guardrail_llm

        llm_fn = call_guardrail_llm

    prompt = build_out_of_scope_prompt(message)

    try:
        raw_response = llm_fn(prompt)
        from graph.nodes.agentic_nodes.guardrail import safe_json_loads

        data = safe_json_loads(raw_response)
    except Exception as exc:
        logger.debug("detect_out_of_scope: llm fallback failed: %s", exc)
        return get_out_of_scope_result(False, "in_scope", domain)

    is_out_of_scope = bool(data.get("is_out_of_scope", False))
    reason = get_out_of_scope_reason(data.get("reason"), is_out_of_scope)
    return get_out_of_scope_result(is_out_of_scope, reason, domain)


def build_out_of_scope_prompt(message: str) -> str:
    return f"""
You classify whether a message is outside SoulBuddy's scope.

SoulBuddy is a wellbeing companion. It can help with emotions, support, journaling,
self-reflection, stress, motivation, relationships, and planning SoulGym.

Mark messages as out of scope when they are:
- trivia or general knowledge
- technical, academic, or factual questions unrelated to wellbeing support
- nonsense or gibberish that does not form a meaningful request

Keep messages in scope when they ask about psychology, emotional wellbeing,
mindfulness, spirituality, or self-reflection in a way that could support the user.

Return ONLY JSON in this exact shape:
{{
  "is_out_of_scope": true or false,
  "reason": "general_knowledge" or "nonsense" or "other_out_of_scope" or "in_scope"
}}

User message: "{message}"
""".strip()


def detect_pattern_reason(message: str) -> Optional[str]:
    message_lower = message.lower().strip()
    if looks_like_nonsense(message_lower):
        return "nonsense"
    if is_support_topic_query(message_lower):
        return None
    if looks_like_general_knowledge(message_lower):
        return "general_knowledge"
    return None


def looks_like_general_knowledge(message_lower: str) -> bool:
    return any(pattern.search(message_lower) for pattern in GENERAL_KNOWLEDGE_PATTERNS)


def is_support_topic_query(message_lower: str) -> bool:
    if any(pattern.search(message_lower) for pattern in TRIVIA_STYLE_PATTERNS):
        return False
    if not any(keyword in message_lower for keyword in IN_SCOPE_KNOWLEDGE_KEYWORDS):
        return False
    return any(pattern.search(message_lower) for pattern in IN_SCOPE_KNOWLEDGE_PATTERNS)


def looks_like_nonsense(message_lower: str) -> bool:
    if looks_like_code_snippet(message_lower):
        return False

    alnum_tokens = re.findall(r"[a-zA-Z0-9]+", message_lower)
    alpha_tokens = re.findall(r"[a-zA-Z]+", message_lower)

    if len(alpha_tokens) >= 4 and all(len(token) == 1 for token in alpha_tokens):
        return True
    if not alnum_tokens:
        return False

    token_scores = [
        get_gibberish_score(token, message_lower)
        for token in alnum_tokens
        if len(token) >= 3
    ]
    if not token_scores:
        return False
    if max(token_scores) >= MESSAGE_GIBBERISH_THRESHOLD:
        return True

    medium_tokens = [score for score in token_scores if score >= 2]
    return len(medium_tokens) >= 3 and sum(medium_tokens) >= 6


def get_gibberish_score(token: str, message_lower: str) -> int:
    normalized = token.lower()
    alpha_only = re.sub(r"[^a-z]", "", normalized)
    if len(alpha_only) < 3 or contains_support_term(alpha_only):
        return 0

    score = 0
    vowel_count = sum(char in VOWELS for char in alpha_only)
    vowel_ratio = vowel_count / len(alpha_only)
    consonant_run = get_longest_consonant_run(alpha_only)
    vowel_run = get_longest_vowel_run(alpha_only)
    common_hits = get_common_chunk_hits(alpha_only)

    if re.fullmatch(r"([a-z])\1{5,}", alpha_only):
        score += 5
    if is_keyboard_smash(alpha_only):
        score += 5
    elif is_single_keyboard_row_token(alpha_only):
        score += 3

    if len(alpha_only) >= 5 and vowel_count == 0:
        score += 3
    elif len(alpha_only) >= 8 and (vowel_ratio < 0.2 or vowel_ratio > 0.75):
        score += 2

    if has_symbol_noise(message_lower) and len(alpha_only) >= 4 and vowel_count == 0:
        score += 3

    if consonant_run >= 5:
        score += 3
    elif consonant_run >= 4:
        score += 2

    if len(alpha_only) >= 12 and consonant_run >= 3 and vowel_run >= 3:
        score += 2

    if has_mixed_alnum_noise(normalized, alpha_only):
        score += 3
        if sum(char.isdigit() for char in normalized) >= 2:
            score += 1

    if any(char.isdigit() for char in normalized) and len(alpha_only) >= 12 and consonant_run >= 3 and vowel_run >= 2:
        score += 1

    if has_symbol_noise(message_lower) and len(alpha_only) >= 10 and consonant_run >= 3 and vowel_run >= 3:
        score += 1

    if len(alpha_only) >= 8 and common_hits == 0:
        score += 1
    if len(alpha_only) >= 10 and common_hits <= 1 and consonant_run >= 3:
        score += 1

    return score


def looks_like_code_snippet(message_lower: str) -> bool:
    marker_hits = sum(marker in message_lower for marker in CODE_HINTS)
    punctuation_hits = sum(char in message_lower for char in "{}[]();")
    return marker_hits >= 1 and punctuation_hits >= 2


def has_symbol_noise(message_lower: str) -> bool:
    return bool(re.search(r"[^a-z0-9\s]", message_lower))


def has_mixed_alnum_noise(token: str, alpha_only: str) -> bool:
    if len(token) < 8 or len(alpha_only) < 5:
        return False
    if not any(char.isalpha() for char in token) or not any(char.isdigit() for char in token):
        return False
    if re.fullmatch(r"[a-z]+[0-9]+", token) or re.fullmatch(r"[0-9]+[a-z]+", token):
        return False
    digit_ratio = sum(char.isdigit() for char in token) / len(token)
    return 0.1 <= digit_ratio <= 0.5


def is_keyboard_smash(token: str) -> bool:
    return len(token) >= 5 and any(token in row or token in row[::-1] for row in KEYBOARD_ROWS)


def is_single_keyboard_row_token(token: str) -> bool:
    if len(token) < 5:
        return False
    letters = set(token)
    return any(letters.issubset(set(row)) for row in KEYBOARD_ROWS)


def get_common_chunk_hits(token: str) -> int:
    return sum(chunk in token for chunk in COMMON_WORD_BITS)


def contains_support_term(token: str) -> bool:
    normalized = re.sub(r"[^a-z]", "", token.lower())
    if not normalized:
        return False
    return any(term in normalized for term in SUPPORT_TERM_TOKENS)


def get_longest_vowel_run(token: str) -> int:
    longest = 0
    current = 0
    for char in token:
        if char in VOWELS:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def get_longest_consonant_run(token: str) -> int:
    longest = 0
    current = 0
    for char in token:
        if char.isalpha() and char not in VOWELS:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def looks_like_in_scope_support(message: str) -> bool:
    message_lower = message.lower()
    if is_support_topic_query(message_lower):
        return True
    if any(keyword in message_lower for keyword in IN_SCOPE_SUPPORT_KEYWORDS):
        return True
    return bool(
        re.search(r"\b(i|i'm|i am|my|me)\b", message_lower)
        and re.search(r"\b(feel|feeling|struggling|cope|help)\b", message_lower)
    )


def get_out_of_scope_reason(raw_reason: Any, is_out_of_scope: bool) -> str:
    normalized = str(raw_reason or "").strip().lower()
    if not is_out_of_scope:
        return "in_scope"
    if normalized in {"general_knowledge", "nonsense", "other_out_of_scope"}:
        return normalized
    return "other_out_of_scope"


def get_out_of_scope_result(
    is_out_of_scope: bool,
    reason: str,
    domain: str,
) -> Dict[str, Any]:
    return {
        "is_out_of_scope": is_out_of_scope,
        "reason": reason if is_out_of_scope else "in_scope",
        "response": get_out_of_scope_response(domain, reason=reason) if is_out_of_scope else "",
    }
