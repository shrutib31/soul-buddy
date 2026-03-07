"""
Response Evaluator

Scores LLM-generated responses on dimensions that matter for a mental-health
support chatbot, then selects the better of two candidates (Ollama vs GPT).

Scoring dimensions
──────────────────
1. Empathy       – presence of empathetic language (+points per phrase, capped)
2. Engagement    – contains an open-ended question to keep the user talking
3. Length        – not too terse, not overwhelming
4. Completeness  – ends with proper punctuation (not cut off mid-sentence)
5. Repetition    – penalise if 4-word n-grams repeat 3+ times (hallucination signal)
6. Robotic lang  – penalise dismissive or clinical AI disclaimers
"""

import re
from collections import Counter
from typing import Tuple


# ── Empathy vocabulary ────────────────────────────────────────────────────────
# Each phrase found in the response adds to the empathy score.

_EMPATHY_PHRASES = [
    "i hear you", "i understand", "that sounds", "i'm sorry", "i'm so sorry",
    "must be", "you're not alone", "you don't have to", "i care",
    "i'm here", "i'm right here", "i'm with you", "that makes sense",
    "it's okay", "it's understandable", "thank you for sharing",
    "thank you for trusting", "that takes courage", "you matter",
    "your feelings are valid", "i can hear", "that must be",
    "so hard", "really tough", "genuinely concerned", "glad you reached out",
    "not your fault", "makes complete sense", "you're doing", "i see you",
]

# ── Engagement ────────────────────────────────────────────────────────────────
_QUESTION_RE = re.compile(r'\?')

# ── Robotic / dismissive patterns ─────────────────────────────────────────────
_ROBOTIC_PHRASES = [
    "as an ai", "i'm just an ai", "i am an ai",
    "i'm not a therapist", "i cannot provide therapy",
    "i cannot provide mental health", "i am unable to",
    "i don't have feelings", "i don't have emotions",
    "please seek professional help",     # penalised when used as the entire response
    "consult a professional",
]

# ── Length thresholds (word count) ────────────────────────────────────────────
_MIN_WORDS = 20       # too terse → unhelpful
_MAX_WORDS = 200      # too long  → overwhelming
_IDEAL_MIN = 30
_IDEAL_MAX = 120

# ── Repetition detection ──────────────────────────────────────────────────────
_NGRAM_SIZE = 4
_REPETITION_THRESHOLD = 3   # if any 4-gram appears ≥3 times → penalise


# ── Individual scorers ────────────────────────────────────────────────────────

def _empathy_score(text: str) -> float:
    lower = text.lower()
    hits = sum(1 for phrase in _EMPATHY_PHRASES if phrase in lower)
    return min(hits * 1.5, 7.0)   # cap so one dimension can't dominate


def _engagement_score(text: str) -> float:
    questions = len(_QUESTION_RE.findall(text))
    if questions == 0:
        return 0.0
    # 1-2 questions is ideal; more than 3 feels interrogative
    return min(questions, 2) * 1.5


def _length_score(text: str) -> float:
    words = len(text.split())
    if words < _MIN_WORDS:
        return -2.0
    if words > _MAX_WORDS:
        return -1.0
    if _IDEAL_MIN <= words <= _IDEAL_MAX:
        return 2.0
    return 1.0   # acceptable but outside ideal band


def _completeness_score(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return -5.0
    return 1.0 if stripped[-1] in ".!?\"'" else -0.5


def _repetition_penalty(text: str) -> float:
    words = text.lower().split()
    if len(words) < _NGRAM_SIZE:
        return 0.0
    ngrams = [" ".join(words[i:i + _NGRAM_SIZE]) for i in range(len(words) - _NGRAM_SIZE + 1)]
    repeated = sum(1 for count in Counter(ngrams).values() if count >= _REPETITION_THRESHOLD)
    return -repeated * 2.0


def _robotic_penalty(text: str) -> float:
    lower = text.lower()
    hits = sum(1 for phrase in _ROBOTIC_PHRASES if phrase in lower)
    return -hits * 3.0


# ── Public API ────────────────────────────────────────────────────────────────

def score_response(response: str) -> float:
    """
    Score a single response on mental-health conversation quality.

    Args:
        response: The response text to evaluate.

    Returns:
        A float score — higher is better. Empty/None responses score 0.0.
    """
    if not response or not response.strip():
        return 0.0

    return (
        _empathy_score(response)
        + _engagement_score(response)
        + _length_score(response)
        + _completeness_score(response)
        + _repetition_penalty(response)
        + _robotic_penalty(response)
    )


def select_best_response(
    ollama_response: str,
    gpt_response: str,
) -> Tuple[str, str, float, float]:
    """
    Compare Ollama and GPT responses and return the better one.

    Falls back to whichever response is non-empty if the other is missing.

    Args:
        ollama_response: Response text from Ollama (may be empty).
        gpt_response:    Response text from GPT (may be empty).

    Returns:
        Tuple of (selected_text, source, ollama_score, gpt_score)
        where source is "ollama" or "gpt".
    """
    ollama_score = score_response(ollama_response)
    gpt_score = score_response(gpt_response)

    if not ollama_response:
        return gpt_response, "gpt", ollama_score, gpt_score
    if not gpt_response:
        return ollama_response, "ollama", ollama_score, gpt_score

    if ollama_score > gpt_score:
        return ollama_response, "ollama", ollama_score, gpt_score

    return gpt_response, "gpt", ollama_score, gpt_score
