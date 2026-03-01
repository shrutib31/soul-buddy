"""
Response Templates

Pre-defined responses for high-confidence, time-sensitive situations where a
readymade response is safer, faster, and more consistent than LLM generation:

  - Greetings : is_greeting == True
  - Crisis    : is_crisis_detected == True
"""

import random
from typing import Optional


# ============================================================================
# GREETING TEMPLATES
# ============================================================================
# Shown when intent is classified as "greeting".
# Variants prevent the bot from feeling scripted on repeated use.

_GREETING_TEMPLATES = {
    "student": [
        "Hey! I'm SoulBuddy, your personal wellness companion. Whether it's exam stress, friendships, or just a tough day — I'm here to listen. How are you feeling today?",
        "Hi there! Really glad you stopped by. Student life can be a lot to handle sometimes. I'm here whenever you're ready to talk. What's on your mind?",
        "Hello! I'm SoulBuddy. I know being a student comes with its own set of pressures. You don't have to face them alone — I'm here. How are you doing today?",
    ],
    "employee": [
        "Hey! I'm SoulBuddy, your wellness companion at work. Whether it's work pressure, burnout, or just needing someone to talk to — I'm here. How are you doing?",
        "Hi there! Good to see you. Workplace stress is real, and it's okay to take a moment for yourself. I'm here to listen. What's on your mind?",
        "Hello! I'm SoulBuddy. Navigating work life can be tough — I'm here as a safe space whenever you need it. How are you feeling today?",
    ],
    "corporate": [
        "Hi! I'm SoulBuddy, your professional wellness companion. I'm here to support your wellbeing — no matter what you're navigating. How are you doing today?",
        "Hello! Great to connect. High-pressure environments can take a toll. I'm here as a confidential space to talk things through. What's on your mind?",
        "Hey! I'm SoulBuddy. Whether it's a difficult project, team dynamics, or just the day catching up with you — I'm here. How are you feeling right now?",
    ],
    "general": [
        "Hey! I'm SoulBuddy, your mental wellness companion. I'm really glad you're here. How are you feeling today?",
        "Hello! It's great to connect with you. I'm here to listen and support you, whatever's on your mind. How are you doing?",
        "Hi there! I'm SoulBuddy. Whenever you're ready to talk, I'm here — no judgment, just a caring space. What's going on with you today?",
    ],
}


# ============================================================================
# MEDIUM RISK TEMPLATES
# ============================================================================
# CRISIS TEMPLATES
# ============================================================================
# Shown when is_crisis_detected == True.
# Goal: immediate warmth, clear safety resources, no dismissal, open door.

_HIGH_RISK_TEMPLATES = [
    (
        "I hear you, and what you've shared matters deeply to me. Please know that you are not alone right now — not even for a second. "
        "If you're in immediate danger, please reach out to iCall at 9152987821 (Mon–Sat, 8am–10pm) or the Vandrevala Foundation at 1860-2662-345 (available 24/7). "
        "I'm right here with you. Can you tell me a little more about what you're feeling right now?"
    ),
    (
        "Thank you for trusting me with this — that took real courage. What you're going through sounds incredibly painful, and your life matters. "
        "Please consider reaching out to AASRA at 91-22-27546669 or iCall at 9152987821 — they're there to help, day or night. "
        "I'm not going anywhere either. Would you like to talk about what's been happening?"
    ),
    (
        "I'm really glad you reached out. I want you to know I'm taking what you've said seriously, and I care about your safety. "
        "If things feel overwhelming right now, please call the Vandrevala Foundation at 1860-2662-345 (24/7) or iCall at 9152987821. "
        "You don't have to face this alone. I'm here — what's going on for you right now?"
    ),
    (
        "What you're sharing is important, and so are you. I'm genuinely concerned and I want to make sure you're safe. "
        "If you're in crisis, please reach out to iCall (9152987821) or AASRA (91-22-27546669) — trained counsellors are there to listen. "
        "And so am I. Can you tell me more about what's been happening?"
    ),
]


# ============================================================================
# PUBLIC API
# ============================================================================

def get_template_response(
    is_crisis_detected: bool,
    is_greeting: bool,
    domain: Optional[str] = "general",
) -> Optional[str]:
    """
    Return a readymade template response if one applies, otherwise None.

    Priority order:
      1. is_crisis_detected → crisis template  (safety-critical, must come first)
      2. is_greeting        → greeting template

    Args:
        is_crisis_detected: True when classification_node detected a crisis.
        is_greeting:        True when classification_node detected a greeting.
        domain:             Conversation domain ("student", "employee", "corporate", "general").

    Returns:
        A template string, or None if no template applies (LLM should be used).
    """
    if is_crisis_detected:
        return random.choice(_HIGH_RISK_TEMPLATES)

    if is_greeting:
        domain_key = domain if domain in _GREETING_TEMPLATES else "general"
        return random.choice(_GREETING_TEMPLATES[domain_key])

    return None
