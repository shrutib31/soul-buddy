"""
Response Templates

Pre-defined responses for high-confidence, time-sensitive situations where a
readymade response is safer, faster, and more consistent than LLM generation:

  - Greetings     : is_greeting == True
  - Crisis        : is_crisis_detected == True
  - Out-of-scope  : is_out_of_scope == True
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
        "Hey! Welcome back. Whether it's academics, social stuff, or just life in general — I'm all ears. What's going on?",
        "Hi! I'm SoulBuddy. Think of me as that friend who's always around when you need to talk. How's your day going?",
        "Hello there! Student life has its ups and downs — and I'm here for all of it. What would you like to talk about today?",
    ],
    "employee": [
        "Hey! I'm SoulBuddy, your wellness companion at work. Whether it's work pressure, burnout, or just needing someone to talk to — I'm here. How are you doing?",
        "Hi there! Good to see you. Workplace stress is real, and it's okay to take a moment for yourself. I'm here to listen. What's on your mind?",
        "Hello! I'm SoulBuddy. Navigating work life can be tough — I'm here as a safe space whenever you need it. How are you feeling today?",
        "Hey! Taking a moment to check in with yourself is a great step. I'm here if you want to talk through anything. How are things going?",
        "Hi! Whether it's deadlines, colleagues, or just feeling drained — I'm here to listen without any judgment. What's up?",
        "Hello! I'm SoulBuddy, here to support your wellbeing at work. No agenda, just a space to be heard. How are you doing right now?",
    ],
    "corporate": [
        "Hi! I'm SoulBuddy, your professional wellness companion. I'm here to support your wellbeing — no matter what you're navigating. How are you doing today?",
        "Hello! Great to connect. High-pressure environments can take a toll. I'm here as a confidential space to talk things through. What's on your mind?",
        "Hey! I'm SoulBuddy. Whether it's a difficult project, team dynamics, or just the day catching up with you — I'm here. How are you feeling right now?",
        "Hi there! Stepping back to check in with yourself shows real self-awareness. I'm here whenever you'd like to talk. How are things?",
        "Hello! I'm SoulBuddy. Leadership, targets, team challenges — whatever's weighing on you, this is a safe space to unpack it. What's on your mind?",
        "Hey! Good to see you here. Balancing performance with wellbeing matters. I'm here to support that balance. How are you feeling?",
    ],
    "general": [
        "Hey! I'm SoulBuddy, your mental wellness companion. I'm really glad you're here. How are you feeling today?",
        "Hello! It's great to connect with you. I'm here to listen and support you, whatever's on your mind. How are you doing?",
        "Hi there! I'm SoulBuddy. Whenever you're ready to talk, I'm here — no judgment, just a caring space. What's going on with you today?",
        "Hey! Welcome. I'm SoulBuddy — think of me as a companion who's here to listen. How are things with you?",
        "Hi! I'm really glad you reached out. Whatever's on your mind, big or small, I'm here for it. How are you feeling right now?",
        "Hello! I'm SoulBuddy. There's no wrong way to start — just say whatever feels right. What's going on?",
    ],
}


# ============================================================================
# CRISIS TEMPLATES
# ============================================================================
# Shown when is_crisis_detected == True.
# Goal: immediate warmth, clear safety resources, no dismissal, open door.

_HIGH_RISK_TEMPLATES = [
    (
        "I hear you, and what you've shared matters deeply to me. Please know that you are not alone right now — not even for a second. "
        "If you're in immediate danger, please reach out to iCall at 9152987821 (MonΓÇôSat, 8amΓÇô10pm) or the Vandrevala Foundation at 1860-2662-345 (available 24/7). "
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
# OUT-OF-SCOPE TEMPLATES
# ============================================================================

_OUT_OF_SCOPE_REASON_OPENERS = {
    "general_knowledge": [
        "That sounds more like a trivia or fact question than something I can answer well.",
        "I'm probably not the best place for general knowledge or factual lookup questions.",
        "That seems like a factual question, and I'm not the most reliable tool for that kind of answer.",
    ],
    "nonsense": [
        "I don't think I quite caught what you meant there.",
        "That message didn't come through clearly enough for me to respond to it well.",
        "I might be missing what you were trying to say there.",
    ],
    "other_out_of_scope": [
        "That's a bit outside what I'm built to help with here.",
        "I'm probably not the right tool for that kind of request.",
        "That goes beyond the kind of support I can give well in this chat.",
    ],
}

_OUT_OF_SCOPE_REDIRECTS = {
    "student": [
        "I can help with student stress, motivation, friendships, journaling, or planning your SoulGym instead. Want to go that way?",
        "I'm better at helping with things like exam stress, feeling overwhelmed, journaling, or planning your SoulGym. If you want, we can switch to that.",
        "If what you really need is support, I can help you talk through stress, motivation, or how you're feeling, or help plan your SoulGym.",
    ],
    "employee": [
        "I can help with work stress, burnout, journaling, or planning your SoulGym instead. Want to shift to that?",
        "I'm better at supporting you with work stress, pressure, burnout, reflection, or planning your SoulGym. We can go there if that helps.",
        "If you want, I can help you talk through work stress, how you're feeling, or plan your SoulGym instead.",
    ],
    "corporate": [
        "I can help with stress, pressure, reflection, journaling, or planning your SoulGym instead. Want to pivot to that?",
        "I'm better at helping with pressure, wellbeing, reflection, or planning your SoulGym. If that's useful, we can switch gears.",
        "If what you need is support, I can help you think through stress, pressure, or how you're feeling, or help plan your SoulGym.",
    ],
    "general": [
        "I can help with journaling, planning your SoulGym, or talking through what you're feeling instead. Want to try one of those?",
        "I'm much better at wellbeing support like reflection, journaling, or planning your SoulGym. If you want, we can go in that direction.",
        "If you were looking for support rather than facts, I can help you sort through what you're feeling or plan your SoulGym.",
    ],
}


# ============================================================================
# CATEGORY-SPECIFIC CRISIS TEMPLATES
# ============================================================================
# PASSIVE_DEATH_WISH: softer, focuses on connection and asking about feelings.
# SELF_HARM: acknowledges the behaviour without shame, focuses on pain underneath.
# SUICIDAL (ideation / plan): urgent, explicit hotlines, affirms life value.
# DEFAULT: fallback for other high-risk states.

_CRISIS_TEMPLATES_BY_CATEGORY = {
    "PASSIVE_DEATH_WISH": [
        (
            "I hear you, and I'm really glad you shared that with me. Feeling like you want to disappear "
            "often means you're carrying something really heavy right now. You don't have to carry it alone. "
            "Can you tell me a bit more about what's been going on for you lately?"
        ),
        (
            "What you just said matters, and so do you. When things feel that exhausting, it can be hard to "
            "see a way forward — but I'm here with you right now. "
            "If you ever feel like you need more support, iCall (9152987821) is a safe space to reach out. "
            "What's been weighing on you the most?"
        ),
        (
            "That sounds like an incredibly heavy place to be, and I want you to know I'm here and I'm listening. "
            "Sometimes when we're that tired, it helps to just say it out loud and have someone hold it with you. "
            "I'm not going anywhere. What's been happening?"
        ),
    ],
    "SELF_HARM": [
        (
            "Thank you for trusting me with that — it takes real courage to share something so personal. "
            "Self-harm often means there's a lot of pain underneath, and I want to understand what you're going through. "
            "Please know you can also reach iCall at 9152987821 — they're trained to support exactly this. "
            "Are you safe right now? Can you tell me more about what's been going on?"
        ),
        (
            "I'm really glad you said something. What you're describing tells me you're hurting deeply, "
            "and you deserve real support — not judgment. "
            "If you're in immediate distress, please reach out to the Vandrevala Foundation at 1860-2662-345 (24/7). "
            "I'm here too. Can we talk about what's been going on?"
        ),
    ],
    "SUICIDAL_IDEATION": [
        (
            "Thank you for trusting me with this — that took real courage. What you're going through sounds "
            "incredibly painful, and your life matters. "
            "Please consider reaching out to AASRA at 91-22-27546669 or iCall at 9152987821 — "
            "they're there to help, day or night. I'm not going anywhere either. "
            "Would you like to talk about what's been happening?"
        ),
        (
            "I hear you, and what you've shared matters deeply to me. Please know that you are not alone "
            "right now — not even for a second. "
            "If you're in immediate danger, please reach out to iCall at 9152987821 (MonΓÇôSat, 8amΓÇô10pm) "
            "or the Vandrevala Foundation at 1860-2662-345 (available 24/7). "
            "I'm right here with you. Can you tell me a little more about what you're feeling right now?"
        ),
        (
            "I'm really glad you reached out. I want you to know I'm taking what you've said seriously, "
            "and I care about your safety. "
            "If things feel overwhelming right now, please call the Vandrevala Foundation at 1860-2662-345 (24/7) "
            "or iCall at 9152987821. "
            "You don't have to face this alone. I'm here — what's going on for you right now?"
        ),
    ],
    "SUICIDAL_PLAN": [
        (
            "I'm very concerned about you right now, and I'm so glad you told me. "
            "Please reach out immediately — AASRA is available 24/7 at 91-22-27546669, "
            "and the Vandrevala Foundation is at 1860-2662-345 (also 24/7). "
            "If you are in immediate danger, please call emergency services (112 in India). "
            "You matter, and help is there right now."
        ),
        (
            "What you're sharing is serious, and I want to make sure you're safe. "
            "Please contact a crisis line immediately — iCall at 9152987821 or AASRA at 91-22-27546669. "
            "If you're in immediate danger, call 112. "
            "You don't have to go through this alone — there are people who want to help you right now."
        ),
    ],
    "DEFAULT": _HIGH_RISK_TEMPLATES,
}


# ============================================================================
# CHAT PREFERENCES TEMPLATES
# ============================================================================
_CHAT_PREFERENCES = {
    "gentle_reflective": (
        "Generate more empathetic response, with thoughtful and nurturing tone, "
        "Goal of quality emotional processing and deep reflection with user"
    ),
    "direct_practical": (
        "Generate clear, actionable advice and concrete strategies, "
        "Goal of problem-solving and giving practical advice to user"
    ),
    "general": (
        "Generate helpful, warm response that helps user feel seen"
    ),
}

# Chat mode — user-selectable per session; shapes the interaction contract.
_CHAT_MODES = {
    "default": (
        "Talk like a close, caring friend — casual, natural, zero jargon. "
        "Keep replies SHORT (1ΓÇô2 sentences max). React the way a friend would: 'oh no, what happened?' or "
        "'ugh that's annoying, tell me more' — never like a counselor. "
        "No clinical phrases like 'it's completely normal', 'I hear you', or 'that sounds really hard'. "
        "Just be real, warm, and human."
    ),
    "reflection": (
        "You are a reflective guide. Your primary role is to help the user explore their own thoughts and feelings. "
        "Ask open-ended, curious questions (e.g. 'What do you think is underneath that feeling?'). "
        "Do NOT offer solutions or advice. Help the user arrive at their own insights through guided self-inquiry."
    ),
    "venting": (
        "You are a compassionate listener in pure listening mode. The user needs to vent — your only job is to "
        "acknowledge, validate, and affirm their feelings without steering or advising. "
        "Use phrases like 'That sounds really hard', 'I hear you', 'It makes sense you feel that way'. "
        "Never offer solutions, suggestions, or silver linings unless the user explicitly asks."
    ),
    "therapist": (
        "You are a supportive, structured mental wellness coach drawing on evidence-based approaches "
        "(CBT, mindfulness, motivational interviewing). Use psychoeducation where helpful. "
        "Gently challenge unhelpful thought patterns. Suggest concrete coping techniques or short exercises. "
        "Maintain a warm but professional tone — structured, grounding, and empowering."
    ),
}

# ============================================================================
# PUBLIC API
# ============================================================================

def get_template_response(
    is_crisis_detected: bool,
    is_greeting: bool,
    domain: Optional[str] = "general",
    is_out_of_scope: bool = False,
    out_of_scope_reason: Optional[str] = None,
    crisis_category: Optional[str] = None,
) -> Optional[str]:
    """
    Return a readymade template response if one applies, otherwise None.

    Priority order:
      1. is_crisis_detected → crisis template  (safety-critical, must come first)
      2. is_out_of_scope    → out-of-scope template
      3. is_greeting        → greeting template

    Args:
        is_crisis_detected: True when classification_node detected a crisis.
        is_greeting:        True when classification_node detected a greeting.
        domain:             Conversation domain ("student", "employee", "corporate", "general").
        is_out_of_scope:    True when classification_node detected an out-of-scope message.
        out_of_scope_reason: Why the message was considered out of scope.
        crisis_category:    The specific crisis category (e.g. SUICIDAL_IDEATION, PASSIVE_DEATH_WISH).

    Returns:
        A template string, or None if no template applies (LLM should be used).
    """
    if is_crisis_detected:
        # Use category-specific template if available, otherwise fall back to generic.
        template_pool = _CRISIS_TEMPLATES_BY_CATEGORY.get(
            crisis_category, _CRISIS_TEMPLATES_BY_CATEGORY["DEFAULT"]
        )
        return random.choice(template_pool)

    if is_out_of_scope:
        return get_out_of_scope_response(domain, reason=out_of_scope_reason)

    if is_greeting:
        domain_key = domain if domain in _GREETING_TEMPLATES else "general"
        return random.choice(_GREETING_TEMPLATES[domain_key])

    return None


def get_out_of_scope_response(
    domain: Optional[str] = "general",
    reason: Optional[str] = "other_out_of_scope",
) -> str:
    domain_key = domain if domain in _OUT_OF_SCOPE_REDIRECTS else "general"
    reason_key = reason if reason in _OUT_OF_SCOPE_REASON_OPENERS else "other_out_of_scope"
    opener = random.choice(_OUT_OF_SCOPE_REASON_OPENERS[reason_key])
    redirect = random.choice(_OUT_OF_SCOPE_REDIRECTS[domain_key])
    return f"{opener} {redirect}"


def get_chat_mode_instructions(chat_mode: str) -> str:
    return _CHAT_MODES.get(chat_mode, _CHAT_MODES["default"])


def get_chat_preference_style(selected_preference):
    if (selected_preference == "gentle_reflective"):
        return _CHAT_PREFERENCES["gentle_reflective"]
    elif (selected_preference == "direct_practical"):
        return _CHAT_PREFERENCES["direct_practical"]
    else:
        return _CHAT_PREFERENCES["general"] #BASE CASE
