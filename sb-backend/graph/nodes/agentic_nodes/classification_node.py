from typing import Dict, Any
import os
import logging
import re

from graph.state import ConversationState
from graph.nodes.function_nodes.out_of_scope import detect_out_of_scope
from graph.nodes.function_nodes.out_of_scope import looks_like_general_knowledge, looks_like_nonsense
from config.settings import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"

# Label mappings

# Expanded intent labels to match detect_crisis
INTENT_LABELS = {
    0: "greeting",
    1: "venting",
    2: "seek_information",
    3: "seek_understanding",
    4: "open_to_solution",
    5: "try_tool",
    6: "seek_support",
    7: "unclear",
    8: "crisis_disclosure",  # Used in detect_crisis
    9: "seek_help",           # Possible future intent
    10: "self_harm_disclosure" # For self-harm cases
}


# Expanded situation labels to match detect_crisis
SITUATION_LABELS = {
    0: "ACADEMIC_COMPARISON",
    1: "EXAM_ANXIETY",
    2: "GENERAL_OVERWHELM",
    3: "LOW_MOTIVATION",
    4: "BELONGING_DOUBT",
    5: "UNLABELED_DISTRESS",
    6: "PASSIVE_DEATH_WISH",
    7: "HEALTH_CONCERNS",
    8: "RELATIONSHIP_ISSUES",
    9: "FINANCIAL_STRESS",
    10: "FUTURE_UNCERTAINTY",
    11: "OTHER",
    12: "NO_SITUATION",
    13: "UNCLEAR",
    14: "SUICIDAL",
    15: "SELF_HARM",           # Used in detect_crisis
    16: "SEVERE_HOPELESSNESS", # Used in detect_crisis
    17: "SEVERE_BURDEN",      # Used in detect_crisis
    18: "SEVERE_DISTRESS",    # Used in detect_crisis
    19: "GIVING_AWAY",        # Used in detect_crisis
    20: "SAYING_GOODBYE"      # Used in detect_crisis
}

SEVERITY_LABELS = {
    0: "low",
    1: "medium",
    2: "high"
}

# Global model instance
_tokenizer = None
_model = None
_model_loaded = False
_torch = None


def load_model():
    """Load the classification model and tokenizer."""
    global _tokenizer, _model, _model_loaded, _torch
    
    if _model_loaded:
        return
    
    try:
        import torch
        from transformers import AutoTokenizer
        from transformer_models.SoulBuddyClassifier import SoulBuddyClassifier

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = SoulBuddyClassifier(MODEL_NAME)
        _torch = torch
        
        # Load model weights if file exists
        model_path = settings.model.weights_path
        if os.path.exists(model_path):
            _model.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info("Classification model weights loaded successfully")
        else:
            logger.warning(f"Model weights file not found at {model_path}, using untrained model")
        
        _model.eval()
        _model_loaded = True
        logger.info("Classification model initialized")
        
    except Exception as e:
        logger.error(f"Failed to load classification model: {str(e)}")
        raise

def detect_greeting(message: str) -> bool:
    """
    Detect if a message is a conversation-initiating greeting. Covers:
    - Common words:      hi, hello, hey, namaste, yo, sup, hiya …
    - Enthusiastic form: hiii, heyyy (repeated trailing chars normalised)
    - Time-based:        good morning/afternoon/evening/night/day, gm, gn
    - Phrases:           hi there, hello there, nice to meet you …
    - Informal:          what's up, wassup, howdy, hola …
    - Check-in openers:  how are you, how r u, how have you been …
    - Casual variants:   what is up baby, what's up bro, whats up buddy …
    """
    message_lower = message.lower().strip()
    if not message_lower:
        return False

    # Normalise repeated trailing characters so "hiii" → "hii", "heyyy" → "heyy"
    message_lower = re.sub(r'(.)\1{2,}', r'\1\1', message_lower)

    # Strip punctuation for clean word comparison
    message_clean = re.sub(r'[^\w\s]', '', message_lower).strip()
    words = message_clean.split()
    if not words:
        return False

    # Rule 1: Exact match against known greeting phrases
    EXACT_GREETINGS = {
        "hi", "hii", "hello", "hey", "heyy", "greetings", "howdy",
        "hi there", "hey there", "hello there", "howdy there",
        "hiya", "hola", "yo", "sup", "wassup", "watsup", "whats up", "what is up",
        "namaste", "namaskar",
        "good morning", "good afternoon", "good evening",
        "good night", "good day", "gm", "gn",
        "morning", "afternoon", "evening",
        "nice to meet you", "great to meet you", "pleased to meet you",
        "long time no see", "hey hey",
        "how are you", "how are you doing", "how are you today",
        "how r u", "how ru", "how are u", "how do you do",
        "how have you been", "hope you are well", "hope youre well",
    }
    if message_clean in EXACT_GREETINGS:
        return True

    # Rule 2: Short messages (<= 4 words) starting with a greeting word
    GREETING_STARTERS = {
        "hi", "hii", "hello", "hey", "heyy", "howdy", "hiya",
        "namaste", "namaskar", "yo", "sup", "hola", "greetings", "morning",
    }
    if len(words) <= 4 and words[0] in GREETING_STARTERS:
        return True

    # Rule 3: Time-based greetings as first two words
    TIME_GREETING_PAIRS = {
        "good morning", "good afternoon", "good evening",
        "good night", "good day",
    }
    if len(words) >= 2 and " ".join(words[:2]) in TIME_GREETING_PAIRS:
        return True

    # Rule 4: "what is/what's up" with optional term of address
    INFORMAL_GREETING_PREFIXES = {"whats up", "what is up"}
    VOCATIVE_WORDS = {
        "baby", "babe", "bro", "bruh", "buddy", "dude", "fam", "friend",
        "girl", "guys", "homie", "king", "love", "man", "mate", "pal",
        "queen", "sir", "sis", "soulbuddy", "team", "there", "yall",
    }
    if len(words) >= 2:
        for prefix in INFORMAL_GREETING_PREFIXES:
            prefix_words = prefix.split()
            if words[:len(prefix_words)] == prefix_words:
                trailing_words = words[len(prefix_words):]
                if not trailing_words:
                    return True
                if len(trailing_words) <= 2 and all(word in VOCATIVE_WORDS for word in trailing_words):
                    return True

    return False


def is_true_negation(msg_lower: str) -> bool:
    """Returns True only if the message clearly negates suicidal action."""
    negated_action_patterns = [
        r"not going to kill myself",
        r"not going to end (it|my life)",
        r"not planning to kill myself",
        r"not planning to end (it|my life)",
        r"will not kill myself",
        r"won't kill myself",
        r"will not commit suicide",
        r"won't commit suicide",
        r"i am not suicidal",
        r"i'm not suicidal"
    ]
    for pattern in negated_action_patterns:
        if re.search(pattern, msg_lower):
            return True
    return False


def detect_crisis(message: str, logger=None) -> Dict[str, Any]:
    """
    Detect crisis or high-risk content in a message.
    Returns a comprehensive classification dict.
    """
    logger_to_use = logger if logger else logging.getLogger(__name__)

    if not message or not isinstance(message, str):
        return {
            "is_crisis": False, "intent": "unclear", "situation": "NO_SITUATION",
            "severity": "low", "risk_level": "low", "risk_score": 0.0,
            "requires_immediate_response": False, "crisis_type": None,
            "matched_patterns": [], "confidence": 0.0,
            "message_length": 0, "word_count": 0
        }

    message_lower = message.lower().strip()

    if is_true_negation(message_lower):
        return {
            "is_crisis": False, "intent": "unclear", "situation": "NO_SITUATION",
            "severity": "low", "risk_level": "low", "risk_score": 0.0,
            "requires_immediate_response": False, "crisis_type": None,
            "crisis_category": None, "matched_patterns": [], "confidence": 0.0,
            "message_length": len(message), "word_count": len(message.split())
        }

    crisis_categories = {
        "SUICIDAL_PLAN": {
            "patterns": [
                r"plan\s+to\s+kill\s+myself", r"going\s+to\s+kill\s+myself",
                r"will\s+kill\s+myself", r"method\s+to\s+kill\s+myself",
                r"how\s+to\s+kill\s+myself", r"bought\s+pills", r"got\s+a\s+gun",
                r"have\s+a\s+rope", r"found\s+a\s+way\s+to\s+end\s+it",
                r"written\s+a\s+note", r"wrote\s+a\s+goodbye\s+letter",
                r"said\s+my\s+goodbyes", r"tonight\s+is\s+the\s+night",
                r"tomorrow\s+i'?ll\s+be\s+gone", r"this\s+weekend\s+i'?m\s+ending\s+it",
                r"(plan|planning|going)\s+to\s+(kill\s+myself|end(\s+it)?(\s+all)?|suicide)",
                r"(tonight|tomorrow|this\s+weekend|soon).*(kill\s+myself|end(\s+it)?(\s+all)?|suicide)"
            ],
            "intent": "crisis_disclosure", "situation": "SUICIDAL", "severity": "high",
            "risk_level": "critical", "risk_score": 1.0, "requires_immediate_response": True,
            "crisis_type": "active_suicidal_with_plan", "base_confidence": 0.95
        },
        "SUICIDAL_IDEATION": {
            "patterns": [
                r"want to kill myself", r"feel like killing myself", r"thinking of suicide",
                r"suicidal thoughts", r"considering suicide", r"thinking about ending it",
                r"want to end my life", r"feel like ending it all", r"i'm going to do it",
                r"ready to die", r"nothing left to live for", r"no reason to live",
                r"no point in living", r"life is not worth living", r"better off dead",
                r"wish i was dead", r"wish i were dead", r"don't want to live anymore",
                r"do not want to live anymore", r"don't want to live", r"do not want to live",
                r"i don't see any reason to live", r"i do not see any reason to live",
                r"want to die", r"want to be dead", r"feel like dying",
                r"feel like i('?m)? dying", r"feel like i should die", r"i should die",
                r"feel like i want to die", r"wish i would die", r"wish i could die",
                r"sometimes i feel like dying", r"sometimes i feel like i should die",
                r"i want to die"
            ],
            "intent": "crisis_disclosure", "situation": "SUICIDAL", "severity": "high",
            "risk_level": "high", "risk_score": 0.9, "requires_immediate_response": True,
            "crisis_type": "active_suicidal_ideation", "base_confidence": 0.9
        },
        "SELF_HARM": {
            "patterns": [
                r"cut myself", r"hurt myself", r"harm myself", r"self[-\s]?harm",
                r"self[-\s]?injury", r"injure myself", r"burn myself", r"hit myself",
                r"bang my head", r"scratch myself", r"pull my hair", r"want to cut",
                r"cutting myself", r"self[-\s]?destructive", r"destroy myself"
            ],
            "intent": "crisis_disclosure", "situation": "SELF_HARM", "severity": "high",
            "risk_level": "high", "risk_score": 0.85, "requires_immediate_response": True,
            "crisis_type": "self_harm", "base_confidence": 0.9
        },
        "PASSIVE_DEATH_WISH": {
            "patterns": [
                r"hope i don't wake up", r"hope i die in my sleep",
                r"wouldn't mind dying", r"wouldn't care if i died",
                r"don't care if i live or die", r"do not care if i live or die",
                r"what's the point of living", r"tired of living", r"tired of life",
                r"don't want to be here anymore", r"do not want to be here anymore",
                r"don't want to exist", r"do not want to exist",
                r"wish i could disappear", r"want to disappear",
                r"would be better if i wasn't here", r"would be easier if i was gone"
            ],
            "intent": "crisis_disclosure", "situation": "PASSIVE_DEATH_WISH", "severity": "high",
            "risk_level": "high", "risk_score": 0.8, "requires_immediate_response": True,
            "crisis_type": "passive_death_wish", "base_confidence": 0.85
        }
    }

    priority_order = {
        "SUICIDAL_PLAN": 10, "SUICIDAL_IDEATION": 9,
        "SELF_HARM": 8, "PASSIVE_DEATH_WISH": 7
    }

    matched_patterns = []
    crisis_type = None
    crisis_details = None
    highest_priority = -1

    negation_pattern = re.compile(
        r"\b(?:not|never|no|can't|cannot|don't)\b|\bdo\s+not\b"
    )

    for category, details in crisis_categories.items():
        for pattern in details["patterns"]:
            for match in re.finditer(pattern, message_lower):
                start = match.start()
                window_start = max(0, start - 50)
                context = message_lower[window_start:start]
                if negation_pattern.search(context):
                    continue
                matched_patterns.append(match.group())
                priority = priority_order.get(category, 0)
                if priority > highest_priority:
                    highest_priority = priority
                    crisis_type = category
                    crisis_details = details
                break

    if not crisis_type:
        high_risk_words = {
            "die": 0.5, "dead": 0.4, "death": 0.4, "suicide": 0.6,
            "suicidal": 0.6, "kill": 0.5, "ending": 0.3, "pain": 0.2,
            "hurt": 0.2, "suffering": 0.2, "hopeless": 0.4, "despair": 0.4
        }
        words = message_lower.split()
        risk_score_combined = 0.0
        matched_words = []
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word in high_risk_words:
                risk_score_combined += high_risk_words[clean_word]
                matched_words.append(clean_word)
        if risk_score_combined >= 0.6 or len(matched_words) >= 3:
            matched_patterns = matched_words
            crisis_type = "SEVERE_DISTRESS"
            crisis_details = {
                "intent": "venting", "situation": "UNLABELED_DISTRESS", "severity": "high",
                "risk_level": "high", "risk_score": min(risk_score_combined, 1.0),
                "requires_immediate_response": True, "crisis_type": "severe_distress",
                "base_confidence": 0.75
            }

    if crisis_type and crisis_details:
        confidence_multiplier = min(1.0, 0.7 + (len(matched_patterns) * 0.1))
        final_confidence = crisis_details["base_confidence"] * confidence_multiplier
        return {
            "is_crisis": True,
            "intent": crisis_details["intent"],
            "situation": crisis_details["situation"],
            "severity": crisis_details["severity"],
            "risk_level": crisis_details["risk_level"],
            "risk_score": crisis_details["risk_score"],
            "requires_immediate_response": crisis_details["requires_immediate_response"],
            "crisis_type": crisis_details["crisis_type"],
            "crisis_category": crisis_type,
            "matched_patterns": matched_patterns,
            "confidence": round(final_confidence, 3),
            "message_length": len(message),
            "word_count": len(message.split())
        }

    return {
        "is_crisis": False, "intent": "unclear", "situation": "NO_SITUATION",
        "severity": "low", "risk_level": "low", "risk_score": 0.0,
        "requires_immediate_response": False, "crisis_type": None, "crisis_category": None,
        "matched_patterns": [], "confidence": 0.0,
        "message_length": len(message), "word_count": len(message.split())
    }


# ============================================================================
# POSITIVE EMOTION DETECTION
# ============================================================================

_POSITIVE_WORDS = {
    # joy / happiness
    "happy", "happiness", "excited", "exciting", "joy", "joyful", "elated", "thrilled",
    "ecstatic", "overjoyed", "delighted", "wonderful", "fantastic", "amazing", "brilliant",
    "awesome", "great", "excellent", "superb", "incredible", "magnificent",
    # positive social
    "love", "loved", "lovely", "blessed", "grateful", "thankful", "lucky", "fortunate",
    "proud", "content", "satisfied", "peaceful", "calm", "relaxed",
    # affirmatives / short enthusiastic replies
    "yes", "yep", "yup", "yeah", "absolutely", "definitely", "totally", "exactly",
    "right", "true", "indeed", "agreed",
    # celebration
    "celebrate", "celebration",
    "fun", "laugh", "laughed", "laughing", "smile", "smiling", "smiled",
    "nice", "good", "cool", "sweet", "rad", "yay", "woah", "wow",
}

# Multi-word positive phrases — tested against the full normalised text, not token-by-token
_POSITIVE_PHRASES = {"of course", "good news", "good day", "great day", "best day"}

_DISTRESS_WORDS = {
    "sad", "depressed", "anxious", "anxiety", "stressed", "stress", "overwhelmed",
    "hopeless", "helpless", "worthless", "alone", "lonely", "scared", "afraid",
    "crying", "cried", "hurt", "pain", "suffer", "suffering", "numb", "empty",
    "miserable", "desperate", "broken", "lost", "stuck", "dying",
    "dead", "kill", "harm", "cut", "hate",
}

# Multi-word distress phrases — tested against the full normalised text
_DISTRESS_PHRASES = {"give up", "gave up"}


def _has_distress(words: list, normalised_text: str) -> bool:
    return (
        any(w in _DISTRESS_WORDS for w in words)
        or any(phrase in normalised_text for phrase in _DISTRESS_PHRASES)
    )


def detect_positive_message(message: str) -> bool:
    """
    Return True when a message is clearly positive/celebratory and
    contains no distress signals.  Used to prevent the ML model from
    misclassifying short, context-dependent affirmatives like "yes so much".
    """
    normalised = re.sub(r"[^\w\s]", "", message.lower())
    words = normalised.split()
    if not words:
        return False
    if _has_distress(words, normalised):
        return False
    positive_count = sum(1 for w in words if w in _POSITIVE_WORDS)
    positive_count += sum(1 for phrase in _POSITIVE_PHRASES if phrase in normalised)
    # Short messages (≤ 6 words): one positive word/phrase is enough
    # Longer messages: require at least two positive words/phrases
    threshold = 1 if len(words) <= 6 else 2
    return positive_count >= threshold


# ============================================================================
# RULE-BASED INTENT CLASSIFICATION
# ============================================================================

# Evaluated in order — first match wins (more specific intents first)
_INTENT_PATTERNS = [
    ("open_to_solution", [
        r"\bwhat should i (do|try)\b",
        r"\bany (advice|suggestions|recommendations)\b",
        r"\bhow (can i|do i) (fix|solve|handle|deal with|improve|overcome)\b",
        r"\bwhat (would|could) (you|i) (suggest|recommend|do)\b",
        r"\blooking for (help|advice|a solution|ways to)\b",
        r"\bwhat are my options\b",
        r"\bcan you (help me|suggest|recommend)\b",
    ]),
    ("try_tool", [
        r"\b(breathing|relaxation|meditation|mindfulness) (exercise|technique|practice)\b",
        r"\b(show|give|suggest).{0,20}(exercise|technique|activity|tool)\b",
        r"\bhelp me (calm|relax|breathe|focus)\b",
        r"\b(coping|grounding) (technique|strategy|exercise)\b",
        r"\bsomething to (do|try|practice)\b",
    ]),
    ("seek_information", [
        r"\bhow (do|can|should) (i|we|you)\b",
        r"\bwhat (is|are|does|should)\b",
        r"\bcan you (explain|tell me|help me understand)\b",
        r"\btell me (about|more|how)\b",
        r"\bwhat (causes|happens|helps)\b",
        r"\bwhy (does|do|is|are)\b.{0,30}(happen|cause|occur)\b",
        r"\bany (tips|advice|resources|information)\b",
    ]),
    ("seek_understanding", [
        r"\bwhy (do|am|is|does) i (feel|think|act|behave)\b",
        r"\bhelp me understand (why|how|what)\b",
        r"\bwhat does (it|this) mean\b",
        r"\bwhy (can't|don't) i\b",
        r"\bis it normal (to|that)\b",
        r"\bam i (normal|okay|fine|weird|wrong)\b",
    ]),
    ("seek_support", [
        r"\bi need (help|support|someone|you)\b",
        r"\bplease help\b",
        r"\bi('m| am) (lost|alone|scared|terrified|desperate|broken)\b",
        r"\bi don't know (what to do|how to|where to)\b",
        r"\bno one (understands|cares|listens|helps)\b",
        r"\bi feel (so alone|completely alone|totally alone)\b",
    ]),
    ("venting", [
        r"\bi('m| am) (so |really |just )?(stressed|overwhelmed|exhausted|tired|frustrated|angry|upset|sad|anxious|depressed|lonely|lost|confused|hurt|devastated|broken|numb)\b",
        r"\bi feel (so |really |very |just )?(bad|awful|terrible|horrible|miserable|empty|hopeless|worthless|useless|like a failure)\b",
        r"\beverything (is|feels|seems) (wrong|bad|terrible|falling apart|too much)\b",
        r"\bi (can't|cannot) (do this|cope|handle|take it|go on)\b",
        r"\bit('s| is) (so hard|too hard|really hard|too much)\b",
        r"\bi('ve| have) been (struggling|having a hard time|going through)\b",
        r"\bi just wanted to (vent|talk|share|say)\b",
        r"\bnobody (cares|understands|listens)\b",
    ]),
]


def classify_intent(message: str) -> str:
    """Classify intent using keyword/pattern matching. Returns first match."""
    msg = message.lower()
    for intent, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, msg):
                return intent
    return "unclear"


# ============================================================================
# RULE-BASED SITUATION CLASSIFICATION
# ============================================================================

# Evaluated in order — first match wins
_SITUATION_PATTERNS = [
    ("EXAM_ANXIETY", [
        r"\b(exam|test|finals|quiz|midterm|assessment|board exam)\b",
        r"\bstudying (for|all)\b",
        r"\bfail(ing)? (the|my|an)? ?(exam|test|class|course)\b",
    ]),
    ("ACADEMIC_COMPARISON", [
        r"\b(grades?|gpa|marks?|score)\b.{0,40}\b(others?|everyone|classmates?|friends?|peers?)\b",
        r"\b(everyone|others?|classmates?|peers?)\b.{0,40}\b(better|smarter|ahead)\b",
        r"\bi('m| am) (falling behind|behind (everyone|others|my class))\b",
        r"\btop(per)? of (the|my) class\b",
    ]),
    ("RELATIONSHIP_ISSUES", [
        r"\b(boyfriend|girlfriend|partner|spouse|husband|wife|ex)\b",
        r"\b(breakup|broke up|cheated|cheating)\b",
        r"\b(best friend|close friend|friend).{0,30}(left|ignored|betrayed|hurt|lost)\b",
        r"\bfamily (issues?|problems?|conflict|fight|tension)\b",
        r"\bparents?.{0,20}(fighting|divorced|pressure|expectations?)\b",
    ]),
    ("FINANCIAL_STRESS", [
        r"\b(money|finances?|debt|broke|loan|rent|bills?)\b",
        r"\bcan'?t afford\b",
        r"\bfinancial (stress|pressure|problem|crisis)\b",
        r"\bno money\b",
    ]),
    ("HEALTH_CONCERNS", [
        r"\b(sick|illness|disease|diagnosis|doctor|hospital|medication)\b",
        r"\b(chronic|panic attack|anxiety disorder)\b",
        r"\b(physical|body|health) (pain|issue|problem|concern)\b",
    ]),
    ("BELONGING_DOUBT", [
        r"\b(don't|do not) (fit in|belong)\b",
        r"\b(outcast|isolated|excluded|left out|invisible)\b",
        r"\bnobody (likes?|wants?|notices?|cares? about) me\b",
        r"\bi (have no|don't have any) friends?\b",
        r"\b(lonely|loneliness)\b",
    ]),
    ("LOW_MOTIVATION", [
        r"\b(can't|cannot|don't want to) (start|do|work|study|get up|get out of bed)\b",
        r"\b(lazy|procrastinat|no motivation|unmotivated|stuck)\b",
        r"\bwhat'?s the point\b",
        r"\b(lost|lack|no) (purpose|direction|goal|drive|motivation)\b",
    ]),
    ("FUTURE_UNCERTAINTY", [
        r"\b(career|job|future|degree|major|college|university|graduation)\b.{0,40}\b(don't know|unsure|worried|scared|confused)\b",
        r"\b(don't know|unsure|confused) (what|where|how).{0,30}(future|career|life|going)\b",
        r"\bwhat (do|should) i do with my (life|future|career)\b",
        r"\blife (after|post|beyond) (college|graduation|school|university)\b",
    ]),
    ("GENERAL_OVERWHELM", [
        r"\b(overwhelmed|burned? out|burnt out)\b",
        r"\btoo (much|many) (things|tasks|responsibilities|problems)\b",
        r"\b(can't keep up|falling apart|too much at once)\b",
    ]),
]


def classify_situation(message: str) -> str:
    """Classify situation using keyword/pattern matching. Returns first match."""
    msg = message.lower()
    for situation, patterns in _SITUATION_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, msg):
                return situation
    return "NO_SITUATION"


# ============================================================================
# RULE-BASED SEVERITY CLASSIFICATION
# ============================================================================

_HIGH_SEVERITY_PATTERNS = [
    r"\b(can't|cannot) (take it|cope|go on|breathe|function|do this) (anymore|any more|at all)\b",
    r"\b(breaking|falling) (down|apart)\b",
    r"\b(desperate|desperation|hopeless|helpless|worthless|useless)\b",
    r"\bnothing (helps?|works?|matters?|is going to change)\b",
    r"\b(gave up|given up|no hope|no way out)\b",
    r"\b(completely|totally|absolutely) (lost|alone|destroyed|shattered|empty|numb)\b",
    r"\bworst.{0,15}(ever|of my life|i'?ve ever)\b",
    r"\bdon'?t (see|have) (any )?(reason|point) to (live|go on)\b",
]

_LOW_SEVERITY_PATTERNS = [
    r"\b(a (bit|little|little bit)|slightly|kind of|sort of|somewhat)\b",
    r"\b(sometimes|occasionally|every now and then|once in a while)\b",
    r"\bnot (too|that|very) (bad|serious|big)\b",
    r"\bjust (wanted|need) to (vent|talk|share)\b",
    r"\b(minor|small|manageable)\b",
]

_MEDIUM_SEVERITY_PATTERNS = [
    r"\b(stressed|stress|anxious|anxiety|worried|worry|overwhelmed|struggling|"
    r"hard time|difficult|depressed|depression|sad|upset|frustrated)\b",
]


def classify_severity(message: str) -> str:
    """Classify severity based on distress language intensity."""
    msg = message.lower()

    for pattern in _HIGH_SEVERITY_PATTERNS:
        if re.search(pattern, msg):
            return "high"

    for pattern in _LOW_SEVERITY_PATTERNS:
        if re.search(pattern, msg):
            return "low"

    for pattern in _MEDIUM_SEVERITY_PATTERNS:
        if re.search(pattern, msg):
            return "medium"

    return "low"


# ============================================================================
# OUT-OF-SCOPE DETECTION
# ============================================================================
# Flags messages where the user is explicitly requesting the bot to do something
# outside the mental wellness domain (cooking, coding, legal, financial advice, etc.).
#
# KEY DESIGN RULE: patterns must combine a REQUEST VERB directed at the bot
# AND an off-domain topic in a single phrase.  Bare topic words (recipe, code, law)
# must NOT appear as standalone patterns — that would incorrectly flag personal
# narratives like "I told my friend about a recipe" or "I was coding all night".

_OUT_OF_SCOPE_PATTERNS = [
    # ── Cooking / Food ────────────────────────────────────────────────────────
    r"\b(give|tell|share|send|show|write|suggest|recommend)\s+(me\s+)?(a\s+|an\s+|some\s+)?(recipe|recipes|ingredients?|cooking instructions?|how to cook|how to make|how to bake|dish|meal plan)\b",
    r"\b(how\s+(do|can|should)\s+i\s+(cook|bake|prepare|make)\b)",
    r"\bwhat\s+(should\s+i\s+(cook|bake|eat|make)|can\s+i\s+cook|can\s+i\s+make)\b",

    # ── Programming / Tech ────────────────────────────────────────────────────
    r"\b(write|create|build|generate|make|fix|debug|code|implement)\s+(me\s+)?(a\s+|an\s+|some\s+)?(code|function|script|program|app|application|algorithm|class|module|snippet)\b",
    r"\b(write|create|build|fix|debug)\s+me\s+.{0,20}(code|function|script|program|app)\b",
    r"\bhelp\s+me\s+(write|build|create|fix|debug).{0,30}(code|function|script|program|app|bug)\b",
    r"\bhelp\s+me\s+debug\b",
    r"\bhow\s+(do|can|should)\s+i\s+(code|program|write\s+(a\s+)?function|build\s+(a\s+)?app|fix\s+(a\s+)?bug)\b",
    r"\bwrite\s+(me\s+)?(a\s+)?(python|javascript|java|sql|html|css|typescript|react|node)\b",
    r"\b(what\s+is|explain)\s+the\s+(syntax|code|algorithm|data structure|design pattern)\s+(for|of|in)\b",

    # ── Legal ─────────────────────────────────────────────────────────────────
    r"\b(give|provide|tell|explain)\s+(me\s+)?(legal\s+(advice|guidance|opinion|help)|your\s+legal\s+opinion)\b",
    r"\bwhat\s+(is\s+the\s+law|are\s+my\s+legal\s+rights|should\s+i\s+do\s+legally)\b",
    r"\b(draft|write|prepare)\s+(me\s+)?(a\s+|an\s+)?(legal\s+document|contract|will|lawsuit|legal\s+letter)\b",
    r"\bam\s+i\s+(legally|liable|entitled\s+to)\b",

    # ── Financial / Investment ────────────────────────────────────────────────
    r"\b(give|provide|tell)\s+(me\s+)?(financial\s+(advice|tips|planning|guidance)|investment\s+(advice|tips|recommendations?))\b",
    r"\b(should\s+i\s+invest|how\s+(do|should)\s+i\s+invest|where\s+to\s+invest)\b",
    r"\b(best\s+)?(stocks?|crypto|mutual\s+funds?|etf)\s+to\s+(buy|invest|pick)\b",
    r"\bwhat\s+stocks?\s+should\s+i\s+(buy|invest|pick)\b",
    r"\bshould\s+i\s+(buy|invest\s+in|pick)\s+(stocks?|crypto|shares?|etf|mutual\s+funds?)\b",
    r"\b(help\s+me\s+)?(file\s+(my\s+)?tax(es)?|do\s+(my\s+)?taxes|calculate\s+(my\s+)?tax)\b",

    # ── Entertainment recommendations ─────────────────────────────────────────
    r"\b(recommend|suggest|tell\s+me)\s+(a\s+|an\s+|some\s+)?(good\s+)?(movie|film|show|series|anime|book|novel|song|playlist|game)\s+to\s+(watch|read|play|listen)\b",
    r"\b(what\s+(movies?|shows?|books?|games?|songs?|anime)\s+should\s+i\s+(watch|read|play|listen))\b",

    # ── Travel / Logistics ────────────────────────────────────────────────────
    r"\b(plan|book|suggest|help\s+me\s+plan)\s+(my\s+|a\s+|an\s+)?(trip|vacation|holiday|travel|itinerary|flight|hotel)\b",
    r"\b(best\s+places?\s+to\s+visit|where\s+should\s+i\s+(travel|go\s+for\s+vacation))\b",

    # ── Academic / Homework (non-wellness) ────────────────────────────────────
    r"\b(solve|answer|do|complete|help\s+me\s+(with|solve|do|answer))\s+(my\s+|this\s+|the\s+)?(homework|assignment|math\s+problem|physics\s+problem|chemistry\s+problem|essay)\b",
    r"\b(write\s+(my|the|an?)\s+(essay|report|thesis|assignment|homework))\b",
    r"\b(solve\s+this\s+(equation|problem|question|sum))\b",

    # ── Math / Calculations ───────────────────────────────────────────────────
    r"\bwhat\s+is\s+(the\s+)?(average|mean|median|mode|sum|total|product|difference|quotient|square\s+root|percentage)\s+of\b",
    r"\b(calculate|compute|find|evaluate)\s+(the\s+|my\s+)?(average|mean|median|mode|sum|total|percentage|interest|profit|loss|gpa|cgpa|discount|tax\s+amount)\b",
    r"\bwhat\s+is\s+\d+\s*[\+\-\*\/\^]\s*\d+\b",    # "what is 5 + 3", "what is 10 / 2"
    r"\bwhat\s+is\s+\d+\s*percent\s+of\b",           # "what is 20 percent of 150"
    r"\b(convert|how\s+many)\s+.{0,20}\s+to\s+(km|miles|kg|pounds|celsius|fahrenheit|dollars|rupees|euros)\b",

    # ── Weather / News / Sports ───────────────────────────────────────────────
    r"\b(what'?s\s+the\s+weather|weather\s+(today|tomorrow|this\s+week|forecast))\b",
    r"\b(latest\s+news|breaking\s+news|what'?s\s+happening\s+in\s+the\s+news)\b",
    r"\b(cricket|football|soccer|basketball|nfl|nba|ipl)\s+(score|match|result|schedule|live)\b",
]


def classify_out_of_scope(message: str) -> bool:
    """
    Return True only when the user is explicitly requesting the bot to perform
    an off-domain task (cooking, coding, legal/financial advice, etc.) or sends
    a general-knowledge query / nonsensical input.

    Personal narratives that merely *mention* off-domain topics are NOT flagged:
      ✗ out-of-scope  →  "Can you give me a recipe for pasta?"
      ✓ in-scope      →  "I spoke to my friend about a recipe today"
      ✓ in-scope      →  "I was coding all night and I'm stressed"
    """
    if not message or not isinstance(message, str):
        return False
    msg = message.lower().strip()
    if looks_like_general_knowledge(msg) or looks_like_nonsense(msg):
        return True
    for pattern in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, msg):
            return True
    return False


# ============================================================================
# MAIN CLASSIFICATION ENTRY POINT
# ============================================================================

def get_classifications(message: str) -> Dict[str, Any]:
    """
    Classify a message.

    Evaluation order:
    1. Empty message  → unclear / NO_SITUATION / low
    2. Greeting       → greeting intent, no situation, low risk
    3. Out-of-scope   → redirect template, no model call
    4. Crisis         → crisis_disclosure, specific situation, high risk
    5. ML model       → SoulBuddyClassifier for intent / situation / severity
    """
    if not message or message.strip() == "":
        logger.warning("Received empty message for classification")
        return {
            "intent": "unclear", "situation": "NO_SITUATION",
            "severity": "low", "risk_score": 0.0, "risk_level": "low",
            "raw_scores": {"situation": 0.0, "severity": 0.0, "intent": 0.0, "risk": 0.0}
        }

    if detect_greeting(message):
        logger.info("Message classified as greeting")
        return {
            "intent": "greeting", "situation": "NO_SITUATION",
            "severity": "low", "risk_score": 0.0, "risk_level": "low",
            "is_greeting": True,
            "raw_scores": {"situation": 0.0, "severity": 0.0, "intent": 1.0, "risk": 0.0}
        }

    if classify_out_of_scope(message):
        logger.info("Message classified as out-of-scope")
        msg_lower = message.lower().strip()
        if looks_like_nonsense(msg_lower):
            oos_reason = "nonsense"
        elif looks_like_general_knowledge(msg_lower):
            oos_reason = "general_knowledge"
        else:
            oos_reason = "other_out_of_scope"
        return {
            "intent": "out_of_scope", "situation": "NO_SITUATION",
            "severity": "low", "risk_score": 0.0, "risk_level": "low",
            "is_out_of_scope": True,
            "out_of_scope_reason": oos_reason,
            "raw_scores": {"situation": 0.0, "severity": 0.0, "intent": 1.0, "risk": 0.0}
        }

    crisis_result = detect_crisis(message)
    logger.info("Crisis detection result: %s", crisis_result)
    if crisis_result["is_crisis"]:
        logger.info("Message classified as crisis")
        return {
            "intent": crisis_result["intent"],
            "situation": crisis_result["situation"],
            "severity": crisis_result["severity"],
            "risk_score": crisis_result["risk_score"],
            "risk_level": crisis_result["risk_level"],
            "is_crisis_detected": True,
            "raw_scores": {
                "situation": 0.0,
                "severity": 1.0 if crisis_result["severity"] == "high" else 0.0,
                "intent": 1.0 if crisis_result["intent"] == "crisis_disclosure" else 0.0,
                "risk": crisis_result["risk_score"]
            }
        }

    out_of_scope_result = detect_out_of_scope(
        message,
        domain="general",
        allow_llm_fallback=False,
    )
    if out_of_scope_result["is_out_of_scope"]:
        logger.info("Message classified as out_of_scope")
        return {
            "intent": "out_of_scope",
            "situation": "NO_SITUATION",
            "severity": "low",
            "risk_score": 0.0,
            "risk_level": "low",
            "is_out_of_scope": True,
            "out_of_scope_reason": out_of_scope_result.get("reason"),
            "raw_scores": {
                "situation": 0.0,
                "severity": 0.0,
                "intent": 1.0,
                "risk": 0.0
            }
        }

    # ── Positive emotion short-circuit ───────────────────────────────────────
    # Prevents the ML model from misclassifying joyful / affirmative messages.
    # "yes so much", "that was amazing", "I'm so excited" etc. must NOT come
    # back as distress — the LLM has full conversation history and will handle
    # these correctly once we return a neutral classification.
    if detect_positive_message(message):
        logger.info("Message classified as positive/celebratory — skipping ML model")
        return {
            "intent": "unclear",
            "situation": "NO_SITUATION",
            "severity": "low",
            "risk_score": 0.0,
            "risk_level": "low",
            "raw_scores": {"situation": 0.0, "severity": 0.0, "intent": 0.0, "risk": 0.0}
        }

    # ── Short-message guard ───────────────────────────────────────────────────
    # Very short messages (≤ 4 words) with no rule-based distress signal are
    # ambiguous out of context.  Rather than letting the ML model guess, return
    # neutral and let the LLM use conversation history to respond correctly.
    word_count = len(message.strip().split())
    if word_count <= 4 and classify_situation(message) == "NO_SITUATION" and classify_severity(message) == "low":
        logger.info("Short message with no distress signal — skipping ML model (word_count=%d)", word_count)
        return {
            "intent": "unclear",
            "situation": "NO_SITUATION",
            "severity": "low",
            "risk_score": 0.0,
            "risk_level": "low",
            "raw_scores": {"situation": 0.0, "severity": 0.0, "intent": 0.0, "risk": 0.0}
        }

    # ── ML model inference ────────────────────────────────────────────────────
    logger.info("Classifying message with ML model: '%s'", message)
    global _tokenizer, _model, _model_loaded, _torch

    if not _model_loaded:
        load_model()

    if _tokenizer is None or _model is None or _torch is None:
        raise RuntimeError("Classification model failed to load")

    try:
        inputs = _tokenizer(message, return_tensors="pt", truncation=True, padding=True)
        model_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
        }

        with _torch.no_grad():
            s_logits, sev_logits, i_logits, r_logits = _model(**model_inputs)

        situation_idx = _torch.argmax(s_logits, dim=1).item()
        severity_idx = _torch.argmax(sev_logits, dim=1).item()
        intent_idx = _torch.argmax(i_logits, dim=1).item()
        risk_score = float(_torch.sigmoid(r_logits).item())

        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"

        classifications = {
            "intent": INTENT_LABELS.get(intent_idx, "unknown"),
            "situation": SITUATION_LABELS.get(situation_idx, "unknown"),
            "severity": SEVERITY_LABELS.get(severity_idx, "unknown"),
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "raw_scores": {
                "situation": float(s_logits.max().item()),
                "severity": float(sev_logits.max().item()),
                "intent": float(i_logits.max().item()),
                "risk": risk_score
            }
        }
        if float(i_logits.max().item()) < 0.5:
            classifications["intent"] = "unclear"
        if float(s_logits.max().item()) < 0.5:
            classifications["situation"] = "unclear"
            classifications["severity"] = "low"

        logger.info(
            "ML model classification: intent=%s severity=%s",
            classifications["intent"], classifications["severity"]
        )
        return classifications

    except Exception as e:
        logger.exception("Error during ML model classification: %s", str(e))
        raise


# ============================================================================
# LANGGRAPH NODE
# ============================================================================

def _escalate_risk_with_user_memory(
    risk_level: str,
    risk_score: float,
    user_memory: Dict[str, Any],
) -> str:
    """
    Adjust the ML-derived risk_level upward when the user's cross-session
    memory contains known risk signals.

    Rules (conservative — only escalate, never de-escalate):
      - risk_signals present (non-empty) + current risk_level == "medium"
        → escalate to "high"
      - emotional_baseline == "high_distress" + current risk_level == "low"
        → escalate to "medium"

    This prevents under-detecting risk in users who have a documented history
    of high-risk disclosures but whose current message looks mild in isolation.
    """
    if not user_memory:
        return risk_level

    risk_signals = user_memory.get("risk_signals")
    emotional_baseline = user_memory.get("emotional_baseline", "")

    # Escalate medium → high when there are known risk signals
    if risk_signals and risk_level == "medium":
        logger.info(
            "Risk escalated medium→high based on user_memory risk_signals | score=%.3f",
            risk_score,
        )
        return "high"

    # Escalate low → medium when baseline is documented as high distress
    if emotional_baseline == "high_distress" and risk_level == "low":
        logger.info(
            "Risk escalated low→medium based on user_memory emotional_baseline=high_distress | score=%.3f",
            risk_score,
        )
        return "medium"

    return risk_level


def classification_node(state: ConversationState) -> Dict[str, Any]:
    """LangGraph node: classify the user message and update state."""
    try:
        message = getattr(state, "user_message", "")
        if not message:
            return {"error": "No user message to classify"}

        classifications = get_classifications(message)

        risk_score = classifications["risk_score"]
        raw_risk_level = (
            "high" if risk_score > 0.7 else "medium" if risk_score > 0.3 else "low"
        )

        # Apply user memory risk escalation for cognito users
        user_memory = getattr(state, "user_memory", None) or {}
        final_risk_level = _escalate_risk_with_user_memory(raw_risk_level, risk_score, user_memory)

        # Derive emotion_intensity from risk_score (0.0–1.0).
        # risk_score is a sigmoid output that captures emotional distress intensity —
        # a reasonable proxy until a dedicated emotion intensity model is added.
        emotion_intensity = round(risk_score, 3)

        return {
            "intent": classifications["intent"],
            "situation": classifications["situation"],
            "severity": classifications["severity"],
            "is_greeting": classifications.get("is_greeting", False),
            "is_crisis_detected": classifications.get("is_crisis_detected", False),
            "is_out_of_scope": classifications.get("is_out_of_scope", False),
            "out_of_scope_reason": classifications.get("out_of_scope_reason"),
            "risk_level": final_risk_level,
            "emotion_intensity": emotion_intensity,
        }

    except Exception as e:
        logger.exception("Classification node failed")
        return {"error": f"Classification failed: {str(e)}"}
