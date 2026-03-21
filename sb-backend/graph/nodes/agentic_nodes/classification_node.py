from typing import Dict, Any
import os
import logging
import re
from graph.state import ConversationState
from graph.nodes.agentic_nodes.guardrail import detect_out_of_scope

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
    """
    Load the classification model and tokenizer.

    Only attempted when model_weights.pt exists — avoids downloading the base
    model from HuggingFace at runtime when there are no trained weights to use.
    Callers should check _model_loaded after calling this; if the weights file
    is absent the function returns early and the ML path stays disabled.
    """
    global _tokenizer, _model, _model_loaded, _torch

    if _model_loaded:
        return

    model_path = os.getenv("MODEL_WEIGHTS_PATH", "model_weights.pt")
    if not os.path.exists(model_path):
        logger.info(
            "Classification model weights not found at %r — ML classification disabled, "
            "rule-based fallback active. Provide a trained model_weights.pt to enable it.",
            model_path,
        )
        return

    try:
        import torch
        from transformers import AutoTokenizer
        from transformer_models.SoulBuddyClassifier import SoulBuddyClassifier

        logger.info("Loading classification model from %r …", model_path)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = SoulBuddyClassifier(MODEL_NAME)
        _torch = torch

        _model.load_state_dict(torch.load(model_path, map_location="cpu"))
        _model.eval()
        _model_loaded = True
        logger.info("Classification model loaded successfully from %r", model_path)

    except Exception as e:
        logger.error("Failed to load classification model: %s", e)
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
    # (one collapse step is enough to reach a known root)
    message_lower = re.sub(r'(.)\1{2,}', r'\1\1', message_lower)

    # Strip punctuation for clean word comparison
    message_clean = re.sub(r'[^\w\s]', '', message_lower).strip()
    words = message_clean.split()
    if not words:
        return False

    # ── Rule 1: Exact match against known greeting phrases ──────────────────
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
        # common check-in phrases used as openers
        "how are you", "how are you doing", "how are you today",
        "how r u", "how ru", "how are u", "how do you do",
        "how have you been", "hope you are well", "hope youre well",
    }
    if message_clean in EXACT_GREETINGS:
        return True

    # ── Rule 2: Short messages (≤ 4 words) starting with a greeting word ────
    GREETING_STARTERS = {
        "hi", "hii", "hello", "hey", "heyy", "howdy", "hiya",
        "namaste", "namaskar", "yo", "sup", "hola", "greetings", "morning",
    }
    if len(words) <= 4 and words[0] in GREETING_STARTERS:
        return True

    # ── Rule 3: Time-based greetings as first two words ─────────────────────
    TIME_GREETING_PAIRS = {
        "good morning", "good afternoon", "good evening",
        "good night", "good day",
    }
    if len(words) >= 2 and " ".join(words[:2]) in TIME_GREETING_PAIRS:
        return True

    # ── Rule 4: "what is/what's up" with optional term of address ──────────
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



def get_classifications(message: str, domain: str = "general") -> Dict[str, Any]:
    """
    Classify a message using the SoulBuddyClassifier model.
    
    Args:
        message: The user message to classify
    
    Returns:
        Dictionary with classification results
    
    Raises:
        RuntimeError: If model is not loaded
    """
    # Check if the message is empty or only whitespace
    if not message or message.strip() == "":
        logger.warning("Received empty message for classification")
        return {
            "intent": "unclear",
            "situation": "unclear",
            "severity": "low",
            "risk_score": 0.0,
            "risk_level": "low",
            "raw_scores": {
                "situation": 0.0,
                "severity": 0.0,
                "intent": 0.0,
                "risk": 0.0
            }
        }
    
    if detect_greeting(message):
        logger.info("Message classified as greeting based on keyword matching")
        return {
            "intent": "greeting",
            "situation": "no situation",
            "severity": "low",
            "risk_score": 0.0,
            "risk_level": "low",
            "is_greeting": True,
            "raw_scores": {
                "situation": 0.0,
                "severity": 0.0,
                "intent": 1.0, # Set intent score to 1 for greetings
                "risk": 0.0
            }
        }
    
    # detect_crisis already handles the high-risk keyword override path
    # for phrases like "die", "kill myself", "end it all", "can't go on",
    # and "no reason to live" before any model inference happens.
    crisis_detect_result = detect_crisis(message)
    #  log crisis detection result for debugging
    logger.info(f"Crisis detection result: {crisis_detect_result}")
    if crisis_detect_result["is_crisis"]:
        logger.info("Message classified as high crisis based on keyword matching")
        return {
            "intent": crisis_detect_result["intent"],
            "situation": crisis_detect_result["situation"],
            "severity": crisis_detect_result["severity"],
            "risk_score": crisis_detect_result["risk_score"],
            "risk_level": crisis_detect_result["risk_level"],
            "is_crisis_detected": True,
            "raw_scores": {
                "situation": 0.0,
                "severity": crisis_detect_result["severity"] == "high" and 1.0 or 0.0, # Set severity score to 1 for high severity cases
                "intent": crisis_detect_result["intent"] == "crisis_disclosure" and 1.0 or 0.0, # Set intent score to 1 for crisis disclosures
                "risk": crisis_detect_result["risk_score"]
            }
        }

    out_of_scope_result = detect_out_of_scope(message, domain=domain)
    if out_of_scope_result["is_out_of_scope"]:
        logger.info("Message classified as out_of_scope")
        return {
            "intent": "out_of_scope",
            "situation": "NO_SITUATION",
            "severity": "low",
            "risk_score": 0.0,
            "risk_level": "low",
            "is_out_of_scope": True,
            "raw_scores": {
                "situation": 0.0,
                "severity": 0.0,
                "intent": 1.0,
                "risk": 0.0
            }
        }

    logger.info(f"Classifying message: '{message}'")
    global _tokenizer, _model, _model_loaded, _torch

    if not _model_loaded:
        load_model()

    if _tokenizer is None or _model is None or _torch is None:
        logger.info("ML model unavailable — returning rule-based default classification")
        return {
            "intent": "seek_information",
            "situation": "NO_SITUATION",
            "severity": "low",
            "risk_score": 0.0,
            "risk_level": "low",
            "is_out_of_scope": False,
            "raw_scores": {
                "situation": 0.0,
                "severity": 0.0,
                "intent": 0.0,
                "risk": 0.0,
            }
        }

    try:
        inputs = _tokenizer(message, return_tensors="pt", truncation=True, padding=True)

        model_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
        }

        # Get predictions
        with _torch.no_grad():
            s_logits, sev_logits, i_logits, r_logits = _model(**model_inputs)

        # Extract predictions
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

        logger.info(f"Classified message: intent={classifications['intent']}, severity={classifications['severity']}")
        return classifications

    except Exception as e:
        logger.exception(f"Error during classification: {str(e)}")
        raise

def is_true_negation(msg_lower: str) -> bool:
    """
    Returns True only if the message clearly negates suicidal action.
    """

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
    Detect crisis or high-risk content in a message and return a comprehensive classification.
    
    Args:
        message: The user message to check
        logger: Optional logger instance for debugging
    
    Returns:
        Dictionary with crisis classification including intent, situation, severity, and risk
    """
    logger_to_use = logger if logger else logging.getLogger(__name__)

    if not message or not isinstance(message, str):
        logger_to_use.warning("Received invalid message for crisis detection")
        return {
            "is_crisis": False,
            "intent": "unclear",
            "situation": "NO_SITUATION",
            "severity": "low",
            "risk_level": "low",
            "risk_score": 0.0,
            "requires_immediate_response": False,
            "crisis_type": None,
            "matched_patterns": [],
            "confidence": 0.0,
            "message_length": 0,
            "word_count": 0
        }
    
    message_lower = message.lower().strip()
    msg_lower = message.lower().strip()

    # Check for true negation first
    if is_true_negation(msg_lower):
        return {
            "is_crisis": False,
            "intent": "unclear",
            "situation": "NO_SITUATION",
            "severity": "low",
            "risk_level": "low",
            "risk_score": 0.0,
            "requires_immediate_response": False,
            "crisis_type": None,
            "crisis_category": None,
            "matched_patterns": [],
            "confidence": 0.0,
            "message_length": len(message),
            "word_count": len(message.split())
        }
    
    # ===== Crisis categories and patterns =====
    crisis_categories = {
        "SUICIDAL_PLAN": {
            "patterns": [
                r"plan\s+to\s+kill\s+myself",
                r"going\s+to\s+kill\s+myself",
                r"will\s+kill\s+myself",
                r"method\s+to\s+kill\s+myself",
                r"how\s+to\s+kill\s+myself",
                r"bought\s+pills",
                r"got\s+a\s+gun",
                r"have\s+a\s+rope",
                r"found\s+a\s+way\s+to\s+end\s+it",
                r"written\s+a\s+note",
                r"wrote\s+a\s+goodbye\s+letter",
                r"said\s+my\s+goodbyes",
                r"tonight\s+is\s+the\s+night",
                r"tomorrow\s+i'?ll\s+be\s+gone",
                r"this\s+weekend\s+i'?m\s+ending\s+it",
                r"(plan|planning|going)\s+to\s+(kill\s+myself|end(\s+it)?(\s+all)?|suicide)",
                r"(tonight|tomorrow|this\s+weekend|soon).*(kill\s+myself|end(\s+it)?(\s+all)?|suicide)"
            ],
            "intent": "crisis_disclosure",
            "situation": "SUICIDAL",
            "severity": "high",
            "risk_level": "critical",
            "risk_score": 1.0,
            "requires_immediate_response": True,
            "crisis_type": "active_suicidal_with_plan",
            "base_confidence": 0.95
        },
        "SUICIDAL_IDEATION": {
            "patterns": [
                r"want to kill myself",
                r"feel like killing myself",
                r"thinking of suicide",
                r"suicidal thoughts",
                r"considering suicide",
                r"thinking about ending it",
                r"want to end my life",
                r"feel like ending it all",
                r"i'm going to do it",
                r"ready to die",
                r"nothing left to live for",
                r"no reason to live",
                r"no point in living",
                r"life is not worth living",
                r"better off dead",
                r"wish i was dead",
                r"wish i were dead",
                r"don't want to live anymore",
                r"do not want to live anymore",
                r"don't want to live",
                r"do not want to live",
                r"i don't see any reason to live",
                r"i do not see any reason to live",
                r"want to die",
                r"want to be dead",
                r"feel like dying",
                r"feel like i('?m)? dying",
                r"feel like i should die",
                r"i should die",
                r"feel like i want to die",
                r"wish i would die",
                r"wish i could die",
                r"sometimes i feel like dying",
                r"sometimes i feel like i should die",
                r"i want to die"
            ],
            "intent": "crisis_disclosure",
            "situation": "SUICIDAL",
            "severity": "high",
            "risk_level": "high",
            "risk_score": 0.9,
            "requires_immediate_response": True,
            "crisis_type": "active_suicidal_ideation",
            "base_confidence": 0.9
        },
        "SELF_HARM": {
            "patterns": [
                r"cut myself",
                r"hurt myself",
                r"harm myself",
                r"self[-\s]?harm",
                r"self[-\s]?injury",
                r"injure myself",
                r"burn myself",
                r"hit myself",
                r"bang my head",
                r"scratch myself",
                r"pull my hair",
                r"want to cut",
                r"cutting myself",
                r"self[-\s]?destructive",
                r"destroy myself"
            ],
            "intent": "crisis_disclosure",
            "situation": "SELF_HARM",
            "severity": "high",
            "risk_level": "high",
            "risk_score": 0.85,
            "requires_immediate_response": True,
            "crisis_type": "self_harm",
            "base_confidence": 0.9
        },
        "PASSIVE_DEATH_WISH": {
            "patterns": [
                r"hope i don't wake up",
                r"hope i die in my sleep",
                r"wouldn't mind dying",
                r"wouldn't care if i died",
                r"don't care if i live or die",
                r"do not care if i live or die",
                r"what's the point of living",
                r"tired of living",
                r"tired of life",
                r"don't want to be here anymore",
                r"do not want to be here anymore",
                r"don't want to exist",
                r"do not want to exist",
                r"wish i could disappear",
                r"want to disappear",
                r"would be better if i wasn't here",
                r"would be easier if i was gone"
            ],
            "intent": "crisis_disclosure",
            "situation": "PASSIVE_DEATH_WISH",
            "severity": "high",
            "risk_level": "high",
            "risk_score": 0.8,
            "requires_immediate_response": True,
            "crisis_type": "passive_death_wish",
            "base_confidence": 0.85
        }
        # You can add additional categories here (PASSIVE_DEATH_WISH, SEVERE_HOPELESSNESS, etc.)
    }

    # Priority order
    priority_order = {
        "SUICIDAL_PLAN": 10,
        "SUICIDAL_IDEATION": 9,
        "SELF_HARM": 8,
        "PASSIVE_DEATH_WISH": 7
    }

    matched_patterns = []
    crisis_type = None
    crisis_details = None
    highest_priority = -1

    # ===== Pattern matching with negation handling =====
    # Use whole-word matching to avoid negations inside other words (e.g. "no" in "know").
    negation_pattern = re.compile(
        r"\b(?:not|never|no|can't|cannot|don't)\b|\bdo\s+not\b"
    )

    for category, details in crisis_categories.items():
        for pattern in details["patterns"]:
            for match in re.finditer(pattern, message_lower):
                start, end = match.span()
                # Check if any negation appears within 5 words before the match
                window_start = max(0, start - 50)  # ~50 chars before
                context = message_lower[window_start:start]
                if negation_pattern.search(context):
                    if logger_to_use:
                        logger_to_use.debug(f"Negated pattern skipped for '{pattern}'")
                    continue
                matched_patterns.append(match.group())
                priority = priority_order.get(category, 0)
                if priority > highest_priority:
                    highest_priority = priority
                    crisis_type = category
                    crisis_details = details
                break  # Only one match per pattern

    # ===== Fallback high-risk word scoring =====
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
                "intent": "venting",
                "situation": "UNLABELED_DISTRESS",
                "severity": "high",
                "risk_level": "high",
                "risk_score": min(risk_score_combined, 1.0),
                "requires_immediate_response": True,
                "crisis_type": "severe_distress",
                "base_confidence": 0.75
            }

    # ===== Prepare output =====
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
        "is_crisis": False,
        "intent": "unclear",
        "situation": "NO_SITUATION",
        "severity": "low",
        "risk_level": "low",
        "risk_score": 0.0,
        "requires_immediate_response": False,
        "crisis_type": None,
        "crisis_category": None,
        "matched_patterns": [],
        "confidence": 0.0,
        "message_length": len(message),
        "word_count": len(message.split())
    }


def classification_node(state: ConversationState) -> Dict[str, Any]:
    """
    LangGraph node for classifying conversation state.
    
    Args:
        state: The conversation state
    
    Returns:
        Updated state with classification results
    """
    try:
        message = getattr(state, "user_message", "")
        
        if not message:
            return {"error": "No user message to classify"}

        classifications = get_classifications(message, domain=getattr(state, "domain", "general"))
        return {
            "intent": classifications["intent"],
            "situation": classifications["situation"],
            "severity": classifications["severity"],
            "is_greeting": classifications.get("is_greeting", False),
            "is_crisis_detected": classifications.get("is_crisis_detected", False),
            "is_out_of_scope": classifications.get("is_out_of_scope", False),
            "risk_level": "high" if classifications["risk_score"] > 0.7 else "medium" if classifications["risk_score"] > 0.3 else "low",
        }

    except Exception as e:
        logger.exception("Classification node failed")
        return {"error": f"Classification failed: {str(e)}"}
