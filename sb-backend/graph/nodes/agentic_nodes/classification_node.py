from transformers import AutoTokenizer
import torch
from typing import Dict, Any
import os
import logging
from transformer_models.SoulBuddyClassifier import SoulBuddyClassifier
import re

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


def load_model():
    """Load the classification model and tokenizer."""
    global _tokenizer, _model, _model_loaded
    
    if _model_loaded:
        return
    
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = SoulBuddyClassifier(MODEL_NAME)
        
        # Load model weights if file exists
        model_path = "model_weights.pt"
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
    Comprehensive greeting detection that handles:
    - Pure greetings
    - Greetings with names
    - Greetings with punctuation
    - Avoids false positives in longer messages
    """
    
    # Define greeting patterns
    simple_greetings = ["hi", "hello", "hey", "greetings", "howdy"]
    time_greetings = ["good morning", "good afternoon", "good evening"]
    phrase_greetings = ["nice to meet you", "hello there", "hi there"]
    
    all_greetings = simple_greetings + time_greetings + phrase_greetings
    
    message_lower = message.lower().strip()
    words = message_lower.split()
    
    # If message is empty or too long, quick check
    if not message_lower:
        return False
    
    # Rule 1: Message is exactly a greeting (with possible punctuation)
    message_clean = re.sub(r'[^\w\s]', '', message_lower)
    if message_clean in all_greetings:
        return True
    
    # Rule 2: Short messages (1-3 words) starting with greeting
    if len(words) <= 3:
        first_word = words[0].rstrip('!,.').lower()
        
        # Check if first word is a simple greeting
        if first_word in simple_greetings:
            return True
        
        # Check first two words for multi-word greetings
        first_two = ' '.join(words[:2]).lower()
        if first_two in time_greetings or first_two in phrase_greetings:
            return True
    
    # Rule 3: Messages that start with greeting + anything (for very short only)
    if len(words) == 2 and words[0].rstrip('!,.').lower() in simple_greetings:
        return True
    
    return False



def get_classifications(message: str) -> Dict[str, Any]:
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
            "raw_scores": {
                "situation": 0.0,
                "severity": 0.0,
                "intent": 1.0, # Set intent score to 1 for greetings
                "risk": 0.0
            }
        }
    
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
            "raw_scores": {
                "situation": 0.0,
                "severity": crisis_detect_result["severity"] == "high" and 1.0 or 0.0, # Set severity score to 1 for high severity cases
                "intent": crisis_detect_result["intent"] == "crisis_disclosure" and 1.0 or 0.0, # Set intent score to 1 for crisis disclosures
                "risk": crisis_detect_result["risk_score"]
            }
        }

    #Checking if the message contains high risk words like "die", "kill myself", "end it all", "can't go on", "no reason to live" - if yes we can consider risk level as high regardless of the risk score from the model
    logger.info(f"Classifying message: '{message}'")
    global _tokenizer, _model, _model_loaded
    
    if not _model_loaded:
        load_model()
    
    if _tokenizer is None or _model is None:
        raise RuntimeError("Classification model failed to load")
    
    try:
        # Tokenize the input
        inputs = _tokenizer(message, return_tensors="pt", truncation=True, padding=True)
        
        model_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
        }

        # Get predictions
        with torch.no_grad():
            s_logits, sev_logits, i_logits, r_logits = _model(**model_inputs)
        
        # Extract predictions
        situation_idx = torch.argmax(s_logits, dim=1).item()
        severity_idx = torch.argmax(sev_logits, dim=1).item()
        
        intent_idx = torch.argmax(i_logits, dim=1).item()
        risk_score = float(torch.sigmoid(r_logits).item())
        
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
        #if intent score is < 0.5, we can consider intent as "unclear" in the graph node logic
        # if the situation score is < 0.5 we can consider situation as "unclear" and severity as low in the graph node logic
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

import re
import logging
from typing import Dict, Any


import logging
import re
from typing import Dict, Any

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
                r"plan to kill myself",
                r"going to kill myself",
                r"will kill myself",
                r"method to kill myself",
                r"how to kill myself",
                r"bought pills",
                r"got a gun",
                r"have a rope",
                r"found a way to end it",
                r"written a note",
                r"wrote a goodbye letter",
                r"said my goodbyes",
                r"tonight is the night",
                r"tomorrow i'll be gone",
                r"this weekend i'm ending it",
                r"(plan|planning|going)\s+to\s+(kill myself|end( it)?( all)?|suicide)",
                r"(tonight|tomorrow|this weekend|soon).*(kill myself|end( it)?( all)?|suicide)"
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
                r"i do not see any reason to live"
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
                r"would be easier if i was gone",
                r"don't want to live anymore",
                r"do not want to live anymore",
                r"don't want to live",
                r"do not want to live",
                r"i don't see any reason to live",
                r"i do not see any reason to live"
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
        "SELF_HARM": 8
        # Add other categories here as needed
    }

    matched_patterns = []
    crisis_type = None
    crisis_details = None
    highest_priority = -1

    # ===== Pattern matching with negation handling =====
    negation_words = ["not", "never", "no", "can't", "cannot", "don't", "do not"]

    for category, details in crisis_categories.items():
        for pattern in details["patterns"]:
            for match in re.finditer(pattern, message_lower):
                start, end = match.span()
                # Check if any negation appears within 5 words before the match
                window_start = max(0, start - 50)  # ~50 chars before
                context = message_lower[window_start:start]
                if any(neg in context for neg in negation_words):
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
            "die": 0.3, "dead": 0.3, "death": 0.3, "suicide": 0.5,
            "suicidal": 0.5, "kill": 0.4, "ending": 0.3, "pain": 0.2,
            "hurt": 0.2, "suffering": 0.2, "hopeless": 0.3, "despair": 0.3
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


def _no_crisis_response(message: str) -> Dict[str, Any]:
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
        "message_length": len(message) if message else 0,
        "word_count": len(message.split()) if message else 0
    }


# Example usage and testing
if __name__ == "__main__":
    test_messages = [
        "Hi, how are you?",
        "I've been feeling really sad lately",
        "I want to kill myself",
        "I have a plan to end it all tonight",
        "Sometimes I wish I could just disappear",
        "I bought some pills and I'm going to use them",
        "No one would care if I died",
        "I'm so overwhelmed with everything",
        "This is goodbye. I love you all.",
        "I can't take this pain anymore",
        "I've been cutting myself to cope",
        "What's the point of living anyway?",
        "I'm a burden to everyone around me"
    ]
    
    for msg in test_messages:
        result = detect_crisis(msg)
        print(f"\nMessage: {msg}")
        print(f"Crisis: {result['is_crisis']}")
        if result['is_crisis']:
            print(f"  Type: {result['crisis_type']}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Risk Score: {result['risk_score']}")
            print(f"  Situation: {result['situation']}")
            print(f"  Intent: {result['intent']}")
            print(f"  Immediate Response: {result['requires_immediate_response']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Matched: {result['matched_patterns'][:2]}...")

def classification_node(state: Dict[str, Any]) -> Dict[str, Any]:
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
            return {
                **state.dict(),
                "error": "No user message to classify"
            }
        
        
        classifications = get_classifications(message)
        return {
            **state.dict(),
            "intent": classifications["intent"],
            "situation": classifications["situation"],
            "severity": classifications["severity"],
            "risk_level": "high" if classifications["risk_score"] > 0.7 else "medium" if classifications["risk_score"] > 0.3 else "low",
            "is_greeting": classifications["intent"] == "greeting",
            "is_high_crisis": classifications["risk_level"] == "high",
            "classification_details": classifications,
            "is_medium_crisis": classifications["risk_level"] == "medium",
        }
        
    except Exception as e:
        logger.exception("Classification node failed")
        return {
            **state.dict(),
            "error": f"Classification failed: {str(e)}"
        }
