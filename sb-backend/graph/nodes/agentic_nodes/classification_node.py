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
INTENT_LABELS = {
    0: "greeting",
    1: "venting",
    2: "seek_information",
    3: "seek_understanding",
    4: "open_to_solution",
    5: "try_tool",
    6: "seek_support",
    7: "unclear"
}

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
    12: "NO_SITUATION", # Added a "no situation" label for cases where no specific situation is detected
    13: "UNCLEAR", # Added an "unclear" label for cases where the situation is not clear
    14: "SUICIDAL" # Added a "suicidal" label for cases where the situation is high risk and may indicate suicidal thoughts or intentions
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
    

    #Checking if the message contains high risk words like "die", "kill myself", "end it all", "can't go on", "no reason to live" - if yes we can consider risk level as high regardless of the risk score from the model
    logger.info(f"Classifying message: '{message}'")
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
            "is_greeting": "true" if classifications["intent"] == "greeting" else "false",
            "is_high_crisis": "true" if classifications["risk_level"] == "high" and classifications["situation"] == "suicidal" else "false",
            "classification_details": classifications,
            "is_medium_crisis": "true" if classifications["risk_level"] == "medium" and classifications["situation"] == "suicidal" else "false",
        }
        
    except Exception as e:
        logger.exception("Classification node failed")
        return {
            **state.dict(),
            "error": f"Classification failed: {str(e)}"
        }
