import logging
from typing import Dict, Any
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# --- CONFIGURATION ---
# Initialize engines once to save loading time
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# 1. SETUP CUSTOM RECOGNIZERS (The "Client Value" part)
# Example: Catching Medical Record Numbers like "MRN-12345" or "ID: 9999"
# You can adjust the Regex pattern to match your client's actual ID format
mrn_pattern = Pattern(
    name="mrn_pattern", 
    regex=r"(?i)\b(MRN|ID|Num|#)\b\s*(?:is|number|:|#)?\s*\d{4,9}\b", 
    score=0.6
)
mrn_recognizer = PatternRecognizer(supported_entity="MEDICAL_RECORD_NUM", patterns=[mrn_pattern])
analyzer.registry.add_recognizer(mrn_recognizer)

# 2. DEFINE WHAT TO HIDE
SENSITIVE_ENTITIES = [
    "PERSON",               # Names (John Doe)
    "PHONE_NUMBER",         # (555) 123-4567
    "EMAIL_ADDRESS",        # john@example.com
    "US_SSN",               # Social Security
    "MEDICAL_RECORD_NUM",   # Our custom MRN recognizer
    "LOCATION",             # Cities/Addresses (Use carefully, can be aggressive)
]

def privacy_masking_node(state) -> Dict[str, Any]:
    """
    Scans state.user_message for PII/PHI and masks it.
    Returns an update to state.user_message.
    """
    text_to_clean = state.user_message
    
    # Safety check for empty input
    if not text_to_clean or not isinstance(text_to_clean, str):
        return {} # No changes

    try:
        # A. ANALYZE: Find the positions of sensitive data
        results = analyzer.analyze(
            text=text_to_clean,
            entities=SENSITIVE_ENTITIES,
            language='en'
        )

        # B. ANONYMIZE: Replace data with placeholders
        anonymized_result = anonymizer.anonymize(
            text=text_to_clean,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "<PRIVATE_DATA>"}),
                "PERSON": OperatorConfig("replace", {"new_value": "<NAME>"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
                "MEDICAL_RECORD_NUM": OperatorConfig("replace", {"new_value": "<MRN_ID>"}),
            }
        )
        
        cleaned_text = anonymized_result.text

        # C. LOGGING (Optional - Good for debugging, disable in Prod)
        if text_to_clean != cleaned_text:
            print(f"🔒 PRIVACY SHIELD: Masked PII in message.")
            print(f"\n   Original: {text_to_clean}\n") 
            print(f"\n   Cleaned:  {cleaned_text}\n")

        # Return the specific field update for the Pydantic state
        return {"user_message": cleaned_text}

    except Exception as e:
        print(f"ERROR in Privacy Node: {e}")
        # Fail open or closed? Here we fail open (return original) to keep chat working, 
        # but in high security, you might return an error state.
        return {"user_message": text_to_clean}