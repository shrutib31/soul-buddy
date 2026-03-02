import logging
from typing import Dict, Any
# from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
# from presidio_anonymizer import AnonymizerEngine
# from presidio_anonymizer.entities import OperatorConfig

# --- CONFIGURATION ---
# Initialize engines once to save loading time
# analyzer = AnalyzerEngine()
# anonymizer = AnonymizerEngine()

# 1. SETUP CUSTOM RECOGNIZERS (The "Client Value" part)
# Example: Catching Medical Record Numbers like "MRN-12345" or "ID: 9999"
# You can adjust the Regex pattern to match your client's actual ID format
# mrn_pattern = Pattern(
#     name="mrn_pattern", 
#     regex=r"(?i)\b(MRN|ID|Num|#)\b\s*(?:is|number|:|#)?\s*\d{4,9}\b", 
#     score=0.6
# )
# mrn_recognizer = PatternRecognizer(supported_entity="MEDICAL_RECORD_NUM", patterns=[mrn_pattern])
# analyzer.registry.add_recognizer(mrn_recognizer)

# 2. DEFINE WHAT TO HIDE
# SENSITIVE_ENTITIES = [
#     "PERSON",               # Names (John Doe)
#     "PHONE_NUMBER",         # (555) 123-4567
#     "EMAIL_ADDRESS",        # john@example.com
#     "US_SSN",               # Social Security
#     "MEDICAL_RECORD_NUM",   # Our custom MRN recognizer
#     "LOCATION",             # Cities/Addresses (Use carefully, can be aggressive)
# ]

# def privacy_masking_node(state) -> Dict[str, Any]:
#     """
#     Scans state.user_message for PII/PHI and masks it.
#     Returns an update to state.user_message.
#     """
#     text_to_clean = state.user_message
    
#     # Safety check for empty input
#     if not text_to_clean or not isinstance(text_to_clean, str):
#         return {} # No changes

#     try:
#         # A. ANALYZE: Find the positions of sensitive data
#         results = analyzer.analyze(
#             text=text_to_clean,
#             entities=SENSITIVE_ENTITIES,
#             language='en'
#         )

#         # B. ANONYMIZE: Replace data with placeholders
#         anonymized_result = anonymizer.anonymize(
#             text=text_to_clean,
#             analyzer_results=results,
#             operators={
#                 "DEFAULT": OperatorConfig("replace", {"new_value": "<PRIVATE_DATA>"}),
#                 "PERSON": OperatorConfig("replace", {"new_value": "<NAME>"}),
#                 "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
#                 "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
#                 "MEDICAL_RECORD_NUM": OperatorConfig("replace", {"new_value": "<MRN_ID>"}),
#             }
#         )
        
#         cleaned_text = anonymized_result.text

#         # C. LOGGING (Optional - Good for debugging, disable in Prod)
#         if text_to_clean != cleaned_text:
#             print(f"🔒 PRIVACY SHIELD: Masked PII in message.")
#             print(f"\n   Original: {text_to_clean}\n") 
#             print(f"\n   Cleaned:  {cleaned_text}\n")

#         # Return the specific field update for the Pydantic state
#         return {"user_message": cleaned_text}

#     except Exception as e:
#         print(f"ERROR in Privacy Node: {e}")
#         # Fail open or closed? Here we fail open (return original) to keep chat working, 
#         # but in high security, you might return an error state.
#         return {"user_message": text_to_clean}
    
## NEW CODE

import spacy
from transformers import pipeline
import re
import logging

logger = logging.getLogger(__name__)

class AdvancedPHIPIIMasker:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
        
        # We use 'simple' aggregation but we will clean up the duplicates manually
        self.ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )
        
    def mask_indian_patterns(self, text: str) -> str:
        # Aadhaar: 12 digits
        text = re.sub(r'\b\d{4}[ -]?\d{4}[ -]?\d{4}\b', "[AADHAAR]", text)
        
        # PAN Card
        text = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', "[PAN_CARD]", text, flags=re.IGNORECASE)
        
        # Phone Numbers: Catches +91..., 0..., and standard 10 digits (including 1234567890)
        text = re.sub(r'\b(?:\+91|0)?[6-9]\d{9}\b|\b\d{10}\b', "[PHONE_NUMBER]", text)
        
        # UPI & Email
        text = re.sub(r'\b[\w\.\-]+@(?:ok\w+|upi|ybl|paytm|ibl|axl)\b', "[UPI_ID]", text, flags=re.IGNORECASE)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]", text)
        
        return text

    def mask_with_transformers(self, text: str) -> str:
        entities = self.ner_pipeline(text)
        # Sort in reverse to keep indices valid
        entities = sorted(entities, key=lambda e: e['start'], reverse=True)
        
        for ent in entities:
            if ent['entity_group'] in ['PER', 'LOC', 'ORG']:
                label = "NAME" if ent['entity_group'] == 'PER' else ent['entity_group']
                text = text[:ent['start']] + f"[{label}]" + text[ent['end']:]
        
        # --- DEDUPLICATION STEP ---
        # Fixes [NAME][NAME] by merging adjacent identical tags
        text = re.sub(r'(\[NAME\])+', '[NAME]', text)
        text = re.sub(r'(\[ORG\])+', '[ORGANIZATION]', text)
        text = re.sub(r'(\[LOC\])+', '[LOCATION]', text)
        
        return text

    def comprehensive_mask(self, text: str) -> str:
        if not text: return ""
        # 1. Regex first (Very precise)
        text = self.mask_indian_patterns(text)
        # 2. AI second (Contextual)
        text = self.mask_with_transformers(text)
        return text

# Global Instance
masker_instance = AdvancedPHIPIIMasker()

def new_masking_node(state):
    # Handle MockState vs Dict
    if isinstance(state, dict):
        original_text = state.get("user_message", "")
    else:
        original_text = getattr(state, "user_message", "")
    
    if not original_text:
        return {"user_message": ""}

    try:
        protected_text = masker_instance.comprehensive_mask(original_text)
        return {"user_message": protected_text}
    except Exception as e:
        logger.error(f"Masking error: {e}")
        return {"user_message": original_text}