"""
from utils.lang_classifier import classify_language_format, ROMANISED, CANONICAL, MIXED

test_cases = [
    ("Mera naam Sayan hai", ROMANISED),
    ("আমার নাম সায়ন", CANONICAL),
    ("I am finding it tiring (overwhelmed)", ROMANISED), # Default for Latin
    ("Today I feel tired but ক্লান্ত", MIXED)
]

print("Starting Language Classifier Test...")
for text, expected in test_cases:
    result = classify_language_format(text)
    print(f"Text: '{text}' | Expected: {expected} | Result: {result} | Match: {result == expected}")
"""

