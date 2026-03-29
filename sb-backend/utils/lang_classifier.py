"""
import re

ROMANISED = 'romanised'
CANONICAL = 'canonical'
MIXED = 'mixed'

def classify_language_format(text: str) -> str:
    # Categorizes Indian language text into:
    # 1. ROMANISED (Latin script for native words)
    # 2. CANONICAL (Native script - Hn, Bn, Mr, etc.)
    # 3. MIXED (Both scripts or code-switching)
    if not text or not text.strip():
        return ROMANISED

    # Native scripts (Devanagari, Bengali, etc.)
    native_script_pattern = re.compile(r'[\u0900-\u0D7F]')
    latin_script_pattern = re.compile(r'[a-zA-Z]')

    has_native = bool(native_script_pattern.search(text))
    has_latin = bool(latin_script_pattern.search(text))

    if has_native and has_latin:
        return MIXED
    
    if has_native:
        return CANONICAL
    
    return ROMANISED # Default for Latin script
"""

