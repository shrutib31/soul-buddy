import re

ROMANISED = 'romanised'
CANONICAL = 'canonical'
MIXED = 'mixed'

# Broad set of common English words to distinguish English from romanised Indian text.
ENGLISH_VOCAB = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
    'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
    'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
    'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well',
    'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were', 'am', 'been', 'has',
    'had', 'did', 'does', 'doing', 'should', 'must', 'need', 'here', 'there', 'every', 'everything', 'everyone', 'something', 'anything',
    'nothing', 'glad', 'sorry', 'thanks', 'thank', 'please', 'hi', 'hello', 'hey', 'okay', 'ok', 'yes', 'no', 'fine', 'good', 'great',
    'happy', 'sad', 'tired', 'tiredness', 'stressed', 'stress', 'mind', 'feel', 'feeling', 'mood', 'today', 'tonight', 'tomorrow',
    'national', 'bird', 'curious', 'sounds', 'personal', 'wellness', 'buddy', 'soul', 'life', 'heart', 'problem', 'thought', 'talk',
}


def _is_english_content(text: str) -> bool:
    """Return True if 40%+ of words match common English vocabulary."""
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return False
    matches = sum(1 for w in words if w in ENGLISH_VOCAB)
    ratio = matches / len(words)
    return ratio >= 0.4 or (len(words) <= 2 and ratio > 0)


def classify_language_format(text: str, language: str | None = None) -> str | None:
    """
    Classify text based on its character composition.

    Returns:
      CANONICAL  — pure native script (Devanagari etc.) OR English latin text
      ROMANISED  — latin-script romanised Indian language (e.g. Hinglish)
      MIXED      — combination of native script and latin characters
      None       — empty or unclassifiable

    The `language` parameter is kept for backward compatibility and does not
    affect the classification result.
    """
    if not text or not text.strip():
        return None

    native_script = bool(re.search(r'[\u0900-\u0D7F]', text))
    latin_script = bool(re.search(r'[a-zA-Z]', text))

    if native_script and latin_script:
        return MIXED

    if native_script:
        return CANONICAL

    if latin_script:
        return CANONICAL if _is_english_content(text) else ROMANISED

    return None
