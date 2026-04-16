"""
Language detection with robust sampling strategy.
"""

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection across runs
DetectorFactory.seed = 0

# ISO 639-1 language code to full name mapping
LANGUAGES = {
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "pt": "Portuguese",
    "cs": "Czech",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "hu": "Hungarian",
    "he": "Hebrew",
    "it": "Italian",
    "no": "Norwegian",
    "nb": "Norwegian",
    "nn": "Norwegian",
    "pl": "Polish",
    "sk": "Slovak",
    "sv": "Swedish",
    "tr": "Turkish",
    "ko": "Korean",
    "en": "English",
    "ca": "Catalan",
    "eu": "Basque",
}


def get_language_name(code: str) -> str:
    """Get the full language name from ISO 639-1 code."""
    return LANGUAGES.get(code, code.capitalize())


def detect_language(text: str) -> str | None:
    """
    Detect the language of the given text using a robust sampling strategy.

    Strategy:
        1. Sample from the middle of the document (more representative of content)
        2. Fall back to beginning of document if middle-sample detection fails

    Args:
        text: The full document text to analyze.

    Returns:
        ISO 639-1 language code (e.g., 'en', 'de', 'fr'), or None if detection fails.
    """
    text = text.strip()
    if len(text) < 20:
        return None

    # Try middle-of-document sample first (more representative)
    sample_start = max(0, len(text) // 4)
    sample_end = min(len(text), sample_start + 1000)
    mid_sample = text[sample_start:sample_end]

    detected = _detect_sample(mid_sample)
    if detected is not None:
        return detected

    # Fall back to beginning of document
    return _detect_sample(text[:1000])


def _detect_sample(sample: str) -> str | None:
    """Detect language from a text sample."""
    try:
        if len(sample.strip()) < 20:
            return None
        return str(detect(sample))
    except LangDetectException:
        return None


"""
FLORES-200 language code mapping for NLLB-200 and M2M-100 models.

These models reject ISO 639-1 codes (e.g. "en") and require FLORES-200 codes
(e.g. "eng_Latn"). Call to_flores() when model_type is "nllb" or "m2m_100".
Helsinki-NLP models do not use lang codes at all.
"""
# ISO 639-1 → FLORES-200
FLORES_CODES: dict[str, str] = {
    "ar": "arb_Arab",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "eu": "eus_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "nb": "nob_Latn",
    "nl": "nld_Latn",
    "nn": "nno_Latn",
    "no": "nob_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sv": "swe_Latn",
    "tr": "tur_Latn",
    "zh": "zho_Hans",
}

def to_flores(code: str) -> str:
    """Convert an ISO 639-1 code to its FLORES-200 equivalent.

    If the code is already a FLORES-200 code (contains '_') or is not in the
    mapping, it is returned unchanged.
    """
    return FLORES_CODES.get(code, code)