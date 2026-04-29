"""
Language detection with robust sampling strategy.
"""

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection across runs
DetectorFactory.seed = 0

# ISO 639-1 language code to full name mapping
LANGUAGES = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "eu": "Basque",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nb": "Norwegian",
    "nl": "Dutch",
    "nn": "Norwegian",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sk": "Slovak",
    "sv": "Swedish",
    "tr": "Turkish",
    "zh": "Chinese",
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
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",
    "az": "azj_Latn",  # North
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "ce": "ceb_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "eo": "epo_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",  # western
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "ga": "gle_Latn",
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "ha": "hau_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "ig": "ibo_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "ky": "kir_Cyrl",
    "ln": "lin_Latn",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",  # standard
    "mi": "mri_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",  # standard
    "mt": "mlt_Latn",
    "my": "mya_Mymr",
    "nb": "nob_Latn",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "nn": "nno_Latn",
    "no": "nob_Latn",
    "oc": "oci_Latn",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sa": "san_Deva",
    "sd": "snd_Arab",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "so": "som_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",  # codespell:ignore te
    "tg": "tgk_Cyrl",
    "th": "tha_Thai",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "yo": "yor_Latn",
    "zh": "zho_Hans",  # Simplified
    "zu": "zul_Latn",
}


def to_flores(code: str) -> str:
    """Convert an ISO 639-1 code to its FLORES-200 equivalent.

    If the code is already a FLORES-200 code (contains '_') or is not in the
    mapping, it is returned unchanged.
    """
    return FLORES_CODES.get(code, code)
