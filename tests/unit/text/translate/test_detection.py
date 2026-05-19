"""Tests for the detection module."""

from tigerflow_ml.text.translate.detection import (
    LANGUAGES,
    _detect_sample,
    detect_language,
    get_language_name,
)


class TestGetLanguageName:
    def test_known_language(self):
        assert get_language_name("en") == "English"
        assert get_language_name("de") == "German"
        assert get_language_name("fr") == "French"
        assert get_language_name("es") == "Spanish"

    def test_unknown_language_capitalized(self):
        assert get_language_name("xyz") == "Xyz"
        assert get_language_name("foo") == "Foo"

    def test_norwegian_variants(self):
        # All Norwegian variants map to "Norwegian"
        assert get_language_name("no") == "Norwegian"
        assert get_language_name("nb") == "Norwegian"
        assert get_language_name("nn") == "Norwegian"


class TestDetectSample:
    def test_short_text_returns_none(self):
        assert _detect_sample("Hi") is None
        assert _detect_sample("Hello world") is None  # < 20 chars

    def test_empty_text_returns_none(self):
        assert _detect_sample("") is None
        assert _detect_sample("   ") is None

    def test_english_text(self):
        text = "This is a sample of English text that should be detected correctly."
        result = _detect_sample(text)
        assert result == "en"

    def test_german_text(self):
        text = (
            "Dies ist ein Beispieltext auf Deutsch, der korrekt erkannt werden sollte."
        )
        result = _detect_sample(text)
        assert result == "de"

    def test_french_text(self):
        text = (
            "Ceci est un exemple de texte en français "
            "qui devrait être détecté correctement."
        )
        result = _detect_sample(text)
        assert result == "fr"

    def test_spanish_text(self):
        text = (
            "Este es un texto de ejemplo en español "
            "que debería detectarse correctamente."
        )
        result = _detect_sample(text)
        assert result == "es"


class TestDetectLanguage:
    def test_short_text_returns_none(self):
        assert detect_language("Hi") is None
        assert detect_language("Short text") is None

    def test_empty_text_returns_none(self):
        assert detect_language("") is None
        assert detect_language("   ") is None

    def test_english_document(self):
        # Longer document to test mid-sample strategy
        text = """
        This is a longer document written in English. It contains multiple
        paragraphs and sentences to ensure that the language detection
        algorithm has enough text to work with. The detection should
        sample from the middle of the document to get a representative
        sample of the content.
        """
        result = detect_language(text)
        assert result == "en"

    def test_german_document(self):
        text = """
        Dies ist ein längeres Dokument auf Deutsch geschrieben. Es enthält
        mehrere Absätze und Sätze, um sicherzustellen, dass der Algorithmus
        zur Spracherkennung genügend Text hat. Die Erkennung sollte aus
        der Mitte des Dokuments eine Stichprobe nehmen.
        """
        result = detect_language(text)
        assert result == "de"

    def test_uses_middle_sample_first(self):
        # Document where beginning is different from middle
        # (This is a bit artificial but tests the strategy)
        text = (
            "x" * 100
            + " This is definitely English text in the middle. " * 20
            + "x" * 100
        )
        result = detect_language(text)
        assert result == "en"


class TestLanguagesConstant:
    def test_common_languages_present(self):
        assert "en" in LANGUAGES
        assert "de" in LANGUAGES
        assert "fr" in LANGUAGES
        assert "es" in LANGUAGES
        assert "it" in LANGUAGES
        assert "pt" in LANGUAGES

    def test_values_are_strings(self):
        for code, name in LANGUAGES.items():
            assert isinstance(code, str)
            assert isinstance(name, str)
            assert len(code) == 2  # ISO 639-1 codes are 2 chars
