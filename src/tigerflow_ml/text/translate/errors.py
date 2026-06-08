class TranslationError(Exception):
    """Raised when translation fails."""

    pass


class AlreadyInTargetLanguageError(Exception):
    """Raised when input text is already in the
    target language"""

    pass
