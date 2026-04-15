"""Unit tests for token-aware chunking helpers (no model downloads needed)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tigerflow_ml.text.translate.chunking import (
    _chunk_by_raw_tokens,
    _chunk_by_sentences,
    chunk_text_by_tokens,
    compute_chunk_size,
    count_tokens,
)

OVERHEAD = 248  # default prompt_overhead


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that counts words as tokens."""
    tokenizer = MagicMock()

    def mock_encode(text, add_special_tokens=True):
        # Simple tokenization: split on whitespace
        tokens = text.split()
        return list(range(len(tokens)))  # Return list of fake token IDs

    def mock_decode(token_ids):
        # For testing, we can't perfectly reconstruct, but this is fine
        # In real usage, decode would return the original text
        return f"[{len(token_ids)} tokens]"

    tokenizer.encode = mock_encode
    tokenizer.decode = mock_decode
    return tokenizer


class TestCountTokens:
    def test_empty_string(self, mock_tokenizer):
        assert count_tokens("", mock_tokenizer) == 0

    def test_single_word(self, mock_tokenizer):
        assert count_tokens("hello", mock_tokenizer) == 1

    def test_multiple_words(self, mock_tokenizer):
        assert count_tokens("hello world foo bar", mock_tokenizer) == 4

    def test_whitespace_only(self, mock_tokenizer):
        assert count_tokens("   ", mock_tokenizer) == 0


class TestChunkTextByTokens:
    def test_empty_text(self, mock_tokenizer):
        chunks = chunk_text_by_tokens("", mock_tokenizer, max_tokens=10)
        assert chunks == []

    def test_whitespace_only(self, mock_tokenizer):
        chunks = chunk_text_by_tokens("   \n\n   ", mock_tokenizer, max_tokens=10)
        assert chunks == []

    def test_single_short_paragraph(self, mock_tokenizer):
        text = "This is a short paragraph."
        chunks = chunk_text_by_tokens(text, mock_tokenizer, max_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_paragraphs_fit_in_one_chunk(self, mock_tokenizer):
        text = "First paragraph.\n\nSecond paragraph."
        chunks = chunk_text_by_tokens(text, mock_tokenizer, max_tokens=20)
        assert len(chunks) == 1
        assert chunks[0] == "First paragraph.\n\nSecond paragraph."

    def test_paragraphs_split_at_boundary(self, mock_tokenizer):
        # Each paragraph is 2 tokens, max is 3, so they can't fit together
        text = "First para.\n\nSecond para.\n\nThird para."
        chunks = chunk_text_by_tokens(text, mock_tokenizer, max_tokens=3)
        assert len(chunks) == 3
        assert chunks[0] == "First para."
        assert chunks[1] == "Second para."
        assert chunks[2] == "Third para."

    def test_long_paragraph_splits_into_sentences(self, mock_tokenizer):
        # Paragraph with multiple sentences, each 3 tokens
        text = "First sentence here. Second sentence here. Third sentence here."
        # Max 5 tokens, so sentences must be split
        chunks = chunk_text_by_tokens(text, mock_tokenizer, max_tokens=5)
        assert len(chunks) == 3
        assert "First sentence here." in chunks[0]
        assert "Second sentence here." in chunks[1]
        assert "Third sentence here." in chunks[2]

    def test_accumulates_sentences_up_to_limit(self, mock_tokenizer):
        # Each sentence is 2 tokens, max is 5, so 2 sentences fit together
        text = "One two. Three four. Five six. Seven eight."
        chunks = chunk_text_by_tokens(text, mock_tokenizer, max_tokens=5)
        assert len(chunks) == 2
        assert "One two. Three four." in chunks[0]
        assert "Five six. Seven eight." in chunks[1]

    def test_windows_line_endings_normalized(self, mock_tokenizer):
        text = "First para.\r\n\r\nSecond para."
        chunks = chunk_text_by_tokens(text, mock_tokenizer, max_tokens=10)
        assert len(chunks) == 1
        # Should be joined with Unix-style newlines
        assert "\r\n" not in chunks[0]

    def test_preserves_paragraph_structure_in_output(self, mock_tokenizer):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text_by_tokens(text, mock_tokenizer, max_tokens=20)
        assert len(chunks) == 1
        assert "\n\n" in chunks[0]


class TestChunkBySentences:
    def test_single_sentence(self, mock_tokenizer):
        def token_count(t):
            return count_tokens(t, mock_tokenizer)

        result = _chunk_by_sentences("Hello world.", token_count, 10, mock_tokenizer)
        assert result == ["Hello world."]

    def test_multiple_sentences_fit(self, mock_tokenizer):
        def token_count(t):
            return count_tokens(t, mock_tokenizer)

        result = _chunk_by_sentences(
            "One. Two. Three.", token_count, 10, mock_tokenizer
        )
        assert len(result) == 1
        assert result[0] == "One. Two. Three."

    def test_sentences_exceed_limit(self, mock_tokenizer):
        def token_count(t):
            return count_tokens(t, mock_tokenizer)

        # Each sentence is 2 tokens, limit is 3
        result = _chunk_by_sentences(
            "Word one. Word two. Word three.", token_count, 3, mock_tokenizer
        )
        assert len(result) == 3

    def test_handles_exclamation_and_question_marks(self, mock_tokenizer):
        def token_count(t):
            return count_tokens(t, mock_tokenizer)

        result = _chunk_by_sentences(
            "Hello! How are you? Fine.", token_count, 10, mock_tokenizer
        )
        assert len(result) == 1

    def test_handles_exclamation_and_question_marks_exceed_limit(self, mock_tokenizer):
        def token_count(t):
            return count_tokens(t, mock_tokenizer)

        result = _chunk_by_sentences(
            "Hello! How are you? Fine.", token_count, 3, mock_tokenizer
        )
        assert len(result) == 3


class TestChunkByRawTokens:
    def test_text_within_limit(self, mock_tokenizer):
        # 3 tokens, limit 10
        result = _chunk_by_raw_tokens("one two three", 10, mock_tokenizer)
        assert len(result) == 1

    def test_text_exceeds_limit(self, mock_tokenizer):
        # 6 tokens, limit 2
        result = _chunk_by_raw_tokens("one two three four five six", 2, mock_tokenizer)
        assert len(result) == 3  # ceil(6/2) = 3 chunks

    def test_exact_multiple(self, mock_tokenizer):
        # 4 tokens, limit 2
        result = _chunk_by_raw_tokens("one two three four", 2, mock_tokenizer)
        assert len(result) == 2


class TestChunkEdgeCases:
    def test_very_long_sentence_gets_token_split(self, mock_tokenizer):
        # A single sentence with many words
        long_sentence = " ".join(["word"] * 20)
        chunks = chunk_text_by_tokens(long_sentence, mock_tokenizer, max_tokens=5)
        # Should be split into 4 chunks of ~5 tokens each
        assert len(chunks) == 4

    def test_mixed_paragraph_lengths(self, mock_tokenizer):
        text = (
            "Short.\n\nThis is a much longer paragraph with many words in it."
            "\n\nAnother short."
        )
        chunks = chunk_text_by_tokens(text, mock_tokenizer, max_tokens=5)
        # First paragraph fits alone, second needs splitting, third fits alone
        assert len(chunks) > 3

    def test_empty_paragraphs_ignored(self, mock_tokenizer):
        text = "Para one.\n\n\n\n\n\nPara two."
        chunks = chunk_text_by_tokens(text, mock_tokenizer, max_tokens=5)
        assert len(chunks) == 1
        # Multiple newlines collapse to single paragraph separator
        assert "Para one." in chunks[0]
        assert "Para two." in chunks[0]

    def test_trailing_whitespace_stripped(self, mock_tokenizer):
        text = "  Paragraph with spaces.  \n\n  Another one.  "
        chunks = chunk_text_by_tokens(text, mock_tokenizer, max_tokens=20)
        assert len(chunks) == 1
        assert not chunks[0].startswith(" ")
        assert not chunks[0].endswith(" ")


def cfg(**kwargs):
    """Build a minimal fake config with only the given attributes."""
    return SimpleNamespace(**kwargs)


def multimodal_cfg(text_kwargs, **top_kwargs):
    """Build a fake multimodal config whose text fields live in text_config."""
    return SimpleNamespace(text_config=SimpleNamespace(**text_kwargs), **top_kwargs)


class TestComputeChunkSize:
    def test_max_position_embeddings(self):
        result = compute_chunk_size(cfg(max_position_embeddings=8192))
        assert result == (8192 - OVERHEAD) // 2

    def test_n_positions(self):
        result = compute_chunk_size(cfg(n_positions=1024))
        assert result == (1024 - OVERHEAD) // 2

    def test_n_ctx(self):
        result = compute_chunk_size(cfg(n_ctx=2048))
        assert result == (2048 - OVERHEAD) // 2

    def test_seq_length(self):
        result = compute_chunk_size(cfg(seq_length=512))
        assert result == (512 - OVERHEAD) // 2

    def test_seq_len(self):
        result = compute_chunk_size(cfg(seq_len=512))
        assert result == (512 - OVERHEAD) // 2

    def test_max_sequence_length(self):
        result = compute_chunk_size(cfg(max_sequence_length=4096))
        assert result == (4096 - OVERHEAD) // 2

    # sliding_window uses the direct formula (no halving)

    def test_sliding_window(self):
        result = compute_chunk_size(cfg(sliding_window=1024))
        assert result == 1024 - OVERHEAD

    def test_sliding_window_takes_priority_over_full_context(self):
        result = compute_chunk_size(
            cfg(sliding_window=1024, max_position_embeddings=131072)
        )
        assert result == 1024 - OVERHEAD

    # Multimodal models (Gemma3, etc.) nest text fields under text_config
    # Top-level config has no context fields; text_config does

    def test_multimodal_config_max_position_embeddings(self):
        result = compute_chunk_size(multimodal_cfg({"max_position_embeddings": 131072}))
        assert result == (131072 - OVERHEAD) // 2

    def test_multimodal_config_sliding_window(self):
        result = compute_chunk_size(multimodal_cfg({"sliding_window": 1024}))
        assert result == 1024 - OVERHEAD

    def test_multimodal_config_sliding_window_priority(self):
        result = compute_chunk_size(
            multimodal_cfg({"sliding_window": 1024, "max_position_embeddings": 131072})
        )
        assert result == 1024 - OVERHEAD

    def test_multimodal_config_none_does_not_crash(self):
        c = SimpleNamespace(text_config=None, max_position_embeddings=8192)
        result = compute_chunk_size(c)
        assert result == (8192 - OVERHEAD) // 2

    def test_no_matching_fields_returns_none(self):
        result = compute_chunk_size(cfg(vocab_size=32000, hidden_size=4096))
        assert result is None

    def test_none_field_value_is_skipped(self):
        # sliding_window present but None — should fall through to full context field
        result = compute_chunk_size(
            cfg(sliding_window=None, max_position_embeddings=8192)
        )
        assert result == (8192 - OVERHEAD) // 2

    def test_zero_field_value_is_skipped(self):
        result = compute_chunk_size(cfg(sliding_window=0, max_position_embeddings=8192))
        assert result == (8192 - OVERHEAD) // 2

    def test_custom_prompt_overhead(self):
        result = compute_chunk_size(
            cfg(max_position_embeddings=4096), prompt_overhead=500
        )
        assert result == (4096 - 500) // 2

    def test_chunk_size_minimum_is_one(self):
        # overhead larger than window should clamp to 1, not go negative
        result = compute_chunk_size(cfg(sliding_window=100), prompt_overhead=500)
        assert result == 1
