"""
Token-aware text chunking for translation.

Splits text into chunks that fit within a token limit while preserving
document structure (paragraphs, sentences) as much as possible.
"""

import re
from collections.abc import Callable
from typing import cast

from transformers import PretrainedConfig, PreTrainedTokenizerBase

from .utils import ConfigParsingError

# TranslateGemma context window is 2048 tokens (input + output).
# Reserve ~248 tokens for the prompt template and split the rest
# evenly between the input chunk and the generated translation.
FALLBACK_MAX_CHUNK_TOKENS = 900

# context windows range in sizes, assuming a 128K token context
# window on the larger end to be the cap.
MAX_CHUNK_TOKENS = 63500
MIN_CHUNK_TOKENS = 250


def compute_chunk_size(config: PretrainedConfig, prompt_overhead: int = 248) -> int:
    """
    Derive max input chunk tokens from a model config.

    Uses the first matching context-window field found, checking the top-level
    config and then text_config (for multimodal models like Gemma3).

    For sliding_window the local window is already small enough to use directly.

    Raises:
        ConfigParsingError if no relevant fields are found
    """
    # Fields that represent the model's full context window
    full_context_fields = [
        "n_ctx",
        "seq_length",
        "seq_len",
        "max_sequence_length",
        "n_positions",
        "max_position_embeddings",
    ]
    # Fields that represent a local attention window — already small, so don't halve
    sliding_fields = ["sliding_window"]

    # For multimodal models (e.g. Gemma3) the relevant fields live in text_config
    configs_to_check = [config]
    if hasattr(config, "text_config") and config.text_config is not None:
        configs_to_check.append(config.text_config)

    def _first_positive(cfg, fields):
        for field in fields:
            value = getattr(cfg, field, None)
            if value is not None and value > 0:
                return value
        return None

    for cfg in configs_to_check:
        if (value := _first_positive(cfg, sliding_fields)) is not None:
            return max(value - prompt_overhead, MIN_CHUNK_TOKENS)
        if (value := _first_positive(cfg, full_context_fields)) is not None:
            return max((value - prompt_overhead) // 2, MIN_CHUNK_TOKENS)

    raise ConfigParsingError(
        "Could not compute an optimal chunk size from the model's config file. "
        "This is likely because of an unexpected format, or missing fields. "
        "Please raise an issue on Github specifying the model being used."
    )


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Count tokens in text, excluding BOS/EOS special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def compute_prompt_overhead(
    prompt_template: str,
    tokenizer: PreTrainedTokenizerBase,
    source_lang: str = "en",
    target_lang: str = "de",
) -> int:
    """Count tokens consumed by the prompt template, excluding the {text} slot."""
    prompt_without_text = prompt_template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        text="",
    )
    prompt_tokens = count_tokens(prompt_without_text, tokenizer)

    if getattr(tokenizer, "chat_template", None):
        empty = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=True,
            add_generation_prompt=True,
        )
        chat_overhead = len(empty)
    else:
        chat_overhead = 0
    return prompt_tokens + chat_overhead


def chunk_text_by_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = FALLBACK_MAX_CHUNK_TOKENS,
) -> list[str]:
    """
    Split text into chunks that fit within a token limit.

    Strategy:
        1. First, try to split at paragraph boundaries (\\n\\n)
        2. If a paragraph is too long, split at sentence boundaries (.!?)
        3. If a sentence is still too long, hard-split by token count

    This preserves document structure as much as possible while guaranteeing
    each chunk fits within the model's context window.

    Args:
        text: The full document text to split.
        tokenizer: HuggingFace tokenizer for counting tokens.
        max_tokens: Maximum tokens per chunk.

    Returns:
        List of text chunks, each guaranteed to be <= max_tokens.
    """
    # Normalize line endings and split into paragraphs
    paragraphs = text.replace("\r\n", "\n").split("\n\n")

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    def token_count(t: str) -> int:
        return count_tokens(t, tokenizer)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = token_count(para)

        if para_tokens > max_tokens:
            # Paragraph exceeds limit - flush current chunk and split paragraph
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split paragraph into sentences
            sentence_chunks = _chunk_by_sentences(
                para, token_count, max_tokens, tokenizer
            )
            chunks.extend(sentence_chunks)

        elif current_tokens + para_tokens > max_tokens:
            # Adding paragraph would exceed limit - flush and start new chunk
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            # Paragraph fits in current chunk
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def _chunk_by_sentences(
    paragraph: str,
    token_count: Callable[[str], int],
    max_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """Split a paragraph by sentences, falling back to token splits if needed."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        sent_tokens = token_count(sent)

        if sent_tokens > max_tokens:
            # Single sentence exceeds limit - flush and hard-split by tokens
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            token_chunks = _chunk_by_raw_tokens(sent, max_tokens, tokenizer)
            chunks.extend(token_chunks)

        elif current_tokens + sent_tokens > max_tokens:
            # Adding sentence would exceed limit - flush and start new
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_tokens = sent_tokens
        else:
            # Sentence fits
            current_chunk.append(sent)
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _chunk_by_raw_tokens(
    text: str,
    max_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """Hard-split text by token count (may split mid-word)."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[str] = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(cast(str, tokenizer.decode(chunk_tokens)))

    return chunks
