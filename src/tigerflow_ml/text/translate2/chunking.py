"""
Token-aware text chunking for translation.

Splits text into chunks that fit within a token limit while preserving
document structure (paragraphs, sentences) as much as possible.
"""

import re
from typing import Callable

from transformers import PreTrainedTokenizerBase

# TranslateGemma context window is 2048 tokens (input + output).
# Reserve ~248 tokens for the prompt template and split the rest
# evenly between the input chunk and the generated translation.
MAX_CHUNK_TOKENS = 900


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Count tokens in text, excluding BOS/EOS special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_text_by_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = MAX_CHUNK_TOKENS,
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
            sentence_chunks = _chunk_by_sentences(para, token_count, max_tokens, tokenizer)
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
        chunks.append(tokenizer.decode(chunk_tokens))

    return chunks