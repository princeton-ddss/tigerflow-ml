"""
Translator protocol and HuggingFace implementation.

Translators handle single-chunk translation only. Chunking, retry logic,
and orchestration are handled by the orchestration module.
"""

from typing import Protocol

from transformers import PreTrainedTokenizerBase

from .chunking import MAX_CHUNK_TOKENS, chunk_text_by_tokens, count_tokens
from .utils import TranslationError


class Translator(Protocol):
    """Protocol for translation backends."""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single chunk of text."""
        ...

    def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        """Translate multiple chunks. Default loops over translate()."""
        ...


class HuggingFaceTranslator:
    """TranslateGemma using Hugging Face Transformers backend."""

    def __init__(
        self,
        model_name: str = "google/translategemma-12b-it",
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
        batch_size: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens

        import torch
        from transformers import pipeline

        print(f"Loading model: {model_name}...")
        print("(This may take a few minutes on first run as the model downloads)")

        self.pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("Model loaded!")

        if batch_size is None:
            self.batch_size = self._auto_batch_size()
            print(f"Auto batch size: {self.batch_size}")
        else:
            self.batch_size = batch_size

    def _auto_batch_size(self) -> int:
        """Estimate batch size from free VRAM and model KV-cache footprint."""
        import torch

        if not torch.cuda.is_available():
            return 1
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            cfg = self.pipe.model.config
            if hasattr(cfg, "text_config"):
                cfg = cfg.text_config
            n_layers = cfg.num_hidden_layers
            n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            # KV cache per sequence: 2 (K+V) * layers * kv_heads * head_dim * tokens * 2 bytes (bfloat16)
            # 1.5x overhead for activations and intermediate buffers
            per_seq_bytes = int(2 * n_layers * n_kv_heads * head_dim * self.max_chunk_tokens * 2 * 1.5)
            return max(1, min(int(free_bytes // per_seq_bytes), 256))
        except:
            return 1

    def _build_messages(self, text: str, source_lang: str, target_lang: str) -> list[dict[str, object]]:
        """Build the message payload for a single chunk."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text,
                    }
                ],
            }
        ]

    def is_truncated(self, text: str) -> bool:
        """Check if output was likely truncated by hitting max_new_tokens."""
        return count_tokens(text, self.tokenizer) >= self.max_chunk_tokens

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate a single chunk of text.

        Raises:
            TranslationError: If translation returns empty or truncated output.
        """
        messages = self._build_messages(text, source_lang, target_lang)
        output = self.pipe(
            text=messages,
            do_sample=False,
            pad_token_id=1,
            max_new_tokens=self.max_chunk_tokens,
        )
        result: str = output[0]["generated_text"][-1]["content"]

        if not result or not result.strip():
            raise TranslationError("Translation returned empty result")

        if self.is_truncated(result):
            raise TranslationError(f"Output truncated (hit {self.max_chunk_tokens} token limit)")

        return result

    def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        """
        Translate multiple chunks using batched inference.

        Raises:
            TranslationError: If any translation returns empty or truncated output.
        """
        results = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]

            if len(texts) > 1:
                batch_end = batch_start + len(batch)
                print(f"    Translating chunks {batch_start + 1}-{batch_end}/{len(texts)}...")

            batch_messages = [self._build_messages(c, source_lang, target_lang) for c in batch]
            outputs = self.pipe(
                text=batch_messages,
                do_sample=False,
                batch_size=len(batch),
                pad_token_id=1,
                max_new_tokens=self.max_chunk_tokens,
            )

            for i, output in enumerate(outputs):
                result = output[0]["generated_text"][-1]["content"]
                if not result or not result.strip():
                    raise TranslationError(f"Translation returned empty result for chunk {batch_start + i + 1}")
                if self.is_truncated(result):
                    result = self._retry_truncated(texts[batch_start + i], source_lang, target_lang)
                results.append(result)

        return results

    def _retry_truncated(self, text: str, source_lang: str, target_lang: str, depth: int = 0) -> str:
        """Recursively retry a truncated chunk by splitting it into smaller sub-chunks."""
        _MAX_DEPTH = 3
        if depth >= _MAX_DEPTH:
            raise TranslationError(
                f"Output still truncated after {_MAX_DEPTH} retry attempts. "
                "Input may be too complex to translate within token limits."
            )

        input_tokens = count_tokens(text, self.tokenizer)
        half = max(int(input_tokens * 0.6), 1)
        print(f"      Output truncated ({input_tokens} tokens), retrying (attempt {depth + 1}/{_MAX_DEPTH})...")

        sub_chunks = chunk_text_by_tokens(text, self.tokenizer, max_tokens=half)
        parts = []
        for j, sub in enumerate(sub_chunks, 1):
            print(f"      Sub-chunk {j}/{len(sub_chunks)} ({count_tokens(sub, self.tokenizer)} tokens)...")
            messages = self._build_messages(sub, source_lang, target_lang)
            output = self.pipe(
                text=messages,
                do_sample=False,
                pad_token_id=1,
                max_new_tokens=self.max_chunk_tokens,
            )
            result: str = output[0]["generated_text"][-1]["content"]
            if not result or not result.strip():
                raise TranslationError("Translation returned empty result in retry")
            if self.is_truncated(result):
                result = self._retry_truncated(sub, source_lang, target_lang, depth + 1)
            parts.append(result)

        return "\n\n".join(parts)