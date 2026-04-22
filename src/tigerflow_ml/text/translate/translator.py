"""
Translator protocol and HuggingFace implementation.

Translators handle single-chunk translation only. Chunking, retry logic,
and orchestration are handled by the orchestration module.
"""

import os
import re
from typing import Any, Protocol

import torch
from tigerflow.logconfig import logger
from transformers import PretrainedConfig, PreTrainedTokenizerBase, pipeline

from .chunking import FALLBACK_MAX_CHUNK_TOKENS, chunk_text_by_tokens, count_tokens
from .utils import TranslationError


def _is_image_text_model(config: PretrainedConfig) -> bool:
    """Return True if the config describes a multimodal (vision+language) model."""
    return hasattr(config, "vision_config") or hasattr(config, "image_token_id")


class Translator(Protocol):
    """Protocol for translation backends."""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single chunk of text."""
        ...

    def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """Translate multiple chunks."""
        ...


class HuggingFaceTranslator:
    """Shared base class for all HuggingFace translation backends."""

    def __init__(
        self,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_chunk_tokens: int = FALLBACK_MAX_CHUNK_TOKENS,
        batch_size: int | None = None,
        fetch: bool = False,
        cache_dir: str | None = None,
        revision: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        # pipeline() doesn't forward cache_dir to its internal AutoConfig call,
        # so set the env var so all HF lookups use the right cache.
        if cache_dir:
            os.environ["HF_HUB_CACHE"] = cache_dir
        self.pipe: Any = self._load_pipeline(
            model_name=model_name, fetch=fetch, cache_dir=cache_dir, revision=revision
        )
        logger.info("Model loaded!")

        if batch_size is None:
            self.batch_size = self._auto_batch_size()
            logger.info(f"Auto batch size: {self.batch_size}")
        else:
            self.batch_size = batch_size

    def _load_pipeline(
        self, model_name: str, fetch: bool, cache_dir: str | None, revision: str | None
    ) -> Any:
        raise NotImplementedError

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError

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
            head_dim = getattr(
                cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
            )
            # KV cache per sequence:
            # 2(K+V) * layers * kv_heads * head_dim * tokens * 2 bytes (bfloat16)
            # 1.5x overhead for activations and intermediate buffers
            per_seq_bytes = int(
                2 * n_layers * n_kv_heads * head_dim * self.max_chunk_tokens * 2 * 1.5
            )
            return max(1, min(int(free_bytes // per_seq_bytes), 256))
        except Exception:
            return 1

    def is_truncated(self, text: str) -> bool:
        """Check if output was likely truncated by hitting max_new_tokens."""
        if self.tokenizer is None:
            return False
        return count_tokens(text, self.tokenizer) >= self.max_chunk_tokens

    def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """
        Translate multiple chunks by calling translate() for each.

        Raises:
            TranslationError: If any translation returns an empty result.

        Note: Subclasses may override this to use true pipeline batching for
        better GPU throughput.
        """
        results = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]

            if len(texts) > 1:
                batch_end = batch_start + len(batch)
                logger.info(
                    "    Translating chunks "
                    f"{batch_start + 1}-{batch_end}/{len(texts)}..."
                )

            for chunk in batch:
                try:
                    result = self.translate(chunk, source_lang, target_lang)
                except TranslationError as e:
                    if "truncated" not in str(e).lower():
                        raise
                    result = self._retry_truncated(chunk, source_lang, target_lang)
                results.append(result)

        return results

    def _retry_truncated(
        self, text: str, source_lang: str, target_lang: str, depth: int = 0
    ) -> str:
        """Recursively retry a truncated chunk by splitting it."""
        _MAX_DEPTH = 3
        if depth >= _MAX_DEPTH:
            raise TranslationError(
                f"Output still truncated after {_MAX_DEPTH} retry attempts. "
                "Input may be too complex to translate within token limits."
            )

        assert self.tokenizer is not None
        input_tokens = count_tokens(text, self.tokenizer)
        half = max(int(input_tokens * 0.6), 1)
        logger.info(
            f"      Output truncated ({input_tokens} tokens), retrying "
            f"(attempt {depth + 1}/{_MAX_DEPTH})..."
        )

        sub_chunks = chunk_text_by_tokens(text, self.tokenizer, max_tokens=half)
        parts = []
        for j, sub in enumerate(sub_chunks, 1):
            logger.info(
                f"      Sub-chunk {j}/{len(sub_chunks)} "
                f"({count_tokens(sub, self.tokenizer)} tokens)..."
            )
            try:
                result = self.translate(sub, source_lang, target_lang)
            except TranslationError as e:
                if "truncated" not in str(e).lower():
                    raise
                result = self._retry_truncated(sub, source_lang, target_lang, depth + 1)
            parts.append(result)

        return "\n\n".join(parts)


class GemmaTranslator(HuggingFaceTranslator):
    """Translation backend for TranslateGemma image-text-to-text models."""

    def _load_pipeline(
        self, model_name: str, fetch: bool, cache_dir: str | None, revision: str | None
    ) -> Any:

        logger.info(f"Loading model: {model_name}...")
        pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            local_files_only=not fetch,
            revision=revision,
            model_kwargs={"cache_dir": cache_dir},
        )
        return pipe

    def _build_messages(
        self, text: str, source_lang: str, target_lang: str
    ) -> list[dict[str, object]]:
        """Build the message payload using TranslateGemma's content format."""
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
            raise TranslationError(
                f"Output truncated (hit {self.max_chunk_tokens} token limit)"
            )

        return result

    def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """
        Translate multiple chunks using batched pipeline inference.

        Raises:
            TranslationError: If any translation returns an empty result.
        """
        results = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]

            if len(texts) > 1:
                batch_end = batch_start + len(batch)
                logger.info(
                    "    Translating chunks "
                    f"{batch_start + 1}-{batch_end}/{len(texts)}..."
                )

            batch_messages = [
                self._build_messages(c, source_lang, target_lang) for c in batch
            ]
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
                    raise TranslationError(
                        "Translation returned empty result for chunk "
                        f"{batch_start + i + 1}"
                    )
                if self.is_truncated(result):
                    result = self._retry_truncated(
                        texts[batch_start + i], source_lang, target_lang
                    )
                results.append(result)

        return results


class ChatTranslator(HuggingFaceTranslator):
    """Translation backend for chat-based LLMs
    (text-generation and image-text-to-text)."""

    def __init__(
        self,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_chunk_tokens: int = FALLBACK_MAX_CHUNK_TOKENS,
        batch_size: int | None = None,
        fetch: bool = False,
        cache_dir: str | None = None,
        revision: str | None = None,
        prompt_template: str = "",
        is_vlm: bool = False,
    ):
        self._is_vlm = is_vlm  # set before super().__init__ calls _load_pipeline
        super().__init__(
            model_name=model_name,
            tokenizer=tokenizer,
            max_chunk_tokens=max_chunk_tokens,
            batch_size=batch_size,
            fetch=fetch,
            cache_dir=cache_dir,
            revision=revision,
        )
        self.prompt_template = prompt_template

    def _load_pipeline(
        self,
        model_name: str,
        fetch: bool,
        cache_dir: str | None = None,
        revision: str | None = None,
    ):
        logger.info(f"Loading model: {model_name}...")
        if self._is_vlm:
            return pipeline(
                "image-text-to-text",
                model=model_name,
                device_map="auto",
                dtype=torch.bfloat16,
                local_files_only=not fetch,
                revision=revision,
                model_kwargs={"cache_dir": cache_dir},
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            local_files_only=not fetch,
            cache_dir=cache_dir,
            revision=revision,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=not fetch,
            cache_dir=cache_dir,
            revision=revision,
        )
        self._has_chat_template = bool(getattr(tokenizer, "chat_template", None))
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"cache_dir": cache_dir},
            revision=revision,
        )

    def _build_prompt(self, text, source_lang, target_lang) -> str:
        return self.prompt_template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text=text,
        )

    def translate(self, text, source_lang, target_lang) -> str:
        prompt = self._build_prompt(text, source_lang, target_lang)
        if self._is_vlm:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            out = self.pipe(
                text=messages, do_sample=False, max_new_tokens=self.max_chunk_tokens
            )
            result: str = out[0]["generated_text"][-1]["content"]
        elif self._has_chat_template:
            out = self.pipe(
                [{"role": "user", "content": prompt}],
                do_sample=False,
                max_new_tokens=self.max_chunk_tokens,
            )
            result = out[0]["generated_text"][-1]["content"]
        else:
            out = self.pipe(
                prompt,
                do_sample=False,
                max_new_tokens=self.max_chunk_tokens,
                return_full_text=False,
            )
            result = out[0]["generated_text"]
        if not result or not result.strip():
            raise TranslationError("Translation returned empty result")
        if self.is_truncated(result):
            raise TranslationError(
                f"Output truncated (hit {self.max_chunk_tokens} token limit)"
            )
        return result


def get_model_type(
    model_name: str,
) -> str:
    """Determine if a model is of type 'tgemma' or 'chat' based on
    the model name."""
    tgemma_pattern = r"translategemma-\d+b-it"
    if re.search(tgemma_pattern, model_name):
        return "tgemma"
    return "chat"


def build_translator(
    model_name: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_chunk_tokens: int,
    batch_size: int | None,
    fetch: bool,
    config: PretrainedConfig,
    backend: str = "auto",
    prompt_template: str = "",
    revision: str | None = None,
    cache_dir: str | None = None,
) -> HuggingFaceTranslator:
    """
    Instantiate the appropriate translation backend.

    Args:
        model_name: HuggingFace model repo ID.
        tokenizer: Pre-loaded tokenizer for the model.
        max_chunk_tokens: Maximum input tokens per chunk.
        batch_size: Pipeline batch size (None = auto).
        fetch: Allow downloading from HuggingFace Hub.
        config: Pre-loaded model config, used for auto-detection.
        backend: One of "auto", "tgemma", or "chat".
        prompt_template: Prompt template for chat backends.
        revision: Model revision (branch, tag, or commit hash)
        cache_dir: HuggingFace cache directory for model files

    Returns:
        A concrete HuggingFaceTranslator subclass.
    """
    kwargs: dict[str, Any] = dict(
        model_name=model_name,
        tokenizer=tokenizer,
        max_chunk_tokens=max_chunk_tokens,
        batch_size=batch_size,
        fetch=fetch,
        cache_dir=cache_dir,
        revision=revision,
    )

    if backend == "tgemma":
        return GemmaTranslator(**kwargs)
    elif backend == "chat":
        is_vlm = _is_image_text_model(config)
        return ChatTranslator(**kwargs, prompt_template=prompt_template, is_vlm=is_vlm)

    # Auto-detect from model name and config
    if get_model_type(model_name) == "tgemma":
        logger.info("Using tgemma image-text-to-text backend")
        return GemmaTranslator(**kwargs)
    else:
        is_vlm = _is_image_text_model(config)
        logger.info(
            "Using chat backend "
            f"({'image-text-to-text' if is_vlm else 'text-generation'})"
        )
        return ChatTranslator(**kwargs, prompt_template=prompt_template, is_vlm=is_vlm)
