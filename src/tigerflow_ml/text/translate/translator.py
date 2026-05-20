"""
Translator protocol and HuggingFace implementation.

Translators handle single-chunk translation only. Chunking, retry logic,
and orchestration are handled by the orchestration module.
"""

import logging
import os
import re
from typing import Any, Protocol, cast

import torch
import transformers
from tigerflow.logconfig import logger
from transformers import PretrainedConfig, PreTrainedTokenizerBase, pipeline, set_seed
from vllm import LLM, SamplingParams

from .chunking import DEFAULT_CHUNK_SIZE, chunk_text_by_tokens, count_tokens
from .utils import TranslationError

_TRANSFORMERS_WARNINGS_TO_IGNORE = (
    "Kwargs passed to `processor.__call__`",
    "`local_files_only` is not a valid argument for this processor",
    "Both `max_new_tokens`",
    "Passing `generation_config` together",
    "Setting `pad_token_id` to `eos_token_id`",
)


class _ProcessorKwargsFilter(logging.Filter):
    """Drop known-harmless per-chunk warnings from ImageTextToTextPipeline."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in _TRANSFORMERS_WARNINGS_TO_IGNORE)


def _log_gpu_info(model) -> None:

    if hasattr(model, "hf_device_map"):
        device_map = model.hf_device_map
        logger.info(f"  device map: {device_map}")
        cpu_layers = [k for k, v in device_map.items() if v in ("cpu", "disk")]
        if cpu_layers:
            logger.warning(
                f" {len(cpu_layers)} layer(s) offloaded to CPU/disk: "
                f"{cpu_layers[:5]}" + (" ..." if len(cpu_layers) > 5 else "")
            )

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            reserved_mem = torch.cuda.memory_reserved(i)
            allocated_mem = torch.cuda.memory_allocated(i)

            driver_free, _ = torch.cuda.mem_get_info(i)
            pytorch_cache_free = reserved_mem - allocated_mem
            usable = driver_free + pytorch_cache_free

            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Total:     {total_mem / (1024**3):.2f} GB")
            logger.info(f"  Allocated: {allocated_mem / (1024**3):.2f} GB")
            logger.info(f"  Cached:    {pytorch_cache_free / (1024**3):.2f} GB")
            logger.info(
                f"  Driver free (all processes): {driver_free / (1024**3):.2f} GB"
            )
            logger.info(f"  Usable (driver free + cache): {usable / (1024**3):.2f} GB")


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
        vram_fraction: float,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_chunk_tokens: int = DEFAULT_CHUNK_SIZE,
        batch_size: int | None = None,
        fetch: bool = False,
        cache_dir: str | None = None,
        revision: str | None = None,
        device: str | int = "auto",
    ):
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        if device == "auto":
            if torch.cuda.is_available():
                self._use_device_map = True
                self.device = None
            else:
                self._use_device_map = False
                self.device = "cpu"
        else:
            self._use_device_map = False
            self.device = device
        # pipeline() doesn't forward cache_dir to its internal AutoConfig call,
        # so set the env var so all HF lookups use the right cache.
        if cache_dir:
            os.environ["HF_HUB_CACHE"] = cache_dir
            logger.warning(f"Setting HF_HUB_CACHE={cache_dir}")

        self.pipe: Any = self._load_pipeline(
            model_name=model_name, fetch=fetch, cache_dir=cache_dir, revision=revision
        )
        logger.info(f"Model loaded on {self.pipe.model.device}")

        _log_gpu_info(self.pipe.model)

        if batch_size is None:
            self.batch_size = self._auto_batch_size(vram_fraction)
            logger.info(f"Auto batch size: {self.batch_size}")
        else:
            self.batch_size = batch_size

    def _load_pipeline(
        self, model_name: str, fetch: bool, cache_dir: str | None, revision: str | None
    ) -> Any:
        raise NotImplementedError

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError

    def _auto_batch_size(self, vram_fraction) -> int:
        """Estimate batch size from free VRAM and model KV-cache footprint."""

        if not torch.cuda.is_available():
            return 1
        try:
            if self._use_device_map and torch.cuda.device_count() > 1:
                usable_per_gpu = []
                for i in range(torch.cuda.device_count()):
                    driver_free, _ = torch.cuda.mem_get_info(i)
                    cache_free = torch.cuda.memory_reserved(
                        i
                    ) - torch.cuda.memory_allocated(i)
                    usable_per_gpu.append(driver_free + cache_free)
                logger.info(f"Usable bytes per gpu: {usable_per_gpu}")
                free_bytes = min(usable_per_gpu)
            else:
                driver_free, _ = torch.cuda.mem_get_info()
                cache_free = (
                    torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                )
                free_bytes = driver_free + cache_free

            free_bytes = free_bytes * vram_fraction

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
            overhead = 1.5
            per_seq_bytes = int(
                2
                * n_layers
                * n_kv_heads
                * head_dim
                * (self.max_chunk_tokens * 2)  # tokens
                * 2
                * overhead
            )
            max_batch = 64
            return max(1, min(int(free_bytes // per_seq_bytes), max_batch))
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
        transformers.logging.disable_progress_bar()
        try:
            if self._use_device_map:
                pipe = pipeline(
                    "image-text-to-text",
                    model=model_name,
                    device_map="auto",
                    dtype=torch.bfloat16,
                    local_files_only=not fetch,
                    revision=revision,
                    model_kwargs={"cache_dir": cache_dir},
                )
            else:
                pipe = pipeline(
                    "image-text-to-text",
                    model=model_name,
                    device=self.device,
                    dtype=torch.bfloat16,
                    local_files_only=not fetch,
                    revision=revision,
                    model_kwargs={"cache_dir": cache_dir},
                )
        finally:
            transformers.logging.enable_progress_bar()

        # filter out warnings from _TRANSFORMERS_WARNINGS_TO_IGNORE
        for handler in logging.getLogger("transformers").handlers:
            handler.addFilter(_ProcessorKwargsFilter())

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
            max_new_tokens=self.max_chunk_tokens,
            generate_kwargs={"do_sample": False},
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
        Translate multiple chunks using batched pipeline inference

        Raises:
            TranslationError: If any translation returns an empty result or is truncated
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
                batch_size=len(batch),
                max_new_tokens=self.max_chunk_tokens,
                generate_kwargs={"do_sample": False},
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


class vllmTranslator:
    def __init__(
        self,
        model_name: str,
        vram_fraction: float,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_chunk_tokens: int = DEFAULT_CHUNK_SIZE,
        fetch: bool = False,
        cache_dir: str | None = None,
        revision: str | None = None,
        device: str = "auto",
        prompt_template: str = "",
        system_message: str = "You are an expert linguist",
    ):
        from huggingface_hub import snapshot_download

        if cache_dir is not None:
            resolved_model = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=not fetch,
                revision=revision,
            )
        else:
            resolved_model = model_name

        tp = torch.cuda.device_count() or 1
        gpu_util = vram_fraction if device != "cpu" else 0.0

        _MAX_LEN_ATTRS = (
            "max_position_embeddings",
            "n_positions",
            "n_ctx",
            "max_seq_len",
            "seq_length",
        )
        model_len_cap = next(
            (getattr(config, a) for a in _MAX_LEN_ATTRS if hasattr(config, a)),
            None,
        )
        max_model_len = max_chunk_tokens * 2.5 + 512  # 2.5 for extra wiggle room
        if model_len_cap is not None:
            max_model_len = min(max_model_len, model_len_cap)
        logger.info(f"   max_model_len={max_model_len}")
        llm_kwargs: dict[str, Any] = dict(
            model=resolved_model,
            gpu_memory_utilization=gpu_util,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            enforce_eager=True,
        )
        if device != "auto":
            llm_kwargs["device"] = device
        self.model = LLM(**llm_kwargs)

        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.sampling_params = SamplingParams(
            max_tokens=max_chunk_tokens, temperature=0, seed=42
        )
        self.prompt_template = prompt_template
        self.system_message = system_message

        logger.info("Model loaded using vLLM!")

    def _build_message(
        self, text: str, source_lang: str, target_lang: str
    ) -> list[dict[str, str]]:
        prompt = self.prompt_template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text=text,
        )
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        message = self._build_message(text, source_lang, target_lang)
        output = self.model.chat(
            cast(Any, message), sampling_params=self.sampling_params, use_tqdm=False
        )
        return output[0].outputs[0].text

    def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        messages = [self._build_message(t, source_lang, target_lang) for t in texts]
        outputs = self.model.chat(
            cast(Any, messages), sampling_params=self.sampling_params, use_tqdm=False
        )
        return [o.outputs[0].text for o in outputs]


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
    vram_fraction: float,
    max_chunk_tokens: int,
    batch_size: int | None,
    fetch: bool,
    config: PretrainedConfig,
    backend: str = "auto",
    prompt_template: str = "",
    system_message: str = "You are an expert linguist",
    revision: str | None = None,
    cache_dir: str | None = None,
    device: str = "auto",
) -> Translator:
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
        device: Device to use (cuda, cpu, or auto)
        vram_fraction: The fraction of free VRAM to use

    Returns:
        A concrete Translator subclass.
    """
    kwargs: dict[str, Any] = dict(
        model_name=model_name,
        max_chunk_tokens=max_chunk_tokens,
        fetch=fetch,
        cache_dir=cache_dir,
        revision=revision,
        device=device,
        vram_fraction=vram_fraction,
        tokenizer=tokenizer,
    )
    set_seed(42)

    if backend == "tgemma":
        return GemmaTranslator(**kwargs, batch_size=batch_size)
    elif backend == "chat":
        return vllmTranslator(
            **kwargs,
            prompt_template=prompt_template,
            system_message=system_message,
            config=config,
        )

    # Auto-detect from model name and config
    if get_model_type(model_name) == "tgemma":
        logger.info("  Using tgemma image-text-to-text backend")
        return GemmaTranslator(**kwargs, batch_size=batch_size)
    else:
        logger.info("  Using chat backend through vLLM")
        return vllmTranslator(
            **kwargs,
            prompt_template=prompt_template,
            system_message=system_message,
            config=config,
        )
