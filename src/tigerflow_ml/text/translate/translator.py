"""
Translator protocol and HuggingFace implementation.

Translators handle single-chunk translation only. Chunking, retry logic,
and orchestration are handled by the orchestration module.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, cast

from tigerflow.logconfig import logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizerBase

from .chunking import DEFAULT_CHUNK_SIZE


class Translator(Protocol):
    """Protocol for translation backends."""

    tokenizer: PreTrainedTokenizerBase
    max_chunk_tokens: int
    prompt_template: str

    def translate(self, text: str, source_lang: str | None, target_lang: str) -> str:
        """Translate a single chunk of text."""
        ...

    def translate_batch(
        self, texts: list[str], source_lang: str | None, target_lang: str
    ) -> list[str]:
        """Translate multiple chunks."""
        ...


class vllmTranslator:
    def __init__(
        self,
        model_name: str,
        config: PretrainedConfig,
        seed: int,
        temperature: float,
        tokenizer: PreTrainedTokenizerBase,
        max_model_len: int | None = None,
        max_chunk_tokens: int = DEFAULT_CHUNK_SIZE,
        fetch: bool = False,
        cache_dir: str | None = None,
        revision: str | None = None,
        prompt_template: str = "",
        system_message: str | None = None,
        user_llm_kwargs: dict = {},
        user_sampling_kwargs: dict = {},
        user_chat_kwargs: dict = {},
    ):
        import torch
        from huggingface_hub import snapshot_download
        from vllm import LLM, SamplingParams

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

        # user set max_model_len overrides all else
        if not max_model_len:
            # identify model context window to ensure max_model_len does not exceed this
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
            # max_model_len should be approx max_chunk_tokens*2 + overhead
            max_model_len = int(
                max_chunk_tokens * 2.5 + 512
            )  # 2.5 for extra wiggle room
            if model_len_cap is not None:
                max_model_len = min(max_model_len, model_len_cap)

        llm_kwargs = dict(
            model=resolved_model,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            enforce_eager=True,
        )
        llm_kwargs.update(user_llm_kwargs)
        logger.info(f"    llm_kwargs={llm_kwargs}")

        sampling_kwargs = dict(
            temperature=temperature,
            seed=seed,
            max_tokens=max_chunk_tokens,
        )
        sampling_kwargs.update(user_sampling_kwargs)
        logger.info(f"    sampling_kwargs={sampling_kwargs}")
        self.sampling_params = SamplingParams(**cast(Any, sampling_kwargs))

        extra_chat_kwargs: dict[str, Any] = {"use_tqdm": False}
        extra_chat_kwargs.update(user_chat_kwargs)
        logger.info(f"    extra_chat_kwargs={extra_chat_kwargs}")
        self.extra_chat_kwargs = extra_chat_kwargs

        self.model = LLM(**cast(Any, llm_kwargs))
        self.max_chunk_tokens = max_chunk_tokens  # used for chunking
        self.prompt_template = prompt_template
        self.system_message = system_message
        self.tokenizer = tokenizer  # used for chunking

        logger.info("Model loaded using vLLM!")

    def _build_message(
        self, text: str, source_lang: str | None, target_lang: str
    ) -> list[dict[str, str]]:

        if source_lang:
            prompt = self.prompt_template.format(
                source_lang=source_lang,
                target_lang=target_lang,
                text=text,
            )
        else:
            prompt = self.prompt_template.format(
                target_lang=target_lang,
                text=text,
            )

        if self.system_message:
            return [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ]
        return [{"role": "user", "content": prompt}]

    def translate(self, text: str, source_lang: str | None, target_lang: str) -> str:
        message = self._build_message(text, source_lang, target_lang)
        output = self.model.chat(
            cast(Any, message),
            sampling_params=self.sampling_params,
            **cast(Any, self.extra_chat_kwargs),
        )
        return output[0].outputs[0].text

    def translate_batch(
        self, texts: list[str], source_lang: str | None, target_lang: str
    ) -> list[str]:
        messages = [self._build_message(t, source_lang, target_lang) for t in texts]
        outputs = self.model.chat(
            cast(Any, messages),
            sampling_params=self.sampling_params,
            **cast(Any, self.extra_chat_kwargs),
        )
        return [o.outputs[0].text for o in outputs]


class TgemmaTranslator(vllmTranslator):
    """Translation for vLLM optimaized translateGemma models
    https://docs.vllm.ai/projects/recipes/en/latest/Google/TranslateGemma.html"""

    def __init__(
        self,
        model_name: str,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        temperature: float,
        max_model_len: int | None = None,
        max_chunk_tokens: int = DEFAULT_CHUNK_SIZE,
        fetch: bool = False,
        cache_dir: str | None = None,
        revision: str | None = None,
        prompt_template: str = "",
        system_message: str | None = None,
        user_llm_kwargs: dict = {},
        user_sampling_kwargs: dict = {},
        user_chat_kwargs: dict = {},
    ):
        super().__init__(
            model_name=model_name,
            config=config,
            tokenizer=tokenizer,
            seed=seed,
            temperature=temperature,
            max_model_len=max_model_len,
            max_chunk_tokens=max_chunk_tokens,
            fetch=fetch,
            cache_dir=cache_dir,
            revision=revision,
            prompt_template=prompt_template,
            system_message=system_message,
            user_llm_kwargs=user_llm_kwargs,
            user_sampling_kwargs=user_sampling_kwargs,
            user_chat_kwargs=user_chat_kwargs,
        )
        self.prompt_template = (
            "<<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{text}"
        )
        self.system_message = None


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
    max_chunk_tokens: int,
    fetch: bool,
    config: PretrainedConfig,
    tokenizer: PreTrainedTokenizerBase,
    seed: int = 42,
    temperature: float = 0,
    max_model_len: int | None = None,
    backend: str = "auto",
    prompt_template: str = "",
    system_message: str = "You are an expert linguist",
    revision: str | None = None,
    cache_dir: str | None = None,
    user_llm_kwargs: dict = {},
    user_sampling_kwargs: dict = {},
    user_chat_kwargs: dict = {},
) -> Translator:
    """
    Instantiate the appropriate translation backend.

    Args:
        model_name: HuggingFace model repo ID.
        max_chunk_tokens: Maximum input tokens per chunk.
        fetch: Allow downloading from HuggingFace Hub.
        config: Pre-loaded model config, used for auto-detection.
        tokenizer: Pre-loaded tokenizer for the model.
        seed: The seed for reproducibility.
        temperature: Model temperature.
        max_model_len: The max length of the model.
        backend: One of "auto", "tgemma", or "chat".
        prompt_template: Prompt template for chat backends.
        system_message: Any system message to be included.
        revision: Model revision (branch, tag, or commit hash).
        cache_dir: HuggingFace cache directory for model files.
        user_llm_kwargs: Any additional args for LLM().
        user_sampling_kwargs: Any additional args for SamplingParams().
        user_chat_kwargs: Any additional args for llm.chat().

    Returns:
        A concrete Translator subclass.
    """

    kwargs: dict[str, Any] = dict(
        model_name=model_name,
        max_chunk_tokens=max_chunk_tokens,
        max_model_len=max_model_len,
        fetch=fetch,
        cache_dir=cache_dir,
        revision=revision,
        config=config,
        seed=seed,
        temperature=temperature,
        tokenizer=tokenizer,
        system_message=system_message,
        prompt_template=prompt_template,
        user_llm_kwargs=user_llm_kwargs,
        user_sampling_kwargs=user_sampling_kwargs,
        user_chat_kwargs=user_chat_kwargs,
    )

    from transformers import set_seed

    set_seed(42)

    if backend == "tgemma":
        return TgemmaTranslator(**kwargs)
    elif backend == "chat":
        return vllmTranslator(**kwargs)

    # Auto-detect from model name and config
    if get_model_type(model_name) == "tgemma":
        logger.info("  Detected a tgemma model")
        return TgemmaTranslator(**kwargs)
    else:
        logger.info("  Using chat backend through vLLM")
        return vllmTranslator(**kwargs)
