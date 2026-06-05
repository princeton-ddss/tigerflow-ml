"""
Translate text documents using Hugging Face models.

Supports Seq2Seq translation models (MADLAD-400, Helsinki-NLP/opus-mt-*, NLLB)
and chat/causal LMs with a translation prompt.

python -m tigerflow_ml.text.translate.slurm --input-dir ../tgemma/tests/input/
--input-ext .txt --output-dir tests/test-outputs/rerun-test/ --output-ext .txt
--max-workers 1 --cpus 1 --memory 10G --time 00:30:00 --gpus 1 --sbatch-option
"--constraint=gpu80"
--setup-command "export HF_HOME=/scratch/gpfs/TITIUNIK/nv5842/github/tgemma/.hf/"
--setup-command "source .venv/bin/activate" --model google/translategemma-27b-it
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import VLLMParams
from tigerflow_ml.utils import (
    ConfigParsingError,
    EmptyFileError,
    parse_kwargs,
    read_file_with_fallback,
)

from .chunking import (
    DEFAULT_CHUNK_SIZE,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    chunk_text_by_tokens,
    count_tokens,
)
from .detection import LANGUAGES, detect_language, get_language_name
from .errors import (
    AlreadyInTargetLanguageError,
    TranslationError,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .translator import Translator

_DEFAULT_PROMPT = (
    "Translate the following text from {source_lang} to {target_lang}. "
    "Output only the translated text, nothing else. Text: {text}"
)

_FALLBACK_PROMPT = (
    "Translate the following text to {target_lang}. "
    "Output only the translated text, nothing else. Text: {text}"
)


class _TranslateBase:
    """Translate documents using Hugging Face models."""

    class Params(VLLMParams):
        source_lang: Annotated[
            str | None,
            typer.Option(help="Source language code (e.g. 'en', 'de', 'zh')"),
        ] = None

        target_lang: Annotated[
            str,
            typer.Option(help="Target language code (e.g. 'de', 'en', 'fr')"),
        ] = "en"

        chunk_size: Annotated[
            int | None,
            typer.Option(
                help="Maximum token sequence length in a batch",
                min=MIN_CHUNK_TOKENS,
                max=MAX_CHUNK_TOKENS,
            ),
        ] = DEFAULT_CHUNK_SIZE

        prompt_template: Annotated[
            str,
            typer.Option(
                help="Prompt template for chat-based translation models "
                "Use {source_lang}, {target_lang}, and {text} as placeholders."
                "This is unused if using a tgemma model."
            ),
        ] = _DEFAULT_PROMPT

        model_backend: Annotated[
            Literal["auto", "chat", "tgemma"],
            typer.Option(help="Translation model backend. 'auto' uses model name."),
        ] = "auto"

        auto_lang_detect: Annotated[
            bool,
            typer.Option(
                help="Auto-detect source language with langdetect. "
                "When disabled, --source-lang is required."
            ),
        ] = True

        use_fallback_prompt: Annotated[
            bool,
            typer.Option(
                help="Use a source-language-free fallback prompt when "
                "no source language can be determined. Not supported for "
                "tgemma models."
            ),
        ] = False

        # override for custom help message
        max_model_len: Annotated[
            int | None,
            typer.Option(
                help="Maximum sequence length (input + output tokens) passed to vLLM. "
                "Defaults to (chunk_size * 2.5 + 512), "
                "capped by the model's configured context window."
            ),
        ] = None

        # override to remove help message
        max_tokens: Annotated[
            int,
            typer.Option(hidden=True),
        ] = int(DEFAULT_CHUNK_SIZE * 2.5)  # unused

    @staticmethod
    def setup(context: SetupContext):
        from transformers import AutoConfig

        from .translator import build_translator

        if "{text}" not in context.prompt_template:
            raise ValueError(
                '--prompt-template must contain "{text}"'
                " as a placeholder for input file contents."
            )
        if context.source_lang == context.target_lang:
            raise ValueError(
                f"--source-lang ({context.source_lang}) is the same as"
                f" --target-lang ({context.target_lang}). No translation required."
            )

        try:
            config = AutoConfig.from_pretrained(
                context.model,
                local_files_only=not context.allow_fetch,
                cache_dir=context.cache_dir,
                revision=context.revision,
            )
        except Exception as e:
            raise ConfigParsingError(f"Failed to load model config: {e}")

        tokenizer = _get_tokenizer(
            context.model,
            fetch=context.allow_fetch,
            cache_dir=context.cache_dir,
            revision=context.revision,
        )

        if context.chunk_size > MAX_CHUNK_TOKENS:
            logger.warning(
                f"Warning: --chunk-size {context.chunk_size} exceeds maximum of"
                f" {MAX_CHUNK_TOKENS}, clamping"
            )
            context.chunk_size = MAX_CHUNK_TOKENS
        else:
            logger.info(f"Chunk size: {context.chunk_size} tokens")

        logger.info(f"Model: {context.model}")
        logger.info("Initializing HuggingFace backend...")

        context.translator = build_translator(
            context.model,
            tokenizer=tokenizer,
            max_chunk_tokens=context.chunk_size,
            max_model_len=context.max_model_len,
            config=config,
            seed=context.seed,
            temperature=context.temperature,
            fetch=context.allow_fetch,
            prompt_template=context.prompt_template,
            system_message=context.system_message,
            backend=context.model_backend,
            revision=context.revision,
            cache_dir=context.cache_dir,
            user_llm_kwargs=parse_kwargs(context.llm_kwargs, name="llm-kwargs"),
            user_sampling_kwargs=parse_kwargs(
                context.sampling_kwargs, name="sampling-kwargs"
            ),
            user_chat_kwargs=parse_kwargs(context.chat_kwargs, name="chat-kwargs"),
        )

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):

        _translate_file(
            context.translator,
            input_file,
            output_file,
            context.source_lang,
            context.target_lang,
            logger.info,
            auto_lang_detect=context.auto_lang_detect,
            use_fallback_prompt=context.use_fallback_prompt,
        )

        logger.info("Translation complete!")


def _get_tokenizer(
    model_name: str,
    fetch: bool,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> PreTrainedTokenizerBase:
    """Load tokenizer, downloading if needed and allowed."""
    try:
        return _load_tokenizer(model_name, cache_dir=cache_dir, revision=revision)
    except OSError:
        if not fetch:
            logger.error(f"Error: Tokenizer for '{model_name}' not found in cache.")
            logger.error("  Run with --fetch to download, or manually with:")
            logger.error(f"    hf download {model_name} --include 'tokenizer*'")
            raise typer.Exit(1)
        logger.info("Downloading tokenizer from HuggingFace Hub...")
        _download_tokenizer(model_name, cache_dir=cache_dir, revision=revision)
        return _load_tokenizer(model_name, cache_dir=cache_dir, revision=revision)


def _load_tokenizer(
    model_name: str, cache_dir: str | None = None, revision: str | None = None
) -> PreTrainedTokenizerBase:
    """
    Load a HuggingFace tokenizer from local cache

    Returns:
        Loaded tokenizer.

    Raises:
        OSError: If tokenizer not found in cache."""

    from transformers import AutoTokenizer

    return cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(
            model_name, local_files_only=True, cache_dir=cache_dir, revision=revision
        ),
    )


def _download_tokenizer(
    model_name: str, cache_dir: str | None = None, revision: str | None = None
) -> None:
    """
    Download tokenizer files from HuggingFace Hub.

    Args:
        model_name: HuggingFace model name.
        cache_dir: Optional cache directory override.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        model_name,
        allow_patterns=["tokenizer*", "special_tokens_map.json"],
        cache_dir=cache_dir,
        revision=revision,
    )


def _resolve_source_lang(
    content: str,
    source_lang: str | None,
    target_lang: str,
    auto_lang_detect: bool,
    on_progress: Callable[..., None],
) -> str | None:
    """
    Resolve the source language for a file.

    Explicit source_lang always takes precedence. When auto_lang_detect is
    True, langdetect fills in a missing source_lang and a mismatch between
    detected and given is warned about. Returns None when no source language
    can be determined; the caller decides whether that's an error.

    Args:
        content: File contents (used for detection).
        source_lang: Explicit source language code, or None.
        target_lang: Target language code.
        auto_lang_detect: Whether to run langdetect.
        on_progress: Callback for progress messages.

    Returns:
        Resolved source language code, or None if unresolvable.

    Raises:
        AlreadyInTargetLanguageError: If resolved language equals target_lang.
        TranslationError: If auto_lang_detect is False and source_lang is None.
    """
    if not auto_lang_detect:
        if source_lang is None:
            raise TranslationError(
                "Source language could not be determined. "
                "Explicitly set --source-lang or enable --auto-lang-detect."
            )
        on_progress(
            f"  Source language: {get_language_name(source_lang)} ({source_lang})"
        )
        resolved: str | None = source_lang
    else:
        detected = detect_language(content)
        if detected is not None:
            on_progress(
                f"  Detected language: {get_language_name(detected)} ({detected})"
            )
        if source_lang is not None:
            on_progress(
                f"  Source language: {get_language_name(source_lang)} ({source_lang})"
            )
            if detected is not None and detected != source_lang:
                on_progress(
                    f"  Warning: detected {detected} but --source-lang is"
                    f" {source_lang}; using --source-lang"
                )
            resolved = source_lang
        else:
            resolved = detected

    if resolved is not None and resolved == target_lang:
        raise AlreadyInTargetLanguageError(
            f"Already in {get_language_name(target_lang)}"
        )
    return resolved


def _translate_file(
    translator: Translator,
    input_file: Path,
    output_file: Path,
    source_lang: str | None = None,
    target_lang: str = "en",
    on_progress: Callable[..., None] = print,
    auto_lang_detect: bool = True,
    use_fallback_prompt: bool = False,
) -> str:
    """
    Translate a single file.

    Args:
        translator: Translator instance.
        input_file: Path to input file.
        output_file: Path to output file.
        source_lang: Source language code. Required when auto_lang_detect is
            False; otherwise overrides auto-detection if provided.
        target_lang: Target language code.
        on_progress: Callback for progress messages.
        auto_lang_detect: Run langdetect on the file. Explicit source_lang
            takes precedence; when False, source_lang is required.
        use_fallback_prompt: When no source language can be determined and
            the translator's prompt template requires {source_lang}, swap to
            a source-language-free fallback prompt for this file. Not
            supported for tgemma models.

    Raises:
        EmptyFileError: If the input file is empty.
        AlreadyInTargetLanguageError: If the input file is already 'translated.'
        TranslationError: If translation fails.
    """
    from .translator import TgemmaTranslator

    on_progress(f"Processing: {input_file.name}")

    content = read_file_with_fallback(input_file)

    if not content.strip():
        raise EmptyFileError("Empty file")

    on_progress(f"  File size: {len(content):,} characters")

    resolved_lang = _resolve_source_lang(
        content, source_lang, target_lang, auto_lang_detect, on_progress
    )

    original_prompt = translator.prompt_template
    use_fallback = False

    if resolved_lang is None:
        if isinstance(translator, TgemmaTranslator):
            raise TranslationError(
                "Source language could not be determined. Explicitly set --source-lang."
            )
        if "{source_lang}" in original_prompt:
            if not use_fallback_prompt:
                raise TranslationError(
                    "Source language could not be determined. "
                    "Explicitly set --source-lang or run with"
                    " --use-fallback-prompt."
                )
            on_progress(f"  Using fallback prompt: {_FALLBACK_PROMPT}")
            use_fallback = True

    if resolved_lang is not None and resolved_lang not in LANGUAGES:
        on_progress(
            f"  Note: '{resolved_lang}' not in common language list,"
            " attempting anyway..."
        )

    on_progress(
        f"  Translating to: {get_language_name(target_lang)} ({target_lang})..."
    )

    try:
        if use_fallback:
            translator.prompt_template = _FALLBACK_PROMPT
        translated = _translate_text(content, translator, resolved_lang, target_lang)
    finally:
        if use_fallback:
            translator.prompt_template = original_prompt

    # Sanity check
    if len(translated) > 100 and resolved_lang != "en":
        if content[:100] == translated[:100]:
            on_progress(
                "  Warning: Output appears identical to input -"
                " translation may have failed"
            )

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(translated)

    on_progress(f"  Output size: {len(translated):,} characters")

    return translated


def _translate_text(
    text: str,
    translator: Translator,
    source_lang: str | None,
    target_lang: str = "en",
    max_retries: int = 3,
) -> str:
    """
    Translate text, chunking if necessary.

    Handles the chunk -> translate -> merge flow, with retry logic for
    chunks that produce truncated output.

    Args:
        text: Full document text to translate.
        translator: Translator backend to use.
        source_lang: Source language code.
        target_lang: Target language code.
        max_retries: Maximum retry attempts for truncated chunks.

    Returns:
        Translated document text.
    """
    tokenizer = translator.tokenizer
    assert tokenizer is not None
    max_tokens = translator.max_chunk_tokens

    if count_tokens(text, tokenizer) <= max_tokens:
        return _translate_chunk_with_retry(
            text,
            translator,
            source_lang,
            target_lang,
            tokenizer,
            max_tokens,
            max_retries,
        )

    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=max_tokens)
    logger.info(f"    Document is long - splitting into {len(chunks)} chunks...")

    return "\n\n".join(translator.translate_batch(chunks, source_lang, target_lang))


def _translate_chunk_with_retry(
    text: str,
    translator: Translator,
    source_lang: str | None,
    target_lang: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    max_retries: int,
    chunk_num: int | None = None,
    total_chunks: int | None = None,
    retry_depth: int = 0,
) -> str:
    """Translate a chunk, retrying with smaller sub-chunks if truncated."""
    try:
        return translator.translate(text, source_lang, target_lang)
    except TranslationError as e:
        if "truncated" not in str(e).lower():
            raise

        if retry_depth >= max_retries:
            raise TranslationError(
                f"Output still truncated after {max_retries} retry attempts. "
                f"Input may be too complex to translate within token limits."
            ) from e

        input_tokens = count_tokens(text, tokenizer)
        half = max(int(input_tokens * 0.6), 1)

        prefix = f"Chunk {chunk_num}/{total_chunks}: " if chunk_num else ""
        logger.info(
            f"      {prefix}Output truncated, splitting ({input_tokens} tokens) "
            f"and retrying (attempt {retry_depth + 1}/{max_retries})..."
        )

        sub_chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=half)
        parts = []
        for j, sub in enumerate(sub_chunks, 1):
            logger.info(
                f"      Sub-chunk {j}/{len(sub_chunks)} "
                f"({count_tokens(sub, tokenizer)} tokens)..."
            )
            result = _translate_chunk_with_retry(
                sub,
                translator,
                source_lang,
                target_lang,
                tokenizer,
                max_tokens,
                max_retries,
                retry_depth=retry_depth + 1,
            )
            parts.append(result)

        return "\n\n".join(parts)
