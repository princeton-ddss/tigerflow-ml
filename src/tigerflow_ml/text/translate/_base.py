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

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, cast

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from tigerflow_ml.params import HFParams

from .chunking import (
    FALLBACK_MAX_CHUNK_TOKENS,
    MAX_CHUNK_TOKENS,
    chunk_text_by_tokens,
    compute_chunk_size,
    count_tokens,
)
from .detection import LANGUAGES, detect_language, get_language_name
from .translator import HuggingFaceTranslator, build_translator
from .utils import SkippedFileError, TranslationError, read_file_with_fallback

_DEFAULT_PROMPT = (
    "Translate the following text from {source_lang} to {target_lang}. "
    "Output only the translated text, nothing else. Text: {text}"
)


class _TranslateBase:
    """Translate documents using Hugging Face models."""

    class Params(HFParams):
        model: Annotated[
            str,
            typer.Option(help="HuggingFace model repo ID"),
        ] = "google/translategemma-12b-it"

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
            typer.Option(help="Maximum tokens per chunk"),
        ] = None

        prompt: Annotated[
            str,
            typer.Option(
                help="Prompt template for text-generation models. "
                "Use {source_lang}, {target_lang}, and {text} as placeholders."
            ),
        ] = _DEFAULT_PROMPT

        batch_size: Annotated[
            int | None,
            typer.Option(help="Chunks to translate in parallel (default: auto)"),
        ] = None

        fetch: Annotated[
            bool,
            typer.Option(help="Allow downloading from HuggingFace Hub"),
        ] = False

    @staticmethod
    def setup(context: SetupContext):
        chunk_size = context.chunk_size

        try:
            config = AutoConfig.from_pretrained(
                context.model, local_files_only=not context.fetch
            )
        except Exception as e:
            logger.error(f"Failed to parse model config file: {e}")

        try:
            computed_chunk_size = compute_chunk_size(config)
            logger.info(f"Calculated max chunk size: {computed_chunk_size} tokens")
            # if user did not provide a chunk size, use calculated
            if chunk_size is None:
                chunk_size = computed_chunk_size
            # if user provided a chunk size that is too large
            else:
                if chunk_size > computed_chunk_size:
                    logger.warning(
                        f"Warning: --chunk-size {chunk_size} exceeds maximum of"
                        f" {computed_chunk_size}, clamping"
                    )
                    chunk_size = computed_chunk_size
        except Exception:
            # if user did not provide a chunk size, use FALLBACK_MAX_CHUNK_TOKENS
            if chunk_size is None:
                chunk_size = FALLBACK_MAX_CHUNK_TOKENS
                logger.info(f"Chunk size: {chunk_size} tokens")

            # if user provided a chunk size that is too large
            else:
                if chunk_size > MAX_CHUNK_TOKENS:
                    logger.warning(
                        f"Warning: --chunk-size {chunk_size} exceeds maximum of"
                        f" {MAX_CHUNK_TOKENS}, clamping"
                    )
                    chunk_size = MAX_CHUNK_TOKENS

        assert chunk_size is not None
        tokenizer = _get_tokenizer(context.model, context.fetch)
        logger.info(f"Model: {context.model}")
        logger.info("Initializing HuggingFace backend...")
        context.translator = build_translator(
            context.model,
            tokenizer=tokenizer,
            max_chunk_tokens=chunk_size,
            config=config,
            batch_size=context.batch_size,
            fetch=context.fetch,
            prompt_template=context.prompt
        )

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        try:
            _translate_file(
                context.translator,
                input_file,
                output_file,
                context.source_lang,
                context.target_lang,
                logger.info,
            )
        except SkippedFileError as e:
            logger.warning(f"  Skipping: {e}")
            return

        logger.info("Translation complete!")


def _get_tokenizer(model_name: str, fetch: bool) -> PreTrainedTokenizerBase:
    """Load tokenizer, downloading if needed and allowed."""
    try:
        return _load_tokenizer(model_name)
    except OSError:
        if not fetch:
            logger.error(f"Error: Tokenizer for '{model_name}' not found in cache.")
            logger.error("  Run with --fetch to download, or manually with:")
            logger.error(f"    hf download {model_name} --include 'tokenizer*'")
            raise typer.Exit(1)
        logger.info("Downloading tokenizer from HuggingFace Hub...")
        _download_tokenizer(model_name)
        return _load_tokenizer(model_name)


def _load_tokenizer(
    model_name: str, cache_dir: str | None = None
) -> PreTrainedTokenizerBase:
    """
    Load a HuggingFace tokenizer from local cache.

    Args:
        model_name: HuggingFace model name.
        cache_dir: Optional cache directory override.

    Returns:
        Loaded tokenizer.

    Raises:
        OSError: If tokenizer not found in cache.
    """
    return cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            model_name, local_files_only=True, cache_dir=cache_dir
        ),
    )


def _download_tokenizer(model_name: str, cache_dir: str | None = None) -> None:
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
    )


def _translate_file(
    translator: HuggingFaceTranslator,
    input_file: Path,
    output_file: Path,
    source_lang: str | None = None,
    target_lang: str = "en",
    on_progress: Callable[..., None] = print,
) -> str:
    """
    Translate a single file.

    Args:
        translator: Translator instance.
        input_file: Path to input file.
        output_file: Path to output file.
        source_lang: Source language code (auto-detect if None).
        target_lang: Target language code.
        on_progress: Callback for progress messages.

    Raises:
        SkippedFileError: If file should be skipped.
        TranslationError: If translation fails.
    """
    on_progress(f"Processing: {input_file.name}")

    content = read_file_with_fallback(input_file)

    if not content.strip():
        raise SkippedFileError("Empty file")

    on_progress(f"  File size: {len(content):,} characters")

    # Detect language
    detected_lang = source_lang
    if detected_lang is None:
        detected_lang = detect_language(content)
        if detected_lang is None:
            raise TranslationError(
                "Could not detect language (text may be too short or mixed)"
            )
        on_progress(
            f"  Detected language: {get_language_name(detected_lang)} ({detected_lang})"
        )
    else:
        on_progress(
            f"  Source language: {get_language_name(detected_lang)} ({detected_lang})"
        )

    if detected_lang == target_lang:
        raise SkippedFileError(f"Already in {get_language_name(target_lang)}")

    if detected_lang not in LANGUAGES:
        on_progress(
            f"  Note: '{detected_lang}' not in common language list,"
            " attempting anyway..."
        )

    # Translate
    on_progress(
        f"  Translating to: {get_language_name(target_lang)} ({target_lang})..."
    )
    translated = _translate_text(content, translator, detected_lang, target_lang)

    # Sanity check
    if len(translated) > 100 and detected_lang != "en":
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
    translator: HuggingFaceTranslator,
    source_lang: str,
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
    translator: HuggingFaceTranslator,
    source_lang: str,
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
