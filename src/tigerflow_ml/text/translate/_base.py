"""
Translate text documents using Hugging Face models.

Supports Seq2Seq translation models (MADLAD-400, Helsinki-NLP/opus-mt-*, NLLB)
and chat/causal LMs with a translation prompt.
"""

import re
from pathlib import Path
from typing import Annotated

import typer
from tigerflow.logconfig import logger
from tigerflow.utils import SetupContext

from tigerflow_ml.params import HFParams

_DEFAULT_PROMPT = (
    "Translate the following text from {source_lang} to {target_lang}. "
    "Output only the translation, nothing else.\n\n{text}"
)


class _TranslateBase:
    """Translate documents using Hugging Face models."""

    class Params(HFParams):
        model: Annotated[
            str,
            typer.Option(help="HuggingFace model repo ID"),
        ] = "google/madlad400-3b-mt"

        source_lang: Annotated[
            str,
            typer.Option(help="Source language code (e.g. 'en', 'de', 'zh')"),
        ] = "en"

        target_lang: Annotated[
            str,
            typer.Option(help="Target language code (e.g. 'de', 'en', 'fr')"),
        ] = "de"

        max_length: Annotated[
            int,
            typer.Option(help="Maximum number of tokens to generate per chunk"),
        ] = 512

        prompt: Annotated[
            str,
            typer.Option(
                help="Prompt template for text-generation models. "
                "Use {source_lang}, {target_lang}, and {text} as placeholders."
            ),
        ] = _DEFAULT_PROMPT

        encoding: Annotated[
            str,
            typer.Option(
                help="Input file encoding (non-UTF-8 may cause lossy tokenization)"
            ),
        ] = "utf-8-sig"

    @staticmethod
    def setup(context: SetupContext):
        import torch
        from transformers import AutoConfig

        logger.info("Setting up translation model...")
        logger.info("Model: {}", context.model)

        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Detect model type
        config = AutoConfig.from_pretrained(
            context.model,
            revision=context.revision,
            cache_dir=context.cache_dir or None,
        )
        is_seq2seq = config.is_encoder_decoder

        if is_seq2seq:
            from transformers import AutoModelForSeq2SeqLM

            logger.info("Using Seq2Seq translation model")

            # Use model-specific tokenizer for Marian models
            if config.model_type == "marian":
                from transformers import MarianTokenizer

                context.tokenizer = MarianTokenizer.from_pretrained(
                    context.model,
                    revision=context.revision,
                    cache_dir=context.cache_dir or None,
                )
            else:
                from transformers import AutoTokenizer

                context.tokenizer = AutoTokenizer.from_pretrained(
                    context.model,
                    revision=context.revision,
                    cache_dir=context.cache_dir or None,
                )

            context.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                context.model,
                revision=context.revision,
                cache_dir=context.cache_dir or None,
            )
            context.translation_model.to(device)
            context.is_seq2seq = True
        else:
            from transformers import pipeline

            logger.info("Using text-generation pipeline for translation")
            context.pipeline = pipeline(
                "text-generation",
                model=context.model,
                revision=context.revision,
                device=device,
                model_kwargs={"cache_dir": context.cache_dir or None},
            )
            context.tokenizer = context.pipeline.tokenizer
            if context.tokenizer is None:
                msg = f"Model {context.model!r} does not have a tokenizer"
                raise ValueError(msg)
            context.is_seq2seq = False

            # Account for prompt overhead when chunking
            template_without_text = context.prompt.replace("{text}", "").format(
                source_lang=context.source_lang,
                target_lang=context.target_lang,
            )
            prompt_overhead = len(
                context.tokenizer.encode(
                    template_without_text, add_special_tokens=False
                )
            )
            context.max_input_tokens = (
                context.tokenizer.model_max_length - prompt_overhead
            )
            context.translation_device = device
            logger.info("Translation model ready on device: {}", device)
            return

        context.translation_device = device
        context.uses_lang_prefix = "madlad" in context.model.lower()
        if context.tokenizer is None:
            msg = f"Model {context.model!r} does not have a tokenizer"
            raise ValueError(msg)
        context.max_input_tokens = context.tokenizer.model_max_length

        logger.info("Translation model ready on device: {}", device)

    @staticmethod
    def run(context: SetupContext, input_file: Path, output_file: Path):
        import torch

        logger.info("Translating: {}", input_file)

        with open(input_file, encoding=context.encoding) as f:
            text = f.read()

        # Account for language prefix tokens in chunk budget
        max_tokens = context.max_input_tokens
        if context.is_seq2seq and getattr(context, "uses_lang_prefix", False):
            prefix = f"<2{context.target_lang}> "
            prefix_tokens = len(
                context.tokenizer.encode(prefix, add_special_tokens=False)
            )
            max_tokens -= prefix_tokens

        chunks = _chunk_text(text, context.tokenizer, max_tokens)
        logger.info("Split into {} chunk(s)", len(chunks))

        translated_parts = []
        for i, chunk in enumerate(chunks):
            if context.is_seq2seq:
                input_text = chunk
                if getattr(context, "uses_lang_prefix", False):
                    input_text = f"<2{context.target_lang}> {chunk}"

                inputs = context.tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=context.max_input_tokens,
                )
                inputs = {
                    k: v.to(context.translation_device) for k, v in inputs.items()
                }

                with torch.no_grad():
                    output_ids = context.translation_model.generate(
                        **inputs,
                        max_new_tokens=context.max_length,
                    )

                output = context.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
            else:
                prompt = context.prompt.format(
                    source_lang=context.source_lang,
                    target_lang=context.target_lang,
                    text=chunk,
                )
                result = context.pipeline(
                    prompt,
                    max_new_tokens=context.max_length,
                    do_sample=False,
                )
                output = result[0]["generated_text"]
                # Remove the prompt from the output
                if output.startswith(prompt):
                    output = output[len(prompt) :].strip()

            translated_parts.append(output)
            logger.info(
                "Chunk {}: {} chars -> {} chars", i + 1, len(chunk), len(output)
            )

        translated = " ".join(translated_parts)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(translated)

        logger.info("Done")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s]


def _chunk_text(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Pack sentences into chunks that fit within the token limit."""
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        token_count = len(tokenizer.encode(sentence, add_special_tokens=False))

        if token_count > max_tokens:
            # Flush current chunk
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            # Force-split the long sentence at token boundaries
            chunks.extend(_split_by_tokens(sentence, tokenizer, max_tokens))
            continue

        if current_tokens + token_count > max_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += token_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Last resort: split text at token boundaries.

    Note: the encode/decode round-trip may not preserve the original
    text exactly (e.g., unknown characters, whitespace normalization,
    Unicode normalization).
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks
