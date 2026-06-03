from tigerflow_ml.text.translate._base import _DEFAULT_PROMPT, _TranslateBase


def test_translate_defaults():
    p = _TranslateBase.Params()
    assert p.source_lang is None
    assert p.chunk_size == 900
    assert p.target_lang == "en"
    assert p.max_model_len is None
    assert p.model_backend == "auto"
    assert p.prompt_template == _DEFAULT_PROMPT

    assert p.allow_fetch is False
    assert p.temperature == 0
    assert p.seed == 42
    assert p.system_message is None
    assert p.revision == "main"
    assert p.cache_dir is None
    assert p.llm_kwargs == "{}"
    assert p.sampling_kwargs == "{}"
    assert p.chat_kwargs == "{}"
