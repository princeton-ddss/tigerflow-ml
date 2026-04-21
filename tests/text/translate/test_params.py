from tigerflow_ml.text.translate._base import _DEFAULT_PROMPT, _TranslateBase


def test_translate_defaults():
    p = _TranslateBase.Params()
    assert p.model == ""
    assert p.source_lang is None
    assert p.chunk_size is None
    assert p.target_lang == "en"
    assert p.batch_size is None
    assert p.fetch is False
    assert p.prompt == _DEFAULT_PROMPT
    assert p.revision == "main"
    assert p.device == "auto"
    assert p.cache_dir == ""
