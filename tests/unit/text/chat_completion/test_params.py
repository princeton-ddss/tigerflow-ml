from tigerflow_ml.text.chat_completion._base import _ChatCompletionBase


def test_translate_defaults():
    p = _ChatCompletionBase.Params()
    assert p.allow_fetch is False
    assert p.system_message is None
    assert p.max_tokens == 512
    assert p.max_model_len == 32_000
    assert p.max_image_size == 1024
    assert p.revision == "main"
    assert p.device == "auto"
    assert p.cache_dir == ""
