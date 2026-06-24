from tigerflow_ml.text.chat_completion._base import _ChatCompletionBase


def test_chat_completion_defaults():
    p = _ChatCompletionBase.Params()
    assert p.max_image_pixels is None
    assert p.temperature == 0.0
