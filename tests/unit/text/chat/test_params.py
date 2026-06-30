from tigerflow_ml.text.chat._base import _ChatBase


def test_chat_defaults():
    p = _ChatBase.Params()
    assert p.max_image_pixels is None
    assert p.temperature == 0.0
    assert p.response_schema is None
