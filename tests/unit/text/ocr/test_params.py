from tigerflow_ml.text.ocr._base import _OCRBase


def test_ocr_defaults():
    p = _OCRBase.Params()
    assert p.max_tokens == 4096
    assert p.json_schema is None
