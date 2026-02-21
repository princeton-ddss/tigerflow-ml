"""Smoke tests: verify default parameter values for each task."""

from tigerflow_ml.audio.transcribe._base import _TranscribeBase
from tigerflow_ml.image.detect._base import _DetectBase
from tigerflow_ml.text.ocr._base import _OCRBase
from tigerflow_ml.text.translate._base import _TranslateBase


def test_ocr_defaults():
    p = _OCRBase.Params()
    assert p.model == "microsoft/trocr-base-printed"
    assert p.max_length == 512
    assert p.batch_size == 4
    assert p.device == "auto"


def test_translate_defaults():
    p = _TranslateBase.Params()
    assert p.model == "Helsinki-NLP/opus-mt-en-de"
    assert p.max_length == 512
    assert p.encoding == "utf-8-sig"


def test_transcribe_defaults():
    p = _TranscribeBase.Params()
    assert p.model == "openai/whisper-large-v3"
    assert p.batch_size == 16
    assert p.chunk_length_s == 30.0
    assert p.stride_length_s == 5.0
    assert p.return_timestamps is False


def test_detect_defaults():
    p = _DetectBase.Params()
    assert p.model == "PekingU/rtdetr_r50vd"
    assert p.threshold == 0.3
    assert p.batch_size == 4
    assert p.sample_fps == 1.0
    assert p.labels == ""
