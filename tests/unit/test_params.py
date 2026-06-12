"""Smoke tests: verify default parameter values for each task."""

from tigerflow_ml.audio.transcribe._base import _TranscribeBase
from tigerflow_ml.image.detect._base import _DetectBase
from tigerflow_ml.params import HFParams, VLLMParams
from tigerflow_ml.text.ocr._base import _OCRBase


def test_ocr_defaults():
    p = _OCRBase.Params()
    assert p.max_length == 4096
    assert p.device == "auto"
    assert p.temperature == 0
    assert p.seed == 42


def test_transcribe_defaults():
    p = _TranscribeBase.Params()
    assert p.batch_size == 16
    assert p.chunk_length_s == 30.0
    assert p.stride_length_s == 5.0
    assert p.return_timestamps is False


def test_detect_defaults():
    p = _DetectBase.Params()
    assert p.threshold == 0.3
    assert p.batch_size == 4
    assert p.sample_fps == 1.0
    assert p.labels == ""


def test_vLLM_defaults():
    p = VLLMParams()
    assert p.revision == "main"
    assert p.cache_dir is None
    assert p.allow_fetch is False
    assert p.system_message is None
    assert p.max_tokens == 512
    assert p.max_model_len is None
    assert p.temperature == 0
    assert p.seed == 42
    assert p.llm_kwargs == "{}"
    assert p.sampling_kwargs == "{}"
    assert p.chat_kwargs == "{}"


def test_hfparams_defaults():
    p = HFParams()
    assert p.revision == "main"
    assert p.cache_dir is None
    assert p.allow_fetch is False
    assert p.device == "auto"
