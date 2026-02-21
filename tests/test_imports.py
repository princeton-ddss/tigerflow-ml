"""Smoke tests: verify package imports and task class structure."""

from tigerflow.tasks import LocalTask, SlurmTask


def test_top_level_imports():
    from tigerflow_ml import Detect, OCR, Transcribe, Translate

    assert OCR is not None
    assert Translate is not None
    assert Transcribe is not None
    assert Detect is not None


def test_local_variants_importable():
    from tigerflow_ml.audio.transcribe.local import Transcribe
    from tigerflow_ml.image.detect.local import Detect
    from tigerflow_ml.text.ocr.local import OCR
    from tigerflow_ml.text.translate.local import Translate

    for cls in (OCR, Translate, Transcribe, Detect):
        assert issubclass(cls, LocalTask)


def test_slurm_variants_importable():
    from tigerflow_ml.audio.transcribe.slurm import Transcribe
    from tigerflow_ml.image.detect.slurm import Detect
    from tigerflow_ml.text.ocr.slurm import OCR
    from tigerflow_ml.text.translate.slurm import Translate

    for cls in (OCR, Translate, Transcribe, Detect):
        assert issubclass(cls, SlurmTask)


def test_each_task_has_params():
    from tigerflow_ml import Detect, OCR, Transcribe, Translate

    for cls in (OCR, Translate, Transcribe, Detect):
        assert hasattr(cls, "Params")
        assert hasattr(cls, "setup")
        assert hasattr(cls, "run")
