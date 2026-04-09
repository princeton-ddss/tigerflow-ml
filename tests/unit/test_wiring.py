"""Verify tigerflow can instantiate all task variants with valid configs."""

import pytest
from tigerflow.models import LocalTaskConfig, SlurmResourceConfig, SlurmTaskConfig
from tigerflow.tasks import LocalTask, SlurmTask

_SLURM_RESOURCES = SlurmResourceConfig(cpus=1, memory="4GB", time="00:10:00")

_TASKS = [
    ("ocr", ".pdf", "tigerflow_ml.text.ocr.local", "tigerflow_ml.text.ocr.slurm"),
    (
        "translate",
        ".txt",
        "tigerflow_ml.text.translate.local",
        "tigerflow_ml.text.translate.slurm",
    ),
    (
        "transcribe",
        ".wav",
        "tigerflow_ml.audio.transcribe.local",
        "tigerflow_ml.audio.transcribe.slurm",
    ),
    (
        "detect",
        ".jpg",
        "tigerflow_ml.image.detect.local",
        "tigerflow_ml.image.detect.slurm",
    ),
]


def _import_task(module_path: str):
    """Import the task class from a module (convention: class name matches module)."""
    import importlib

    module = importlib.import_module(module_path)
    # Each module exports a single task class (OCR, Translate, Transcribe, or Detect)
    classes = [
        v
        for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, (LocalTask, SlurmTask)) and v not in (LocalTask, SlurmTask)
    ]
    assert len(classes) == 1, f"Expected 1 task class in {module_path}, found {len(classes)}"
    return classes[0]


@pytest.mark.parametrize("name,input_ext,local_mod,slurm_mod", _TASKS)
def test_local_task_instantiation(name, input_ext, local_mod, slurm_mod):
    cls = _import_task(local_mod)
    config = LocalTaskConfig(
        name=f"{name}-test",
        module=cls.get_module_path(),
        input_ext=input_ext,
        kind="local",
    )
    task = cls(config)
    assert isinstance(task, LocalTask)
    assert hasattr(cls, "Params")
    assert hasattr(cls, "setup")
    assert hasattr(cls, "run")


@pytest.mark.parametrize("name,input_ext,local_mod,slurm_mod", _TASKS)
def test_slurm_task_instantiation(name, input_ext, local_mod, slurm_mod):
    cls = _import_task(slurm_mod)
    config = SlurmTaskConfig(
        name=f"{name}-test",
        module=cls.get_module_path(),
        input_ext=input_ext,
        kind="slurm",
        max_workers=1,
        worker_resources=_SLURM_RESOURCES,
    )
    task = cls(config)
    assert isinstance(task, SlurmTask)
    assert hasattr(cls, "Params")
    assert hasattr(cls, "setup")
    assert hasattr(cls, "run")
