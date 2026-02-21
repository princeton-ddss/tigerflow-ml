"""Smoke tests: verify tigerflow.tasks entry points are registered."""

from importlib.metadata import entry_points


def test_entry_points_registered():
    eps = entry_points(group="tigerflow.tasks")
    names = {ep.name for ep in eps}
    expected = {
        "ocr",
        "ocr-local",
        "translate",
        "translate-local",
        "transcribe",
        "transcribe-local",
        "detect",
        "detect-local",
    }
    assert expected.issubset(names), f"Missing entry points: {expected - names}"


def test_entry_points_loadable():
    eps = entry_points(group="tigerflow.tasks")
    for ep in eps:
        cls = ep.load()
        assert cls is not None, f"Failed to load entry point: {ep.name}"
