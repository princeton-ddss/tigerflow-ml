"""Root test configuration."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: run on HPC cluster with real models and test data",
    )
