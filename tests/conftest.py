"""Pytest configuration: non-interactive backend and warning filters."""

import matplotlib

matplotlib.use("Agg")


def pytest_configure(config):
    """Ignore deprecation warnings from sklearn/scipy (outside our code)."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:sklearn.linear_model.*",
    )
