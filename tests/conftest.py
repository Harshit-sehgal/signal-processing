"""Shared pytest configuration."""

import multiprocessing


def pytest_configure(config):  # noqa: ANN001, ARG001
    """Use ``spawn`` so PyEMD's internal Pool avoids fork-in-thread warnings."""
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
