"""Separate inference validation from opt-in training smoke experiments."""

import os

import pytest


def pytest_ignore_collect(collection_path, config):
    # Training smoke executes backward and optimizer steps; opt in explicitly.
    if collection_path.name == "test_fbdnet_nwd_smoke.py":
        return os.getenv("RUN_TRAINING_TESTS") != "1"
    return None


@pytest.fixture(scope="session", autouse=True)
def bounded_cpu_threads():
    import torch

    torch.set_num_threads(2)
