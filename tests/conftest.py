"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_X():
    """Sample feature data."""
    np.random.seed(42)
    return np.random.randn(100, 4)


@pytest.fixture
def sample_y():
    """Sample binary labels."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100)


@pytest.fixture
def results_dir(tmp_path):
    """Temporary results directory."""
    return tmp_path / "results"


@pytest.fixture(autouse=True)
def set_seeds():
    """Set seeds for reproducibility before each test."""
    np.random.seed(42)
    import random
    random.seed(42)
