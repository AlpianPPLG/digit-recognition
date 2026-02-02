"""
Test Configuration - Pytest Fixtures

This module contains shared fixtures for all tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests"""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_input():
    """Create sample input data (batch_size=10, features=784)"""
    np.random.seed(42)
    return np.random.randn(10, 784).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Create sample one-hot encoded labels"""
    labels = np.zeros((10, 10), dtype=np.float32)
    for i in range(10):
        labels[i, i % 10] = 1.0
    return labels


@pytest.fixture
def sample_image():
    """Create sample 28x28 grayscale image"""
    np.random.seed(42)
    return np.random.rand(28, 28).astype(np.float32)


@pytest.fixture
def mnist_sample():
    """Create a small sample mimicking MNIST data"""
    np.random.seed(42)
    X = np.random.rand(100, 784).astype(np.float32)
    y = np.zeros((100, 10), dtype=np.float32)
    for i in range(100):
        y[i, i % 10] = 1.0
    return X, y
