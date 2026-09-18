"""Pytest configuration and shared fixtures for ELR tests."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.1,
    )
    ids = np.arange(len(y))
    return X, y, ids


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame for testing."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=8, n_redundant=2, n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["id"] = np.arange(len(y))
    df["target"] = y
    return df
