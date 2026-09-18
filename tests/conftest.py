"""Pytest configuration and shared fixtures for ELR tests."""

import sqlite3

import pytest
from hypothesis import settings
from sklearn.datasets import make_classification

import numpy as np
import pandas as pd

# CI runners are slow; hypothesis' default 200ms/example deadline flakes there.
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


def _sqlite_has_math_functions() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("SELECT EXP(1.0), CEIL(1.5)").fetchall()
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


SQLITE_MATH_FUNCTIONS = _sqlite_has_math_functions()

requires_sqlite_math = pytest.mark.skipif(
    not SQLITE_MATH_FUNCTIONS,
    reason="generated SQL uses EXP/CEIL; this sqlite build lacks math functions",
)


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
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["id"] = np.arange(len(y))
    df["target"] = y
    return df
