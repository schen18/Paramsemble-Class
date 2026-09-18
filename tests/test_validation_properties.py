"""Property-based tests for parameter validation."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume
from paramsemble_class.utils.validation import ParameterValidator

# Feature: elr-package, Property 29: Invalid parameter rejection
# Validates: Requirements 10.1


@given(m=st.integers(max_value=0))
def test_property_invalid_m_rejection(m):
    """
    Property 29: Invalid parameter rejection - m parameter.

    For any m < 1, the system should raise a descriptive ValueError.
    """
    params = {"m": m}
    with pytest.raises(ValueError, match="Parameter 'm' must be >= 1"):
        ParameterValidator.validate_parameters(params)


@given(f=st.integers(max_value=0))
def test_property_invalid_f_rejection(f):
    """
    Property 29: Invalid parameter rejection - f parameter.

    For any f < 1, the system should raise a descriptive ValueError.
    """
    params = {"f": f}
    with pytest.raises(ValueError, match="Parameter 'f' must be >= 1"):
        ParameterValidator.validate_parameters(params)


@given(sample=st.text(min_size=1).filter(lambda x: x not in ["unique", "replace"]))
def test_property_invalid_sample_rejection(sample):
    """
    Property 29: Invalid parameter rejection - sample parameter.

    For any sample value not in ["unique", "replace"],
    the system should raise a descriptive ValueError.
    """
    params = {"sample": sample}
    with pytest.raises(ValueError, match="Parameter 'sample' must be one of"):
        ParameterValidator.validate_parameters(params)


@given(d=st.integers().filter(lambda x: x < 1 or x > 10))
def test_property_invalid_d_rejection(d):
    """
    Property 29: Invalid parameter rejection - d parameter.

    For any d not between 1 and 10 inclusive,
    the system should raise a descriptive ValueError.
    """
    params = {"d": d}
    with pytest.raises(ValueError, match="Parameter 'd' must be between 1 and 10"):
        ParameterValidator.validate_parameters(params)


@given(method=st.text(min_size=1).filter(lambda x: x not in ["intersect", "venn", "ensemble"]))
def test_property_invalid_method_rejection(method):
    """
    Property 29: Invalid parameter rejection - method parameter.

    For any method value not in ["intersect", "venn", "ensemble"],
    the system should raise a descriptive ValueError.
    """
    params = {"method": method}
    with pytest.raises(ValueError, match="Parameter 'method' must be one of"):
        ParameterValidator.validate_parameters(params)


@given(spread=st.integers(max_value=0))
def test_property_invalid_spread_rejection(spread):
    """
    Property 29: Invalid parameter rejection - spread parameter.

    For any spread < 1, the system should raise a descriptive ValueError.
    """
    params = {"spread": spread}
    with pytest.raises(ValueError, match="Parameter 'spread' must be >= 1"):
        ParameterValidator.validate_parameters(params)


@given(
    solver=st.text(min_size=1).filter(
        lambda x: x
        not in ["auto", "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    )
)
def test_property_invalid_solver_rejection(solver):
    """
    Property 29: Invalid parameter rejection - solver parameter.

    For any solver value not in the valid solver list,
    the system should raise a descriptive ValueError.
    """
    params = {"solver": solver}
    with pytest.raises(ValueError, match="Parameter 'solver' must be one of"):
        ParameterValidator.validate_parameters(params)


@given(
    n_samples=st.integers(min_value=1, max_value=100),
    n_features=st.integers(min_value=1, max_value=20),
)
def test_property_missing_values_rejection(n_samples, n_features):
    """
    Property 29: Invalid parameter rejection - missing values in data.

    For any dataset with missing values in feature columns,
    the system should raise a descriptive ValueError.
    """
    # Create data with at least one missing value
    X = np.random.randn(n_samples, n_features)
    # Randomly insert NaN values
    nan_row = np.random.randint(0, n_samples)
    nan_col = np.random.randint(0, n_features)
    X[nan_row, nan_col] = np.nan

    y = np.random.randint(0, 2, size=n_samples)

    with pytest.raises(ValueError, match="contains missing values"):
        ParameterValidator.validate_data(X, y)


@given(
    n_samples_X=st.integers(min_value=1, max_value=100),
    n_samples_y=st.integers(min_value=1, max_value=100),
    n_features=st.integers(min_value=1, max_value=20),
)
def test_property_shape_mismatch_rejection(n_samples_X, n_samples_y, n_features):
    """
    Property 29: Invalid parameter rejection - shape mismatch.

    For any X and y with different number of samples,
    the system should raise a descriptive ValueError.
    """
    assume(n_samples_X != n_samples_y)

    X = np.random.randn(n_samples_X, n_features)
    y = np.random.randint(0, 2, size=n_samples_y)

    with pytest.raises(ValueError, match="Shape mismatch"):
        ParameterValidator.validate_data(X, y)


@given(
    n_samples=st.integers(min_value=1, max_value=100),
    n_features=st.integers(min_value=1, max_value=20),
    id_column=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=20
    ),
)
def test_property_missing_id_column_rejection(n_samples, n_features, id_column):
    """
    Property 29: Invalid parameter rejection - missing ID column.

    For any DataFrame without the specified ID column,
    the system should raise a descriptive ValueError.
    """
    # Create DataFrame without the specified ID column
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = np.random.randint(0, 2, size=n_samples)

    # Ensure the id_column doesn't exist in the DataFrame
    assume(id_column not in X.columns)

    with pytest.raises(ValueError) as exc_info:
        ParameterValidator.validate_data(X, y, id_column=id_column)

    assert "does not exist" in str(exc_info.value)


@given(f=st.integers(min_value=1, max_value=100), n_features=st.integers(min_value=1, max_value=50))
def test_property_f_exceeds_features_rejection(f, n_features):
    """
    Property 29: Invalid parameter rejection - f exceeds available features.

    For any f > n_features, the system should raise a descriptive ValueError.
    """
    assume(f > n_features)

    with pytest.raises(ValueError, match="exceeds the number of available features"):
        ParameterValidator.validate_f_against_features(f, n_features)


@given(
    m=st.integers(min_value=1, max_value=1000),
    f=st.integers(min_value=1, max_value=50),
    spread=st.integers(min_value=1, max_value=100),
    d=st.integers(min_value=1, max_value=10),
    sample=st.sampled_from(["unique", "replace"]),
    method=st.sampled_from(["intersect", "venn", "ensemble"]),
    solver=st.sampled_from(["auto", "lbfgs", "liblinear", "newton-cg", "sag", "saga"]),
)
def test_property_valid_parameters_accepted(m, f, spread, d, sample, method, solver):
    """
    Property 29: Valid parameter acceptance.

    For any valid parameter values, the system should not raise an error.
    """
    params = {
        "m": m,
        "f": f,
        "spread": spread,
        "d": d,
        "sample": sample,
        "method": method,
        "solver": solver,
    }

    # Should not raise any exception
    ParameterValidator.validate_parameters(params)


@given(
    n_samples=st.integers(min_value=1, max_value=100),
    n_features=st.integers(min_value=1, max_value=20),
)
def test_property_valid_data_accepted(n_samples, n_features):
    """
    Property 29: Valid data acceptance.

    For any valid dataset without missing values and matching shapes,
    the system should not raise an error.
    """
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)

    # Should not raise any exception
    ParameterValidator.validate_data(X, y)
