"""Property-based tests for constituent model module."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from sklearn.datasets import make_classification
from paramsemble_class.core.constituent_model import ConstituentModel
from paramsemble_class.core.feature_sampler import FeatureSampler

# Feature: elr-package, Property 10: Constituent model count matches featureset count
# Validates: Requirements 3.1


@given(
    m=st.integers(min_value=1, max_value=20),
    f=st.integers(min_value=2, max_value=5),
    n_features=st.integers(min_value=5, max_value=15),
    n_samples=st.integers(min_value=50, max_value=200),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=5000)
def test_property_constituent_model_count(m, f, n_features, n_samples, random_state):
    """
    Property 10: Constituent model count matches featureset count.

    For any m featuresets generated, training should produce
    exactly m logistic regression models.
    """
    assume(f <= n_features)

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(f, n_features - 1),
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )

    # Split into train/test
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Generate feature combinations
    sampler = FeatureSampler(n_features, f, m, "unique", random_state)
    combinations = sampler.generate_combinations()

    # Train constituent models
    models = []
    for feature_indices in combinations:
        try:
            model = ConstituentModel(
                feature_indices=feature_indices, solver="lbfgs", random_state=random_state
            )
            model.fit(X_train, y_train)
            models.append(model)
        except Exception as e:
            # If training fails, skip this model
            # (e.g., due to singular matrix or convergence issues)
            pass

    # The number of successfully trained models should equal the number of featuresets
    # (assuming no training failures, which is reasonable for well-conditioned data)
    assert len(models) == len(
        combinations
    ), f"Expected {len(combinations)} models, got {len(models)}"


# Feature: elr-package, Property 11: Specified solver is used consistently
# Validates: Requirements 3.3


@given(
    f=st.integers(min_value=2, max_value=5),
    n_features=st.integers(min_value=5, max_value=10),
    n_samples=st.integers(min_value=50, max_value=100),
    solver=st.sampled_from(["lbfgs", "liblinear", "saga"]),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=5000)
def test_property_solver_consistency(f, n_features, n_samples, solver, random_state):
    """
    Property 11: Specified solver is used consistently.

    For any valid solver name specified (not "auto"),
    all trained logistic regression models should use that exact solver.
    """
    assume(f <= n_features)

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(f, n_features - 1),
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )

    # Split into train/test
    split_idx = int(0.7 * n_samples)
    X_train = X[:split_idx]
    y_train = y[:split_idx]

    # Generate a few feature combinations
    sampler = FeatureSampler(n_features, f, 3, "unique", random_state)
    combinations = sampler.generate_combinations()

    # Train models with specified solver
    for feature_indices in combinations:
        model = ConstituentModel(
            feature_indices=feature_indices, solver=solver, random_state=random_state
        )
        model.fit(X_train, y_train)

        # Verify the actual solver used matches the specified solver
        assert (
            model._actual_solver == solver
        ), f"Expected solver '{solver}', but model used '{model._actual_solver}'"


# Feature: elr-package, Property 12: Constituent model metrics completeness
# Validates: Requirements 3.4


@given(
    f=st.integers(min_value=2, max_value=5),
    n_features=st.integers(min_value=5, max_value=10),
    n_samples=st.integers(min_value=50, max_value=100),
    d=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=5000)
def test_property_metrics_completeness(f, n_features, n_samples, d, random_state):
    """
    Property 12: Constituent model metrics completeness.

    For any trained constituent model, evaluation should produce
    all required metrics: PLR, FNR, DRP, DRS, and DPS.
    """
    assume(f <= n_features)

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(f, n_features - 1),
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )

    # Split into train/test
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Create IDs for test samples
    ids = np.arange(len(y_test))

    # Generate feature combination
    sampler = FeatureSampler(n_features, f, 1, "unique", random_state)
    combinations = sampler.generate_combinations()
    feature_indices = combinations[0]

    # Train and evaluate model
    model = ConstituentModel(
        feature_indices=feature_indices, solver="lbfgs", random_state=random_state
    )
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test, ids, d)

    # Verify all required metrics are present
    required_metrics = ["plr", "fnr", "drp", "drs", "dps"]
    for metric_name in required_metrics:
        assert metric_name in metrics, f"Missing required metric: {metric_name}"

    # Verify metric types
    assert isinstance(metrics["plr"], (int, float)), "PLR should be numeric"
    assert isinstance(metrics["fnr"], (int, float)), "FNR should be numeric"
    assert isinstance(metrics["drp"], (int, float)), "DRP should be numeric"
    assert isinstance(metrics["drs"], set), "DRS should be a set"
    assert isinstance(metrics["dps"], set), "DPS should be a set"


# Feature: elr-package, Property 13: Equation dictionary structure
# Validates: Requirements 3.5


@given(
    f=st.integers(min_value=2, max_value=5),
    n_features=st.integers(min_value=5, max_value=10),
    n_samples=st.integers(min_value=50, max_value=100),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=5000)
def test_property_equation_dict_structure(f, n_features, n_samples, random_state):
    """
    Property 13: Equation dictionary structure.

    For any trained constituent model with feature subset,
    the equation dictionary should contain exactly one key per feature
    in the subset plus a "constant" key for the intercept.
    """
    assume(f <= n_features)

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(f, n_features - 1),
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )

    # Split into train/test
    split_idx = int(0.7 * n_samples)
    X_train = X[:split_idx]
    y_train = y[:split_idx]

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Generate feature combination
    sampler = FeatureSampler(n_features, f, 1, "unique", random_state)
    combinations = sampler.generate_combinations()
    feature_indices = combinations[0]

    # Train model
    model = ConstituentModel(
        feature_indices=feature_indices, solver="lbfgs", random_state=random_state
    )
    model.fit(X_train, y_train)

    # Get equation dictionary
    equation_dict = model.get_equation_dict(feature_names)

    # Verify structure
    # Should have exactly f feature keys + 1 constant key
    assert (
        len(equation_dict) == f + 1
    ), f"Expected {f + 1} keys (f features + constant), got {len(equation_dict)}"

    # Verify constant key exists
    assert "constant" in equation_dict, "Missing 'constant' key for intercept"

    # Verify all selected features are present
    for feature_idx in feature_indices:
        feature_name = feature_names[feature_idx]
        assert (
            feature_name in equation_dict
        ), f"Missing feature '{feature_name}' in equation dictionary"

    # Verify all values are numeric
    for key, value in equation_dict.items():
        assert isinstance(
            value, (int, float)
        ), f"Value for '{key}' should be numeric, got {type(value)}"
