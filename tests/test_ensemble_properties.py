"""Property-based tests for ensemble ensemble method."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from paramsemble_class.ensemble.ensemble import EnsembleMethod
from paramsemble_class.core.constituent_model import ConstituentModel


# Helper function to generate constituent results
def generate_constituent_results(n_models, n_ids, random_state):
    """Generate random constituent model results for testing."""
    np.random.seed(random_state)

    results = []
    for i in range(n_models):
        # Generate random metrics
        plr = np.random.uniform(0.5, 10.0)
        fnr = np.random.uniform(0.0, 0.5)
        drp = np.random.uniform(0.5, 2.0)

        # Generate random DRS (subset of available IDs)
        n_drs = np.random.randint(1, n_ids + 1)
        drs = set(np.random.choice(n_ids, size=n_drs, replace=False))

        # Generate random DPS (subset of DRS)
        n_dps = np.random.randint(0, len(drs) + 1)
        dps = set(np.random.choice(list(drs), size=n_dps, replace=False)) if n_dps > 0 else set()

        results.append({"plr": plr, "fnr": fnr, "drp": drp, "drs": drs, "dps": dps})

    return results


# Helper function to create mock constituent models
def create_mock_constituent_models(n_models, n_features, n_samples, random_state):
    """Create trained constituent models for testing."""
    np.random.seed(random_state)

    # Generate synthetic training data
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, size=n_samples)

    models = []
    for i in range(n_models):
        # Select random feature subset
        n_model_features = np.random.randint(1, min(n_features, 5) + 1)
        feature_indices = list(np.random.choice(n_features, size=n_model_features, replace=False))

        # Create and train model
        model = ConstituentModel(
            feature_indices=feature_indices, solver="lbfgs", random_state=random_state + i
        )

        try:
            model.fit(X_train, y_train)
            models.append(model)
        except:
            # If training fails, skip this model
            continue

    return models


# Feature: elr-package, Property 23: Ensemble method probability generation
# Validates: Requirements 7.3


@given(
    n_models=st.integers(min_value=3, max_value=10),
    n_features=st.integers(min_value=3, max_value=10),
    n_samples=st.integers(min_value=50, max_value=100),
    spread=st.integers(min_value=1, max_value=5),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_property_ensemble_probability_generation(
    n_models, n_features, n_samples, spread, random_state
):
    """
    Property 23: Ensemble method probability generation.

    For any selected models with method="ensemble",
    each model should generate predicted probabilities for all test samples,
    with values in range [0, 1].
    """
    np.random.seed(random_state)

    # Create constituent models
    constituent_models = create_mock_constituent_models(
        n_models, n_features, n_samples, random_state
    )

    # Skip if not enough models were successfully created
    assume(len(constituent_models) >= spread)

    # Generate test data
    X_test = np.random.randn(n_samples, n_features)
    y_test = np.random.randint(0, 2, size=n_samples)
    ids = np.arange(n_samples)

    # Generate constituent results
    constituent_results = generate_constituent_results(
        len(constituent_models), n_samples, random_state
    )

    # Generate baseline results (mediocre performance)
    baseline_results = {
        "plr": 2.0,
        "fnr": 0.3,
        "drp": 1.0,
        "drs": set(range(5)),
        "dps": set(range(3)),
    }

    # Run ensemble method
    result_df, meta_equation, selected_indices = EnsembleMethod.select_and_combine(
        constituent_models=constituent_models,
        X_test=X_test,
        y_test=y_test,
        ids=ids,
        constituent_results=constituent_results,
        baseline_results=baseline_results,
        spread=spread,
        random_state=random_state,
    )

    # If no models selected, skip
    if len(selected_indices) == 0:
        return

    # Verify that each selected model generates probabilities in [0, 1]
    for idx in selected_indices:
        model = constituent_models[idx]
        proba = model.predict_proba(X_test)[:, 1]

        # Check all probabilities are in valid range
        assert np.all(proba >= 0.0), f"Model {idx} generated probabilities < 0: min={proba.min()}"
        assert np.all(proba <= 1.0), f"Model {idx} generated probabilities > 1: max={proba.max()}"

        # Check we have probability for each sample
        assert (
            len(proba) == n_samples
        ), f"Model {idx} generated {len(proba)} probabilities, expected {n_samples}"


# Feature: elr-package, Property 24: Ensemble method output structure
# Validates: Requirements 7.5


@given(
    n_models=st.integers(min_value=3, max_value=10),
    n_features=st.integers(min_value=3, max_value=10),
    n_samples=st.integers(min_value=50, max_value=100),
    spread=st.integers(min_value=1, max_value=5),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_property_ensemble_output_structure(n_models, n_features, n_samples, spread, random_state):
    """
    Property 24: Ensemble method output structure.

    For any ensemble execution, the output dataframe should contain
    all test set IDs and predicted probabilities in range [0, 1].
    """
    np.random.seed(random_state)

    # Create constituent models
    constituent_models = create_mock_constituent_models(
        n_models, n_features, n_samples, random_state
    )

    # Skip if not enough models were successfully created
    assume(len(constituent_models) >= spread)

    # Generate test data
    X_test = np.random.randn(n_samples, n_features)
    y_test = np.random.randint(0, 2, size=n_samples)
    ids = np.arange(n_samples)

    # Generate constituent results
    constituent_results = generate_constituent_results(
        len(constituent_models), n_samples, random_state
    )

    # Generate baseline results (mediocre performance)
    baseline_results = {
        "plr": 2.0,
        "fnr": 0.3,
        "drp": 1.0,
        "drs": set(range(5)),
        "dps": set(range(3)),
    }

    # Run ensemble method
    result_df, meta_equation, selected_indices = EnsembleMethod.select_and_combine(
        constituent_models=constituent_models,
        X_test=X_test,
        y_test=y_test,
        ids=ids,
        constituent_results=constituent_results,
        baseline_results=baseline_results,
        spread=spread,
        random_state=random_state,
    )

    # Verify DataFrame structure
    assert "id" in result_df.columns, "Output DataFrame missing 'id' column"
    assert "predicted" in result_df.columns, "Output DataFrame missing 'predicted' column"

    # Verify all test IDs are present
    assert len(result_df) == n_samples, f"Expected {n_samples} rows, got {len(result_df)}"

    # Verify IDs match
    assert set(result_df["id"].values) == set(ids), "Output IDs do not match input IDs"

    # Verify predicted probabilities are in valid range [0, 1]
    predicted = result_df["predicted"].values
    assert np.all(predicted >= 0.0), f"Found predicted probabilities < 0: min={predicted.min()}"
    assert np.all(predicted <= 1.0), f"Found predicted probabilities > 1: max={predicted.max()}"


# Feature: elr-package, Property 25: Ensemble modeljson completeness
# Validates: Requirements 7.6


@given(
    n_models=st.integers(min_value=3, max_value=10),
    n_features=st.integers(min_value=3, max_value=10),
    n_samples=st.integers(min_value=50, max_value=100),
    spread=st.integers(min_value=1, max_value=5),
    d=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_property_ensemble_modeljson_completeness(
    n_models, n_features, n_samples, spread, d, random_state
):
    """
    Property 25: Ensemble modeljson completeness.

    For any ensemble method execution with modeljson specified,
    the saved JSON should include method name, all constituent model
    equation dictionaries, and the meta-model equation dictionary.
    """
    import tempfile
    import os
    from paramsemble_class.utils.model_io import ModelIO

    np.random.seed(random_state)

    # Create constituent models
    constituent_models = create_mock_constituent_models(
        n_models, n_features, n_samples, random_state
    )

    # Skip if not enough models were successfully created
    assume(len(constituent_models) >= spread)

    # Generate test data
    X_test = np.random.randn(n_samples, n_features)
    y_test = np.random.randint(0, 2, size=n_samples)
    ids = np.arange(n_samples)

    # Generate constituent results
    constituent_results = generate_constituent_results(
        len(constituent_models), n_samples, random_state
    )

    # Generate baseline results (mediocre performance)
    baseline_results = {
        "plr": 2.0,
        "fnr": 0.3,
        "drp": 1.0,
        "drs": set(range(5)),
        "dps": set(range(3)),
    }

    # Run ensemble method
    result_df, meta_equation, selected_indices = EnsembleMethod.select_and_combine(
        constituent_models=constituent_models,
        X_test=X_test,
        y_test=y_test,
        ids=ids,
        constituent_results=constituent_results,
        baseline_results=baseline_results,
        spread=spread,
        random_state=random_state,
    )

    # Skip if no models selected
    assume(len(selected_indices) > 0)

    # Extract equation dictionaries for selected models
    feature_names = [f"feature_{i}" for i in range(n_features)]
    selected_equations = []
    for idx in selected_indices:
        equation_dict = constituent_models[idx].get_equation_dict(feature_names)
        selected_equations.append(equation_dict)

    # Create temporary file path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        # Export to JSON
        ModelIO.export_selected_models(
            method="ensemble",
            d=d,
            selected_models=selected_equations,
            meta_model=meta_equation,
            filepath=filepath,
        )

        # Load from JSON
        loaded_config = ModelIO.load_model(filepath)

        # Verify method name
        assert (
            loaded_config["method"] == "ensemble"
        ), f"Method mismatch: expected 'ensemble', got '{loaded_config['method']}'"

        # Verify d parameter
        assert (
            loaded_config["d"] == d
        ), f"d parameter mismatch: expected {d}, got {loaded_config['d']}"

        # Verify constituent models are present
        assert "models" in loaded_config, "Missing 'models' key in JSON"
        assert len(loaded_config["models"]) == len(
            selected_indices
        ), f"Model count mismatch: expected {len(selected_indices)}, got {len(loaded_config['models'])}"

        # Verify meta-model is present
        assert "meta_model" in loaded_config, "Missing 'meta_model' key in JSON"
        assert (
            loaded_config["meta_model"] is not None
        ), "meta_model should not be None for ensemble method"

        # Verify meta-model structure
        meta_model_loaded = loaded_config["meta_model"]
        assert "constant" in meta_model_loaded, "Meta-model missing 'constant' key"

        # Verify meta-model has coefficients for each constituent model
        for i in range(len(selected_indices)):
            key = f"model_{i}_prob"
            assert key in meta_model_loaded, f"Meta-model missing coefficient for '{key}'"

        # Verify meta-model equation matches what was generated
        for key in meta_equation.keys():
            assert key in meta_model_loaded, f"Meta-model key '{key}' not found in loaded JSON"
            assert np.isclose(
                meta_equation[key], meta_model_loaded[key], rtol=1e-9
            ), f"Meta-model coefficient mismatch for '{key}': expected {meta_equation[key]}, got {meta_model_loaded[key]}"

    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)
