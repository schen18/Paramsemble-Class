"""Property-based tests for model scorer."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from hypothesis import given, strategies as st, assume, settings
from paramsemble_class.scoring.scorer import ModelScorer
from paramsemble_class.utils.model_io import ModelIO


# Helper function to generate synthetic data
def generate_synthetic_data(n_samples, n_features, random_state):
    """Generate synthetic feature data and IDs for testing."""
    np.random.seed(random_state)

    # Generate feature matrix
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Generate IDs
    ids = np.arange(n_samples)

    return X, ids


# Helper function to generate equation dictionaries
def generate_equation_dict(feature_names, random_state):
    """Generate a random equation dictionary."""
    np.random.seed(random_state)

    # Select random subset of features
    n_features_to_use = np.random.randint(1, len(feature_names) + 1)
    selected_features = np.random.choice(feature_names, size=n_features_to_use, replace=False)

    equation_dict = {}
    for feature in selected_features:
        equation_dict[feature] = float(np.random.randn())

    equation_dict["constant"] = float(np.random.randn())

    return equation_dict


# Feature: elr-package, Property 26: Scoring decile restriction for intersect/venn
# Validates: Requirements 8.2


@given(
    n_samples=st.integers(min_value=50, max_value=200),
    n_features=st.integers(min_value=3, max_value=10),
    n_models=st.integers(min_value=2, max_value=5),
    d=st.integers(min_value=1, max_value=10),
    method=st.sampled_from(["intersect", "venn"]),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_scoring_decile_restriction(
    n_samples, n_features, n_models, d, method, random_state
):
    """
    Property 26: Scoring decile restriction for intersect/venn.

    For any scoring dataset with method="intersect" or "venn",
    output should only include IDs from the top d deciles of
    each constituent model's predictions.
    """
    # Generate synthetic data
    X, ids = generate_synthetic_data(n_samples, n_features, random_state)

    # Generate equation dictionaries for models
    feature_names = X.columns.tolist()
    models = []
    for i in range(n_models):
        equation_dict = generate_equation_dict(feature_names, random_state + i)
        models.append(equation_dict)

    # Create temporary model JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        # Export model configuration
        ModelIO.export_selected_models(
            method=method, d=d, selected_models=models, meta_model=None, filepath=filepath
        )

        # Create scorer and score data
        scorer = ModelScorer(filepath)
        result_df = scorer.score(X, ids)

        # If result is empty, that's valid
        if result_df.empty:
            return

        # For each model, calculate which IDs should be in top d deciles
        all_valid_ids = set()

        for model_equation in models:
            # Apply equation to get probabilities
            probabilities = scorer._apply_logistic_regression(X, model_equation)

            # Rank predictions
            score_df = pd.DataFrame({"id": ids, "score": probabilities})
            score_df = score_df.sort_values("score", ascending=False).reset_index(drop=True)

            # Calculate top d deciles
            samples_per_decile = n_samples / 10.0
            top_d_cutoff = int(np.ceil(d * samples_per_decile))

            # Get IDs from top d deciles
            top_d_ids = set(score_df.iloc[:top_d_cutoff]["id"].values)
            all_valid_ids.update(top_d_ids)

        # Verify all IDs in result are from top d deciles of at least one model
        result_ids = set(result_df["id"].values)

        assert result_ids.issubset(all_valid_ids), (
            f"Result contains IDs not in top {d} deciles of any model: "
            f"{result_ids - all_valid_ids}"
        )

    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)


# Feature: elr-package, Property 27: Ensemble scoring consistency
# Validates: Requirements 8.3


@given(
    n_samples=st.integers(min_value=50, max_value=150),
    n_features=st.integers(min_value=3, max_value=8),
    n_models=st.integers(min_value=2, max_value=5),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_ensemble_scoring_consistency(n_samples, n_features, n_models, random_state):
    """
    Property 27: Ensemble scoring consistency.

    For any test dataset used in training with method="ensemble",
    scoring that same dataset should produce predictions matching
    the original fit predictions.

    This test verifies that the scorer produces the same results
    as the training process when given the same data.
    """
    np.random.seed(random_state)

    # Generate synthetic data
    X, ids = generate_synthetic_data(n_samples, n_features, random_state)

    # Generate constituent model equations
    feature_names = X.columns.tolist()
    constituent_models = []
    for i in range(n_models):
        equation_dict = generate_equation_dict(feature_names, random_state + i)
        constituent_models.append(equation_dict)

    # Generate meta-model equation
    # Meta-model takes constituent probabilities as input
    meta_model = {}
    for i in range(n_models):
        meta_model[f"model_{i}_prob"] = float(np.random.randn())
    meta_model["constant"] = float(np.random.randn())

    # Create temporary model JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        # Export model configuration
        ModelIO.export_selected_models(
            method="ensemble",
            d=2,  # d is not used for ensemble, but required in JSON
            selected_models=constituent_models,
            meta_model=meta_model,
            filepath=filepath,
        )

        # Create scorer and score data
        scorer = ModelScorer(filepath)
        result_df = scorer.score(X, ids)

        # Manually compute expected predictions using the same logic
        # Step 1: Generate constituent probabilities
        constituent_probs = []
        for model_equation in constituent_models:
            probs = scorer._apply_logistic_regression(X, model_equation)
            constituent_probs.append(probs)

        # Step 2: Stack into meta-features
        X_meta = np.column_stack(constituent_probs)
        meta_df = pd.DataFrame(X_meta, columns=[f"model_{i}_prob" for i in range(n_models)])

        # Step 3: Apply meta-model
        expected_predictions = scorer._apply_logistic_regression(meta_df, meta_model)

        # Verify results match
        assert (
            len(result_df) == n_samples
        ), f"Expected {n_samples} predictions, got {len(result_df)}"

        assert (
            "id" in result_df.columns and "predicted" in result_df.columns
        ), f"Expected columns ['id', 'predicted'], got {result_df.columns.tolist()}"

        # Verify IDs match
        assert np.array_equal(result_df["id"].values, ids), "IDs in result do not match input IDs"

        # Verify predictions match expected values
        actual_predictions = result_df["predicted"].values
        assert np.allclose(actual_predictions, expected_predictions, rtol=1e-9), (
            f"Predictions do not match expected values. "
            f"Max difference: {np.max(np.abs(actual_predictions - expected_predictions))}"
        )

        # Verify all predictions are in valid probability range [0, 1]
        assert np.all((actual_predictions >= 0) & (actual_predictions <= 1)), (
            f"Predictions outside [0, 1] range: "
            f"min={np.min(actual_predictions)}, max={np.max(actual_predictions)}"
        )

    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)


# Feature: elr-package, Property 28: Scoring output format consistency
# Validates: Requirements 8.4


@given(
    n_samples=st.integers(min_value=50, max_value=150),
    n_features=st.integers(min_value=3, max_value=8),
    n_models=st.integers(min_value=2, max_value=5),
    d=st.integers(min_value=1, max_value=10),
    method=st.sampled_from(["intersect", "venn", "ensemble"]),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_scoring_output_format_consistency(
    n_samples, n_features, n_models, d, method, random_state
):
    """
    Property 28: Scoring output format consistency.

    For any method type, the scoring output structure (column names and types)
    should match the training output structure for that method.

    - intersect/venn: DataFrame with ['id', 'sets'] columns
    - ensemble: DataFrame with ['id', 'predicted'] columns
    """
    np.random.seed(random_state)

    # Generate synthetic data
    X, ids = generate_synthetic_data(n_samples, n_features, random_state)

    # Generate constituent model equations
    feature_names = X.columns.tolist()
    constituent_models = []
    for i in range(n_models):
        equation_dict = generate_equation_dict(feature_names, random_state + i)
        constituent_models.append(equation_dict)

    # Generate meta-model if needed
    meta_model = None
    if method == "ensemble":
        meta_model = {}
        for i in range(n_models):
            meta_model[f"model_{i}_prob"] = float(np.random.randn())
        meta_model["constant"] = float(np.random.randn())

    # Create temporary model JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        # Export model configuration
        ModelIO.export_selected_models(
            method=method,
            d=d,
            selected_models=constituent_models,
            meta_model=meta_model,
            filepath=filepath,
        )

        # Create scorer and score data
        scorer = ModelScorer(filepath)
        result_df = scorer.score(X, ids)

        # Verify output format based on method
        if method in ["intersect", "venn"]:
            # Expected columns: ['id', 'sets', 'score']
            expected_columns = ["id", "sets", "score"]

            assert list(result_df.columns) == expected_columns, (
                f"For {method} method, expected columns {expected_columns}, "
                f"got {list(result_df.columns)}"
            )

            # Verify column types
            # 'id' should match input id type (int in this case)
            # 'sets' should be integer (count)
            if not result_df.empty:
                assert result_df["sets"].dtype in [
                    np.int32,
                    np.int64,
                    int,
                ], f"'sets' column should be integer type, got {result_df['sets'].dtype}"

                # Verify sets values are positive integers
                assert np.all(
                    result_df["sets"] > 0
                ), "All 'sets' values should be positive integers"

                # Verify sets values are <= number of models
                assert np.all(
                    result_df["sets"] <= n_models
                ), f"'sets' values should not exceed number of models ({n_models})"

                # Verify no duplicate IDs
                assert result_df["id"].nunique() == len(
                    result_df
                ), "Result should not contain duplicate IDs"

        elif method == "ensemble":
            # Expected columns: ['id', 'predicted']
            expected_columns = ["id", "predicted"]

            assert list(result_df.columns) == expected_columns, (
                f"For {method} method, expected columns {expected_columns}, "
                f"got {list(result_df.columns)}"
            )

            # Verify column types
            # 'id' should match input id type
            # 'predicted' should be float (probability)
            assert result_df["predicted"].dtype in [
                np.float32,
                np.float64,
                float,
            ], f"'predicted' column should be float type, got {result_df['predicted'].dtype}"

            # Verify predicted values are in [0, 1] range
            assert np.all((result_df["predicted"] >= 0) & (result_df["predicted"] <= 1)), (
                f"'predicted' values should be in [0, 1] range. "
                f"Got min={result_df['predicted'].min()}, max={result_df['predicted'].max()}"
            )

            # Verify all input IDs are present
            assert (
                len(result_df) == n_samples
            ), f"Expected {n_samples} predictions, got {len(result_df)}"

            assert np.array_equal(
                result_df["id"].values, ids
            ), "IDs in result should match input IDs in order"

        # Verify result is a DataFrame
        assert isinstance(
            result_df, pd.DataFrame
        ), f"Result should be a pandas DataFrame, got {type(result_df)}"

    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)
