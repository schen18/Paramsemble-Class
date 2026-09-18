"""Property-based tests for intersect ensemble method."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from paramsemble_class.ensemble.intersect import IntersectMethod


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


# Feature: elr-package, Property 15: Intersect method ranking consistency
# Validates: Requirements 5.1


@given(
    n_models=st.integers(min_value=3, max_value=20),
    n_ids=st.integers(min_value=10, max_value=50),
    spread=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_intersect_ranking_consistency(n_models, n_ids, spread, random_state):
    """
    Property 15: Intersect method ranking consistency.

    For any set of constituent models with method="intersect",
    models should be ranked such that higher PLR, lower FNR,
    and higher DRP result in better rankings.
    """
    # Generate constituent results
    constituent_results = generate_constituent_results(n_models, n_ids, random_state)

    # Generate baseline results (mediocre performance)
    baseline_results = {
        "plr": 2.0,
        "fnr": 0.3,
        "drp": 1.0,
        "drs": set(range(5)),
        "dps": set(range(3)),
    }

    # Get ranked indices
    ranked_indices = IntersectMethod._rank_models(constituent_results, baseline_results)

    # If no models outperform baseline, that's valid
    if len(ranked_indices) == 0:
        return

    from scipy.stats import rankdata

    # Every ranked model must pass the gate: beat the baseline on at
    # least one metric (higher PLR, lower FNR, or higher DRP)
    for idx in ranked_indices:
        result = constituent_results[idx]
        better = (
            result["plr"] > baseline_results["plr"],
            result["fnr"] < baseline_results["fnr"],
            result["drp"] > baseline_results["drp"],
        )
        assert any(better), f"Model {idx} ranked without beating the baseline on any metric"

    # Ranking is a Borda count over ranks of the three metrics. Verify
    # the Borda sum is non-decreasing along the ranking.
    plr = np.array([constituent_results[i]["plr"] for i in ranked_indices], dtype=float)
    fnr = np.array([constituent_results[i]["fnr"] for i in ranked_indices], dtype=float)
    drp = np.array([constituent_results[i]["drp"] for i in ranked_indices], dtype=float)

    borda = (
        rankdata(-plr, method="average")
        + rankdata(fnr, method="average")
        + rankdata(-drp, method="average")
    )

    for i in range(len(ranked_indices) - 1):
        assert (
            borda[i] <= borda[i + 1] + 1e-9
        ), f"Ranking inconsistency at position {i}: Borda {borda[i]} > {borda[i + 1]}"


# Feature: elr-package, Property 16: Intersect method selection count
# Validates: Requirements 5.2


@given(
    n_models=st.integers(min_value=5, max_value=20),
    n_ids=st.integers(min_value=10, max_value=50),
    spread=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_intersect_selection_count(n_models, n_ids, spread, random_state):
    """
    Property 16: Intersect method selection count.

    For any spread value n and set of models with method="intersect",
    exactly n models should be selected (or fewer if insufficient models
    outperform baseline).
    """
    # Generate constituent results
    constituent_results = generate_constituent_results(n_models, n_ids, random_state)

    # Generate baseline results
    baseline_results = {
        "plr": 2.0,
        "fnr": 0.3,
        "drp": 1.0,
        "drs": set(range(5)),
        "dps": set(range(3)),
    }

    # Get result DataFrame
    result_df = IntersectMethod.select_and_combine(constituent_results, baseline_results, spread)

    # Count how many models outperform baseline on any metric (the gate)
    outperforming_count = sum(
        1
        for result in constituent_results
        if (
            result["plr"] > baseline_results["plr"]
            or result["fnr"] < baseline_results["fnr"]
            or result["drp"] > baseline_results["drp"]
        )
    )

    # Expected number of selected models
    expected_selected = min(spread, outperforming_count)

    # If no models selected, DataFrame should be empty
    if expected_selected == 0:
        assert (
            len(result_df) == 0 or result_df.empty
        ), "Expected empty DataFrame when no models outperform baseline"
        return

    # Count unique models that contributed to the result
    # We can infer this from the maximum 'sets' value
    # (a model can contribute at most once per ID)
    if not result_df.empty:
        max_sets = result_df["sets"].max()
        # max_sets should be <= expected_selected
        assert (
            max_sets <= expected_selected
        ), f"Max sets ({max_sets}) exceeds expected selected models ({expected_selected})"


# Feature: elr-package, Property 17: Intersect method ID deduplication
# Validates: Requirements 5.3


@given(
    n_models=st.integers(min_value=3, max_value=20),
    n_ids=st.integers(min_value=10, max_value=50),
    spread=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_intersect_deduplication(n_models, n_ids, spread, random_state):
    """
    Property 17: Intersect method ID deduplication.

    For any selected models with method="intersect",
    the output dataframe should contain no duplicate IDs.
    """
    # Generate constituent results
    constituent_results = generate_constituent_results(n_models, n_ids, random_state)

    # Generate baseline results
    baseline_results = {
        "plr": 2.0,
        "fnr": 0.3,
        "drp": 1.0,
        "drs": set(range(5)),
        "dps": set(range(3)),
    }

    # Get result DataFrame
    result_df = IntersectMethod.select_and_combine(constituent_results, baseline_results, spread)

    # Check for duplicate IDs
    if not result_df.empty:
        id_counts = result_df["id"].value_counts()
        duplicates = id_counts[id_counts > 1]

        assert len(duplicates) == 0, f"Found duplicate IDs: {duplicates.to_dict()}"

        # Also verify that number of rows equals number of unique IDs
        assert (
            len(result_df) == result_df["id"].nunique()
        ), "Number of rows does not match number of unique IDs"


# Feature: elr-package, Property 18: Intersect method sets count accuracy
# Validates: Requirements 5.4


@given(
    n_models=st.integers(min_value=3, max_value=20),
    n_ids=st.integers(min_value=10, max_value=50),
    spread=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_intersect_sets_count_accuracy(n_models, n_ids, spread, random_state):
    """
    Property 18: Intersect method sets count accuracy.

    For any selected models with method="intersect",
    the sets count for each ID should equal the number of
    selected model DRS that contain that ID.
    """
    # Generate constituent results
    constituent_results = generate_constituent_results(n_models, n_ids, random_state)

    # Generate baseline results
    baseline_results = {
        "plr": 2.0,
        "fnr": 0.3,
        "drp": 1.0,
        "drs": set(range(5)),
        "dps": set(range(3)),
    }

    # Get result DataFrame
    result_df = IntersectMethod.select_and_combine(constituent_results, baseline_results, spread)

    # If empty, nothing to check
    if result_df.empty:
        return

    # Get ranked indices to determine which models were selected
    ranked_indices = IntersectMethod._rank_models(constituent_results, baseline_results)
    n_to_select = min(spread, len(ranked_indices))
    selected_indices = ranked_indices[:n_to_select]

    # Manually count occurrences for each ID
    expected_counts = {}
    for idx in selected_indices:
        drs = constituent_results[idx]["drs"]
        for id_value in drs:
            expected_counts[id_value] = expected_counts.get(id_value, 0) + 1

    # Verify each ID in result has correct count
    for _, row in result_df.iterrows():
        id_value = row["id"]
        actual_count = row["sets"]
        expected_count = expected_counts.get(id_value, 0)

        assert (
            actual_count == expected_count
        ), f"ID {id_value}: expected count {expected_count}, got {actual_count}"

    # Verify all IDs with counts are in result
    for id_value, expected_count in expected_counts.items():
        assert (
            id_value in result_df["id"].values
        ), f"ID {id_value} with count {expected_count} missing from result"


# Feature: elr-package, Property 19: Intersect modeljson round-trip
# Validates: Requirements 5.5


@given(
    n_models=st.integers(min_value=3, max_value=10),
    n_features=st.integers(min_value=2, max_value=10),
    spread=st.integers(min_value=1, max_value=5),
    d=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_intersect_modeljson_roundtrip(n_models, n_features, spread, d, random_state):
    """
    Property 19: Intersect modeljson round-trip.

    For any intersect method execution with modeljson specified,
    saving and loading the JSON should preserve method name,
    d parameter, and all equation dictionaries.
    """
    import tempfile
    import os
    from paramsemble_class.utils.model_io import ModelIO

    np.random.seed(random_state)

    # Generate equation dictionaries for selected models
    selected_models = []
    for i in range(min(n_models, spread)):
        equation_dict = {}
        # Generate random coefficients for random subset of features
        n_model_features = np.random.randint(1, n_features + 1)
        feature_indices = np.random.choice(n_features, size=n_model_features, replace=False)

        for feat_idx in feature_indices:
            equation_dict[f"feature_{feat_idx}"] = float(np.random.randn())

        equation_dict["constant"] = float(np.random.randn())
        selected_models.append(equation_dict)

    # Skip if no models to export
    assume(len(selected_models) > 0)

    # Create temporary file path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        # Export to JSON
        ModelIO.export_selected_models(
            method="intersect",
            d=d,
            selected_models=selected_models,
            meta_model=None,
            filepath=filepath,
        )

        # Load from JSON
        loaded_config = ModelIO.load_model(filepath)

        # Verify method name
        assert (
            loaded_config["method"] == "intersect"
        ), f"Method mismatch: expected 'intersect', got '{loaded_config['method']}'"

        # Verify d parameter
        assert (
            loaded_config["d"] == d
        ), f"d parameter mismatch: expected {d}, got {loaded_config['d']}"

        # Verify number of models
        assert len(loaded_config["models"]) == len(
            selected_models
        ), f"Model count mismatch: expected {len(selected_models)}, got {len(loaded_config['models'])}"

        # Verify each equation dictionary
        for i, (original, loaded) in enumerate(zip(selected_models, loaded_config["models"])):
            # Check that all keys match
            assert set(original.keys()) == set(
                loaded.keys()
            ), f"Model {i} keys mismatch: expected {set(original.keys())}, got {set(loaded.keys())}"

            # Check that all values match
            for key in original.keys():
                assert np.isclose(
                    original[key], loaded[key], rtol=1e-9
                ), f"Model {i} coefficient mismatch for '{key}': expected {original[key]}, got {loaded[key]}"

        # Verify no meta_model for intersect method
        assert (
            "meta_model" not in loaded_config or loaded_config.get("meta_model") is None
        ), "Intersect method should not have meta_model in JSON"

    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)
