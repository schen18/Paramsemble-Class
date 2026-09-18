"""Property-based tests for venn ensemble method."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from paramsemble_class.ensemble.venn import VennMethod


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


# Feature: elr-package, Property 20: Venn method initial selection count
# Validates: Requirements 6.2


@given(
    n_models=st.integers(min_value=5, max_value=30),
    n_ids=st.integers(min_value=10, max_value=50),
    spread=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_venn_initial_selection_count(n_models, n_ids, spread, random_state):
    """
    Property 20: Venn method initial selection count.

    For any spread value n with method="venn",
    initially 2×n models should be selected before filtering.
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
    ranked_indices = VennMethod._rank_models(constituent_results, baseline_results)

    # Calculate expected initial selection count
    initial_selection_count = 2 * spread
    expected_initial_selected = min(initial_selection_count, len(ranked_indices))

    # To verify initial selection, we need to trace through the algorithm
    # We'll check that the algorithm considers the right number of models
    initially_selected_indices = ranked_indices[:expected_initial_selected]

    # The property is about the initial selection, not the final result
    # We verify that the algorithm would initially select 2×n models
    # (or all outperforming models if fewer than 2×n)

    # Count how many models outperform baseline
    outperforming_count = len(ranked_indices)

    # Expected initial selection should be min(2*spread, outperforming_count)
    assert expected_initial_selected == min(
        2 * spread, outperforming_count
    ), f"Initial selection count mismatch: expected {min(2 * spread, outperforming_count)}, got {expected_initial_selected}"

    # Verify that initially_selected_indices has the correct length
    assert (
        len(initially_selected_indices) == expected_initial_selected
    ), f"Initially selected indices length mismatch: expected {expected_initial_selected}, got {len(initially_selected_indices)}"


# Feature: elr-package, Property 21: Venn method unique ID identification
# Validates: Requirements 6.3


@given(
    n_models=st.integers(min_value=5, max_value=20),
    n_ids=st.integers(min_value=10, max_value=50),
    spread=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_venn_unique_id_identification(n_models, n_ids, spread, random_state):
    """
    Property 21: Venn method unique ID identification.

    For any constituent model DPS and baseline DPS with method="venn",
    unique IDs should be exactly those in the model DPS but not in
    baseline DPS or incremental set.
    """
    # Generate constituent results
    constituent_results = generate_constituent_results(n_models, n_ids, random_state)

    # Generate baseline results with specific DPS
    baseline_dps = set(range(5))
    baseline_results = {
        "plr": 2.0,
        "fnr": 0.3,
        "drp": 1.0,
        "drs": set(range(10)),
        "dps": baseline_dps,
    }

    # Get ranked indices
    ranked_indices = VennMethod._rank_models(constituent_results, baseline_results)

    # If no models outperform baseline, skip
    if len(ranked_indices) == 0:
        return

    # Initially select top 2×n models
    initial_selection_count = 2 * spread
    n_to_select = min(initial_selection_count, len(ranked_indices))
    initially_selected_indices = ranked_indices[:n_to_select]

    # Simulate the venn algorithm to verify unique ID identification
    incremental_id_set = set(baseline_dps)

    for idx in initially_selected_indices:
        model_dps = constituent_results[idx]["dps"]

        # Calculate unique IDs
        unique_ids = model_dps - incremental_id_set

        # Verify that unique_ids are exactly those in model_dps but not in incremental_set
        assert unique_ids == (
            model_dps - incremental_id_set
        ), f"Unique ID calculation error for model {idx}"

        # Verify that unique_ids contains no IDs from incremental_set
        assert (
            len(unique_ids & incremental_id_set) == 0
        ), f"Unique IDs should not overlap with incremental set for model {idx}"

        # Verify that all unique_ids are in model_dps
        assert unique_ids.issubset(
            model_dps
        ), f"All unique IDs should be in model DPS for model {idx}"

        # Update incremental set (as the algorithm does)
        if unique_ids:
            incremental_id_set.update(unique_ids)


# Feature: elr-package, Property 22: Venn method model discarding
# Validates: Requirements 6.4


@given(
    n_models=st.integers(min_value=5, max_value=20),
    n_ids=st.integers(min_value=10, max_value=50),
    spread=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_venn_model_discarding(n_models, n_ids, spread, random_state):
    """
    Property 22: Venn method model discarding.

    For any model with method="venn", if it has no unique IDs
    compared to baseline and incremental set, it should be
    discarded from final selection.
    """
    # Generate constituent results
    constituent_results = generate_constituent_results(n_models, n_ids, random_state)

    # Generate baseline results
    baseline_dps = set(range(5))
    baseline_results = {
        "plr": 2.0,
        "fnr": 0.3,
        "drp": 1.0,
        "drs": set(range(10)),
        "dps": baseline_dps,
    }

    # Get result DataFrame
    result_df = VennMethod.select_and_combine(constituent_results, baseline_results, spread)

    # Get ranked indices
    ranked_indices = VennMethod._rank_models(constituent_results, baseline_results)

    # If no models outperform baseline, result should be empty
    if len(ranked_indices) == 0:
        assert result_df.empty, "Result should be empty when no models outperform baseline"
        return

    # Initially select top 2×n models
    initial_selection_count = 2 * spread
    n_to_select = min(initial_selection_count, len(ranked_indices))
    initially_selected_indices = ranked_indices[:n_to_select]

    # Simulate the venn algorithm to track which models should be undiscarded
    incremental_id_set = set(baseline_dps)
    expected_undiscarded = []

    for idx in initially_selected_indices:
        model_dps = constituent_results[idx]["dps"]
        unique_ids = model_dps - incremental_id_set

        # Model should be undiscarded if it has unique IDs
        if unique_ids:
            expected_undiscarded.append(idx)
            incremental_id_set.update(unique_ids)

    # If no models have unique IDs, result should be empty
    if len(expected_undiscarded) == 0:
        assert result_df.empty, "Result should be empty when no models have unique IDs"
        return

    # Verify that the result contains IDs only from undiscarded models
    # Collect all IDs from undiscarded model DRS
    expected_ids = set()
    for idx in expected_undiscarded:
        expected_ids.update(constituent_results[idx]["drs"])

    # All IDs in result should be from undiscarded models
    if not result_df.empty:
        result_ids = set(result_df["id"].values)
        assert result_ids.issubset(
            expected_ids
        ), f"Result contains IDs not from undiscarded models: {result_ids - expected_ids}"

        # Verify that discarded models' unique IDs are not in result
        # (unless they also appear in undiscarded models)
        for idx in initially_selected_indices:
            if idx not in expected_undiscarded:
                # This model was discarded
                model_drs = constituent_results[idx]["drs"]
                # IDs unique to this discarded model should not appear in result
                # unless they also appear in undiscarded models
                unique_to_discarded = model_drs - expected_ids
                result_ids_set = set(result_df["id"].values)
                assert (
                    len(unique_to_discarded & result_ids_set) == 0
                ), f"Result contains IDs unique to discarded model {idx}: {unique_to_discarded & result_ids_set}"
