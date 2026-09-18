"""Property-based tests for feature sampling module."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.special import comb
from paramsemble_class.core.feature_sampler import FeatureSampler

# Feature: elr-package, Property 1: Feature combination generation produces correct count
# Validates: Requirements 1.1


@given(
    m=st.integers(min_value=1, max_value=100),
    f=st.integers(min_value=1, max_value=10),
    n_features=st.integers(min_value=1, max_value=20),
    sample=st.sampled_from(["unique", "replace"]),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_feature_combination_count(m, f, n_features, sample, random_state):
    """
    Property 1: Feature combination generation produces correct count.

    For any valid m, f, and feature list, generating feature combinations
    should produce exactly m featuresets (or max possible if m exceeds it),
    each containing exactly f features.
    """
    assume(f <= n_features)

    sampler = FeatureSampler(n_features, f, m, sample, random_state)
    combinations = sampler.generate_combinations()

    # Calculate expected count (m or max combinations, whichever is smaller)
    if sample == "unique":
        max_combinations = int(comb(n_features, f, exact=True))
    else:  # replace
        # Use multicombination formula: C(n + f - 1, f)
        max_combinations = int(comb(n_features + f - 1, f, exact=True))

    expected_count = min(m, max_combinations)

    # Assert correct number of combinations generated
    assert (
        len(combinations) == expected_count
    ), f"Expected {expected_count} combinations, got {len(combinations)}"

    # Assert each combination has exactly f features
    for i, combo in enumerate(combinations):
        assert len(combo) == f, f"Combination {i} has {len(combo)} features, expected {f}"


# Feature: elr-package, Property 2: Unique sampling prevents duplicates within featuresets
# Validates: Requirements 1.2


@given(
    m=st.integers(min_value=1, max_value=50),
    f=st.integers(min_value=2, max_value=10),
    n_features=st.integers(min_value=2, max_value=20),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_unique_sampling_no_duplicates(m, f, n_features, random_state):
    """
    Property 2: Unique sampling prevents duplicates within featuresets.

    For any featureset generated with sample="unique",
    no feature should appear more than once within that featureset.
    """
    assume(f <= n_features)

    sampler = FeatureSampler(n_features, f, m, "unique", random_state)
    combinations = sampler.generate_combinations()

    # Check each combination for duplicate features
    for i, combo in enumerate(combinations):
        unique_features = set(combo)
        assert len(unique_features) == len(
            combo
        ), f"Combination {i} contains duplicate features: {combo}"

        # Also verify all features are within valid range
        for feature_idx in combo:
            assert (
                0 <= feature_idx < n_features
            ), f"Feature index {feature_idx} out of range [0, {n_features})"


# Feature: elr-package, Property 3: Maximum combinations override
# Validates: Requirements 1.4


@given(
    f=st.integers(min_value=1, max_value=5),
    n_features=st.integers(min_value=1, max_value=10),
    sample=st.sampled_from(["unique", "replace"]),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=50, deadline=2000)  # Reduce examples and increase deadline
def test_property_max_combinations_override(f, n_features, sample, random_state):
    """
    Property 3: Maximum combinations override.

    For any m value that exceeds the maximum possible combinations
    for given f and sample method, the system should automatically
    cap m at the calculated maximum.
    """
    assume(f <= n_features)

    # Calculate max combinations
    if sample == "unique":
        max_combinations = int(comb(n_features, f, exact=True))
    else:  # replace
        # Use multicombination formula: C(n + f - 1, f)
        max_combinations = int(comb(n_features + f - 1, f, exact=True))

    # Skip if max_combinations is too large (would take too long to generate)
    assume(max_combinations <= 1000)

    # Set m to exceed max combinations
    m = max_combinations + 100

    sampler = FeatureSampler(n_features, f, m, sample, random_state)
    combinations = sampler.generate_combinations()

    # Should return exactly max_combinations, not m
    assert (
        len(combinations) == max_combinations
    ), f"Expected {max_combinations} combinations (capped), got {len(combinations)}"


# Feature: elr-package, Property 4: Maximum combinations calculation correctness
# Validates: Requirements 1.5


@given(f=st.integers(min_value=1, max_value=10), n_features=st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_property_max_combinations_calculation(f, n_features):
    """
    Property 4: Maximum combinations calculation correctness.

    For any number of features n, f value, and sample method,
    the calculated maximum should equal C(n,f) for "unique" sampling
    or n^f for "replace" sampling.
    """
    assume(f <= n_features)

    # Test unique sampling
    sampler_unique = FeatureSampler(n_features, f, 1, "unique")
    max_unique = sampler_unique._calculate_max_combinations()
    expected_unique = int(comb(n_features, f, exact=True))

    assert (
        max_unique == expected_unique
    ), f"Unique: Expected C({n_features},{f})={expected_unique}, got {max_unique}"

    # Test replace sampling
    sampler_replace = FeatureSampler(n_features, f, 1, "replace")
    max_replace = sampler_replace._calculate_max_combinations()
    # Use multicombination formula: C(n + f - 1, f)
    expected_replace = int(comb(n_features + f - 1, f, exact=True))

    assert (
        max_replace == expected_replace
    ), f"Replace: Expected C({n_features}+{f}-1,{f})={expected_replace}, got {max_replace}"
