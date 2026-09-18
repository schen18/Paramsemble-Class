"""Property-based tests for performance metrics module."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from sklearn.metrics import confusion_matrix
from paramsemble_class.metrics.performance import PerformanceMetrics

# Feature: elr-package, Property 5: PLR calculation formula
# Validates: Requirements 2.2


@given(
    n_samples=st.integers(min_value=10, max_value=200),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_plr_calculation_formula(n_samples, random_state):
    """
    Property 5: PLR calculation formula.

    For any confusion matrix with predictions and true labels,
    the calculated Positive Likelihood Ratio should equal
    (True Positive Rate) / (False Positive Rate).
    """
    np.random.seed(random_state)

    # Generate random binary labels and predictions
    y_true = np.random.randint(0, 2, size=n_samples)
    y_pred = np.random.randint(0, 2, size=n_samples)

    # Ensure we have at least some positives and negatives
    assume(np.sum(y_true == 1) > 0)
    assume(np.sum(y_true == 0) > 0)

    # Calculate PLR using our implementation
    plr = PerformanceMetrics.positive_likelihood_ratio(y_true, y_pred)

    # Calculate expected PLR manually
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if fpr == 0:
        expected_plr = np.inf if tpr > 0 else 0.0
    else:
        expected_plr = tpr / fpr

    # Assert PLR matches expected value
    if np.isinf(expected_plr):
        assert np.isinf(plr), f"Expected inf, got {plr}"
    else:
        assert np.isclose(
            plr, expected_plr, rtol=1e-9
        ), f"PLR mismatch: expected {expected_plr}, got {plr}"


# Feature: elr-package, Property 6: FNR calculation formula
# Validates: Requirements 2.3


@given(
    n_samples=st.integers(min_value=10, max_value=200),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_fnr_calculation_formula(n_samples, random_state):
    """
    Property 6: FNR calculation formula.

    For any confusion matrix with predictions and true labels,
    the calculated False Negative Rate should equal FN / (FN + TP).
    """
    np.random.seed(random_state)

    # Generate random binary labels and predictions
    y_true = np.random.randint(0, 2, size=n_samples)
    y_pred = np.random.randint(0, 2, size=n_samples)

    # Ensure we have at least some positives
    assume(np.sum(y_true == 1) > 0)

    # Calculate FNR using our implementation
    fnr = PerformanceMetrics.false_negative_rate(y_true, y_pred)

    # Calculate expected FNR manually
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    expected_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Assert FNR matches expected value
    assert np.isclose(
        fnr, expected_fnr, rtol=1e-9
    ), f"FNR mismatch: expected {expected_fnr}, got {fnr}"


# Feature: elr-package, Property 7: DRP calculation formula
# Validates: Requirements 2.4, 2.5


@given(
    n_samples=st.integers(min_value=20, max_value=200),
    d=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_drp_calculation_formula(n_samples, d, random_state):
    """
    Property 7: DRP calculation formula.

    For any predictions, true labels, and d value,
    the calculated Decile Ranked Performance should equal
    (TPR in top d deciles) / (TPR in entire test set).
    """
    np.random.seed(random_state)

    # Generate random binary labels and scores
    y_true = np.random.randint(0, 2, size=n_samples)
    y_score = np.random.rand(n_samples)

    # Ensure we have at least some positives
    assume(np.sum(y_true == 1) > 0)

    # Calculate DRP using our implementation
    drp = PerformanceMetrics.decile_ranked_performance(y_true, y_score, d)

    # Calculate expected DRP manually
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_y_true = y_true[sorted_indices]

    decile_size = n_samples / 10.0
    top_d_size = int(np.ceil(d * decile_size))

    top_d_labels = sorted_y_true[:top_d_size]

    tp_top_d = np.sum(top_d_labels == 1)
    total_samples_top_d = len(top_d_labels)
    tpr_top_d = tp_top_d / total_samples_top_d if total_samples_top_d > 0 else 0.0

    total_positives = np.sum(y_true == 1)
    tpr_full = total_positives / n_samples if n_samples > 0 else 0.0

    expected_drp = tpr_top_d / tpr_full if tpr_full > 0 else 0.0

    # Assert DRP matches expected value
    assert np.isclose(
        drp, expected_drp, rtol=1e-9
    ), f"DRP mismatch: expected {expected_drp}, got {drp}"


# Feature: elr-package, Property 8: DRS extraction correctness
# Validates: Requirements 2.6


@given(
    n_samples=st.integers(min_value=20, max_value=200),
    d=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_drs_extraction_correctness(n_samples, d, random_state):
    """
    Property 8: DRS extraction correctness.

    For any predictions, IDs, and d value, the Decile Ranked Set
    should contain exactly the IDs corresponding to records in the
    top d deciles when sorted by prediction score descending.
    """
    np.random.seed(random_state)

    # Generate random IDs and scores
    ids = np.arange(n_samples)
    y_score = np.random.rand(n_samples)

    # Extract DRS using our implementation
    drs = PerformanceMetrics.extract_decile_ranked_set(ids, y_score, d)

    # Calculate expected DRS manually
    sorted_indices = np.argsort(y_score)[::-1]

    decile_size = n_samples / 10.0
    top_d_size = int(np.ceil(d * decile_size))

    top_d_indices = sorted_indices[:top_d_size]
    expected_drs = set(ids[top_d_indices])

    # Assert DRS matches expected set
    assert drs == expected_drs, f"DRS mismatch: expected {expected_drs}, got {drs}"

    # Assert correct size
    assert len(drs) == top_d_size, f"DRS size mismatch: expected {top_d_size}, got {len(drs)}"


# Feature: elr-package, Property 9: DPS extraction correctness
# Validates: Requirements 2.7


@given(
    n_samples=st.integers(min_value=20, max_value=200),
    d=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_dps_extraction_correctness(n_samples, d, random_state):
    """
    Property 9: DPS extraction correctness.

    For any predictions, true labels, IDs, and d value,
    the Decile Positive Set should contain only IDs that are
    both in the top d deciles and are true positives.
    """
    np.random.seed(random_state)

    # Generate random IDs, labels, and scores
    ids = np.arange(n_samples)
    y_true = np.random.randint(0, 2, size=n_samples)
    y_score = np.random.rand(n_samples)

    # Ensure we have at least some positives
    assume(np.sum(y_true == 1) > 0)

    # Extract DPS using our implementation
    dps = PerformanceMetrics.extract_decile_positive_set(ids, y_true, y_score, d)

    # Calculate expected DPS manually
    sorted_indices = np.argsort(y_score)[::-1]

    decile_size = n_samples / 10.0
    top_d_size = int(np.ceil(d * decile_size))

    top_d_indices = sorted_indices[:top_d_size]

    # Filter for true positives only
    true_positive_mask = y_true[top_d_indices] == 1
    expected_dps = set(ids[top_d_indices][true_positive_mask])

    # Assert DPS matches expected set
    assert dps == expected_dps, f"DPS mismatch: expected {expected_dps}, got {dps}"

    # Assert all IDs in DPS are true positives
    for id_val in dps:
        id_index = np.where(ids == id_val)[0][0]
        assert y_true[id_index] == 1, f"ID {id_val} in DPS is not a true positive"

    # Assert all IDs in DPS are in top d deciles
    top_d_ids = set(ids[top_d_indices])
    assert dps.issubset(top_d_ids), f"DPS contains IDs not in top {d} deciles"
