"""Performance metrics for ELR classification models.

This module is the single source of truth for decile-based logic. All
decile boundaries in the package (training evaluation, prediction,
JSON scoring and SQL generation) use :func:`top_d_count`, so every
scoring path ranks and cuts identically for a given dataset.

Note on DRP
-----------
``decile_ranked_performance`` computes *lift at the top d deciles*:

    DRP = (positives in top d deciles / size of top d deciles)
          / (total positives / dataset size)
        = precision(top d) / prevalence

A DRP of 1.0 means the top deciles capture positives at the base rate;
higher values mean positives are concentrated at the top of the ranking.
"""

import math
import numpy as np
from typing import Set, Any
from sklearn.metrics import confusion_matrix


def top_d_count(n_samples: int, d: int) -> int:
    """Return the number of samples in the top ``d`` deciles.

    The cutoff is ``ceil(d * n / 10)``, capped at ``n``. Every decile
    computation in the package must go through this function so that
    training, prediction, JSON scoring and SQL generation agree.

    Parameters
    ----------
    n_samples : int
        Number of samples being ranked.
    d : int
        Number of top deciles (1-10).

    Returns
    -------
    int
        Size of the top-d slice.
    """
    if not 1 <= d <= 10:
        raise ValueError(f"Parameter 'd' must be between 1 and 10, got {d}")
    if n_samples < 0:
        raise ValueError(f"'n_samples' must be non-negative, got {n_samples}")
    return min(n_samples, int(math.ceil(d * n_samples / 10.0)))


def top_d_row_indices(y_score, d: int) -> np.ndarray:
    """Return row positions belonging to the top ``d`` deciles by score.

    Samples are ordered by score descending using a stable sort, so ties
    are broken by original row order. The SQL generator documents that
    databases may break ties arbitrarily; scores tied exactly at the
    cutoff can therefore select a different boundary row in SQL.

    Parameters
    ----------
    y_score : array-like of shape (n_samples,)
        Predicted scores (e.g. positive-class probabilities).
    d : int
        Number of top deciles (1-10).

    Returns
    -------
    np.ndarray
        Indices of the top-d rows (unordered positions into ``y_score``).
    """
    y_score = np.asarray(y_score)
    count = top_d_count(len(y_score), d)
    order = np.argsort(-y_score, kind="stable")
    return order[:count]


class PerformanceMetrics:
    """Calculate specialized classification metrics for ELR models."""

    @staticmethod
    def positive_likelihood_ratio(y_true, y_pred) -> float:
        """
        Calculate Positive Likelihood Ratio (PLR).

        PLR = True Positive Rate / False Positive Rate
        PLR = Sensitivity / (1 - Specificity)

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_pred : array-like of shape (n_samples,)
            Predicted binary labels (0 or 1).

        Returns
        -------
        float
            Positive Likelihood Ratio. Returns np.inf if FPR is 0 and
            TPR is positive; returns 0.0 if both are 0.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Calculate TPR (True Positive Rate / Sensitivity)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Calculate FPR (False Positive Rate)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Handle edge case where FPR is 0
        if fpr == 0:
            return np.inf if tpr > 0 else 0.0

        return tpr / fpr

    @staticmethod
    def false_negative_rate(y_true, y_pred) -> float:
        """
        Calculate False Negative Rate (FNR).

        FNR = FN / (FN + TP)
        FNR = 1 - Sensitivity

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_pred : array-like of shape (n_samples,)
            Predicted binary labels (0 or 1).

        Returns
        -------
        float
            False Negative Rate. Returns 0.0 when there are no positives.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Calculate FNR
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return fnr

    @staticmethod
    def decile_ranked_performance(y_true, y_score, d: int) -> float:
        """
        Calculate Decile Ranked Performance (DRP) — lift at the top d deciles.

        DRP = precision(top d deciles) / prevalence(entire dataset)

        where ``precision(top d)`` is the fraction of the top-d slice that
        is truly positive, and ``prevalence`` is the fraction of positives
        in the whole dataset. A value of 1.0 means the top deciles capture
        positives at the base rate; values greater than 1.0 indicate
        concentration of positives at the top of the ranking. The maximum
        reachable value is ``10 / d``.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_score : array-like of shape (n_samples,)
            Predicted scores (probabilities or decision function values).
        d : int
            Number of top deciles to consider (1-10).

        Returns
        -------
        float
            Lift ratio. Returns 0.0 when the dataset has no positives.
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        n_samples = len(y_true)
        if n_samples == 0:
            return 0.0

        # Validate d parameter
        if not 1 <= d <= 10:
            raise ValueError(f"Parameter 'd' must be between 1 and 10, got {d}")

        top_d_indices = top_d_row_indices(y_score, d)
        top_d_labels = y_true[top_d_indices]

        # Precision within the top d deciles
        tp_top_d = np.sum(top_d_labels == 1)
        precision_top_d = tp_top_d / len(top_d_labels) if len(top_d_labels) > 0 else 0.0

        # Prevalence in the entire dataset
        prevalence = np.sum(y_true == 1) / n_samples

        if prevalence == 0:
            return 0.0

        return float(precision_top_d / prevalence)

    @staticmethod
    def extract_decile_ranked_set(ids, y_score, d: int) -> Set[Any]:
        """
        Extract IDs from top d deciles.

        Parameters
        ----------
        ids : array-like of shape (n_samples,)
            ID values for each sample.
        y_score : array-like of shape (n_samples,)
            Predicted scores (probabilities or decision function values).
        d : int
            Number of top deciles to consider (1-10).

        Returns
        -------
        Set[Any]
            Set of IDs from top d deciles.
        """
        ids = np.asarray(ids)

        top_d_indices = top_d_row_indices(y_score, d)
        return set(ids[top_d_indices])

    @staticmethod
    def extract_decile_positive_set(ids, y_true, y_score, d: int) -> Set[Any]:
        """
        Extract true positive IDs from top d deciles.

        Parameters
        ----------
        ids : array-like of shape (n_samples,)
            ID values for each sample.
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_score : array-like of shape (n_samples,)
            Predicted scores (probabilities or decision function values).
        d : int
            Number of top deciles to consider (1-10).

        Returns
        -------
        Set[Any]
            Set of true positive IDs from top d deciles.
        """
        ids = np.asarray(ids)
        y_true = np.asarray(y_true)

        top_d_indices = top_d_row_indices(y_score, d)

        # Filter for true positives only
        true_positive_mask = y_true[top_d_indices] == 1
        return set(ids[top_d_indices][true_positive_mask])
