"""Ensemble ensemble method for ELR."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from paramsemble_class.ensemble._common import rank_models

logger = logging.getLogger(__name__)

# Number of folds used to generate out-of-fold predictions for the
# meta-model. Reduced automatically when the selection set is small.
META_FOLDS = 5


class EnsembleMethod:
    """
    Ensemble ensemble method for creating a meta-model from top performers.

    The ensemble method combines predictions from top-performing constituent
    models by training a meta-model (logistic regression) that uses their
    predicted probabilities as features.

    The method:
    1. Keeps models that beat the baseline on any of the 3 metrics
       (PLR, FNR, DRP) and ranks them by a Borda count
    2. Selects the top ``spread`` models
    3. Trains the meta-model on *out-of-fold* predictions: the selection set
       is split with stratified K-fold, each selected model predicts on its
       held-out folds, and the meta-model is fit on those out-of-fold
       probabilities. This prevents the leakage of fitting and evaluating
       the meta-model on the same rows.
    4. Generates final predictions by applying the meta-model to the
       selected models' full predictions

    Returns a DataFrame with IDs and predicted probabilities, plus the
    meta-model equation.
    """

    @staticmethod
    def _rank_models(
        constituent_results: List[Dict[str, Any]], baseline_results: Dict[str, Any]
    ) -> List[int]:
        """
        Rank constituent models by performance metrics.

        Models must outperform the baseline on at least one of the metrics
        (higher PLR, lower FNR, higher DRP); eligible models are ordered
        by a Borda count over the three metrics.

        Parameters
        ----------
        constituent_results : List[Dict[str, Any]]
            Metric dicts with keys 'plr', 'fnr', 'drp', 'drs', 'dps'.
        baseline_results : Dict[str, Any]
            Baseline metric dict with the same keys.

        Returns
        -------
        List[int]
            Indices of eligible models sorted by rank (best first).
        """
        return rank_models(constituent_results, baseline_results)

    @staticmethod
    def _meta_feature_matrix(constituent_models, X, selected_indices) -> np.ndarray:
        """Stack positive-class probabilities of the selected models, shape (n, k)."""
        columns = [constituent_models[idx].predict_proba(X)[:, 1] for idx in selected_indices]
        if not columns:
            return np.empty((len(X), 0))
        return np.column_stack(columns)

    @staticmethod
    def _out_of_fold_meta_features(
        constituent_models,
        X,
        y,
        selected_indices,
        n_splits: int,
        random_state: Optional[int] = None,
        solver: str = "lbfgs",
        class_weight: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        """
        Build genuine out-of-fold meta features for the selection set.

        For each fold, the *selected* constituents are REFIT on the fold's
        training portion (of the selection set) and predict the held-out
        rows. Each row of the returned matrix therefore contains
        probabilities from models that never saw that row, so the
        meta-model is fit without target leakage. Merely predicting with
        already-fit models per fold would reproduce the full prediction
        matrix exactly and provide no protection.

        Returns None when the data is too small to split.
        """
        n = len(X)
        class_counts = np.bincount(np.asarray(y).astype(int))
        min_class = class_counts[class_counts > 0].min() if len(class_counts) else 0
        n_splits = int(min(n_splits, min_class, n))
        if n_splits < 2:
            return None

        X_meta = np.full((n, len(selected_indices)), np.nan)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, held_idx in splitter.split(X, y):
            for col, model_idx in enumerate(selected_indices):
                template = constituent_models[model_idx]
                fold_model = LogisticRegression(
                    solver=solver, max_iter=1000, class_weight=class_weight
                )
                fold_model.fit(X[train_idx][:, template.feature_indices], y[train_idx])
                proba = fold_model.predict_proba(X[held_idx][:, template.feature_indices])[:, 1]
                X_meta[held_idx, col] = proba
        return X_meta

    @staticmethod
    def select_and_combine(
        constituent_models: List[Any],  # List of ConstituentModel objects
        X_test,
        y_test,
        ids,
        constituent_results: List[Dict[str, Any]],
        baseline_results: Dict[str, Any],
        spread: int,
        random_state: Optional[int] = None,
        class_weight: Optional[Dict] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]], List[int]]:
        """
        Select top models and train an out-of-fold meta-model.

        Parameters
        ----------
        constituent_models : List[ConstituentModel]
            List of trained ConstituentModel objects.
        X_test : array-like of shape (n_samples, n_features)
            Selection-set feature matrix (used to fit and score the meta-model
            via out-of-fold predictions).
        y_test : array-like of shape (n_samples,)
            Selection-set target labels.
        ids : array-like of shape (n_samples,)
            ID values for each sample.
        constituent_results : List[Dict[str, Any]]
            Metric dicts with keys 'plr', 'fnr', 'drp', 'drs', 'dps'.
        baseline_results : Dict[str, Any]
            Baseline metric dict with the same keys.
        spread : int
            Number of top models to select.
        random_state : int, optional
            Random seed for fold shuffling and meta-model training.
        class_weight : dict, optional
            Passed through to the meta-model logistic regression.

        Returns
        -------
        Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]], List[int]]
            - DataFrame with columns 'id' and 'predicted' (probabilities),
              or None when no models pass the baseline gate
            - Meta-model equation dict (keys ``model_i_prob`` and
              ``constant``), or None when no models were selected
            - List of selected model indices (empty when none selected)
        """
        ranked = EnsembleMethod._rank_models(constituent_results, baseline_results)
        selected_indices = ranked[: min(spread, len(ranked))]

        # No eligible models: signal the caller (the classifier falls back
        # to the baseline model) rather than fabricating zero predictions.
        if not selected_indices:
            return None, None, []

        # Out-of-fold probabilities for honest meta-model fitting
        X_meta_oof = EnsembleMethod._out_of_fold_meta_features(
            constituent_models,
            X_test,
            y_test,
            selected_indices,
            n_splits=META_FOLDS,
            random_state=random_state,
            class_weight=class_weight,
        )
        if X_meta_oof is None:
            logger.warning(
                "Selection set too small for out-of-fold meta-model training; "
                "falling back to in-sample fitting."
            )
            X_meta_oof = EnsembleMethod._meta_feature_matrix(
                constituent_models, X_test, selected_indices
            )

        meta_model = LogisticRegression(
            solver="lbfgs", random_state=random_state, max_iter=1000, class_weight=class_weight
        )
        meta_model.fit(X_meta_oof, y_test)

        # Final predictions: meta-model applied to each selected model's
        # full-dataset probabilities. The selected models were trained on
        # the training split, so these are out-of-sample for them; only the
        # meta-model coefficients were fit (out-of-fold) on this data.
        X_meta_full = EnsembleMethod._meta_feature_matrix(
            constituent_models, X_test, selected_indices
        )
        final_predictions = meta_model.predict_proba(X_meta_full)[:, 1]

        result_df = pd.DataFrame({"id": ids, "predicted": final_predictions})

        # Extract meta-model equation dictionary
        meta_equation = {
            f"model_{i}_prob": float(coef) for i, coef in enumerate(meta_model.coef_[0])
        }
        meta_equation["constant"] = float(meta_model.intercept_[0])

        return result_df, meta_equation, selected_indices
