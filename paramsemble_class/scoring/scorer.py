"""Model scoring module for ELR."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, cast

from ..utils.model_io import ModelIO
from ..metrics.performance import top_d_row_indices


class ModelScorer:
    """
    Scores new datasets using previously trained ELR models.

    The ModelScorer loads model configurations from JSON files (modeljson format)
    and applies the saved equations to score new data. It supports all three
    ensemble methods (intersect, venn, ensemble) and produces output in the
    same format as :meth:`~paramsemble_class.ELRClassifier.predict_detailed`,
    using the identical decile rule, so Python scoring and SQL scoring of the
    same data agree.

    For intersect/venn methods:
    - Applies each constituent model equation
    - Ranks predictions and keeps the top d deciles (``ceil(d * n / 10)``
      rows, ties broken by row order — the same rule as training)
    - Compiles results across models
    - Returns DataFrame with IDs and occurrence counts

    For ensemble method:
    - Applies constituent model equations to generate probabilities
    - Applies meta-model equation to constituent probabilities
    - Returns DataFrame with IDs and predicted probabilities

    Parameters
    ----------
    modeljson_path : str
        Path to the model configuration JSON file.

    Attributes
    ----------
    config : Dict[str, Any]
        Loaded model configuration containing method, d, models, and meta_model.
    method : str
        Ensemble method: "intersect", "venn", or "ensemble".
    d : int
        Number of top deciles to consider (for intersect/venn methods).
    models : List[Dict[str, float]]
        List of equation dictionaries for constituent models.
    meta_model : Dict[str, float], optional
        Meta-model equation dictionary (only for ensemble method).
    feature_names : List[str] or None
        Ordered training feature names, when present in the JSON. Required
        to score plain numpy arrays; DataFrames are matched by column name.

    Examples
    --------
    >>> scorer = ModelScorer('model_config.json')
    >>> predictions = scorer.score(X_new, ids_new)
    >>> print(predictions.head())
    """

    def __init__(self, modeljson_path: str):
        """
        Initialize ModelScorer by loading model configuration.

        Parameters
        ----------
        modeljson_path : str
            Path to the model configuration JSON file.

        Raises
        ------
        IOError
            If file cannot be read.
        ValueError
            If JSON structure is invalid.
        """
        # Load model configuration
        self.config = ModelIO.load_model(modeljson_path)

        # Extract configuration components
        self.method = self.config["method"]
        self.d = self.config["d"]
        self.models = self.config["models"]
        self.meta_model = self.config.get("meta_model", None)
        self.feature_names: Optional[List[str]] = self.config.get("feature_names", None)
        self.weights: Optional[List[float]] = self.config.get("weights", None)

    def score(self, X, ids) -> pd.DataFrame:
        """
        Score dataset using loaded model configuration.

        The scoring process depends on the ensemble method:

        - For intersect/venn: Applies each constituent model equation,
          ranks predictions, filters to top d deciles, and compiles results
          with ID occurrence counts.

        - For ensemble: Applies constituent model equations to generate
          probabilities, then applies meta-model equation to produce
          final predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pd.DataFrame
            Feature matrix to score. DataFrames are matched to equation
            feature names by column. Numpy arrays require the exported
            JSON to contain ``feature_names`` (any ELRClassifier export
            does) with columns in the same order as training.
        ids : array-like of shape (n_samples,)
            ID values for each sample.

        Returns
        -------
        pd.DataFrame
            For intersect/venn methods:
                - 'id': Deduplicated IDs from top d deciles
                - 'sets': Count of how many model DRS contain each ID

            For ensemble method:
                - 'id': All input IDs
                - 'predicted': Predicted probabilities from meta-model

        Raises
        ------
        ValueError
            If X is missing required features or has invalid shape.
        """
        X = self._to_dataframe(X)
        ids = np.asarray(ids)

        # Validate input
        if len(X) != len(ids):
            raise ValueError(f"X and ids must have same length. Got X: {len(X)}, ids: {len(ids)}")

        # Route to appropriate scoring method
        if self.method in ["intersect", "venn"]:
            return self._score_intersect_venn(X, ids)
        elif self.method == "ensemble":
            return self._score_ensemble(X, ids)
        else:
            raise ValueError(
                f"Unknown method '{self.method}'. " "Expected 'intersect', 'venn', or 'ensemble'."
            )

    def _to_dataframe(self, X) -> pd.DataFrame:
        """Normalize X to a DataFrame with usable column names.

        Numpy arrays are only usable when the export recorded the training
        feature order; otherwise there is no way to map positions to the
        equation feature names.
        """
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}.")
        if self.feature_names is None:
            raise ValueError(
                "Cannot score a numpy array: the model JSON does not contain "
                "'feature_names'. Re-export with ELRClassifier (which records "
                "them) or pass a pandas DataFrame with named columns."
            )
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but the model was trained on "
                f"{len(self.feature_names)} features."
            )
        return pd.DataFrame(X, columns=self.feature_names)

    def _apply_logistic_regression(self, X: pd.DataFrame, equation_dict: dict) -> np.ndarray:
        """
        Apply logistic regression equation to features.

        Formula: 1 / (1 + exp(-(constant + sum(feature_i * coef_i))))

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with column names matching equation dict keys.
        equation_dict : dict
            Dictionary mapping feature names to coefficients,
            with 'constant' key for intercept.

        Returns
        -------
        np.ndarray
            Predicted probabilities for positive class.

        Raises
        ------
        ValueError
            If required features are missing from X.
        """
        # Extract constant (intercept)
        constant = equation_dict.get("constant", 0.0)

        # Calculate linear combination
        linear_combination = np.full(len(X), constant, dtype=float)

        if "bins" in equation_dict:
            # Binned (WoE) equation: contribution = coef * woe[bin(x)]
            for feature_name, spec in equation_dict["bins"].items():
                if feature_name not in X.columns:
                    raise ValueError(
                        f"Required feature '{feature_name}' not found in input data. "
                        f"Available features: {list(X.columns)}"
                    )
                splits = np.asarray(spec["splits"], dtype=float)
                woe = np.asarray(spec["woe"], dtype=float)
                idx = np.clip(
                    np.searchsorted(splits, X[feature_name].values, side="right"),
                    0,
                    len(woe) - 1,
                )
                linear_combination += woe[idx] * spec["coef"]
        else:
            # Plain linear equation
            for feature_name, coef in equation_dict.items():
                if feature_name == "constant":
                    continue

                # Check if feature exists in X
                if feature_name not in X.columns:
                    raise ValueError(
                        f"Required feature '{feature_name}' not found in input data. "
                        f"Available features: {list(X.columns)}"
                    )

                # Add feature contribution
                linear_combination += X[feature_name].values * coef

        # Apply logistic function (clipped to avoid overflow)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(linear_combination, -500.0, 500.0)))

        return cast(np.ndarray, probabilities)

    def _score_intersect_venn(self, X: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
        """
        Score dataset using intersect or venn method.

        Steps:
        1. Apply each constituent model equation
        2. Rank predictions and identify top d deciles for each model
           (identical rule to training: ceil(d * n / 10) rows by score
           descending, ties by row order)
        3. Compile IDs from top d deciles across all models
        4. Count occurrences of each ID

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to score.
        ids : np.ndarray
            ID values for each sample.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['id', 'sets'].
        """
        # Track IDs, occurrence counts and weighted scores
        weights = self.weights if self.weights is not None else [1.0] * len(self.models)
        id_counts: Dict[Any, int] = {}
        id_scores: Dict[Any, float] = {}

        # Process each constituent model (one weighted vote per containing
        # model, using the exported Borda weights; absent -> weight 1)
        for model_equation, weight in zip(self.models, weights):
            # Apply equation to get predicted probabilities
            probabilities = self._apply_logistic_regression(X, model_equation)

            # Compile IDs from the top d deciles (shared decile rule)
            for id_value in ids[top_d_row_indices(probabilities, self.d)]:
                id_counts[id_value] = id_counts.get(id_value, 0) + 1
                id_scores[id_value] = id_scores.get(id_value, 0.0) + weight

        if not id_counts:
            return pd.DataFrame(columns=["id", "sets", "score"])

        result_df = pd.DataFrame(
            [
                {"id": id_value, "sets": count, "score": id_scores[id_value]}
                for id_value, count in id_counts.items()
            ]
        )

        # Sort by sets desc, score desc, then id for consistency
        result_df = result_df.sort_values(
            ["sets", "score", "id"], ascending=[False, False, True]
        ).reset_index(drop=True)

        return result_df

    def _score_ensemble(self, X: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
        """
        Score dataset using ensemble method.

        Steps:
        1. Apply each constituent model equation to generate probabilities
        2. Apply meta-model equation to constituent probabilities
        3. Return final predicted probabilities

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to score.
        ids : np.ndarray
            ID values for each sample.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['id', 'predicted'].
        """
        # Generate probabilities from each constituent model
        constituent_probabilities = []

        for model_equation in self.models:
            # Apply equation to get predicted probabilities
            probabilities = self._apply_logistic_regression(X, model_equation)
            constituent_probabilities.append(probabilities)

        # Stack into matrix: rows are samples, columns are model probabilities
        X_meta = np.column_stack(constituent_probabilities)

        # Create DataFrame with proper column names for meta-model
        meta_df = pd.DataFrame(X_meta, columns=[f"model_{i}_prob" for i in range(len(self.models))])

        # Apply meta-model equation (validated non-None by ModelIO.load_model)
        assert self.meta_model is not None
        final_probabilities = self._apply_logistic_regression(meta_df, self.meta_model)

        # Create result DataFrame
        result_df = pd.DataFrame({"id": ids, "predicted": final_probabilities})

        return result_df
