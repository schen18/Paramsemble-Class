"""Parameter and data validation for ELR package."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union


class ParameterValidator:
    """Validates parameters and data for ELR classifier."""

    VALID_SAMPLE_METHODS = ["unique", "replace"]
    VALID_ENSEMBLE_METHODS = ["intersect", "venn", "ensemble"]
    VALID_SOLVERS = ["auto", "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]

    @staticmethod
    def validate_parameters(params: Dict[str, Any]) -> None:
        """
        Validate all ELR parameters.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters to validate

        Raises
        ------
        ValueError
            If any parameter is invalid with descriptive message
        """
        # Validate m parameter
        m = params.get("m")
        if m is not None and m < 1:
            raise ValueError(
                f"Parameter 'm' must be >= 1, got {m}. "
                "The 'm' parameter specifies the number of feature combinations to generate."
            )

        # Validate f parameter
        f = params.get("f")
        if f is not None and f < 1:
            raise ValueError(
                f"Parameter 'f' must be >= 1, got {f}. "
                "The 'f' parameter specifies the number of features per combination."
            )

        # Validate sample parameter
        sample = params.get("sample")
        if sample is not None and sample not in ParameterValidator.VALID_SAMPLE_METHODS:
            raise ValueError(
                f"Parameter 'sample' must be one of "
                f"{ParameterValidator.VALID_SAMPLE_METHODS}, got '{sample}'. "
                "Use 'unique' for sampling without replacement or "
                "'replace' for sampling with replacement."
            )

        # Validate d parameter
        d = params.get("d")
        if d is not None and (d < 1 or d > 10):
            raise ValueError(
                f"Parameter 'd' must be between 1 and 10 inclusive, got {d}. "
                "The 'd' parameter specifies the number of top deciles to consider."
            )

        # Validate method parameter
        method = params.get("method")
        if method is not None and method not in ParameterValidator.VALID_ENSEMBLE_METHODS:
            raise ValueError(
                f"Parameter 'method' must be one of "
                f"{ParameterValidator.VALID_ENSEMBLE_METHODS}, got '{method}'. "
                "Choose 'intersect' for ID intersection, 'venn' for "
                "unique ID discovery, or 'ensemble' for the meta-model."
            )

        # Validate spread parameter
        spread = params.get("spread")
        if spread is not None and spread < 1:
            raise ValueError(
                f"Parameter 'spread' must be >= 1, got {spread}. "
                "The 'spread' parameter specifies the number of top models to select."
            )

        # Validate solver parameter
        solver = params.get("solver")
        if solver is not None and solver not in ParameterValidator.VALID_SOLVERS:
            raise ValueError(
                f"Parameter 'solver' must be one of {ParameterValidator.VALID_SOLVERS}, "
                f"got '{solver}'. Use 'auto' for automatic solver selection or specify "
                "a scikit-learn logistic regression solver."
            )

    @staticmethod
    def validate_data(
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        id_column: Optional[str] = None,
        ids: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """
        Validate training or test data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
        y : np.ndarray or pd.Series
            Target labels
        id_column : str, optional
            Name of ID column if X is a DataFrame
        ids : np.ndarray or pd.Series, optional
            ID array if provided separately

        Raises
        ------
        ValueError
            If data is invalid with descriptive message
        """
        # Check for empty datasets
        if len(X) == 0:
            raise ValueError("Dataset is empty. Please provide non-empty training/test data.")

        # Check shape compatibility
        if len(X) != len(y):
            raise ValueError(
                f"Shape mismatch: X has {len(X)} samples but y has {len(y)} samples. "
                "The number of samples in X and y must match."
            )

        # Convert to appropriate format for validation
        if isinstance(X, pd.DataFrame):
            X_array = X.select_dtypes(include=[np.number]).values
            feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            X_array = X
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Check for missing values in features
        if np.any(np.isnan(X_array)):
            # Identify columns with missing values
            if isinstance(X, pd.DataFrame):
                missing_cols = X.columns[X.isna().any()].tolist()
            else:
                missing_mask = np.isnan(X_array).any(axis=0)
                missing_cols = [
                    feature_names[i] for i, has_missing in enumerate(missing_mask) if has_missing
                ]

            raise ValueError(
                f"Training/test data contains missing values in columns: "
                f"{missing_cols}. Please handle missing values before fitting "
                "the model (e.g., imputation or removal)."
            )

        # Check for missing values in target
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        if np.any(np.isnan(y_array)):
            raise ValueError(
                "Target variable contains missing values. "
                "Please handle missing values in the target before fitting the model."
            )

        # Validate ID column if specified
        if id_column is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    f"ID column '{id_column}' specified but X is not a DataFrame. "
                    "Please provide X as a pandas DataFrame when using id_column parameter."
                )

            if id_column not in X.columns:
                raise ValueError(
                    f"ID column '{id_column}' does not exist in the dataset. "
                    f"Available columns: {X.columns.tolist()}"
                )

        # Validate IDs if provided separately
        if ids is not None:
            if len(ids) != len(X):
                raise ValueError(
                    f"Shape mismatch: X has {len(X)} samples but ids has {len(ids)} values. "
                    "The number of IDs must match the number of samples."
                )

    @staticmethod
    def validate_f_against_features(f: int, n_features: int) -> None:
        """
        Validate that f does not exceed the number of available features.

        Parameters
        ----------
        f : int
            Number of features per combination
        n_features : int
            Total number of features available

        Raises
        ------
        ValueError
            If f exceeds n_features
        """
        if f > n_features:
            raise ValueError(
                f"Parameter 'f' ({f}) exceeds the number of available features ({n_features}). "
                "The 'f' parameter must be less than or equal to the total number of features."
            )
