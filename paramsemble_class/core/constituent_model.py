"""Constituent logistic regression model for ELR ensemble."""

from typing import Dict, List, Optional, Any
from sklearn.linear_model import LogisticRegression
from paramsemble_class.metrics.performance import PerformanceMetrics


class ConstituentModel:
    """
    Individual logistic regression model trained on a feature subset.

    Each constituent model is trained on a specific subset of features
    and evaluated using specialized metrics (PLR, FNR, DRP, DRS, DPS).
    The model's coefficients can be extracted as an equation dictionary
    for JSON export and SQL generation.

    Parameters
    ----------
    feature_indices : List[int]
        Indices of features to use for this model.
    solver : str
        Scikit-learn solver name or "auto" for automatic selection.
    random_state : int, optional
        Random seed for reproducibility.
    class_weight : dict or "balanced", optional
        Weights associated with classes, passed to LogisticRegression.
        Use "balanced" for imbalanced targets.

    Attributes
    ----------
    model_ : LogisticRegression
        The trained logistic regression classifier.
    metrics_ : Dict
        Dictionary containing model metrics after evaluation.
    feature_indices : List[int]
        Indices of features used by this model.
    """

    def __init__(
        self,
        feature_indices: List[int],
        solver: str = "lbfgs",
        random_state: Optional[int] = None,
        class_weight: Optional[Any] = None,
    ):
        """
        Initialize constituent logistic regression model.

        Parameters
        ----------
        feature_indices : List[int]
            Indices of features to use for this model.
        solver : str, default="lbfgs"
            Scikit-learn solver name or "auto" for automatic selection.
        random_state : int, optional
            Random seed for reproducibility.
        class_weight : dict or "balanced", optional
            Class weights passed to LogisticRegression.
        """
        self.feature_indices = feature_indices
        self.solver = solver
        self.random_state = random_state
        self.class_weight = class_weight
        self.model_ = None
        self.metrics_ = None
        self._actual_solver = None

    def _select_solver(self, n_samples: int, n_features: int) -> str:
        """
        Select the solver to use.

        ``solver="auto"`` resolves to ``"lbfgs"``: constituents are small
        L2-penalized logistic regressions, which lbfgs handles robustly
        and deterministically without the feature scaling that
        gradient-based solvers like saga require. Pass an explicit solver
        name to override.

        Parameters
        ----------
        n_samples : int
            Number of training samples (kept for API compatibility).
        n_features : int
            Number of features in the subset (kept for API compatibility).

        Returns
        -------
        str
            Selected solver name.
        """
        if self.solver != "auto":
            return self.solver
        return "lbfgs"

    def fit(self, X_train, y_train):
        """
        Train logistic regression on feature subset.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix (all features).
        y_train : array-like of shape (n_samples,)
            Training target labels.

        Returns
        -------
        self : ConstituentModel
            Fitted constituent model.

        Raises
        ------
        ValueError
            If training fails due to data issues.
        """
        try:
            # Extract feature subset
            X_subset = X_train[:, self.feature_indices]

            # Select solver
            n_samples, n_features = X_subset.shape
            self._actual_solver = self._select_solver(n_samples, n_features)

            # Train logistic regression
            self.model_ = LogisticRegression(
                solver=self._actual_solver,
                random_state=self.random_state,
                max_iter=1000,
                class_weight=self.class_weight,
            )
            self.model_.fit(X_subset, y_train)

            return self

        except Exception as e:
            raise ValueError(
                f"Failed to train constituent model with features {self.feature_indices}: {str(e)}"
            )

    def evaluate(self, X_test, y_test, ids, d: int) -> Dict[str, Any]:
        """
        Evaluate model on selection data and calculate all metrics.

        Calculates specialized metrics:
        - PLR: Positive Likelihood Ratio
        - FNR: False Negative Rate
        - DRP: Decile Ranked Performance (lift at the top d deciles)
        - DRS: Decile Ranked Set (IDs in top d deciles)
        - DPS: Decile Positive Set (true positive IDs in top d deciles)

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Feature matrix (all features).
        y_test : array-like of shape (n_samples,)
            Target labels.
        ids : array-like of shape (n_samples,)
            ID values for each sample.
        d : int
            Number of top deciles to consider (1-10).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'plr', 'fnr', 'drp', 'drs', 'dps'.

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before evaluation. Call fit() first.")

        # Extract feature subset
        X_subset = X_test[:, self.feature_indices]

        # Generate predictions
        y_pred = self.model_.predict(X_subset)

        # Generate probability scores for ranking
        y_score = self.model_.predict_proba(X_subset)[:, 1]  # Probability of positive class

        # Calculate metrics
        plr = PerformanceMetrics.positive_likelihood_ratio(y_test, y_pred)
        fnr = PerformanceMetrics.false_negative_rate(y_test, y_pred)
        drp = PerformanceMetrics.decile_ranked_performance(y_test, y_score, d)
        drs = PerformanceMetrics.extract_decile_ranked_set(ids, y_score, d)
        dps = PerformanceMetrics.extract_decile_positive_set(ids, y_test, y_score, d)

        # Store metrics
        self.metrics_ = {"plr": plr, "fnr": fnr, "drp": drp, "drs": drs, "dps": dps}

        return self.metrics_

    def get_equation_dict(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract coefficients as dictionary for JSON export.

        Creates a dictionary mapping feature names to their coefficients,
        with the intercept stored under the "constant" key.

        Parameters
        ----------
        feature_names : List[str]
            Names of all features in the dataset.

        Returns
        -------
        Dict[str, float]
            Dictionary with structure:
            {
                "feature_name_1": coefficient_1,
                "feature_name_2": coefficient_2,
                ...
                "constant": intercept
            }

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before extracting equation. Call fit() first.")

        equation_dict = {}

        # Extract coefficients for selected features
        coefficients = self.model_.coef_[0]  # Shape: (n_features_subset,)

        for i, feature_idx in enumerate(self.feature_indices):
            feature_name = feature_names[feature_idx]
            equation_dict[feature_name] = float(coefficients[i])

        # Add intercept as "constant"
        equation_dict["constant"] = float(self.model_.intercept_[0])

        return equation_dict

    def predict(self, X):
        """
        Generate predictions for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (all features).

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted labels.

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        X_subset = X[:, self.feature_indices]
        return self.model_.predict(X_subset)

    def predict_proba(self, X):
        """
        Generate probability predictions for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (all features).

        Returns
        -------
        array-like of shape (n_samples, 2)
            Predicted probabilities for each class.

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        X_subset = X[:, self.feature_indices]
        return self.model_.predict_proba(X_subset)
