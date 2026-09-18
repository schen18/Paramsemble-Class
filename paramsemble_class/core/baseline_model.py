"""Baseline models for ELR comparison: Random Forest or linear (WoE) LR.

The baseline serves two roles: the gate reference (constituents must beat
it on at least one metric to be selected) and the fallback model (used for
all outputs when nothing passes the gate).

``baseline="woe_logreg"`` (the ELR default) is a full-feature logistic
regression on WoE-binned features — the correct comparison class for
linear constituents ("does the sparse ensemble add anything over one full
linear model?") and itself a linear equation, so the fallback is
SQL-deployable. It requires ``optbinning``; when that library is not
installed the baseline degrades to an unbinned full-feature logistic
regression (logged, and reflected in ``actual_kind_``).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from paramsemble_class.metrics.performance import PerformanceMetrics

logger = logging.getLogger(__name__)

VALID_BASELINES = ["woe_logreg", "logreg", "random_forest"]

try:  # pragma: no cover - import presence depends on environment
    from optbinning import OptimalBinning

    _HAS_OPTBINNING = True
except ImportError:  # pragma: no cover
    OptimalBinning = None  # type: ignore[assignment, misc]
    _HAS_OPTBINNING = False


class BaselineModel:
    """
    Baseline Random Forest model for establishing performance benchmarks.

    The baseline model trains a Random Forest classifier on all features
    and calculates specialized metrics (PLR, FNR, DRP, DRS, DPS) for
    comparison with constituent logistic regression models.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility.
    n_jobs : int, optional
        Number of parallel jobs for fitting/predicting (default 1).
        Use -1 for all cores.
    class_weight : dict or "balanced", optional
        Weights associated with classes, passed to RandomForestClassifier.

    Attributes
    ----------
    model_ : RandomForestClassifier
        The trained Random Forest classifier.
    metrics_ : Dict
        Dictionary containing baseline metrics after evaluation.
    """

    kind = "random_forest"

    def __init__(
        self,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = 1,
        class_weight: Optional[Any] = None,
    ):
        """
        Initialize baseline Random Forest model.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility.
        n_jobs : int, optional
            Number of parallel jobs for the forest (default 1, -1 for all cores).
        class_weight : dict or "balanced", optional
            Class weights passed to RandomForestClassifier.
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.model_ = None
        self.metrics_ = None

    def fit(self, X_train, y_train):
        """
        Train Random Forest classifier on all features.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training target labels.

        Returns
        -------
        self : BaselineModel
            Fitted baseline model.
        """
        self.model_ = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
        )
        self.model_.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test, ids, d: int) -> Dict[str, Any]:
        """
        Evaluate baseline model on selection data and calculate all metrics.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Feature matrix.
        y_test : array-like of shape (n_samples,)
            Target labels.
        ids : array-like of shape (n_samples,)
            ID values for each sample.
        d : int
            Number of top deciles to consider (1-10).

        Returns
        -------
        Dict[str, Any]
            Metrics 'plr', 'fnr', 'drp', 'drs', 'dps'.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before evaluation. Call fit() first.")

        y_pred = self.model_.predict(X_test)
        y_score = self.model_.predict_proba(X_test)[:, 1]

        self.metrics_ = {
            "plr": PerformanceMetrics.positive_likelihood_ratio(y_test, y_pred),
            "fnr": PerformanceMetrics.false_negative_rate(y_test, y_pred),
            "drp": PerformanceMetrics.decile_ranked_performance(y_test, y_score, d),
            "drs": PerformanceMetrics.extract_decile_ranked_set(ids, y_score, d),
            "dps": PerformanceMetrics.extract_decile_positive_set(ids, y_test, y_score, d),
        }
        return self.metrics_

    def predict(self, X):
        """Predict class labels (requires fit first)."""
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        return self.model_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities (requires fit first)."""
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        return self.model_.predict_proba(X)

    def get_equation_dict(self, feature_names: List[str]) -> Dict[str, Any]:
        """Linear equations cannot be extracted from a Random Forest."""
        raise ValueError(
            "The Random Forest baseline has no exportable linear equation; "
            "use baseline='woe_logreg' or 'logreg' for a SQL-deployable fallback."
        )


class LinearBaselineModel:
    """
    Full-feature logistic regression baseline, optionally on WoE-binned features.

    ``kind="woe_logreg"`` fits one ``optbinning.OptimalBinning`` per feature
    on the training split, transforms features to Weight-of-Evidence values,
    and fits a logistic regression on the transformed matrix. Bins are fit
    on the training regime only. When optbinning is not installed the model
    degrades to an unbinned logistic regression on the raw features
    (``actual_kind_="logreg"``), keeping the SQL-deployable linear fallback
    available on every environment.

    Parameters
    ----------
    kind : str
        "woe_logreg" (bin if optbinning is available) or "logreg" (never bin).
    random_state : int, optional
        Random seed (kept for interface parity; lbfgs is deterministic).
    class_weight : dict or "balanced", optional
        Passed to LogisticRegression.

    Attributes
    ----------
    model_ : LogisticRegression
        The trained logistic regression (on WoE features when binned).
    binning_ : list of (splits, woe) or None
        Per-feature bin definitions when binning was applied.
    actual_kind_ : str
        The kind actually used ("woe_logreg" or "logreg").
    """

    def __init__(
        self,
        kind: str = "woe_logreg",
        random_state: Optional[int] = None,
        class_weight: Optional[Any] = None,
    ):
        self.kind = kind
        self.random_state = random_state
        self.class_weight = class_weight
        self.model_ = None
        self.binning_: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.metrics_: Optional[Dict[str, Any]] = None
        self.actual_kind_: Optional[str] = None

    # -- binning ---------------------------------------------------------

    def _fit_bins(self, X, y) -> List[Tuple[np.ndarray, np.ndarray]]:
        bins = []
        for j in range(X.shape[1]):
            ob = OptimalBinning(dtype="numerical", max_n_bins=10, min_bin_size=0.05).fit(X[:, j], y)
            splits = np.asarray(ob.splits, dtype=float)
            table = ob.binning_table.build()
            woe = table["WoE"].iloc[: len(splits) + 1].astype(float).values
            bins.append((splits, woe))
        return bins

    @staticmethod
    def _apply_bins(X, bins) -> np.ndarray:
        """Map features to WoE via the bin lookup (right-closed on splits)."""
        Z = np.empty(X.shape, dtype=float)
        for j, (splits, woe) in enumerate(bins):
            idx = np.clip(np.searchsorted(splits, X[:, j], side="right"), 0, len(woe) - 1)
            Z[:, j] = woe[idx]
        return Z

    # -- scikit-learn-style interface ------------------------------------

    def fit(self, X_train, y_train):
        X = np.asarray(X_train, dtype=float)
        self.actual_kind_ = self.kind
        if self.kind == "woe_logreg" and _HAS_OPTBINNING:
            self.binning_ = self._fit_bins(X, y_train)
            X_fit = self._apply_bins(X, self.binning_)
        else:
            if self.kind == "woe_logreg":
                logger.warning(
                    "optbinning is not installed; the 'woe_logreg' baseline "
                    "degrades to an unbinned full-feature logistic regression "
                    "(actual_kind_='logreg'). Install with: "
                    "pip install paramsemble-class[binning]"
                )
                self.actual_kind_ = "logreg"
            self.binning_ = None
            X_fit = X

        self.model_ = LogisticRegression(
            max_iter=2000, random_state=self.random_state, class_weight=self.class_weight
        ).fit(X_fit, y_train)
        return self

    def _transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.binning_ is not None:
            return self._apply_bins(X, self.binning_)
        return cast(np.ndarray, X)

    def evaluate(self, X_test, y_test, ids, d: int) -> Dict[str, Any]:
        if self.model_ is None:
            raise ValueError("Model must be fitted before evaluation. Call fit() first.")
        X = self._transform(X_test)
        y_pred = self.model_.predict(X)
        y_score = self.model_.predict_proba(X)[:, 1]
        self.metrics_ = {
            "plr": PerformanceMetrics.positive_likelihood_ratio(y_test, y_pred),
            "fnr": PerformanceMetrics.false_negative_rate(y_test, y_pred),
            "drp": PerformanceMetrics.decile_ranked_performance(y_test, y_score, d),
            "drs": PerformanceMetrics.extract_decile_ranked_set(ids, y_score, d),
            "dps": PerformanceMetrics.extract_decile_positive_set(ids, y_test, y_score, d),
        }
        return self.metrics_

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        return self.model_.predict(self._transform(X))

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        return self.model_.predict_proba(self._transform(X))

    # -- export ----------------------------------------------------------

    def get_equation_dict(self, feature_names: List[str]) -> Dict[str, Any]:
        """Export the baseline as a scoring equation dictionary.

        Binned model: ``{"bins": {feat: {"splits": [...], "woe": [...],
        "coef": c}}, "constant": b}`` — score = constant + sum over features
        of coef * woe[bin(x)]. Unbinned: the plain ``{feat: coef, ...,
        "constant": b}`` format used by raw constituent equations.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before export. Call fit() first.")
        coefs = self.model_.coef_[0]
        if self.binning_ is not None:
            bins_out = {}
            for j, name in enumerate(feature_names):
                splits, woe = self.binning_[j]
                bins_out[name] = {
                    "splits": [float(s) for s in splits],
                    "woe": [float(w) for w in woe],
                    "coef": float(coefs[j]),
                }
            return {"bins": bins_out, "constant": float(self.model_.intercept_[0])}
        equation = {name: float(coefs[j]) for j, name in enumerate(feature_names)}
        equation["constant"] = float(self.model_.intercept_[0])
        return equation


def build_baseline(
    kind: str,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = 1,
    class_weight: Optional[Any] = None,
):
    """Construct the baseline model for ``kind`` in {"woe_logreg", "logreg",
    "random_forest"}."""
    if kind == "random_forest":
        return BaselineModel(random_state=random_state, n_jobs=n_jobs, class_weight=class_weight)
    if kind in ("woe_logreg", "logreg"):
        return LinearBaselineModel(kind=kind, random_state=random_state, class_weight=class_weight)
    raise ValueError(f"Parameter 'baseline' must be one of {VALID_BASELINES}, got {kind!r}.")
