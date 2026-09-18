"""Main ELRClassifier for ensemble logistic regression."""

import logging
import warnings

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List, Tuple, cast
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

from paramsemble_class.core.feature_sampler import FeatureSampler
from paramsemble_class.core.baseline_model import build_baseline, VALID_BASELINES
from paramsemble_class.core.constituent_model import ConstituentModel
from paramsemble_class.ensemble.intersect import IntersectMethod
from paramsemble_class.ensemble.venn import VennMethod
from paramsemble_class.ensemble.ensemble import EnsembleMethod
from paramsemble_class.ensemble._common import (
    build_sets_dataframe,
    weights_for_selection,
    VALID_SELECTION_STRATEGIES,
)
from paramsemble_class.metrics.performance import top_d_row_indices
from paramsemble_class.utils.validation import ParameterValidator
from paramsemble_class.utils.model_io import ModelIO

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Fraction of the input held back as the selection set when the
# classifier is fitted in two-argument (scikit-learn) mode.
SKLEARN_MODE_SELECTION_FRACTION = 0.25


def _sigmoid(linear: np.ndarray) -> np.ndarray:
    """Numerically safe logistic function."""
    return cast(np.ndarray, 1.0 / (1.0 + np.exp(-np.clip(linear, -500.0, 500.0))))


def _train_constituent(
    feature_indices: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_sel: np.ndarray,
    y_sel: np.ndarray,
    ids_sel: np.ndarray,
    d: int,
    solver: str,
    random_state: Optional[int],
    class_weight: Optional[Any],
    feature_names: List[str],
) -> Optional[Tuple[ConstituentModel, Dict[str, Any]]]:
    """Fit and evaluate one constituent model (worker for parallel training).

    Module-level so joblib can pickle it on Windows (spawn-based backends).
    Returns None (with a logged warning) when the featureset fails to train,
    e.g. due to multicollinearity, so one bad combination cannot abort the
    whole fit.
    """
    try:
        model = ConstituentModel(
            feature_indices=feature_indices,
            solver=solver,
            random_state=random_state,
            class_weight=class_weight,
        )
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_sel, y_sel, ids_sel, d)
        equation_dict = model.get_equation_dict(feature_names)
    except Exception as e:
        logger.warning("Failed to train constituent model with features %s: %s", feature_indices, e)
        return None

    result = {
        "plr": metrics["plr"],
        "fnr": metrics["fnr"],
        "drp": metrics["drp"],
        "drs": metrics["drs"],
        "dps": metrics["dps"],
        "equation_dict": equation_dict,
        "feature_indices": list(feature_indices),
    }
    return model, result


class ELRClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble Logistic Regression Classifier.

    ELR trains multiple logistic regression models on diverse feature subsets,
    establishes baseline performance using Random Forest, and provides three
    distinct ensemble strategies (intersect, venn, ensemble) for model selection
    and combination.

    **Data-flow semantics.** ELR is a *selection* method: the second dataset
    passed to :meth:`fit` is used to evaluate and select constituent models,
    so metrics measured on it are optimistically biased by the selection
    itself. For honest performance reporting, evaluate the fitted model on a
    third, untouched holdout (``predict_proba`` supports this for every
    method), or rely on the built-in out-of-fold training of the ensemble
    method's meta-model.

    **Two calling conventions.**

    - ELR-style (explicit selection set)::

        clf.fit(X_train, y_train, X_sel, y_sel, ids_sel)

    - scikit-learn style (selection set split off internally, stratified)::

        clf.fit(X, y)                       # works with cross_val_score, GridSearchCV
        clf.predict(X)                      # class labels
        clf.predict_proba(X)                # probabilities for every method

    Method-specific results (ID/sets tables, ID/probability tables) are
    available from :meth:`predict_detailed`.

    For intersect/venn, ``predict`` labels an instance positive when it falls
    in the top-d deciles of a strict majority of the selected models, and
    ``predict_proba`` returns the fraction of selected models that place it
    there. If no constituent passes the baseline gate, the classifier falls
    back to the baseline Random Forest and logs a warning.

    Parameters
    ----------
    m : int, default=100
        Number of feature combinations to generate.
    f : int, default=5
        Number of features per combination.
    sample : str, default="unique"
        Sampling method: "unique" (no replacement) or "replace" (with replacement).
    d : int, default=2
        Number of top deciles to consider (1-10).
    method : str, default="intersect"
        Ensemble method: "intersect", "venn", or "ensemble".
    spread : int, default=10
        Number of top models to select.
    solver : str, default="auto"
        Logistic regression solver or "auto" (resolves to "lbfgs").
    id_column : str, default="id"
        Name of the ID column in DataFrames. When present in an input
        DataFrame the column is excluded from features (and, if no explicit
        IDs are passed to ``fit``, used as the selection-set IDs).
    elr2json : str, optional
        Path to export all model metrics (JSON format).
    modeljson : str, optional
        Path to export selected model equations (JSON format).
    class_weight : dict or "balanced", optional
        Class weights passed to the logistic regressions, the meta-model
        and the baseline Random Forest. Use "balanced" for imbalanced targets.
    n_jobs : int, default=1
        Parallelism for training the m constituent models and the baseline
        forest. Use -1 for all cores.
    f_range : list of int, optional
        Train constituents at several feature widths (e.g. [4, 8, 12])
        instead of a single ``f``; the m combinations are split across the
        widths and the gate/ranking selects across all of them, letting the
        method choose the right constituent scale for the data.
    selection : str, default="rank"
        "rank": select the top ``spread`` models by Borda count (classic).
        "coverage": greedily select models maximizing marginal coverage of
        true-positive IDs on the selection set (diversity-aware — avoids
        spending slots on redundant, correlated models). Applies to the
        intersect/venn methods.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : np.ndarray
        Class labels seen during fit (scikit-learn convention).
    baseline_model_ : BaselineModel
        Trained baseline Random Forest model.
    constituent_models_ : List[ConstituentModel]
        List of trained constituent logistic regression models.
    constituent_results_ : List[Dict]
        Metrics for all constituent models.
    baseline_results_ : Dict
        Metrics for baseline model.
    selected_indices_ : List[int]
        Indices of selected constituent models (empty when the baseline
        fallback is active).
    fell_back_to_baseline_ : bool
        True when no constituent passed the baseline gate.
    result_df_ : pd.DataFrame
        Method-specific results for the selection set.
    meta_equation_ : Dict, optional
        Meta-model equation (only for ensemble method).
    feature_names_ : List[str]
        Names of features used in training.
    n_features_ : int
        Number of features in training data.

    Examples
    --------
    >>> from paramsemble_class import ELRClassifier
    >>> import numpy as np
    >>>
    >>> X_train = np.random.randn(100, 10)
    >>> y_train = np.random.randint(0, 2, 100)
    >>> X_sel = np.random.randn(50, 10)
    >>> y_sel = np.random.randint(0, 2, 50)
    >>> ids_sel = np.arange(50)
    >>>
    >>> clf = ELRClassifier(m=20, f=3, method='intersect', spread=5)
    >>> clf.fit(X_train, y_train, X_sel, y_sel, ids_sel)
    >>> targeting = clf.predict_detailed(X_sel, ids_sel)  # [id, sets] table
    """

    def __init__(
        self,
        m: int = 100,
        f: int = 5,
        sample: str = "unique",
        d: int = 2,
        method: str = "intersect",
        spread: int = 10,
        solver: str = "auto",
        id_column: str = "id",
        elr2json: Optional[str] = None,
        modeljson: Optional[str] = None,
        class_weight: Optional[Any] = None,
        n_jobs: int = 1,
        f_range: Optional[List[int]] = None,
        selection: str = "rank",
        baseline: str = "woe_logreg",
        random_state: Optional[int] = None,
    ):
        """Initialize ELR Classifier."""
        self.m = m
        self.f = f
        self.sample = sample
        self.d = d
        self.method = method
        self.spread = spread
        self.solver = solver
        self.id_column = id_column
        self.elr2json = elr2json
        self.modeljson = modeljson
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.f_range = f_range
        self.selection = selection
        self.baseline = baseline
        self.random_state = random_state

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None,
        ids_test: Optional[Union[np.ndarray, pd.Series, List]] = None,
    ):
        """
        Train ELR classifier.

        This method orchestrates the entire training workflow:
        1. Validates inputs
        2. Generates feature combinations
        3. Trains baseline Random Forest
        4. Trains m logistic regression models (optionally in parallel)
        5. Applies the ensemble method
        6. Exports JSON if specified

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training target labels.
        X_test : array-like of shape (n_samples, n_features), optional
            Selection-set feature matrix. Constituent models are evaluated
            and selected on this data. When omitted (scikit-learn mode),
            a stratified 25% split of the training data is used.
        y_test : array-like of shape (n_samples,), optional
            Selection-set target labels. Required when ``X_test`` is given.
        ids_test : array-like of shape (n_samples,), optional
            ID values for the selection samples. Defaults to the
            ``id_column`` of ``X_test`` when it is a DataFrame containing
            that column, otherwise row positions.

        Returns
        -------
        self : ELRClassifier
            Fitted classifier.

        Raises
        ------
        ValueError
            If parameters or data are invalid.
        """
        logger.info("Starting ELR classifier training...")

        # Step 1: Validate parameters
        params = {
            "m": self.m,
            "f": self.f,
            "sample": self.sample,
            "d": self.d,
            "method": self.method,
            "spread": self.spread,
            "solver": self.solver,
        }
        ParameterValidator.validate_parameters(params)

        if self.selection not in VALID_SELECTION_STRATEGIES:
            raise ValueError(
                f"Parameter 'selection' must be one of "
                f"{VALID_SELECTION_STRATEGIES}, got {self.selection!r}."
            )
        if self.f_range is not None:
            if not isinstance(self.f_range, (list, tuple)) or not self.f_range:
                raise ValueError(
                    "Parameter 'f_range' must be a non-empty list of feature "
                    "counts per constituent (e.g. [4, 8, 12])."
                )
            bad = [w for w in self.f_range if not isinstance(w, (int, np.integer)) or w < 1]
            if bad:
                raise ValueError(f"Parameter 'f_range' entries must be integers >= 1, got {bad}.")
        if self.baseline not in VALID_BASELINES:
            raise ValueError(
                f"Parameter 'baseline' must be one of {VALID_BASELINES}, " f"got {self.baseline!r}."
            )

        # scikit-learn mode: split off a stratified selection set
        if X_test is None:
            X_arr, y_arr = self._to_arrays(X_train, y_train)
            stratify = y_arr if np.bincount(y_arr.astype(int)).min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr,
                y_arr,
                test_size=SKLEARN_MODE_SELECTION_FRACTION,
                random_state=self.random_state,
                stratify=stratify,
            )
            logger.info(
                "scikit-learn mode: split off %d selection samples from training data.", len(X_test)
            )
        elif y_test is None:
            raise ValueError(
                "y_test must be provided when X_test is given. "
                "Call fit(X, y) for automatic internal splitting."
            )

        # Step 2: Prepare data
        logger.info("Preparing data...")
        X_train_arr, y_train_arr, feature_names = self._prepare_train_data(X_train, y_train)
        X_test_arr, y_test_arr = self._prepare_score_data(X_test, y_test, feature_names)

        # Resolve selection-set IDs: explicit > id_column > row positions
        if ids_test is None:
            ids_test = self._extract_ids_from_dataframe(X_test)
        if ids_test is None:
            ids_test = np.arange(len(X_test_arr))
        else:
            ids_test = np.asarray(ids_test)
        if len(ids_test) != len(X_test_arr):
            raise ValueError(
                f"Shape mismatch: X_test has {len(X_test_arr)} samples but "
                f"ids_test has {len(ids_test)} values."
            )

        # Validate data
        ParameterValidator.validate_data(X_train_arr, y_train_arr)
        ParameterValidator.validate_data(X_test_arr, y_test_arr)

        # Store feature information
        self.feature_names_ = feature_names
        self.n_features_ = X_train_arr.shape[1]
        self.n_features_in_ = X_train_arr.shape[1]
        self.classes_ = np.unique(y_train_arr)

        # Validate f against number of features
        ParameterValidator.validate_f_against_features(self.f, self.n_features_)
        if self.f_range is not None:
            for width in self.f_range:
                ParameterValidator.validate_f_against_features(int(width), self.n_features_)

        # Step 3: Generate feature combinations. With f_range set, the m
        # combinations are split across the requested widths so the gate
        # and ranking can select the right constituent scale per dataset.
        widths = list(self.f_range) if self.f_range is not None else [self.f]
        logger.info(
            "Generating %d feature combinations with widths %s, sample=%s...",
            self.m,
            widths,
            self.sample,
        )
        feature_combinations: List[List[int]] = []
        base, remainder = divmod(self.m, len(widths))
        for i, width in enumerate(widths):
            width_m = base + (1 if i < remainder else 0)
            if width_m <= 0:
                continue
            sampler = FeatureSampler(
                n_features=self.n_features_,
                f=int(width),
                m=width_m,
                sample=self.sample,
                # decorrelate the per-width streams while staying
                # reproducible from random_state
                random_state=None if self.random_state is None else self.random_state + i,
            )
            feature_combinations.extend(sampler.generate_combinations())
        actual_m = len(feature_combinations)

        if actual_m < self.m:
            logger.warning("Generated %d combinations (less than requested %d).", actual_m, self.m)
        else:
            logger.info("Generated %d feature combinations.", actual_m)

        # Step 4: Train baseline model
        logger.info("Training %s baseline model...", self.baseline)
        self.baseline_model_ = build_baseline(
            self.baseline,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
        )
        self.baseline_model_.fit(X_train_arr, y_train_arr)
        self.baseline_kind_ = getattr(self.baseline_model_, "actual_kind_", "random_forest")
        self.baseline_results_ = self.baseline_model_.evaluate(
            X_test_arr, y_test_arr, ids_test, self.d
        )
        logger.info(
            "Baseline model - PLR: %s, FNR: %.3f, DRP: %.3f",
            self.baseline_results_["plr"],
            self.baseline_results_["fnr"],
            self.baseline_results_["drp"],
        )

        # Step 5: Train constituent models (optionally in parallel)
        logger.info("Training %d constituent logistic regression models...", actual_m)
        parallel = Parallel(n_jobs=self.n_jobs)
        outcomes = parallel(
            delayed(_train_constituent)(
                feature_indices,
                X_train_arr,
                y_train_arr,
                X_test_arr,
                y_test_arr,
                ids_test,
                self.d,
                self.solver,
                self.random_state,
                self.class_weight,
                feature_names,
            )
            for feature_indices in feature_combinations
        )

        self.constituent_models_ = []
        self.constituent_results_ = []
        for i, outcome in enumerate(outcomes):
            if outcome is None:
                continue
            model, result = outcome
            self.constituent_models_.append(model)
            self.constituent_results_.append(result)

        successful_models = len(self.constituent_models_)
        failed_models = actual_m - successful_models
        logger.info("Successfully trained %d models (%d failed).", successful_models, failed_models)

        if successful_models == 0:
            raise ValueError(
                "All constituent models failed to train. " "Please check your data and parameters."
            )

        # Step 6: Apply ensemble method
        logger.info("Applying %s ensemble method with spread=%d...", self.method, self.spread)
        self._apply_ensemble_method(X_test_arr, y_test_arr, ids_test)

        # Step 7: Export JSON if specified
        if self.elr2json:
            logger.info("Exporting all model metrics to %s...", self.elr2json)
            ModelIO.export_all_models(self.constituent_results_, self.elr2json)

        if self.modeljson:
            if self.fell_back_to_baseline_ and self.method != "ensemble":
                logger.info(
                    "Baseline fallback active; exporting the %s baseline "
                    "equation to %s as a single-model configuration...",
                    self.baseline_kind_,
                    self.modeljson,
                )
                self._export_fallback_baseline()
            elif self.fell_back_to_baseline_:
                logger.warning(
                    "Baseline fallback active for the ensemble method; "
                    "skipping modeljson export (the baseline has no meta-model "
                    "equation). Predictions remain available in Python."
                )
            else:
                logger.info("Exporting selected model equations to %s...", self.modeljson)
                self._export_selected_models()

        logger.info("ELR classifier training complete!")

        return self

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _to_arrays(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Convert X and y to plain numpy arrays."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
        return np.asarray(X_arr, dtype=float), y_arr

    def _numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop the ID column and non-numeric columns (with warnings)."""
        X = X.copy()
        if self.id_column in X.columns:
            X = X.drop(columns=[self.id_column])
        numeric = X.select_dtypes(include=[np.number])
        dropped = [c for c in X.columns if c not in numeric.columns]
        if dropped:
            logger.warning(
                "Dropping non-numeric columns (logistic regression requires "
                "numeric features): %s",
                dropped,
            )
        return numeric

    def _prepare_train_data(self, X, y) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare the training split; determines the feature names."""
        if isinstance(X, pd.DataFrame):
            numeric = self._numeric_frame(X)
            feature_names = numeric.columns.tolist()
            X_array = numeric.values.astype(float)
        else:
            X_array = np.asarray(X, dtype=float)
            feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

        y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)
        return X_array, y_array, feature_names

    def _prepare_score_data(self, X, y, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare the selection/prediction split, aligned to training features."""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            if self.id_column in X.columns:
                X = X.drop(columns=[self.id_column])
            missing = [name for name in feature_names if name not in X.columns]
            if missing:
                raise ValueError(
                    f"Input data is missing required feature columns: {missing}. "
                    f"Available columns: {X.columns.tolist()}"
                )
            X_array = X[feature_names].values.astype(float)
        else:
            X_array = np.asarray(X, dtype=float)
            if X_array.shape[1] != len(feature_names):
                raise ValueError(
                    f"Expected {len(feature_names)} features, got "
                    f"{X_array.shape[1]}. When passing numpy arrays, the "
                    "column order must match the training features."
                )

        y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)
        return X_array, y_array

    def _extract_ids_from_dataframe(self, X) -> Optional[np.ndarray]:
        """Pull selection IDs from ``id_column`` when X is a DataFrame."""
        if isinstance(X, pd.DataFrame) and self.id_column in X.columns:
            return cast(np.ndarray, X[self.id_column].values)
        return None

    # ------------------------------------------------------------------
    # Ensemble application
    # ------------------------------------------------------------------

    def _apply_ensemble_method(self, X_test: np.ndarray, y_test: np.ndarray, ids_test: np.ndarray):
        """Apply the selected ensemble method and store the results."""
        self.fell_back_to_baseline_ = False
        self.selection_weights_: List[float] = []

        if self.method == "intersect":
            ranked = IntersectMethod._rank_models(self.constituent_results_, self.baseline_results_)
            self.selected_indices_ = IntersectMethod.select(
                self.constituent_results_, self.baseline_results_, self.spread, self.selection
            )
            self.selection_weights_ = weights_for_selection(self.selected_indices_, ranked)
            self.result_df_ = self._weighted_vote_table(X_test, ids_test)

        elif self.method == "venn":
            ranked = VennMethod._rank_models(self.constituent_results_, self.baseline_results_)
            self.selected_indices_ = VennMethod.select(
                self.constituent_results_, self.baseline_results_, self.spread, self.selection
            )
            self.selection_weights_ = weights_for_selection(self.selected_indices_, ranked)
            self.result_df_ = self._weighted_vote_table(X_test, ids_test)

        elif self.method == "ensemble":
            result_df, meta_equation, selected = EnsembleMethod.select_and_combine(
                self.constituent_models_,
                X_test,
                y_test,
                ids_test,
                self.constituent_results_,
                self.baseline_results_,
                self.spread,
                self.random_state,
                class_weight=self.class_weight,
            )
            self.selected_indices_ = selected
            self.result_df_ = result_df
            self.meta_equation_ = meta_equation

        else:  # pragma: no cover - validation catches this earlier
            raise ValueError(f"Unknown method: {self.method}")

        if not self.selected_indices_:
            self._fallback_to_baseline(X_test, ids_test)

        logger.info(
            "Selected %d models for %s method%s.",
            len(self.selected_indices_),
            self.method,
            " (baseline fallback)" if self.fell_back_to_baseline_ else "",
        )

    def _fallback_to_baseline(self, X_test: np.ndarray, ids_test: np.ndarray):
        """Adopt the baseline Random Forest when no constituent qualifies.

        The selection methods only keep models that beat the baseline; if
        none does, the baseline is by construction the best available model.
        Linear baselines ("woe_logreg"/"logreg") export their equation to
        modeljson as a single-model fallback configuration, so the fallback
        remains SQL-deployable; the Random Forest fallback is Python-only.
        """
        self.fell_back_to_baseline_ = True
        self.selected_indices_ = []
        self.selection_weights_ = []
        logger.warning(
            "No constituent models passed the baseline gate; falling back "
            "to the baseline Random Forest for all predictions."
        )
        if self.method == "ensemble":
            proba = self.baseline_model_.predict_proba(X_test)[:, 1]
            self.result_df_ = pd.DataFrame({"id": ids_test, "predicted": proba})
            self.meta_equation_ = None
        else:
            # A single "model" (the baseline) contributes its decile set
            id_counts = {id_value: 1 for id_value in self.baseline_results_["drs"]}
            self.result_df_ = build_sets_dataframe(id_counts)

    def _weighted_vote_table(self, X: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
        """Weighted [id, sets, score] table from the selected models' deciles.

        Recomputes each selected model's top-d deciles on ``X`` (same rule
        as training) and accumulates one vote of weight ``w_k`` per model
        containing the ID, mirroring :func:`combine_drs`.
        """
        id_counts: Dict[Any, int] = {}
        id_scores: Dict[Any, float] = {}
        for idx, weight in zip(self.selected_indices_, self.selection_weights_):
            y_score = self.constituent_models_[idx].predict_proba(X)[:, 1]
            for id_value in ids[top_d_row_indices(y_score, self.d)]:
                id_counts[id_value] = id_counts.get(id_value, 0) + 1
                id_scores[id_value] = id_scores.get(id_value, 0.0) + weight
        return build_sets_dataframe(id_counts, id_scores)

    def _count_selection_votes(self, X: np.ndarray, ids=None) -> Dict[Any, int]:
        """Count, per ID, in how many selected models' top-d deciles it lands."""
        id_counts: Dict[Any, int] = {}
        for idx in self.selected_indices_:
            y_score = self.constituent_models_[idx].predict_proba(X)[:, 1]
            for id_value in ids[top_d_row_indices(y_score, self.d)]:
                id_counts[id_value] = id_counts.get(id_value, 0) + 1
        return id_counts

    def _row_vote_fractions(self, X: np.ndarray) -> np.ndarray:
        """Per-row weighted vote share of the selected models' top-d deciles.

        Each model contributes its Borda-position weight when the row lands
        in its top-d deciles; the total is normalized by the sum of all
        weights, yielding a continuous score in [0, 1] usable for ranking.
        """
        total_weight = float(sum(self.selection_weights_))
        if total_weight <= 0:
            return np.zeros(len(X))
        weighted = np.zeros(len(X))
        for idx, weight in zip(self.selected_indices_, self.selection_weights_):
            y_score = self.constituent_models_[idx].predict_proba(X)[:, 1]
            weighted[top_d_row_indices(y_score, self.d)] += weight
        return weighted / total_weight

    def _ensemble_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probabilities from the meta-model equation."""
        if self.meta_equation_ is None or not self.selected_indices_:
            # Baseline fallback
            return cast(np.ndarray, self.baseline_model_.predict_proba(X)[:, 1])
        meta_features = [
            self.constituent_models_[idx].predict_proba(X)[:, 1] for idx in self.selected_indices_
        ]
        X_meta = np.column_stack(meta_features)
        linear = np.full(len(X), self.meta_equation_["constant"])
        for i in range(len(self.selected_indices_)):
            linear += self.meta_equation_[f"model_{i}_prob"] * X_meta[:, i]
        return _sigmoid(linear)

    def _export_fallback_baseline(self):
        """Export the (linear) baseline equation as a single-model modeljson.

        The fallback behaves exactly like a one-model intersect/venn
        selection with weight 1.0, so the standard scoring paths (Python
        scorer, SQL generator) reproduce the fallback output, including
        WoE CASE WHEN ladders for binned baselines.
        """
        try:
            baseline_equation = self.baseline_model_.get_equation_dict(self.feature_names_)
        except ValueError:
            logger.warning("Baseline fallback has no exportable equation; skipping modeljson.")
            return
        ModelIO.export_selected_models(
            method=self.method,
            d=self.d,
            selected_models=[baseline_equation],
            meta_model=None,
            filepath=self.modeljson,
            feature_names=self.feature_names_,
            weights=[1.0],
            fallback=True,
        )

    def _export_selected_models(self):
        """Export selected model equations to JSON."""
        selected_equations = [
            self.constituent_results_[idx]["equation_dict"] for idx in self.selected_indices_
        ]
        ModelIO.export_selected_models(
            method=self.method,
            d=self.d,
            selected_models=selected_equations,
            meta_model=self.meta_equation_ if self.method == "ensemble" else None,
            filepath=self.modeljson,
            feature_names=self.feature_names_,
            weights=self.selection_weights_ if self.method != "ensemble" else None,
        )

    # ------------------------------------------------------------------
    # Prediction (scikit-learn convention)
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if (
            getattr(self, "constituent_models_", None) is None
            or getattr(self, "baseline_model_", None) is None
        ):
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

    def _prepare_predict_input(self, X) -> np.ndarray:
        """Prepare feature data for prediction (aligned with training)."""
        self._check_fitted()
        X_array, _ = self._prepare_score_data(X, np.zeros(max(len(X), 0)), self.feature_names_)
        return X_array

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        ids: Optional[Union[np.ndarray, pd.Series, List]] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Predict class labels (scikit-learn convention).

        - ensemble: positive class when the meta-model probability >= 0.5
        - intersect/venn: positive class when the instance falls in the
          top-d deciles of a strict majority of the selected models
        - baseline fallback: the baseline Random Forest's prediction

        Use :meth:`predict_detailed` for the method-specific targeting
        tables ELR is designed around.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        ids : array-like of shape (n_samples,), optional
            **Deprecated.** When given, emits a ``DeprecationWarning`` and
            returns ``predict_detailed(X, ids)`` — the 0.5.x two-argument
            behaviour. Omit it to get class labels.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels (or a DataFrame when the deprecated
            ``ids`` argument is supplied).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if ids is not None:
            warnings.warn(
                "ELRClassifier.predict(X, ids) is deprecated and will be "
                "removed in a future release. Use predict_detailed(X, ids) "
                "for the [id, sets]/[id, predicted] targeting tables, or "
                "predict(X) for class labels.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.predict_detailed(X, ids)

        X_array = self._prepare_predict_input(X)

        if self.fell_back_to_baseline_ or not self.selected_indices_:
            return np.asarray(self.baseline_model_.predict(X_array))

        if self.method == "ensemble":
            proba = self._ensemble_probabilities(X_array)
            positive = proba >= 0.5
        else:
            # Strict majority of the weighted votes (share of total weight)
            positive = 2 * self._row_vote_fractions(X_array) > 1.0

        return np.where(positive, self.classes_[-1], self.classes_[0])

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities (scikit-learn convention).

        - ensemble: meta-model probability
        - intersect/venn: fraction of selected models whose top-d deciles
          contain the instance (a vote-share score usable for ranking/AUC)
        - baseline fallback: baseline Random Forest probabilities

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Probabilities for [negative, positive] classes.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        X_array = self._prepare_predict_input(X)

        if self.method == "ensemble":
            proba_positive = self._ensemble_probabilities(X_array)
        elif self.fell_back_to_baseline_ or not self.selected_indices_:
            proba_positive = self.baseline_model_.predict_proba(X_array)[:, 1]
        else:
            proba_positive = self._row_vote_fractions(X_array)

        proba_negative = 1.0 - proba_positive
        return np.column_stack([proba_negative, proba_positive])

    # ------------------------------------------------------------------
    # Prediction (ELR targeting tables)
    # ------------------------------------------------------------------

    def predict_detailed(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        ids: Optional[Union[np.ndarray, pd.Series, List]] = None,
    ) -> pd.DataFrame:
        """
        Generate the method-specific targeting output.

        The output format depends on the ensemble method:
        - intersect/venn: DataFrame with 'id' and 'sets' columns, sorted by
          sets descending; only IDs inside the top-d deciles of at least one
          selected model are included
        - ensemble: DataFrame with 'id' and 'predicted' (probability) columns

        This is the exact scoring logic used during fit (the same decile
        rule as :class:`~paramsemble_class.scoring.scorer.ModelScorer` and
        the SQL generator), so scoring the selection set reproduces
        ``result_df_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        ids : array-like of shape (n_samples,), optional
            ID values for samples. Defaults to the ``id_column`` of a
            DataFrame input containing that column, otherwise row positions.

        Returns
        -------
        pd.DataFrame
            Method-specific predictions.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        X_array = self._prepare_predict_input(X)

        if ids is None:
            ids = self._extract_ids_from_dataframe(X)
        if ids is None:
            ids = np.arange(len(X_array))
        ids = np.asarray(ids)
        if len(ids) != len(X_array):
            raise ValueError(f"X has {len(X_array)} samples but ids has {len(ids)} values.")

        if self.method == "ensemble":
            proba = self._ensemble_probabilities(X_array)
            return pd.DataFrame({"id": ids, "predicted": proba})

        if self.fell_back_to_baseline_ or not self.selected_indices_:
            # Baseline fallback: the baseline forest is the single "model";
            # its top-d deciles get a sets count of 1, mirroring result_df_.
            y_score = self.baseline_model_.predict_proba(X_array)[:, 1]
            id_counts = {id_value: 1 for id_value in ids[top_d_row_indices(y_score, self.d)]}
            return build_sets_dataframe(id_counts)
        return self._weighted_vote_table(X_array, ids)
