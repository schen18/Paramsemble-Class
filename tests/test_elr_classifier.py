"""Unit tests for ELRClassifier API."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from paramsemble_class.core.elr_classifier import ELRClassifier


class TestELRClassifierAPI:
    """Test ELRClassifier API and sklearn compatibility."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        # Generate synthetic binary classification data
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )

        # Split into train and test
        split_idx = 100
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        ids_test = np.arange(len(X_test))

        return X_train, y_train, X_test, y_test, ids_test

    def test_init_with_default_parameters(self):
        """Test ELRClassifier initialization with default parameters."""
        clf = ELRClassifier()

        assert clf.m == 100
        assert clf.f == 5
        assert clf.sample == "unique"
        assert clf.d == 2
        assert clf.method == "intersect"
        assert clf.spread == 10
        assert clf.solver == "auto"
        assert clf.id_column == "id"
        assert clf.elr2json is None
        assert clf.modeljson is None
        assert clf.random_state is None

    def test_init_with_custom_parameters(self):
        """Test ELRClassifier initialization with custom parameters."""
        clf = ELRClassifier(
            m=50,
            f=3,
            sample="replace",
            d=3,
            method="venn",
            spread=5,
            solver="lbfgs",
            id_column="custom_id",
            elr2json="models.json",
            modeljson="config.json",
            random_state=42,
        )

        assert clf.m == 50
        assert clf.f == 3
        assert clf.sample == "replace"
        assert clf.d == 3
        assert clf.method == "venn"
        assert clf.spread == 5
        assert clf.solver == "lbfgs"
        assert clf.id_column == "custom_id"
        assert clf.elr2json == "models.json"
        assert clf.modeljson == "config.json"
        assert clf.random_state == 42

    def test_fit_accepts_correct_parameters(self, sample_data):
        """Test that fit method accepts correct parameters."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        clf = ELRClassifier(m=10, f=3, random_state=42)

        # Should not raise any errors
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Check that model attributes are set
        assert clf.baseline_model_ is not None
        assert clf.constituent_models_ is not None
        assert clf.constituent_results_ is not None
        assert clf.baseline_results_ is not None
        assert clf.selected_indices_ is not None
        assert clf.result_df_ is not None
        assert clf.feature_names_ is not None
        assert clf.n_features_ == 10

    def test_fit_with_dataframe_input(self, sample_data):
        """Test fit method with pandas DataFrame input."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        # Convert to DataFrames
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        y_train_series = pd.Series(y_train)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        y_test_series = pd.Series(y_test)
        ids_test_series = pd.Series(ids_test)

        clf = ELRClassifier(m=10, f=3, random_state=42)
        clf.fit(X_train_df, y_train_series, X_test_df, y_test_series, ids_test_series)

        assert clf.feature_names_ == feature_names
        assert clf.n_features_ == 10

    def test_predict_returns_predictions(self, sample_data):
        """Test that predict_detailed returns method-specific output."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        clf = ELRClassifier(m=10, f=3, method="intersect", random_state=42)
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Get predictions
        predictions = clf.predict_detailed(X_test, ids_test)

        # Check output format for intersect method
        assert isinstance(predictions, pd.DataFrame)
        assert "id" in predictions.columns
        assert "sets" in predictions.columns
        assert len(predictions) > 0

    def test_predict_without_ids(self, sample_data):
        """Test predict_detailed without providing IDs (row positions used)."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        clf = ELRClassifier(m=10, f=3, method="ensemble", random_state=42)
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Predict without IDs (should use indices)
        predictions = clf.predict_detailed(X_test)

        assert isinstance(predictions, pd.DataFrame)
        assert "id" in predictions.columns
        assert len(predictions) == len(X_test)

    def test_predict_returns_class_labels(self, sample_data):
        """Test that predict returns sklearn-style class labels for all methods."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        for method in ["intersect", "venn", "ensemble"]:
            clf = ELRClassifier(m=10, f=3, method=method, random_state=42)
            clf.fit(X_train, y_train, X_test, y_test, ids_test)

            labels = clf.predict(X_test)
            assert isinstance(labels, np.ndarray)
            assert labels.shape == (len(X_test),)
            assert set(np.unique(labels)).issubset({0, 1})
            assert hasattr(clf, "classes_")

    def test_sklearn_two_argument_fit(self, sample_data):
        """Test that fit(X, y) works via an internal stratified split."""
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=300, n_features=10, random_state=7)

        clf = ELRClassifier(m=8, f=3, method="ensemble", random_state=0)
        clf.fit(X, y)

        assert clf.classes_ is not None
        labels = clf.predict(X)
        assert len(labels) == len(X)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predict raises error if called before fit."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        clf = ELRClassifier(m=10, f=3, random_state=42)

        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            clf.predict(X_test)

    def test_sklearn_base_estimator_compatibility(self):
        """Test sklearn BaseEstimator compatibility."""
        from sklearn.base import BaseEstimator, ClassifierMixin

        clf = ELRClassifier()

        # Check inheritance
        assert isinstance(clf, BaseEstimator)
        assert isinstance(clf, ClassifierMixin)

    def test_get_params(self):
        """Test get_params method for sklearn compatibility."""
        clf = ELRClassifier(
            m=50, f=3, sample="replace", d=3, method="venn", spread=5, random_state=42
        )

        params = clf.get_params()

        assert params["m"] == 50
        assert params["f"] == 3
        assert params["sample"] == "replace"
        assert params["d"] == 3
        assert params["method"] == "venn"
        assert params["spread"] == 5
        assert params["random_state"] == 42

    def test_set_params(self):
        """Test set_params method for sklearn compatibility."""
        clf = ELRClassifier()

        clf.set_params(m=30, f=4, random_state=123)

        assert clf.m == 30
        assert clf.f == 4
        assert clf.random_state == 123

    def test_intersect_method_workflow(self, sample_data):
        """Test end-to-end workflow for intersect method."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        clf = ELRClassifier(m=15, f=4, method="intersect", spread=5, random_state=42)

        # Fit
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Predict
        predictions = clf.predict_detailed(X_test, ids_test)

        # Validate output structure
        assert isinstance(predictions, pd.DataFrame)
        assert "id" in predictions.columns
        assert "sets" in predictions.columns
        assert predictions["sets"].min() >= 1
        assert predictions["sets"].max() <= len(clf.selected_indices_)

    def test_venn_method_workflow(self, sample_data):
        """Test end-to-end workflow for venn method."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        clf = ELRClassifier(m=15, f=4, method="venn", spread=5, random_state=42)

        # Fit
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Predict
        predictions = clf.predict_detailed(X_test, ids_test)

        # Validate output structure
        assert isinstance(predictions, pd.DataFrame)
        assert "id" in predictions.columns
        assert "sets" in predictions.columns

    def test_ensemble_method_workflow(self, sample_data):
        """Test end-to-end workflow for ensemble method."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        clf = ELRClassifier(m=15, f=4, method="ensemble", spread=5, random_state=42)

        # Fit
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Predict
        predictions = clf.predict_detailed(X_test, ids_test)

        # Validate output structure
        assert isinstance(predictions, pd.DataFrame)
        assert "id" in predictions.columns
        assert "predicted" in predictions.columns
        assert len(predictions) == len(X_test)
        assert predictions["predicted"].min() >= 0
        assert predictions["predicted"].max() <= 1

        # Check meta-model equation exists
        assert clf.meta_equation_ is not None
        assert "constant" in clf.meta_equation_

    def test_predict_proba_for_ensemble_method(self, sample_data):
        """Test predict_proba method for ensemble method."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        clf = ELRClassifier(m=10, f=3, method="ensemble", spread=5, random_state=42)

        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Get probability predictions
        proba = clf.predict_proba(X_test)

        # Validate output
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(proba >= 0) and np.all(proba <= 1)  # Valid probabilities

    def test_predict_proba_vote_share_for_intersect(self, sample_data):
        """Test that predict_proba returns vote-share scores for intersect."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        clf = ELRClassifier(m=10, f=3, method="intersect", random_state=42)

        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_json_export(self, sample_data):
        """Test JSON export functionality."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            elr2json_path = os.path.join(tmpdir, "all_models.json")
            modeljson_path = os.path.join(tmpdir, "selected_models.json")

            clf = ELRClassifier(
                m=10,
                f=3,
                method="intersect",
                spread=5,
                elr2json=elr2json_path,
                modeljson=modeljson_path,
                random_state=42,
            )

            clf.fit(X_train, y_train, X_test, y_test, ids_test)

            # Check that files were created
            assert os.path.exists(elr2json_path)
            assert os.path.exists(modeljson_path)

            # Verify file contents are valid JSON
            import json

            with open(elr2json_path, "r") as f:
                all_models = json.load(f)
                assert isinstance(all_models, list)
                assert len(all_models) > 0

            with open(modeljson_path, "r") as f:
                selected_models = json.load(f)
                assert "method" in selected_models
                assert "d" in selected_models
                assert "models" in selected_models
                assert selected_models["method"] == "intersect"

    def test_invalid_parameters_raise_errors(self):
        """Test that invalid parameters raise appropriate errors."""
        # Invalid m
        with pytest.raises(ValueError, match="Parameter 'm' must be >= 1"):
            clf = ELRClassifier(m=0)
            clf.fit(
                np.random.randn(10, 5),
                np.random.randint(0, 2, 10),
                np.random.randn(5, 5),
                np.random.randint(0, 2, 5),
                np.arange(5),
            )

        # Invalid f
        with pytest.raises(ValueError, match="Parameter 'f' must be >= 1"):
            clf = ELRClassifier(f=0)
            clf.fit(
                np.random.randn(10, 5),
                np.random.randint(0, 2, 10),
                np.random.randn(5, 5),
                np.random.randint(0, 2, 5),
                np.arange(5),
            )

        # Invalid sample
        with pytest.raises(ValueError, match="Parameter 'sample' must be one of"):
            clf = ELRClassifier(sample="invalid")
            clf.fit(
                np.random.randn(10, 5),
                np.random.randint(0, 2, 10),
                np.random.randn(5, 5),
                np.random.randint(0, 2, 5),
                np.arange(5),
            )

        # Invalid d
        with pytest.raises(ValueError, match="Parameter 'd' must be between 1 and 10"):
            clf = ELRClassifier(d=11)
            clf.fit(
                np.random.randn(10, 5),
                np.random.randint(0, 2, 10),
                np.random.randn(5, 5),
                np.random.randint(0, 2, 5),
                np.arange(5),
            )

        # Invalid method
        with pytest.raises(ValueError, match="Parameter 'method' must be one of"):
            clf = ELRClassifier(method="invalid")
            clf.fit(
                np.random.randn(10, 5),
                np.random.randint(0, 2, 10),
                np.random.randn(5, 5),
                np.random.randint(0, 2, 5),
                np.arange(5),
            )

    def test_f_exceeds_features_raises_error(self, sample_data):
        """Test that f exceeding number of features raises error."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        clf = ELRClassifier(m=10, f=20, random_state=42)  # f > n_features

        with pytest.raises(
            ValueError, match="Parameter 'f' .* exceeds the number of available features"
        ):
            clf.fit(X_train, y_train, X_test, y_test, ids_test)

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that results are reproducible with same random_state."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        # Train two models with same random state
        clf1 = ELRClassifier(m=10, f=3, method="ensemble", random_state=42)
        clf1.fit(X_train, y_train, X_test, y_test, ids_test)
        pred1 = clf1.predict_detailed(X_test, ids_test)

        clf2 = ELRClassifier(m=10, f=3, method="ensemble", random_state=42)
        clf2.fit(X_train, y_train, X_test, y_test, ids_test)
        pred2 = clf2.predict_detailed(X_test, ids_test)

        # Results should be identical
        pd.testing.assert_frame_equal(pred1, pred2)

    def test_different_random_states_produce_different_results(self, sample_data):
        """Test that different random states produce different results."""
        X_train, y_train, X_test, y_test, ids_test = sample_data

        # Train two models with different random states
        clf1 = ELRClassifier(m=10, f=3, method="ensemble", random_state=42)
        clf1.fit(X_train, y_train, X_test, y_test, ids_test)
        pred1 = clf1.predict_detailed(X_test, ids_test)

        clf2 = ELRClassifier(m=10, f=3, method="ensemble", random_state=123)
        clf2.fit(X_train, y_train, X_test, y_test, ids_test)
        pred2 = clf2.predict_detailed(X_test, ids_test)

        # Results should be different
        assert not pred1["predicted"].equals(pred2["predicted"])
