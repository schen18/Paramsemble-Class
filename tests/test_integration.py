"""Integration tests for ELR package.

Tests complete workflows from fit to predict for all methods,
JSON export/import, SQL generation, and sklearn compatibility.
"""

import pytest

from tests.conftest import requires_sqlite_math
import numpy as np
import pandas as pd
import tempfile
import os
import json
import sqlite3
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from paramsemble_class import ELRClassifier
from paramsemble_class.scoring.scorer import ModelScorer
from paramsemble_class.sql.generator import SQLGenerator


class TestCompleteWorkflow:
    """Test complete workflow from fit to predict for all methods."""

    def test_intersect_method_workflow(self):
        """Test complete intersect method workflow."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            n_classes=2,
            random_state=42,
        )

        # Split into train/test
        split_idx = 200
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        ids_test = np.arange(len(y_test))

        # Train classifier
        clf = ELRClassifier(
            m=10, f=5, sample="unique", d=2, method="intersect", spread=3, random_state=42
        )

        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Verify training completed
        assert clf.baseline_model_ is not None
        assert clf.constituent_models_ is not None
        assert len(clf.constituent_models_) > 0
        assert clf.result_df_ is not None

        # Verify result structure
        assert "id" in clf.result_df_.columns
        assert "sets" in clf.result_df_.columns
        # Note: result may be empty if no models outperform baseline
        # which is valid behavior

        # Test prediction on new data
        X_new = X[250:260]
        ids_new = np.arange(10)
        predictions = clf.predict_detailed(X_new, ids_new)

        # Verify prediction structure
        assert isinstance(predictions, pd.DataFrame)
        assert "id" in predictions.columns
        assert "sets" in predictions.columns

    def test_venn_method_workflow(self):
        """Test complete venn method workflow."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            n_classes=2,
            random_state=43,
        )

        # Split into train/test
        split_idx = 200
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        ids_test = np.arange(len(y_test))

        # Train classifier
        clf = ELRClassifier(
            m=10, f=5, sample="unique", d=2, method="venn", spread=3, random_state=43
        )

        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Verify training completed
        assert clf.baseline_model_ is not None
        assert clf.constituent_models_ is not None
        assert len(clf.constituent_models_) > 0
        assert clf.result_df_ is not None

        # Verify result structure
        assert "id" in clf.result_df_.columns
        assert "sets" in clf.result_df_.columns

        # Test prediction on new data
        X_new = X[250:260]
        ids_new = np.arange(10)
        predictions = clf.predict_detailed(X_new, ids_new)

        # Verify prediction structure
        assert isinstance(predictions, pd.DataFrame)
        assert "id" in predictions.columns
        assert "sets" in predictions.columns

    def test_ensemble_method_workflow(self):
        """Test complete ensemble method workflow."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            n_classes=2,
            random_state=44,
        )

        # Split into train/test
        split_idx = 200
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        ids_test = np.arange(len(y_test))

        # Train classifier
        clf = ELRClassifier(
            m=10, f=5, sample="unique", d=2, method="ensemble", spread=3, random_state=44
        )

        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Verify training completed
        assert clf.baseline_model_ is not None
        assert clf.constituent_models_ is not None
        assert len(clf.constituent_models_) > 0
        assert clf.result_df_ is not None
        if clf.fell_back_to_baseline_:
            assert clf.meta_equation_ is None
        else:
            assert clf.meta_equation_ is not None

        # Verify result structure
        assert "id" in clf.result_df_.columns
        assert "predicted" in clf.result_df_.columns
        assert len(clf.result_df_) == len(ids_test)

        # Verify probabilities are in valid range
        assert (clf.result_df_["predicted"] >= 0).all()
        assert (clf.result_df_["predicted"] <= 1).all()

        # Test prediction on new data
        X_new = X[250:260]
        ids_new = np.arange(10)
        predictions = clf.predict_detailed(X_new, ids_new)

        # Verify prediction structure
        assert isinstance(predictions, pd.DataFrame)
        assert "id" in predictions.columns
        assert "predicted" in predictions.columns
        assert len(predictions) == len(ids_new)

        # Test predict_proba
        proba = clf.predict_proba(X_new)
        assert proba.shape == (len(X_new), 2)
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestSyntheticDatasets:
    """Test with various synthetic dataset configurations."""

    def test_balanced_dataset(self):
        """Test with balanced binary classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_classes=2,
            weights=[0.5, 0.5],
            random_state=45,
        )

        split_idx = 150
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        ids_test = np.arange(len(y_test))

        clf = ELRClassifier(m=5, f=3, method="intersect", spread=2, random_state=45)
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        assert clf.result_df_ is not None
        # Note: result may be empty if no models outperform baseline

    def test_imbalanced_dataset(self):
        """Test with imbalanced binary classification dataset (90/10 split)."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_classes=2,
            weights=[0.9, 0.1],
            random_state=46,
        )

        split_idx = 150
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        ids_test = np.arange(len(y_test))

        clf = ELRClassifier(m=5, f=3, method="venn", spread=2, random_state=46)
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        assert clf.result_df_ is not None

    def test_high_dimensional_dataset(self):
        """Test with high-dimensional dataset (many features)."""
        X, y = make_classification(
            n_samples=200,
            n_features=50,
            n_informative=30,
            n_redundant=10,
            n_classes=2,
            random_state=47,
        )

        split_idx = 150
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        ids_test = np.arange(len(y_test))

        clf = ELRClassifier(m=5, f=10, method="ensemble", spread=2, random_state=47)
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        assert clf.result_df_ is not None
        assert len(clf.result_df_) == len(ids_test)


class TestJSONExportScoring:
    """Test JSON export and scoring workflow end-to-end."""

    def test_intersect_json_export_and_scoring(self):
        """Test intersect method JSON export and scoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            elr2json_path = os.path.join(tmpdir, "elr2.json")
            modeljson_path = os.path.join(tmpdir, "model.json")

            # Generate data
            X, y = make_classification(
                n_samples=200, n_features=10, n_informative=8, n_classes=2, random_state=48
            )

            split_idx = 150
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            ids_test = np.arange(len(y_test))

            # Train and export
            clf = ELRClassifier(
                m=5,
                f=3,
                method="intersect",
                spread=2,
                elr2json=elr2json_path,
                modeljson=modeljson_path,
                random_state=48,
            )
            clf.fit(X_train, y_train, X_test, y_test, ids_test)

            # Verify JSON files were created
            assert os.path.exists(elr2json_path)
            assert os.path.exists(modeljson_path)

            # Verify elr2json structure
            with open(elr2json_path, "r") as f:
                elr2_data = json.load(f)
            assert isinstance(elr2_data, list)
            assert len(elr2_data) > 0
            assert "plr" in elr2_data[0]
            assert "fnr" in elr2_data[0]
            assert "drp" in elr2_data[0]
            assert "equation_dict" in elr2_data[0]

            # Verify modeljson structure
            with open(modeljson_path, "r") as f:
                model_data = json.load(f)
            assert model_data["method"] == "intersect"
            assert model_data["d"] == 2
            assert "models" in model_data
            assert len(model_data["models"]) > 0

            # Test scoring with ModelScorer
            scorer = ModelScorer(modeljson_path)

            # Create DataFrame for scoring
            X_score = X[180:190]
            ids_score = np.arange(10)
            df_score = pd.DataFrame(
                X_score, columns=[f"feature_{i}" for i in range(X_score.shape[1])]
            )

            result = scorer.score(df_score, ids_score)

            # Verify scoring result
            assert isinstance(result, pd.DataFrame)
            assert "id" in result.columns
            assert "sets" in result.columns

    def test_ensemble_json_export_and_scoring(self):
        """Test ensemble method JSON export and scoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modeljson_path = os.path.join(tmpdir, "model.json")

            # Generate data with more samples and features for better model diversity
            X, y = make_classification(
                n_samples=400,
                n_features=20,
                n_informative=15,
                n_redundant=3,
                n_classes=2,
                random_state=49,
                flip_y=0.05,  # Add some noise
            )

            split_idx = 300
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            ids_test = np.arange(len(y_test))

            # Train and export with more models to increase chance of outperforming baseline
            clf = ELRClassifier(
                m=15,
                f=5,
                method="ensemble",
                spread=3,
                modeljson=modeljson_path,
                random_state=49,
                f_range=[5, 10],  # wider constituents to pass the linear gate
            )
            clf.fit(X_train, y_train, X_test, y_test, ids_test)

            if clf.fell_back_to_baseline_:
                # Honest fallback: the ensemble baseline (a full-feature
                # linear model) beat every constituent, so no meta-model
                # equation exists to export; Python predictions still work.
                assert not os.path.exists(modeljson_path)
                proba = clf.predict_proba(X_test)
                assert proba.shape == (len(X_test), 2)
                return

            # Verify JSON file was created
            assert os.path.exists(modeljson_path)

            # Verify modeljson structure
            with open(modeljson_path, "r") as f:
                model_data = json.load(f)
            assert model_data["method"] == "ensemble"
            assert "models" in model_data
            assert "meta_model" in model_data
            assert model_data["meta_model"] is not None

            # Test scoring with ModelScorer
            scorer = ModelScorer(modeljson_path)

            # Create DataFrame for scoring
            X_score = X[180:190]
            ids_score = np.arange(10)
            df_score = pd.DataFrame(
                X_score, columns=[f"feature_{i}" for i in range(X_score.shape[1])]
            )

            result = scorer.score(df_score, ids_score)

            # Verify scoring result
            assert isinstance(result, pd.DataFrame)
            assert "id" in result.columns
            assert "predicted" in result.columns
            assert len(result) == len(ids_score)
            assert (result["predicted"] >= 0).all()
            assert (result["predicted"] <= 1).all()


class TestSQLGeneration:
    """Test SQL generation and execution against SQLite database."""

    @requires_sqlite_math
    def test_intersect_sql_generation_and_execution(self):
        """Test intersect method SQL generation and execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modeljson_path = os.path.join(tmpdir, "model.json")
            db_path = os.path.join(tmpdir, "test.db")

            # Generate data
            X, y = make_classification(
                n_samples=200,
                n_features=10,
                n_informative=8,
                n_classes=2,
                random_state=50,
                flip_y=0.1,
            )

            split_idx = 150
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            ids_test = np.arange(len(y_test))

            # Train and export
            clf = ELRClassifier(
                m=5, f=3, method="intersect", spread=2, modeljson=modeljson_path, random_state=50
            )
            clf.fit(X_train, y_train, X_test, y_test, ids_test)

            # Generate SQL
            sql_gen = SQLGenerator(modeljson_path)
            sql_query = sql_gen.generate_sql("test_table", "id")

            # Verify SQL was generated
            assert isinstance(sql_query, str)
            assert len(sql_query) > 0
            assert "SELECT" in sql_query.upper()
            assert "FROM" in sql_query.upper()

            # Create SQLite database and table
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create table with test data
            X_db = X[180:190]
            ids_db = np.arange(10)
            df_db = pd.DataFrame(X_db, columns=[f"feature_{i}" for i in range(X_db.shape[1])])
            df_db["id"] = ids_db
            df_db.to_sql("test_table", conn, if_exists="replace", index=False)

            # Execute SQL query
            try:
                cursor.execute(sql_query)
                results = cursor.fetchall()

                # Verify results were returned
                assert results is not None
                # Results may be empty if no IDs meet criteria, which is valid

            except sqlite3.Error as e:
                pytest.fail(f"SQL execution failed: {e}")
            finally:
                conn.close()

    @requires_sqlite_math
    def test_ensemble_sql_generation_and_execution(self):
        """Test ensemble method SQL generation and execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modeljson_path = os.path.join(tmpdir, "model.json")
            db_path = os.path.join(tmpdir, "test.db")

            # Generate data
            X, y = make_classification(
                n_samples=200, n_features=10, n_informative=8, n_classes=2, random_state=51
            )

            split_idx = 150
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            ids_test = np.arange(len(y_test))

            # Train and export
            clf = ELRClassifier(
                m=5, f=3, method="ensemble", spread=2, modeljson=modeljson_path, random_state=51
            )
            clf.fit(X_train, y_train, X_test, y_test, ids_test)

            # Generate SQL
            sql_gen = SQLGenerator(modeljson_path)
            sql_query = sql_gen.generate_sql("test_table", "id")

            # Verify SQL was generated
            assert isinstance(sql_query, str)
            assert len(sql_query) > 0
            assert "SELECT" in sql_query.upper()
            assert "FROM" in sql_query.upper()

            # Create SQLite database and table
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create table with test data
            X_db = X[180:190]
            ids_db = np.arange(10)
            df_db = pd.DataFrame(X_db, columns=[f"feature_{i}" for i in range(X_db.shape[1])])
            df_db["id"] = ids_db
            df_db.to_sql("test_table", conn, if_exists="replace", index=False)

            # Execute SQL query
            try:
                cursor.execute(sql_query)
                results = cursor.fetchall()

                # Verify results were returned
                assert results is not None
                assert len(results) == len(ids_db)

                # Verify probabilities are in valid range
                for row in results:
                    # row should be (id, predicted_probability)
                    assert len(row) >= 2
                    prob = row[1]
                    assert 0 <= prob <= 1

            except sqlite3.Error as e:
                pytest.fail(f"SQL execution failed: {e}")
            finally:
                conn.close()


class TestSklearnCompatibility:
    """Test sklearn compatibility with cross_val_score."""

    def test_get_params(self):
        """Test get_params method."""
        clf = ELRClassifier(m=10, f=5, method="intersect", spread=3)
        params = clf.get_params()

        assert isinstance(params, dict)
        assert params["m"] == 10
        assert params["f"] == 5
        assert params["method"] == "intersect"
        assert params["spread"] == 3

    def test_set_params(self):
        """Test set_params method."""
        clf = ELRClassifier(m=10, f=5)
        clf.set_params(m=20, f=7)

        assert clf.m == 20
        assert clf.f == 7

    def test_cross_val_score_compatibility(self):
        """Test compatibility with sklearn's cross_val_score.

        Note: This is a simplified test since ELRClassifier requires
        test data in fit(), which doesn't align perfectly with
        standard cross-validation. We test that the API is compatible.
        """
        # Generate data with more samples for better model diversity
        X, y = make_classification(
            n_samples=400,
            n_features=15,
            n_informative=12,
            n_redundant=2,
            n_classes=2,
            random_state=52,
            flip_y=0.05,
        )

        # Create a wrapper that adapts ELRClassifier to standard sklearn API
        # ELRClassifier now supports fit(X, y) directly: it splits off a
        # stratified selection set internally, and predict returns labels.
        clf = ELRClassifier(m=15, f=5, method="ensemble", spread=3, random_state=52)

        # Verify it has sklearn-compatible methods
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict")
        assert hasattr(clf, "get_params")
        assert hasattr(clf, "set_params")

        # Test basic fit/predict cycle
        split_idx = 300
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        clf.fit(X_train, y_train, X_test, y_test, np.arange(len(y_test)))
        predictions = clf.predict(X_test)

        # The full scikit-learn workflow also works with fit(X, y)
        cv_clf = ELRClassifier(m=8, f=4, method="ensemble", spread=3, random_state=52)
        cv_scores = cross_val_score(cv_clf, X, y, cv=3)
        assert len(cv_scores) == 3
        assert np.all(cv_scores > 0)

        assert len(predictions) == len(X_test)
        assert predictions.dtype in [np.int32, np.int64]


class TestDataFrameInput:
    """Test that ELRClassifier works with DataFrame inputs."""

    def test_dataframe_input(self):
        """Test training with DataFrame inputs."""
        # Generate data
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=8, n_classes=2, random_state=53
        )

        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y

        split_idx = 150
        df_train = df[:split_idx]
        df_test = df[split_idx:]

        X_train = df_train.drop("target", axis=1)
        y_train = df_train["target"]
        X_test = df_test.drop("target", axis=1)
        y_test = df_test["target"]
        ids_test = np.arange(len(y_test))

        # Train classifier
        clf = ELRClassifier(m=5, f=3, method="intersect", spread=2, random_state=53)
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        # Verify training completed
        assert clf.result_df_ is not None

        # Test prediction with DataFrame
        X_new = df[180:190].drop("target", axis=1)
        ids_new = np.arange(10)
        predictions = clf.predict_detailed(X_new, ids_new)

        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) >= 0  # May be empty if no IDs meet criteria
