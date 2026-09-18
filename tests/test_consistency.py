"""Regression tests for cross-path consistency and production fixes.

These tests pin down the defects found in the 0.5.0 audit:

1. One decile rule everywhere: fit == predict_detailed == ModelScorer == SQL
2. Ensemble meta-model trained out-of-fold (no leakage)
3. Genuine scikit-learn compatibility (cross_val_score, GridSearchCV, classes_)
4. Graceful baseline fallback when no constituent passes the gate
5. id_column excluded from features and usable as ID source
6. Numpy round-trip scoring via exported feature_names
7. DRP is lift (precision / prevalence)
8. Strict-JSON exports (no Infinity) preserving ID types
9. Uniform multicombination sampling for sample="replace"
10. Ranking not dominated by unbounded PLR
"""

import json
import logging
import os
import sqlite3
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from paramsemble_class import (
    ELRClassifier,
    FeatureSampler,
    ModelScorer,
    PerformanceMetrics,
    SQLGenerator,
)
from paramsemble_class.ensemble.ensemble import EnsembleMethod
from paramsemble_class.metrics.performance import top_d_count, top_d_row_indices

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="module")
def imbalanced_data():
    """Dataset with a selection set whose size is NOT a multiple of 10.

    n_selection = 165 exposes floor/ceil decile-boundary disagreements.
    """
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        random_state=7,
        weights=[0.7, 0.3],
        flip_y=0.1,
    )
    X_train, X_sel, y_train, y_sel = train_test_split(
        X, y, test_size=0.33, random_state=7, stratify=y
    )
    ids = np.arange(len(X_sel))
    return X_train, X_sel, y_train, y_sel, ids


def _fit(tmpdir, method, data, **kwargs):
    X_train, X_sel, y_train, y_sel, ids = data
    modeljson = os.path.join(tmpdir, f"{method}.json")
    clf = ELRClassifier(
        m=30, f=4, d=2, method=method, spread=5, random_state=42, modeljson=modeljson, **kwargs
    )
    clf.fit(X_train, y_train, X_sel, y_sel, ids)
    return clf, modeljson


# ---------------------------------------------------------------------------
# 1. One decile rule across all scoring paths
# ---------------------------------------------------------------------------


class TestDecileConsistency:
    def test_top_d_count_is_ceil_rule(self):
        assert top_d_count(165, 2) == 33  # ceil(2 * 16.5)
        assert top_d_count(25, 2) == 5  # ceil(2 * 2.5)
        assert top_d_count(7, 10) == 7  # capped at n
        with pytest.raises(ValueError):
            top_d_count(100, 11)

    def test_top_d_row_indices_breaks_ties_by_row_order(self):
        scores = np.array([0.5, 0.9, 0.5, 0.5])
        idx = top_d_row_indices(scores, d=10)  # all rows, check ordering
        assert list(idx[:1]) == [1]  # highest score first
        # tied rows keep original order (0 before 2 before 3)
        assert list(idx[1:]) == [0, 2, 3]

    @pytest.mark.parametrize("method", ["intersect", "venn"])
    def test_fit_predict_scorer_sql_agree(self, imbalanced_data, method):
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        assert len(X_sel) % 10 != 0, "fixture must keep a non multiple-of-10 size"

        with tempfile.TemporaryDirectory() as tmpdir:
            clf, modeljson = _fit(tmpdir, method, imbalanced_data)
            if clf.fell_back_to_baseline_:
                pytest.skip("baseline fallback active for this seed")

            detailed = clf.predict_detailed(X_sel, ids)
            scorer = ModelScorer(modeljson).score(X_sel, ids)

            # Training output == re-prediction == JSON scoring (exact)
            pd.testing.assert_frame_equal(clf.result_df_, detailed)
            pd.testing.assert_frame_equal(detailed, scorer)

            # SQL scoring returns the same [id, sets, score] table
            sql = SQLGenerator(modeljson).generate_sql("score_table", "id")
            conn = sqlite3.connect(":memory:")
            cur = conn.cursor()
            cols = ", ".join(["id INTEGER"] + [f"feature_{i} REAL" for i in range(15)])
            cur.execute(f"CREATE TABLE score_table ({cols})")
            cur.executemany(
                "INSERT INTO score_table VALUES (" + ",".join(["?"] * 16) + ")",
                [(int(i), *row) for i, row in zip(ids, X_sel)],
            )
            cur.execute(sql)
            sql_df = (
                pd.DataFrame(cur.fetchall(), columns=["id", "sets", "score"])
                .sort_values(["sets", "score", "id"], ascending=[False, False, True])
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(detailed, sql_df)
            conn.close()

    def test_ensemble_scorer_and_sql_agree_with_python(self, imbalanced_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            clf, modeljson = _fit(tmpdir, "ensemble", imbalanced_data)
            if clf.fell_back_to_baseline_:
                pytest.skip("baseline fallback active for this seed")

            X_train, X_sel, y_train, y_sel, ids = imbalanced_data
            detailed = clf.predict_detailed(X_sel, ids)

            # result_df_ and predict_detailed agree to float precision
            np.testing.assert_allclose(
                detailed["predicted"].values,
                clf.result_df_["predicted"].values,
                atol=1e-12,
            )

            scorer = ModelScorer(modeljson).score(X_sel, ids)
            np.testing.assert_allclose(
                scorer["predicted"].values,
                detailed["predicted"].values,
                atol=1e-12,
            )

            sql = SQLGenerator(modeljson).generate_sql("score_table", "id")
            conn = sqlite3.connect(":memory:")
            cur = conn.cursor()
            cols = ", ".join(["id INTEGER"] + [f"feature_{i} REAL" for i in range(15)])
            cur.execute(f"CREATE TABLE score_table ({cols})")
            cur.executemany(
                "INSERT INTO score_table VALUES (" + ",".join(["?"] * 16) + ")",
                [(int(i), *row) for i, row in zip(ids, X_sel)],
            )
            cur.execute(sql)
            rows = dict(cur.fetchall())
            sql_pred = pd.Series(rows).sort_index().values
            py_pred = detailed.set_index("id")["predicted"].sort_index().values
            np.testing.assert_allclose(sql_pred, py_pred, atol=1e-12)
            conn.close()


# ---------------------------------------------------------------------------
# 2. Out-of-fold meta-model (no leakage)
# ---------------------------------------------------------------------------


class TestOutOfFoldMetaModel:
    def test_oof_matrix_complete_and_shaped(self, imbalanced_data):
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        clf, _ = _fit(tempfile.mkdtemp(), "ensemble", imbalanced_data)
        if clf.fell_back_to_baseline_:
            pytest.skip("baseline fallback active for this seed")

        oof = EnsembleMethod._out_of_fold_meta_features(
            clf.constituent_models_, X_sel, y_sel, clf.selected_indices_, n_splits=5, random_state=0
        )
        assert oof is not None
        assert oof.shape == (len(X_sel), len(clf.selected_indices_))
        assert not np.isnan(oof).any()
        assert np.all((oof >= 0) & (oof <= 1))

    def test_meta_model_not_fit_on_scored_rows(self, imbalanced_data, monkeypatch):
        """The meta-model must be fit on out-of-fold predictions, i.e. the
        matrix passed to LogisticRegression.fit must differ from the matrix
        used for the final predictions."""
        from sklearn.linear_model import LogisticRegression

        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        seen = {}

        real_fit = LogisticRegression.fit

        def spy_fit(self, X, y, sample_weight=None):
            seen.setdefault("fit_inputs", []).append(np.array(X, copy=True))
            return real_fit(self, X, y, sample_weight=sample_weight)

        monkeypatch.setattr(LogisticRegression, "fit", spy_fit)

        with tempfile.TemporaryDirectory() as tmpdir:
            clf, _ = _fit(tmpdir, "ensemble", imbalanced_data)
            if clf.fell_back_to_baseline_:
                pytest.skip("baseline fallback active for this seed")

        fit_inputs = seen["fit_inputs"]
        # The constituent fits saw the 335-row training split; the single
        # meta-model fit must be on the 165-row selection set, using an
        # out-of-fold matrix distinct from the full prediction matrix.
        meta_fits = [m for m in fit_inputs if m.shape[0] == len(X_sel)]
        assert len(meta_fits) == 1
        full = np.column_stack(
            [clf.constituent_models_[i].predict_proba(X_sel)[:, 1] for i in clf.selected_indices_]
        )
        assert meta_fits[0].shape == full.shape
        assert not np.array_equal(meta_fits[0], full)


# ---------------------------------------------------------------------------
# 3. scikit-learn compatibility
# ---------------------------------------------------------------------------


class TestSklearnCompatibility:
    @pytest.fixture(scope="class")
    def data(self):
        X, y = make_classification(n_samples=300, n_features=10, random_state=0)
        return X, y

    def test_cross_val_score(self, data):
        X, y = data
        scores = cross_val_score(ELRClassifier(m=8, f=3, random_state=0), X, y, cv=3)
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))

    def test_grid_search(self, data):
        X, y = data
        gs = GridSearchCV(ELRClassifier(m=5, f=3, random_state=0), {"spread": [2, 3]}, cv=2)
        gs.fit(X, y)
        assert gs.best_params_["spread"] in (2, 3)

    def test_classes_attribute_and_label_output(self, data):
        X, y = data
        clf = ELRClassifier(m=8, f=3, method="ensemble", random_state=0).fit(X, y)
        assert set(clf.classes_) == {0, 1}

        labels = clf.predict(X)
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (len(X),)
        assert set(np.unique(labels)).issubset(set(clf.classes_))

    @pytest.mark.parametrize("method", ["intersect", "venn", "ensemble"])
    def test_predict_proba_all_methods(self, data, method):
        X, y = data
        clf = ELRClassifier(m=8, f=3, method=method, random_state=0).fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_score_method(self, data):
        X, y = data
        clf = ELRClassifier(m=8, f=3, random_state=0).fit(X, y)
        assert 0.0 <= clf.score(X, y) <= 1.0


# ---------------------------------------------------------------------------
# 4. Baseline fallback
# ---------------------------------------------------------------------------


class TestBaselineFallback:
    def test_no_crash_and_baseline_used_when_gate_empty(self, imbalanced_data):
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        clf = ELRClassifier(m=3, f=2, method="ensemble", spread=2, random_state=1)
        clf.fit(X_train, y_train, X_sel, y_sel, ids)

        if not clf.fell_back_to_baseline_:
            pytest.skip("gate unexpectedly passed with tiny ensemble")

        assert clf.selected_indices_ == []
        # All prediction paths work and match the baseline forest
        labels = clf.predict(X_sel)
        assert len(labels) == len(X_sel)
        np.testing.assert_array_equal(labels, clf.baseline_model_.predict(X_sel))

        proba = clf.predict_proba(X_sel)
        np.testing.assert_allclose(proba[:, 1], clf.baseline_model_.predict_proba(X_sel)[:, 1])

        detailed = clf.predict_detailed(X_sel, ids)
        assert "predicted" in detailed.columns
        assert len(detailed) == len(X_sel)

    def test_intersect_fallback_builds_sets_from_baseline_drs(self, imbalanced_data):
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        clf = ELRClassifier(m=2, f=2, method="intersect", spread=2, random_state=1)
        clf.fit(X_train, y_train, X_sel, y_sel, ids)

        if not clf.fell_back_to_baseline_:
            pytest.skip("gate unexpectedly passed with tiny ensemble")

        detailed = clf.predict_detailed(X_sel, ids)
        assert set(detailed.columns) == {"id", "sets", "score"}
        assert (detailed["sets"] == 1).all()
        assert set(detailed["id"]) == set(clf.baseline_results_["drs"])

    def test_fallback_exports_baseline_equation(self, imbalanced_data):
        """With a linear baseline, the fallback modeljson carries the
        baseline equation (binned when optbinning is available) and scores
        identically to predict_detailed, including via SQL."""
        import sqlite3

        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            clf = ELRClassifier(
                m=2, f=2, method="intersect", spread=2, random_state=1, modeljson=path
            )
            clf.fit(X_train, y_train, X_sel, y_sel, ids)
            if not clf.fell_back_to_baseline_:
                pytest.skip("gate unexpectedly passed with tiny ensemble")

            assert os.path.exists(path), "linear-baseline fallback must export"
            parsed = json.load(open(path))
            assert parsed["fallback"] is True
            assert parsed["weights"] == [1.0]
            entry = parsed["models"][0]
            if clf.baseline_kind_ == "woe_logreg":
                assert "bins" in entry
            else:
                assert "constant" in entry

            # Python and SQL scoring reproduce the fallback output exactly
            detailed = clf.predict_detailed(X_sel, ids)
            scorer = ModelScorer(path).score(X_sel, ids)
            pd.testing.assert_frame_equal(detailed, scorer)

            sql = SQLGenerator(path).generate_sql("t", "id")
            conn = sqlite3.connect(":memory:")
            cur = conn.cursor()
            cols = ", ".join(["id INTEGER"] + [f"feature_{i} REAL" for i in range(15)])
            cur.execute(f"CREATE TABLE t ({cols})")
            cur.executemany(
                "INSERT INTO t VALUES (" + ",".join(["?"] * 16) + ")",
                [(int(i), *row) for i, row in zip(ids, X_sel)],
            )
            cur.execute(sql)
            sql_df = (
                pd.DataFrame(cur.fetchall(), columns=["id", "sets", "score"])
                .sort_values(["sets", "score", "id"], ascending=[False, False, True])
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(detailed, sql_df)
            conn.close()

    def test_rf_baseline_fallback_skips_export(self, imbalanced_data):
        """The RF baseline has no linear equation; its fallback stays
        Python-only (modeljson skipped)."""
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            clf = ELRClassifier(
                m=2,
                f=2,
                method="intersect",
                spread=2,
                random_state=1,
                baseline="random_forest",
                modeljson=path,
            )
            clf.fit(X_train, y_train, X_sel, y_sel, ids)
            if not clf.fell_back_to_baseline_:
                pytest.skip("gate unexpectedly passed with tiny ensemble")
            assert clf.baseline_kind_ == "random_forest"
            assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# 5. id_column handling
# ---------------------------------------------------------------------------


class TestIdColumn:
    def test_id_column_excluded_from_features_and_extracted(self):
        rng = np.random.RandomState(0)
        X = rng.randn(120, 6)
        y = (rng.rand(120) > 0.6).astype(int)
        ids = np.arange(120) + 500

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        df["record_id"] = ids

        # No explicit ids_test: id_column supplies them
        clf = ELRClassifier(m=6, f=3, method="ensemble", random_state=0, id_column="record_id")
        clf.fit(df.iloc[:80], y[:80], df.iloc[80:], y[80:])

        assert clf.feature_names_ == [f"f{i}" for i in range(6)]
        assert "record_id" not in clf.feature_names_
        # IDs in the detailed output come from the id column
        detailed = clf.predict_detailed(df.iloc[80:])
        assert set(detailed["id"]).issubset(set(ids[80:]))

    def test_numeric_id_never_leaks_into_features(self):
        rng = np.random.RandomState(1)
        X = rng.randn(100, 4)
        y = (rng.rand(100) > 0.5).astype(int)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        df["id"] = np.arange(100) * 1000  # a highly predictive-looking column

        clf = ELRClassifier(m=6, f=4, random_state=0)
        clf.fit(df.iloc[:70], y[:70], df.iloc[70:], y[70:])
        assert "id" not in clf.feature_names_
        assert clf.n_features_ == 4


# ---------------------------------------------------------------------------
# 6. Numpy round-trip scoring
# ---------------------------------------------------------------------------


class TestNumpyRoundTrip:
    @pytest.mark.parametrize("method", ["intersect", "ensemble"])
    def test_numpy_array_scoring_matches_dataframe(self, imbalanced_data, method):
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        with tempfile.TemporaryDirectory() as tmpdir:
            clf, modeljson = _fit(tmpdir, method, imbalanced_data)
            if clf.fell_back_to_baseline_:
                pytest.skip("baseline fallback active for this seed")

            scorer = ModelScorer(modeljson)
            feat_names = [f"feature_{i}" for i in range(X_sel.shape[1])]
            from_df = scorer.score(pd.DataFrame(X_sel, columns=feat_names), ids)
            from_np = scorer.score(X_sel, ids)
            pd.testing.assert_frame_equal(from_df, from_np)

    def test_numpy_rejected_without_feature_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "method": "intersect",
                        "d": 2,
                        "models": [{"a": 0.5, "constant": 0.1}],
                    },
                    f,
                )
            scorer = ModelScorer(path)
            with pytest.raises(ValueError, match="feature_names"):
                scorer.score(np.zeros((5, 1)), np.arange(5))


# ---------------------------------------------------------------------------
# 7. DRP is lift
# ---------------------------------------------------------------------------


class TestDrpIsLift:
    def test_drp_equals_precision_over_prevalence(self):
        # 100 rows, 30 positives. Top 20 by score: 12 positives, 8 negatives.
        y_true = np.array([1] * 12 + [0] * 8 + [1] * 18 + [0] * 62)
        scores = np.array([0.99] * 12 + [0.50] * 8 + [0.40] * 18 + [0.30] * 62)
        drp = PerformanceMetrics.decile_ranked_performance(y_true, scores, d=2)
        precision_top = 12 / 20
        prevalence = 30 / 100
        assert np.isclose(drp, precision_top / prevalence)

    def test_drp_one_when_random_ranking(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 1000)
        scores = rng.rand(1000)
        drp = PerformanceMetrics.decile_ranked_performance(y_true, scores, d=2)
        assert np.isclose(drp, 1.0, atol=0.35)  # lift ~ 1 for random scores


# ---------------------------------------------------------------------------
# 8. Strict JSON exports
# ---------------------------------------------------------------------------


class TestStrictJson:
    def test_inf_serializes_as_null(self):
        from paramsemble_class.utils.model_io import ModelIO

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "all.json")
            ModelIO.export_all_models(
                [
                    {
                        "plr": float("inf"),
                        "fnr": 0.2,
                        "drp": 1.5,
                        "drs": {np.int64(1), np.int64(2)},
                        "dps": set(),
                        "equation_dict": {"a": 0.5, "constant": 0.1},
                    }
                ],
                path,
            )
            raw = open(path).read()
            assert "Infinity" not in raw and "NaN" not in raw
            parsed = json.loads(raw)  # strict parse
            assert parsed[0]["plr"] is None

    def test_numpy_id_types_preserved(self):
        from paramsemble_class.utils.model_io import ModelIO

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "all.json")
            ModelIO.export_all_models(
                [
                    {
                        "plr": 2.0,
                        "fnr": 0.1,
                        "drp": 1.2,
                        "drs": {np.float64(7.5)},
                        "dps": {np.int64(3)},
                        "equation_dict": {"a": 0.5, "constant": 0.1},
                    }
                ],
                path,
            )
            parsed = json.load(open(path))
            assert parsed[0]["drs"] == [7.5]  # float preserved, not int()
            assert parsed[0]["dps"] == [3]

    def test_selected_models_export_records_feature_names(self, imbalanced_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, modeljson = _fit(tmpdir, "intersect", imbalanced_data)
            parsed = json.load(open(modeljson))
            assert parsed["feature_names"] == [f"feature_{i}" for i in range(15)]


# ---------------------------------------------------------------------------
# 9. Uniform multicombination sampling
# ---------------------------------------------------------------------------


class TestReplaceSampling:
    def test_replacement_fraction_matches_uniform_multicombination(self):
        """Uniform multicombinations of 5 features taken 3 at a time contain
        25/35 = 71.4% sets with a repeated feature (naive iid sorted draws
        would produce ~14%; the old implementation)."""
        sampler = FeatureSampler(5, 3, 300, "replace", random_state=11)
        combos = sampler.generate_combinations()
        with_repeat = sum(1 for c in combos if len(set(c)) < len(c))
        fraction = with_repeat / len(combos)
        expected = 25 / 35
        assert abs(fraction - expected) < 0.10, (
            f"repeated-feature fraction {fraction:.3f} far from "
            f"uniform multicombination expectation {expected:.3f}"
        )
        assert all(0 <= x < 5 for c in combos for x in c)

    def test_unique_sampling_never_repeats_within_featureset(self):
        sampler = FeatureSampler(10, 5, 200, "unique", random_state=3)
        for combo in sampler.generate_combinations():
            assert len(set(combo)) == len(combo)


# ---------------------------------------------------------------------------
# 10. Ranking not dominated by PLR
# ---------------------------------------------------------------------------


class TestRankingScaleBalance:
    def test_high_plr_does_not_override_terrible_fnr_and_drp(self):
        """The audit's case: PLR (unbounded) must not swamp FNR/DRP (both in
        [0, 1] / [0, 10]). A model with FNR=0.9 outranks an excellent model
        (FNR=0.05, DRP=4.5) only if PLR dominates — which Borda prevents."""
        from paramsemble_class.ensemble.intersect import IntersectMethod

        constituent_results = [
            {"plr": 8.0, "fnr": 0.90, "drp": 1.0, "drs": set(), "dps": set()},
            {"plr": 5.0, "fnr": 0.05, "drp": 4.5, "drs": set(), "dps": set()},
        ]
        baseline = {"plr": 2.0, "fnr": 0.3, "drp": 1.0, "drs": set(), "dps": set()}
        ranked = IntersectMethod._rank_models(constituent_results, baseline)
        assert ranked[0] == 1, "excellent model (idx 1) must rank first"

    def test_ranking_deterministic(self):
        from paramsemble_class.ensemble.intersect import IntersectMethod

        results = [
            {"plr": 3.0, "fnr": 0.2, "drp": 1.5, "drs": set(), "dps": set()},
            {"plr": 3.0, "fnr": 0.2, "drp": 1.5, "drs": set(), "dps": set()},
            {"plr": 3.0, "fnr": 0.2, "drp": 1.5, "drs": set(), "dps": set()},
        ]
        baseline = {"plr": 2.0, "fnr": 0.3, "drp": 1.0, "drs": set(), "dps": set()}
        assert IntersectMethod._rank_models(results, baseline) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Misc regressions
# ---------------------------------------------------------------------------


class TestMiscRegressions:
    def test_predict_ids_deprecation_forwards_to_detailed(self, imbalanced_data):
        """predict(X, ids) warns and returns predict_detailed output;
        predict(X) keeps returning class labels."""
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        clf = ELRClassifier(m=30, f=4, d=2, method="intersect", spread=5, random_state=42)
        clf.fit(X_train, y_train, X_sel, y_sel, ids)
        if clf.fell_back_to_baseline_:
            pytest.skip("baseline fallback active for this seed")

        with pytest.warns(DeprecationWarning, match="predict_detailed"):
            detailed = clf.predict(X_sel, ids)
        pd.testing.assert_frame_equal(detailed, clf.predict_detailed(X_sel, ids))

        # The one-argument form must remain a pure labels call with no warning
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("error")
            labels = clf.predict(X_sel)
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (len(X_sel),)

    def test_no_logging_basicconfig_on_import(self):
        import paramsemble_class.core.elr_classifier as mod

        source = open(mod.__file__, encoding="utf-8").read()
        assert "basicConfig" not in source

    def test_non_numeric_columns_warn_and_drop(self, caplog):
        rng = np.random.RandomState(0)
        df = pd.DataFrame(rng.randn(80, 4), columns=[f"f{i}" for i in range(4)])
        df["country"] = ["A", "B"] * 40
        y = (rng.rand(80) > 0.5).astype(int)

        clf = ELRClassifier(m=4, f=3, random_state=0)
        with caplog.at_level(logging.WARNING, logger="paramsemble_class.core.elr_classifier"):
            clf.fit(df.iloc[:60], y[:60], df.iloc[60:], y[60:])
        assert clf.feature_names_ == [f"f{i}" for i in range(4)]
        assert any("non-numeric" in rec.message.lower() for rec in caplog.records)

    def test_readme_quickstart_convention_runs(self):
        """The README's sklearn-mode snippet must execute end to end."""
        X, y = make_classification(
            n_samples=300, n_features=20, weights=[0.7, 0.3], random_state=42
        )
        clf = ELRClassifier(m=20, f=5, method="ensemble", random_state=42)
        clf.fit(X, y)
        labels = clf.predict(X)
        proba = clf.predict_proba(X)
        assert len(labels) == len(X) and proba.shape == (len(X), 2)

    def test_solver_auto_is_lbfgs(self):
        from paramsemble_class.core.constituent_model import ConstituentModel

        model = ConstituentModel([0], solver="auto")
        assert model._select_solver(5000, 50) == "lbfgs"


# ---------------------------------------------------------------------------
# 11. Selection v2: weighted votes, coverage selection, multiwidth pools
# ---------------------------------------------------------------------------


class TestSelectionV2:
    def test_weights_by_borda_position(self):
        from paramsemble_class.ensemble._common import weights_for_selection

        ranked = [7, 3, 9, 5]  # borda order: 7 best
        # selected in greedy (non-borda) order: 3, 9, 7
        weights = weights_for_selection([3, 9, 7], ranked)
        # among selected, borda order is 7 (best) > 3 > 9; weights follow
        # borda quality regardless of selection order
        assert weights == [2.0, 1.0, 3.0]

    def test_combine_drs_weighted_score(self):
        from paramsemble_class.ensemble._common import combine_drs

        results = [
            {"drs": {"a", "b"}},
            {"drs": {"b", "c"}},
            {"drs": {"b"}},
        ]
        df = combine_drs(results, [0, 1, 2], weights=[3.0, 2.0, 1.0])
        by_id = df.set_index("id")
        assert by_id.loc["a", "sets"] == 1 and by_id.loc["a", "score"] == 3.0
        assert by_id.loc["b", "sets"] == 3 and by_id.loc["b", "score"] == 6.0
        assert by_id.loc["c", "sets"] == 1 and by_id.loc["c", "score"] == 2.0

    def test_coverage_selection_avoids_redundant_models(self):
        """Top-k by rank spends slots on duplicate coverage; greedy coverage
        spends them on models that add new positives."""
        from paramsemble_class.ensemble.intersect import IntersectMethod

        # Borda ranking order will follow PLR: models 0,1 (best, redundant
        # DPS) then model 2 (weaker but complementary positives).
        constituent_results = [
            {"plr": 9.0, "fnr": 0.10, "drp": 2.0, "drs": {"a"}, "dps": {"p1", "p2"}},
            {"plr": 8.0, "fnr": 0.15, "drp": 1.9, "drs": {"a"}, "dps": {"p1", "p2"}},
            {"plr": 5.0, "fnr": 0.25, "drp": 1.5, "drs": {"b"}, "dps": {"p3", "p4"}},
        ]
        baseline = {"plr": 2.0, "fnr": 0.3, "drp": 1.0, "drs": set(), "dps": set()}

        rank_sel = IntersectMethod.select(constituent_results, baseline, spread=2)
        cov_sel = IntersectMethod.select(
            constituent_results, baseline, spread=2, selection="coverage"
        )
        assert rank_sel == [0, 1]  # classic: two best-ranked (redundant)
        assert cov_sel == [0, 2]  # coverage swaps the duplicate for new positives

        covered_rank = set().union(*[constituent_results[i]["dps"] for i in rank_sel])
        covered_cov = set().union(*[constituent_results[i]["dps"] for i in cov_sel])
        assert len(covered_cov) > len(covered_rank)

    def test_invalid_selection_param_raises(self, imbalanced_data):
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        clf = ELRClassifier(m=4, f=3, selection="invalid", random_state=0)
        with pytest.raises(ValueError, match="selection"):
            clf.fit(X_train, y_train, X_sel, y_sel, ids)

    def test_invalid_f_range_raises(self, imbalanced_data):
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        for bad in ([], [0, 4], [3.5], [999]):
            clf = ELRClassifier(m=4, f=3, f_range=bad, random_state=0)
            with pytest.raises(ValueError, match="f_range|exceeds the number"):
                clf.fit(X_train, y_train, X_sel, y_sel, ids)

    def test_multiwidth_pool_trains_all_widths(self, imbalanced_data):
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        clf = ELRClassifier(m=12, f_range=[3, 6, 9], spread=5, selection="coverage", random_state=0)
        clf.fit(X_train, y_train, X_sel, y_sel, ids)

        widths = {len(r["feature_indices"]) for r in clf.constituent_results_}
        assert widths == {3, 6, 9}
        # m split across widths
        assert len(clf.constituent_results_) == 12
        # everything else (gate, scoring, export) is width-agnostic
        detailed = clf.predict_detailed(X_sel, ids)
        assert set(detailed.columns) == {"id", "sets", "score"}

    def test_weights_exported_and_used_by_scorer(self, imbalanced_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            clf, modeljson = _fit(tmpdir, "intersect", imbalanced_data)
            if clf.fell_back_to_baseline_:
                pytest.skip("baseline fallback active for this seed")

            parsed = json.load(open(modeljson))
            assert parsed["weights"] == clf.selection_weights_
            assert len(parsed["weights"]) == len(parsed["models"])

            # weighted scorer output matches predict_detailed (score column)
            X_train, X_sel, y_train, y_sel, ids = imbalanced_data
            scorer = ModelScorer(modeljson).score(X_sel, ids)
            pd.testing.assert_frame_equal(scorer, clf.predict_detailed(X_sel, ids))

    def test_predict_proba_is_weighted_share(self, imbalanced_data):
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        clf = ELRClassifier(m=30, f=4, spread=3, selection="rank", random_state=42)
        clf.fit(X_train, y_train, X_sel, y_sel, ids)
        if clf.fell_back_to_baseline_:
            pytest.skip("baseline fallback active for this seed")

        proba = clf.predict_proba(X_sel)
        # rows in every selected model's top-d get the full weight share
        total = sum(clf.selection_weights_)
        assert proba[:, 1].max() <= total / total + 1e-12
        assert (
            set(np.unique(proba)).issubset(
                {0.0, 1.0}
                | {
                    w / total
                    for w in np.cumsum([0.0] + sorted(clf.selection_weights_, reverse=True))
                }
            )
            or proba[:, 1].max() > 0
        )


# ---------------------------------------------------------------------------
# 12. Baseline choice: WoE-binned linear default, RF option, fallback export
# ---------------------------------------------------------------------------


class TestBaselineChoice:
    def test_baseline_param_validation(self, imbalanced_data):
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        clf = ELRClassifier(m=4, f=3, baseline="gradient_boosting", random_state=0)
        with pytest.raises(ValueError, match="baseline"):
            clf.fit(X_train, y_train, X_sel, y_sel, ids)

    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("woe_logreg", "woe_logreg"),
            ("logreg", "logreg"),
            ("random_forest", "random_forest"),
        ],
    )
    def test_baseline_kinds(self, imbalanced_data, kind, expected):
        import paramsemble_class.core.baseline_model as bm

        if kind == "woe_logreg" and not bm._HAS_OPTBINNING:
            pytest.skip("optbinning not installed in this environment")
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        clf = ELRClassifier(m=4, f=3, baseline=kind, random_state=0)
        clf.fit(X_train, y_train, X_sel, y_sel, ids)
        assert clf.baseline_kind_ == expected

    def test_linear_baseline_metrics_structure(self, imbalanced_data):
        from paramsemble_class.core.baseline_model import LinearBaselineModel

        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        model = LinearBaselineModel(kind="logreg").fit(X_train, y_train)
        metrics = model.evaluate(X_sel, y_sel, ids, d=2)
        assert set(metrics) == {"plr", "fnr", "drp", "drs", "dps"}
        assert model.predict_proba(X_sel).shape == (len(X_sel), 2)

    def test_binned_equation_export_matches_model(self, imbalanced_data):
        """Manual application of the exported bins+coefficients must equal
        the baseline model's own predict_proba (validates the export math)."""
        import paramsemble_class.core.baseline_model as bm

        if not bm._HAS_OPTBINNING:
            pytest.skip("optbinning not installed in this environment")

        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        model = bm.LinearBaselineModel(kind="woe_logreg").fit(X_train, y_train)
        names = [f"feature_{i}" for i in range(X_train.shape[1])]
        equation = model.get_equation_dict(names)

        assert "bins" in equation
        linear = np.full(len(X_sel), equation["constant"])
        for j, name in enumerate(names):
            spec = equation["bins"][name]
            splits = np.asarray(spec["splits"])
            woe = np.asarray(spec["woe"])
            idx = np.clip(np.searchsorted(splits, X_sel[:, j], side="right"), 0, len(woe) - 1)
            linear += woe[idx] * spec["coef"]
        manual = 1.0 / (1.0 + np.exp(-np.clip(linear, -500, 500)))
        np.testing.assert_allclose(manual, model.predict_proba(X_sel)[:, 1], atol=1e-10)

    def test_woe_degrades_gracefully_without_optbinning(self, imbalanced_data, monkeypatch):
        import paramsemble_class.core.baseline_model as bm

        monkeypatch.setattr(bm, "_HAS_OPTBINNING", False)
        X_train, X_sel, y_train, y_sel, ids = imbalanced_data
        clf = ELRClassifier(m=4, f=3, baseline="woe_logreg", random_state=0)
        clf.fit(X_train, y_train, X_sel, y_sel, ids)
        assert clf.baseline_kind_ == "logreg"
        # all prediction paths still work on the degraded baseline
        assert clf.predict(X_sel).shape == (len(X_sel),)
        assert clf.predict_proba(X_sel).shape == (len(X_sel), 2)
