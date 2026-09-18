"""Unit tests for SQL generator module."""

import json
import os
import tempfile
import sqlite3
import pytest

from tests.conftest import requires_sqlite_math
from paramsemble_class.sql.generator import SQLGenerator


class TestSQLGenerator:
    """Unit tests for SQLGenerator class."""

    def test_sql_generation_for_intersect_method(self):
        """Test SQL generation for intersect method with sample model JSON."""
        # Create sample model JSON
        config = {
            "method": "intersect",
            "d": 2,
            "models": [
                {"feature1": 0.5, "feature2": -0.3, "constant": 0.1},
                {"feature1": 0.7, "feature3": 0.2, "constant": -0.2},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            json_path = f.name

        try:
            generator = SQLGenerator(json_path)
            sql = generator.generate_sql("customers", "customer_id")

            # Verify SQL structure
            assert isinstance(sql, str)
            assert len(sql) > 0
            assert "WITH" in sql
            assert "model_0" in sql
            assert "model_1" in sql
            assert "ROW_NUMBER()" in sql
            assert "CEIL(2 *" in sql
            assert "row_num <= cutoff" in sql
            assert "COUNT(*)" in sql
            assert "GROUP BY" in sql

        finally:
            os.unlink(json_path)

    def test_sql_generation_for_venn_method(self):
        """Test SQL generation for venn method with sample model JSON."""
        # Create sample model JSON
        config = {
            "method": "venn",
            "d": 3,
            "models": [
                {"feature1": 0.4, "feature2": 0.6, "constant": 0.0},
                {"feature2": 0.8, "feature3": -0.5, "constant": 0.3},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            json_path = f.name

        try:
            generator = SQLGenerator(json_path)
            sql = generator.generate_sql("users", "user_id")

            # Verify SQL structure (venn uses same structure as intersect)
            assert isinstance(sql, str)
            assert len(sql) > 0
            assert "WITH" in sql
            assert "model_0" in sql
            assert "model_1" in sql
            assert "ROW_NUMBER()" in sql
            assert "CEIL(3 *" in sql
            assert "row_num <= cutoff" in sql
            assert "COUNT(*)" in sql

        finally:
            os.unlink(json_path)

    def test_sql_generation_for_ensemble_method(self):
        """Test SQL generation for ensemble method with sample model JSON."""
        # Create sample model JSON
        config = {
            "method": "ensemble",
            "d": 2,
            "models": [
                {"feature1": 0.5, "feature2": -0.3, "constant": 0.1},
                {"feature1": 0.7, "feature3": 0.2, "constant": -0.2},
            ],
            "meta_model": {"model_0_prob": 0.6, "model_1_prob": 0.4, "constant": -0.1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            json_path = f.name

        try:
            generator = SQLGenerator(json_path)
            sql = generator.generate_sql("transactions", "transaction_id")

            # Verify SQL structure
            assert isinstance(sql, str)
            assert len(sql) > 0
            assert "WITH" in sql
            assert "model_0" in sql
            assert "model_1" in sql
            assert "constituent_predictions" in sql
            assert "model_0_prob" in sql
            assert "model_1_prob" in sql
            assert "predicted" in sql

        finally:
            os.unlink(json_path)

    @requires_sqlite_math
    def test_generated_sql_can_be_parsed(self):
        """Test that generated SQL can be parsed (syntax check)."""
        # Create sample model JSON
        config = {
            "method": "intersect",
            "d": 2,
            "models": [{"feature1": 0.5, "feature2": -0.3, "constant": 0.1}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            json_path = f.name

        try:
            generator = SQLGenerator(json_path)
            sql = generator.generate_sql("test_table", "id")

            # Create in-memory SQLite database
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()

            # Create test table
            cursor.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    feature1 REAL,
                    feature2 REAL
                )
            """)

            # Insert test data
            cursor.execute("INSERT INTO test_table VALUES (1, 0.5, 0.3)")
            cursor.execute("INSERT INTO test_table VALUES (2, 0.7, 0.1)")
            conn.commit()

            # Try to execute the generated SQL
            # This will raise an exception if SQL is invalid
            cursor.execute(sql)
            results = cursor.fetchall()

            # Verify we got results
            assert results is not None

            conn.close()

        finally:
            os.unlink(json_path)

    @requires_sqlite_math
    def test_feature_names_are_properly_escaped(self):
        """Test that feature names are properly escaped in SQL."""
        # Create model with feature names that need escaping
        config = {
            "method": "intersect",
            "d": 2,
            "models": [{"feature_1": 0.5, "feature_2": -0.3, "constant": 0.1}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            json_path = f.name

        try:
            generator = SQLGenerator(json_path)
            sql = generator.generate_sql("test_table", "id")

            # Verify feature names are quoted
            assert '"feature_1"' in sql
            assert '"feature_2"' in sql

            # Create in-memory SQLite database
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()

            # Create test table with matching column names
            cursor.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    feature_1 REAL,
                    feature_2 REAL
                )
            """)

            # Insert test data
            cursor.execute("INSERT INTO test_table VALUES (1, 0.5, 0.3)")
            conn.commit()

            # Execute the generated SQL
            cursor.execute(sql)
            results = cursor.fetchall()

            # Verify execution succeeded
            assert results is not None

            conn.close()

        finally:
            os.unlink(json_path)

    def test_invalid_method_raises_error(self):
        """Test that invalid method in JSON raises error."""
        config = {
            "method": "invalid_method",
            "d": 2,
            "models": [{"feature1": 0.5, "constant": 0.1}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            json_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid method"):
                generator = SQLGenerator(json_path)
        finally:
            os.unlink(json_path)

    def test_missing_meta_model_for_ensemble_raises_error(self):
        """Test that missing meta-model for ensemble method raises error."""
        config = {
            "method": "ensemble",
            "d": 2,
            "models": [{"feature1": 0.5, "constant": 0.1}],
            # Missing meta_model
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            json_path = f.name

        try:
            with pytest.raises(ValueError, match="meta_model"):
                generator = SQLGenerator(json_path)
                sql = generator.generate_sql("test_table", "id")
        finally:
            os.unlink(json_path)

    def test_logistic_regression_formula_in_sql(self):
        """Test that SQL contains proper logistic regression formula."""
        config = {
            "method": "intersect",
            "d": 2,
            "models": [{"feature1": 0.5, "feature2": -0.3, "constant": 0.1}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            json_path = f.name

        try:
            generator = SQLGenerator(json_path)
            sql = generator.generate_sql("test_table", "id")

            # Verify logistic regression formula components
            assert "EXP" in sql
            assert "1.0 / (1.0 + EXP" in sql or "1 / (1 + EXP" in sql
            assert "0.1" in sql  # constant
            assert "0.5" in sql  # feature1 coefficient
            assert "-0.3" in sql or "0.3" in sql  # feature2 coefficient

        finally:
            os.unlink(json_path)
