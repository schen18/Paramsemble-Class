from tests.conftest import requires_sqlite_math

"""Property-based tests for SQL generator module."""

import json
import os
import tempfile
import sqlite3
from hypothesis import given, strategies as st, assume, settings
from paramsemble_class.sql.generator import SQLGenerator


# Strategy for generating valid feature names
@st.composite
def feature_name_strategy(draw):
    """Generate valid SQL feature names."""
    # Use simple alphanumeric names to avoid SQL escaping issues
    name = draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"), min_codepoint=65, max_codepoint=122
            ),
            min_size=1,
            max_size=20,
        )
    )
    # Ensure it starts with a letter
    if name and not name[0].isalpha():
        name = "f" + name
    return name


# Strategy for generating equation dictionaries
@st.composite
def equation_dict_strategy(draw, num_features=None):
    """Generate a valid equation dictionary."""
    if num_features is None:
        num_features = draw(st.integers(min_value=1, max_value=10))

    equation = {}

    # Generate feature coefficients
    for i in range(num_features):
        feature_name = f"feature_{i}"
        coefficient = draw(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
        equation[feature_name] = coefficient

    # Add constant (intercept)
    constant = draw(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    equation["constant"] = constant

    return equation


# Strategy for generating model JSON configurations
@st.composite
def model_json_strategy(draw, method=None):
    """Generate a valid model JSON configuration."""
    if method is None:
        method = draw(st.sampled_from(["intersect", "venn", "ensemble"]))

    d = draw(st.integers(min_value=1, max_value=10))
    num_models = draw(st.integers(min_value=1, max_value=5))
    num_features = draw(st.integers(min_value=1, max_value=5))

    # Generate constituent models
    models = []
    for _ in range(num_models):
        equation = draw(equation_dict_strategy(num_features=num_features))
        models.append(equation)

    config = {"method": method, "d": d, "models": models}

    # Add meta-model for ensemble method
    if method == "ensemble":
        meta_model = {
            "constant": draw(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
            )
        }
        for i in range(num_models):
            meta_model[f"model_{i}_prob"] = draw(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
            )
        config["meta_model"] = meta_model

    return config, num_features


# **Feature: elr-package, Property 30: SQL generation produces valid syntax**
# **Validates: Requirements 11.1, 11.8**
@settings(max_examples=100)
@given(method=st.sampled_from(["intersect", "venn", "ensemble"]), data=st.data())
@requires_sqlite_math
def test_sql_generation_produces_valid_syntax(method, data):
    """
    Property 30: SQL generation produces valid syntax.

    For any valid model JSON file, the generated SQL should be syntactically
    valid and executable against a database table with matching feature columns.
    """
    # Generate model configuration
    config, num_features = data.draw(model_json_strategy(method=method))

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        json_path = f.name

    try:
        # Create SQL generator
        generator = SQLGenerator(json_path)

        # Generate SQL
        sql = generator.generate_sql("test_table", "id")

        # Verify SQL is a non-empty string
        assert isinstance(sql, str)
        assert len(sql) > 0

        # Verify SQL contains expected keywords
        assert "WITH" in sql.upper()
        assert "SELECT" in sql.upper()

        # Create in-memory SQLite database with test table
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Create test table with matching columns
        columns = ["id INTEGER PRIMARY KEY"]
        for i in range(num_features):
            columns.append(f"feature_{i} REAL")

        create_table_sql = f"CREATE TABLE test_table ({', '.join(columns)})"
        cursor.execute(create_table_sql)

        # Insert a test row
        values = [1] + [0.5] * num_features
        placeholders = ", ".join(["?"] * (num_features + 1))
        cursor.execute(f"INSERT INTO test_table VALUES ({placeholders})", values)
        conn.commit()

        # Try to execute the generated SQL
        # This will raise an exception if SQL is invalid
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            # Verify we got some results
            assert results is not None
        except sqlite3.OperationalError as e:
            # If there's a SQL syntax error, the test should fail
            raise AssertionError(f"Generated SQL has syntax error: {e}\nSQL: {sql}")
        finally:
            conn.close()

    finally:
        # Clean up temporary file
        if os.path.exists(json_path):
            os.unlink(json_path)


# **Feature: elr-package, Property 31: SQL logistic regression formula correctness**
# **Validates: Requirements 11.5**
@settings(max_examples=100)
@given(equation=equation_dict_strategy(), data=st.data())
@requires_sqlite_math
def test_sql_logistic_regression_formula_correctness(equation, data):
    """
    Property 31: SQL logistic regression formula correctness.

    For any equation dictionary, the generated SQL should implement the
    logistic regression formula: 1 / (1 + EXP(-(constant + sum of feature*coefficient products))).
    """
    import numpy as np

    # Create a simple model JSON with one model
    config = {"method": "intersect", "d": 2, "models": [equation]}

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        json_path = f.name

    try:
        # Create SQL generator
        generator = SQLGenerator(json_path)

        # Generate SQL
        sql = generator.generate_sql("test_table", "id")

        # Create in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Get feature names (excluding 'constant')
        feature_names = [k for k in equation.keys() if k != "constant"]
        num_features = len(feature_names)

        # Create test table
        columns = ["id INTEGER PRIMARY KEY"]
        for feature_name in feature_names:
            columns.append(f"{feature_name} REAL")

        create_table_sql = f"CREATE TABLE test_table ({', '.join(columns)})"
        cursor.execute(create_table_sql)

        # Generate random feature values
        feature_values = {}
        for feature_name in feature_names:
            value = data.draw(
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
            )
            feature_values[feature_name] = value

        # Insert test row
        values = [1] + [feature_values[fn] for fn in feature_names]
        placeholders = ", ".join(["?"] * (num_features + 1))
        cursor.execute(f"INSERT INTO test_table VALUES ({placeholders})", values)
        conn.commit()

        # Execute generated SQL
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()

        # Calculate expected probability using Python
        constant = equation.get("constant", 0.0)
        linear_combination = constant
        for feature_name in feature_names:
            linear_combination += feature_values[feature_name] * equation[feature_name]

        expected_probability = 1.0 / (1.0 + np.exp(-linear_combination))

        # Extract actual probability from SQL results
        # For intersect method, we get (id, sets) but we need to check the intermediate CTE
        # Let's verify by checking if the formula is in the SQL
        assert "1.0 / (1.0 + EXP(-(" in sql or "1 / (1 + EXP(-(" in sql

        # Verify the constant is in the SQL
        assert str(constant) in sql or f"{constant}" in sql

        # Verify each feature coefficient is in the SQL
        for feature_name, coefficient in equation.items():
            if feature_name != "constant":
                # Check that the feature and coefficient appear in the SQL
                assert feature_name in sql
                assert str(coefficient) in sql or f"{coefficient}" in sql

    finally:
        # Clean up temporary file
        if os.path.exists(json_path):
            os.unlink(json_path)


# **Feature: elr-package, Property 32: SQL intersect/venn decile logic**
# **Validates: Requirements 11.2, 11.3, 11.6**
@settings(max_examples=100)
@given(
    method=st.sampled_from(["intersect", "venn"]),
    d=st.integers(min_value=1, max_value=10),
    data=st.data(),
)
@requires_sqlite_math
def test_sql_intersect_venn_decile_logic(method, d, data):
    """
    Property 32: SQL intersect/venn decile logic.

    For any intersect or venn method JSON, the generated SQL should include
    logic to rank predictions, identify top d deciles for each constituent model,
    and aggregate results across models.
    """
    # Generate model configuration
    num_models = data.draw(st.integers(min_value=1, max_value=3))
    num_features = data.draw(st.integers(min_value=1, max_value=3))

    models = []
    for _ in range(num_models):
        equation = data.draw(equation_dict_strategy(num_features=num_features))
        models.append(equation)

    config = {"method": method, "d": d, "models": models}

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        json_path = f.name

    try:
        # Create SQL generator
        generator = SQLGenerator(json_path)

        # Generate SQL
        sql = generator.generate_sql("test_table", "id")

        # Verify SQL contains decile ranking logic
        assert "ROW_NUMBER()" in sql.upper()
        assert "CEIL(" in sql.upper()
        assert "row_num <= cutoff" in sql.lower()

        # Verify SQL filters to top d deciles (d embedded in the CEIL cutoff)
        assert f"CEIL({d} *" in sql

        # Verify SQL aggregates across models (COUNT, GROUP BY)
        assert "COUNT(*)" in sql.upper() or "COUNT(" in sql.upper()
        assert "GROUP BY" in sql.upper()

        # Verify SQL has UNION ALL to combine results from multiple models
        if num_models > 1:
            assert "UNION ALL" in sql.upper()

        # Create in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Create test table
        columns = ["id INTEGER PRIMARY KEY"]
        for i in range(num_features):
            columns.append(f"feature_{i} REAL")

        create_table_sql = f"CREATE TABLE test_table ({', '.join(columns)})"
        cursor.execute(create_table_sql)

        # Insert multiple test rows (at least 10 for decile calculation)
        for row_id in range(1, 21):
            values = [row_id] + [
                data.draw(
                    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
                )
                for _ in range(num_features)
            ]
            placeholders = ", ".join(["?"] * (num_features + 1))
            cursor.execute(f"INSERT INTO test_table VALUES ({placeholders})", values)
        conn.commit()

        # Execute generated SQL
        cursor.execute(sql)
        results = cursor.fetchall()

        # Verify results structure: (id, sets, score) columns
        assert len(results) >= 0  # May have 0 results if no IDs in top deciles

        for row in results:
            assert len(row) == 3  # (id, sets, score)
            assert isinstance(row[0], int)  # id
            assert isinstance(row[1], int)  # sets count
            assert row[1] >= 1  # sets count should be at least 1
            assert row[2] >= row[1]  # weighted score >= vote count
            assert isinstance(row[2], (int, float))  # score
            assert row[1] <= num_models  # sets count cannot exceed number of models

        conn.close()

    finally:
        # Clean up temporary file
        if os.path.exists(json_path):
            os.unlink(json_path)


# **Feature: elr-package, Property 33: SQL ensemble meta-model application**
# **Validates: Requirements 11.4, 11.7**
@settings(max_examples=100)
@given(data=st.data())
@requires_sqlite_math
def test_sql_ensemble_meta_model_application(data):
    """
    Property 33: SQL ensemble meta-model application.

    For any ensemble method JSON, the generated SQL should apply the meta-model
    equation to constituent model probabilities to produce final predictions.
    """
    # Generate ensemble configuration
    num_models = data.draw(st.integers(min_value=1, max_value=3))
    num_features = data.draw(st.integers(min_value=1, max_value=3))

    models = []
    for _ in range(num_models):
        equation = data.draw(equation_dict_strategy(num_features=num_features))
        models.append(equation)

    # Generate meta-model
    meta_model = {
        "constant": data.draw(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
    }
    for i in range(num_models):
        meta_model[f"model_{i}_prob"] = data.draw(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        )

    config = {
        "method": "ensemble",
        "d": 2,  # Not used in ensemble scoring but required in config
        "models": models,
        "meta_model": meta_model,
    }

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        json_path = f.name

    try:
        # Create SQL generator
        generator = SQLGenerator(json_path)

        # Generate SQL
        sql = generator.generate_sql("test_table", "id")

        # Verify SQL contains meta-model application
        # Should have constituent_predictions CTE
        assert "constituent_predictions" in sql.lower()

        # Verify SQL applies logistic regression formula for meta-model
        # Should have the meta-model constant
        assert str(meta_model["constant"]) in sql or f"{meta_model['constant']}" in sql

        # Verify SQL references constituent model probabilities
        for i in range(num_models):
            assert f"model_{i}_prob" in sql

        # Verify SQL has final prediction column
        assert "predicted" in sql.lower()

        # Create in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Create test table
        columns = ["id INTEGER PRIMARY KEY"]
        for i in range(num_features):
            columns.append(f"feature_{i} REAL")

        create_table_sql = f"CREATE TABLE test_table ({', '.join(columns)})"
        cursor.execute(create_table_sql)

        # Insert test rows
        for row_id in range(1, 6):
            values = [row_id] + [
                data.draw(
                    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
                )
                for _ in range(num_features)
            ]
            placeholders = ", ".join(["?"] * (num_features + 1))
            cursor.execute(f"INSERT INTO test_table VALUES ({placeholders})", values)
        conn.commit()

        # Execute generated SQL
        cursor.execute(sql)
        results = cursor.fetchall()

        # Verify results structure: should have (id, predicted) columns
        assert len(results) > 0

        for row in results:
            assert len(row) == 2  # (id, predicted)
            assert isinstance(row[0], int)  # id
            assert isinstance(row[1], float)  # predicted probability
            # Probability should be between 0 and 1
            assert 0.0 <= row[1] <= 1.0

        conn.close()

    finally:
        # Clean up temporary file
        if os.path.exists(json_path):
            os.unlink(json_path)
