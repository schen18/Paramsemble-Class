"""Property-based tests for model I/O operations."""

import pytest
import os
import tempfile
import json
from hypothesis import given, strategies as st, assume
from paramsemble_class.utils.model_io import ModelIO

# Feature: elr-package, Property 14: JSON export round-trip consistency
# Validates: Requirements 4.2


# Custom strategies for generating model data
@st.composite
def equation_dict_strategy(draw):
    """Generate a valid equation dictionary."""
    # Generate 1-10 features
    n_features = draw(st.integers(min_value=1, max_value=10))

    equation = {}
    for i in range(n_features):
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


@st.composite
def model_data_strategy(draw):
    """Generate a valid model data dictionary."""
    plr = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    fnr = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    drp = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    equation_dict = draw(equation_dict_strategy())

    return {"plr": plr, "fnr": fnr, "drp": drp, "equation_dict": equation_dict}


@st.composite
def models_list_strategy(draw):
    """Generate a list of model data dictionaries."""
    n_models = draw(st.integers(min_value=1, max_value=20))
    models = [draw(model_data_strategy()) for _ in range(n_models)]
    return models


@given(models_data=models_list_strategy())
def test_property_export_all_models_round_trip(models_data):
    """
    Property 14: JSON export round-trip consistency for export_all_models.

    For any set of trained models, exporting to JSON and reading back
    should preserve all PLR, FNR, DRP values and equation dictionaries.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_models.json")

        # Export models
        ModelIO.export_all_models(models_data, filepath)

        # Read back the JSON file
        with open(filepath, "r") as f:
            loaded_data = json.load(f)

        # Verify the data matches
        assert len(loaded_data) == len(models_data)

        for original, loaded in zip(models_data, loaded_data):
            # Check PLR
            assert abs(original["plr"] - loaded["plr"]) < 1e-10

            # Check FNR
            assert abs(original["fnr"] - loaded["fnr"]) < 1e-10

            # Check DRP
            assert abs(original["drp"] - loaded["drp"]) < 1e-10

            # Check equation dictionary
            assert original["equation_dict"] == loaded["equation_dict"]


@given(
    method=st.sampled_from(["intersect", "venn"]),
    d=st.integers(min_value=1, max_value=10),
    selected_models=st.lists(equation_dict_strategy(), min_size=1, max_size=20),
)
def test_property_export_selected_models_round_trip_intersect_venn(method, d, selected_models):
    """
    Property 14: JSON export round-trip consistency for intersect/venn methods.

    For any intersect or venn method configuration, exporting to JSON and
    loading back should preserve method name, d parameter, and all equation
    dictionaries.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_config.json")

        # Export selected models
        ModelIO.export_selected_models(
            method=method, d=d, selected_models=selected_models, meta_model=None, filepath=filepath
        )

        # Load back the configuration
        loaded_config = ModelIO.load_model(filepath)

        # Verify the data matches
        assert loaded_config["method"] == method
        assert loaded_config["d"] == d
        assert len(loaded_config["models"]) == len(selected_models)

        for original, loaded in zip(selected_models, loaded_config["models"]):
            assert original == loaded

        # Verify no meta_model for intersect/venn
        assert "meta_model" not in loaded_config or loaded_config.get("meta_model") is None


@given(
    d=st.integers(min_value=1, max_value=10),
    selected_models=st.lists(equation_dict_strategy(), min_size=1, max_size=20),
    meta_model=equation_dict_strategy(),
)
def test_property_export_selected_models_round_trip_ensemble(d, selected_models, meta_model):
    """
    Property 14: JSON export round-trip consistency for ensemble method.

    For any ensemble method configuration, exporting to JSON and loading back
    should preserve method name, d parameter, constituent model equations,
    and meta-model equation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_config.json")

        # Export selected models with meta-model
        ModelIO.export_selected_models(
            method="ensemble",
            d=d,
            selected_models=selected_models,
            meta_model=meta_model,
            filepath=filepath,
        )

        # Load back the configuration
        loaded_config = ModelIO.load_model(filepath)

        # Verify the data matches
        assert loaded_config["method"] == "ensemble"
        assert loaded_config["d"] == d
        assert len(loaded_config["models"]) == len(selected_models)

        for original, loaded in zip(selected_models, loaded_config["models"]):
            assert original == loaded

        # Verify meta_model is present and matches
        assert "meta_model" in loaded_config
        assert loaded_config["meta_model"] == meta_model


@given(models_data=models_list_strategy())
def test_property_export_all_models_preserves_structure(models_data):
    """
    Property 14: JSON export preserves data structure.

    For any set of models, the exported JSON should maintain the exact
    structure and all fields should be accessible.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_models.json")

        # Export models
        ModelIO.export_all_models(models_data, filepath)

        # Load and verify structure
        loaded_config = (
            ModelIO.load_model(filepath) if False else None
        )  # Skip load_model for this test

        # Read directly to check structure
        with open(filepath, "r") as f:
            loaded_data = json.load(f)

        # Verify it's a list
        assert isinstance(loaded_data, list)

        # Verify each model has required fields
        for model in loaded_data:
            assert "plr" in model
            assert "fnr" in model
            assert "drp" in model
            assert "equation_dict" in model

            # Verify types
            assert isinstance(model["plr"], (int, float))
            assert isinstance(model["fnr"], (int, float))
            assert isinstance(model["drp"], (int, float))
            assert isinstance(model["equation_dict"], dict)

            # Verify equation_dict has constant
            assert "constant" in model["equation_dict"]


@given(
    method=st.sampled_from(["intersect", "venn", "ensemble"]),
    d=st.integers(min_value=1, max_value=10),
    selected_models=st.lists(equation_dict_strategy(), min_size=1, max_size=20),
    meta_model=st.one_of(st.none(), equation_dict_strategy()),
)
def test_property_load_model_validates_structure(method, d, selected_models, meta_model):
    """
    Property 14: Loading model validates JSON structure.

    For any valid model configuration, load_model should successfully
    parse and validate the structure.
    """
    # Skip invalid combinations
    if method == "ensemble" and meta_model is None:
        assume(False)
    if method in ["intersect", "venn"] and meta_model is not None:
        meta_model = None

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_config.json")

        # Export configuration
        ModelIO.export_selected_models(
            method=method,
            d=d,
            selected_models=selected_models,
            meta_model=meta_model,
            filepath=filepath,
        )

        # Load should succeed without errors
        loaded_config = ModelIO.load_model(filepath)

        # Verify required fields are present
        assert "method" in loaded_config
        assert "d" in loaded_config
        assert "models" in loaded_config

        # Verify types
        assert isinstance(loaded_config["method"], str)
        assert isinstance(loaded_config["d"], int)
        assert isinstance(loaded_config["models"], list)

        # Verify method-specific requirements
        if method == "ensemble":
            assert "meta_model" in loaded_config
            assert isinstance(loaded_config["meta_model"], dict)


@given(
    models_data=models_list_strategy(),
    method=st.sampled_from(["intersect", "venn", "ensemble"]),
    d=st.integers(min_value=1, max_value=10),
)
def test_property_round_trip_preserves_equation_coefficients(models_data, method, d):
    """
    Property 14: Round-trip preserves all equation coefficients exactly.

    For any model equations, the coefficients should be preserved exactly
    through export and load operations (within floating point precision).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_config.json")

        # Extract equation dictionaries
        selected_models = [model["equation_dict"] for model in models_data]

        # For ensemble, use first equation as meta_model
        meta_model = selected_models[0] if method == "ensemble" else None

        # Export
        ModelIO.export_selected_models(
            method=method,
            d=d,
            selected_models=selected_models,
            meta_model=meta_model,
            filepath=filepath,
        )

        # Load
        loaded_config = ModelIO.load_model(filepath)

        # Verify all coefficients match
        for original, loaded in zip(selected_models, loaded_config["models"]):
            # Check all features
            assert set(original.keys()) == set(loaded.keys())

            for feature_name in original.keys():
                original_coef = original[feature_name]
                loaded_coef = loaded[feature_name]

                # Allow small floating point differences
                assert (
                    abs(original_coef - loaded_coef) < 1e-10
                ), f"Coefficient mismatch for {feature_name}: {original_coef} != {loaded_coef}"

        # Verify meta_model if present
        if method == "ensemble":
            assert set(meta_model.keys()) == set(loaded_config["meta_model"].keys())
            for feature_name in meta_model.keys():
                assert (
                    abs(meta_model[feature_name] - loaded_config["meta_model"][feature_name])
                    < 1e-10
                )


# Error handling tests
def test_export_all_models_empty_list_raises_error():
    """Test that exporting empty models list raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")

        with pytest.raises(ValueError, match="models_data cannot be empty"):
            ModelIO.export_all_models([], filepath)


def test_export_selected_models_empty_list_raises_error():
    """Test that exporting empty selected models raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")

        with pytest.raises(ValueError, match="selected_models cannot be empty"):
            ModelIO.export_selected_models(
                method="intersect", d=2, selected_models=[], meta_model=None, filepath=filepath
            )


def test_export_selected_models_invalid_method_raises_error():
    """Test that invalid method raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")

        with pytest.raises(ValueError, match="Invalid method"):
            ModelIO.export_selected_models(
                method="invalid_method",
                d=2,
                selected_models=[{"feature1": 0.5, "constant": 0.1}],
                meta_model=None,
                filepath=filepath,
            )


def test_export_selected_models_ensemble_without_meta_model_raises_error():
    """Test that ensemble method without meta_model raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")

        with pytest.raises(ValueError, match="meta_model is required"):
            ModelIO.export_selected_models(
                method="ensemble",
                d=2,
                selected_models=[{"feature1": 0.5, "constant": 0.1}],
                meta_model=None,
                filepath=filepath,
            )


def test_load_model_nonexistent_file_raises_error():
    """Test that loading nonexistent file raises IOError."""
    with pytest.raises(IOError, match="not found"):
        ModelIO.load_model("/nonexistent/path/model.json")


def test_load_model_invalid_json_raises_error():
    """Test that loading invalid JSON raises IOError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "invalid.json")

        # Write invalid JSON
        with open(filepath, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(IOError, match="Failed to parse JSON"):
            ModelIO.load_model(filepath)


def test_load_model_missing_required_fields_raises_error():
    """Test that loading JSON with missing fields raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "incomplete.json")

        # Write JSON missing required fields
        with open(filepath, "w") as f:
            json.dump({"method": "intersect"}, f)

        with pytest.raises(ValueError, match="missing required fields"):
            ModelIO.load_model(filepath)


def test_load_model_invalid_method_raises_error():
    """Test that loading JSON with invalid method raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "invalid_method.json")

        # Write JSON with invalid method
        config = {
            "method": "invalid_method",
            "d": 2,
            "models": [{"feature1": 0.5, "constant": 0.1}],
        }
        with open(filepath, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="Invalid method"):
            ModelIO.load_model(filepath)


def test_load_model_ensemble_missing_meta_model_raises_error():
    """Test that loading ensemble config without meta_model raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "no_meta.json")

        # Write ensemble config without meta_model
        config = {"method": "ensemble", "d": 2, "models": [{"feature1": 0.5, "constant": 0.1}]}
        with open(filepath, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="meta_model.*required"):
            ModelIO.load_model(filepath)
