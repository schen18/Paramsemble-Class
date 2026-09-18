"""Model I/O utilities for JSON export and import."""

import json
import math
import os
from typing import Dict, List, Optional, Any


def _json_scalar(value: Any) -> Any:
    """Convert one value to a strict-JSON-safe Python scalar.

    numpy scalars become Python scalars; non-finite floats (e.g. a PLR of
    infinity when a model has FPR=0) become None, because ``Infinity`` is
    not valid JSON and breaks non-Python consumers.
    """
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ModelIO:
    """
    Handles JSON export and import for ELR models.

    Provides methods to:
    - Export all constituent model metrics to JSON (elr2json)
    - Export selected model equations to JSON (modeljson)
    - Load model configurations from JSON

    The JSON structure is designed to support SQL equation reconstruction
    and model scoring in production environments. All exports are strict
    JSON (non-finite floats serialize as null).
    """

    @staticmethod
    def export_all_models(models_data: List[Dict[str, Any]], filepath: str) -> None:
        """
        Export all constituent model metrics to JSON (elr2json).

        This method saves comprehensive metrics for all trained constituent models,
        including PLR, FNR, DRP, DRS, DPS, and equation dictionaries. DRS/DPS ID
        sets are serialized as lists. Non-finite metric values (e.g. infinite PLR)
        are written as null to keep the file valid strict JSON.

        Parameters
        ----------
        models_data : List[Dict[str, Any]]
            List of dictionaries, each containing:
            - 'plr': Positive Likelihood Ratio (float)
            - 'fnr': False Negative Rate (float)
            - 'drp': Decile Ranked Performance (float)
            - 'drs': Decile Ranked Set (set of IDs)
            - 'dps': Decile Positive Set (set of IDs)
            - 'equation_dict': Dictionary mapping feature names to coefficients
            - 'feature_indices': List of feature indices used (optional)
        filepath : str
            Path where the JSON file will be saved.

        Raises
        ------
        IOError
            If file cannot be written with descriptive message.
        ValueError
            If models_data is empty or invalid.
        """
        if not models_data:
            raise ValueError("models_data cannot be empty. Provide at least one model to export.")

        try:
            serializable_data = []
            for model in models_data:
                model_copy: Dict[str, Any] = {}
                for key, value in model.items():
                    if isinstance(value, set):
                        model_copy[key] = [_json_scalar(x) for x in value]
                    elif isinstance(value, (list, tuple)):
                        model_copy[key] = [_json_scalar(x) for x in value]
                    elif isinstance(value, dict):
                        model_copy[key] = {k: _json_scalar(v) for k, v in value.items()}
                    else:
                        model_copy[key] = _json_scalar(value)
                serializable_data.append(model_copy)

            ModelIO._write_json(serializable_data, filepath)

        except (IOError, OSError) as e:
            raise IOError(f"Failed to write model data to '{filepath}'. " f"Error: {str(e)}")
        except Exception as e:
            raise IOError(
                f"Unexpected error while exporting models to '{filepath}'. " f"Error: {str(e)}"
            )

    @staticmethod
    def export_selected_models(
        method: str,
        d: int,
        selected_models: List[Dict[str, float]],
        meta_model: Optional[Dict[str, float]],
        filepath: str,
        feature_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        fallback: bool = False,
    ) -> None:
        """
        Export selected model equations to JSON (modeljson).

        This method saves the configuration needed for model scoring,
        including the ensemble method, decile parameter, and equation
        dictionaries for selected models. For ensemble method, it also
        includes the meta-model equation. When ``feature_names`` is given
        (the training feature list, in column order), it is stored so that
        :class:`~paramsemble_class.scoring.scorer.ModelScorer` can score
        plain numpy arrays as well as named DataFrames.

        Parameters
        ----------
        method : str
            Ensemble method: "intersect", "venn", or "ensemble".
        d : int
            Number of top deciles to consider (1-10).
        selected_models : List[Dict[str, float]]
            List of equation dictionaries for selected constituent models.
            Each dictionary maps feature names to coefficients with
            'constant' key for intercept.
        meta_model : Dict[str, float], optional
            Meta-model equation dictionary (only for ensemble method).
            Maps constituent model probability names to coefficients.
        filepath : str
            Path where the JSON file will be saved.
        feature_names : List[str], optional
            Full ordered list of training feature names. Enables scoring
            numpy arrays whose column order matches training input.

        Raises
        ------
        IOError
            If file cannot be written with descriptive message.
        ValueError
            If method is invalid or selected_models is empty.
        """
        # Validate method
        valid_methods = ["intersect", "venn", "ensemble"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}.")

        # Validate selected_models
        if not selected_models:
            raise ValueError(
                "selected_models cannot be empty. Provide at least one model to export."
            )

        # Validate d parameter
        if not (1 <= d <= 10):
            raise ValueError(f"Parameter 'd' must be between 1 and 10 inclusive, got {d}.")

        # Validate meta_model for ensemble method before trying to write
        if method == "ensemble":
            if meta_model is None:
                raise ValueError("meta_model is required for ensemble method but was not provided.")

        try:
            model_config = {
                "method": method,
                "d": d,
                "models": [{k: _json_scalar(v) for k, v in eq.items()} for eq in selected_models],
            }
            if method == "ensemble":
                assert meta_model is not None  # validated above
                model_config["meta_model"] = {k: _json_scalar(v) for k, v in meta_model.items()}
            if feature_names is not None:
                model_config["feature_names"] = list(feature_names)
            if weights is not None:
                model_config["weights"] = [float(w) for w in weights]
            if fallback:
                model_config["fallback"] = True

            ModelIO._write_json(model_config, filepath)

        except (IOError, OSError) as e:
            raise IOError(
                f"Failed to write model configuration to '{filepath}'. " f"Error: {str(e)}"
            )
        except Exception as e:
            raise IOError(
                f"Unexpected error while exporting model configuration to '{filepath}'. "
                f"Error: {str(e)}"
            )

    @staticmethod
    def _write_json(payload, filepath: str) -> None:
        """Write strict JSON (allow_nan=False), creating directories as needed."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2, allow_nan=False)

    @staticmethod
    def load_model(filepath: str) -> Dict[str, Any]:
        """
        Load model configuration from JSON file.

        Reads a model configuration JSON file (modeljson format) and
        returns the parsed dictionary containing method, d parameter,
        constituent model equations, an optional meta-model equation, and
        an optional ordered feature_names list.

        Parameters
        ----------
        filepath : str
            Path to the JSON file to load.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'method': Ensemble method string
            - 'd': Decile parameter (int)
            - 'models': List of equation dictionaries
            - 'meta_model': Meta-model equation dict (ensemble method only)
            - 'feature_names': Ordered training feature names (optional)

        Raises
        ------
        IOError
            If file cannot be read with descriptive message.
        ValueError
            If JSON structure is invalid or missing required fields.
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise IOError(
                f"Model configuration file not found: '{filepath}'. "
                "Please provide a valid file path."
            )

        try:
            # Read JSON file
            with open(filepath, "r") as f:
                model_config: Dict[str, Any] = json.load(f)

            # Validate required fields
            required_fields = ["method", "d", "models"]
            missing_fields = [field for field in required_fields if field not in model_config]

            if missing_fields:
                raise ValueError(
                    f"Invalid model configuration: missing required fields {missing_fields}. "
                    f"Expected fields: {required_fields}"
                )

            # Validate method
            valid_methods = ["intersect", "venn", "ensemble"]
            if model_config["method"] not in valid_methods:
                raise ValueError(
                    f"Invalid method '{model_config['method']}' in configuration. "
                    f"Must be one of {valid_methods}."
                )

            # Validate d parameter
            if not isinstance(model_config["d"], int) or not (1 <= model_config["d"] <= 10):
                raise ValueError(
                    f"Invalid 'd' parameter in configuration: {model_config['d']}. "
                    "Must be an integer between 1 and 10 inclusive."
                )

            # Validate models list
            if not isinstance(model_config["models"], list) or not model_config["models"]:
                raise ValueError(
                    "Invalid 'models' field in configuration. "
                    "Must be a non-empty list of equation dictionaries."
                )
            for entry in model_config["models"]:
                if not isinstance(entry, dict):
                    raise ValueError("Invalid 'models' entry: each model must be a dict.")
                if "bins" in entry:
                    # Binned (WoE) equation: bins -> {feat: {splits, woe, coef}}
                    bins = entry["bins"]
                    if not isinstance(bins, dict) or not bins:
                        raise ValueError("Invalid 'bins' in model equation.")
                    for feat, spec in bins.items():
                        if (
                            not isinstance(spec, dict)
                            or "splits" not in spec
                            or "woe" not in spec
                            or "coef" not in spec
                            or len(spec["woe"]) != len(spec["splits"]) + 1
                        ):
                            raise ValueError(
                                f"Invalid binning spec for feature {feat!r}: needs "
                                "'splits', 'woe' (len = len(splits)+1) and 'coef'."
                            )
                if "constant" not in entry:
                    raise ValueError("Model equation missing 'constant'.")

            # Validate optional weights (vote weights aligned with models)
            if "weights" in model_config and model_config["weights"] is not None:
                wts = model_config["weights"]
                if (
                    not isinstance(wts, list)
                    or not all(isinstance(w, (int, float)) for w in wts)
                    or len(wts) != len(model_config["models"])
                ):
                    raise ValueError(
                        "Invalid 'weights' field in configuration. Must be a "
                        "list of numbers, one per model."
                    )

            # Validate optional feature_names
            if "feature_names" in model_config and model_config["feature_names"] is not None:
                if not isinstance(model_config["feature_names"], list) or not all(
                    isinstance(name, str) for name in model_config["feature_names"]
                ):
                    raise ValueError(
                        "Invalid 'feature_names' field in configuration. "
                        "Must be a list of feature name strings."
                    )

            # Validate meta_model for ensemble method
            if model_config["method"] == "ensemble":
                if "meta_model" not in model_config:
                    raise ValueError(
                        "Invalid configuration for ensemble method: "
                        "'meta_model' field is required but missing."
                    )
                if not isinstance(model_config["meta_model"], dict):
                    raise ValueError(
                        "Invalid 'meta_model' field in configuration. "
                        "Must be a dictionary mapping features to coefficients."
                    )

            return model_config

        except json.JSONDecodeError as e:
            raise IOError(
                f"Failed to parse JSON from '{filepath}'. "
                f"The file may be corrupted or contain invalid JSON. "
                f"Error: {str(e)}"
            )
        except (IOError, OSError) as e:
            raise IOError(
                f"Failed to read model configuration from '{filepath}'. " f"Error: {str(e)}"
            )
