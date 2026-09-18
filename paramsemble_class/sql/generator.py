"""SQL generation from model JSON for database scoring."""

from typing import Any, Dict, Optional

from ..utils.model_io import ModelIO


def _quote_ident(name: str) -> str:
    """Quote an SQL identifier, escaping any embedded double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


class SQLGenerator:
    """
    Generates SQL queries from model JSON files for database scoring.

    This class converts trained ELR model configurations (stored in JSON format)
    into executable SQL queries that can be run directly against database tables.
    It supports all three ensemble methods: intersect, venn, and ensemble.

    The generated SQL uses Common Table Expressions (CTEs) to organize the logic
    and implements the logistic regression formula for probability calculation.

    **Decile rule.** Python scoring keeps the top ``ceil(d * n / 10)`` rows by
    predicted probability; the generated SQL reproduces this with
    ``ROW_NUMBER() ... <= CEIL(d * COUNT(*) OVER () / 10.0)``. Rows with
    *exactly tied* scores at the cutoff can be assigned differently by the
    database (arbitrary tie order) than by Python (ties broken by row order).

    **Requirements on the target table.** Identifiers (table and column names)
    are quoted and may contain spaces or mixed case. The ID column must be
    unique — the ensemble query joins one CTE per constituent model on it.

    Attributes
    ----------
    model_config : Dict[str, Any]
        Loaded model configuration containing method, d parameter, and equations.
    method : str
        Ensemble method: "intersect", "venn", or "ensemble".
    d : int
        Number of top deciles to consider (1-10).
    models : List[Dict[str, float]]
        List of constituent model equation dictionaries.
    meta_model : Dict[str, float], optional
        Meta-model equation dictionary (only for ensemble method).

    Examples
    --------
    >>> generator = SQLGenerator('model_config.json')
    >>> sql = generator.generate_sql('customer_data', 'customer_id')
    >>> print(sql)
    """

    def __init__(self, modeljson_path: str):
        """
        Initialize SQL generator with model JSON file.

        Parameters
        ----------
        modeljson_path : str
            Path to the model JSON file (modeljson format).

        Raises
        ------
        IOError
            If file cannot be read.
        ValueError
            If JSON structure is invalid.
        """
        self.model_config = ModelIO.load_model(modeljson_path)
        self.method = self.model_config["method"]
        self.d = self.model_config["d"]
        self.models = self.model_config["models"]
        self.meta_model: Optional[Dict[str, float]] = self.model_config.get("meta_model", None)
        self.weights: Optional[list] = self.model_config.get("weights", None)

    def generate_sql(self, table_name: str, id_column: str = "id") -> str:
        """
        Generate SQL query for model scoring.

        Creates a complete SQL query that can be executed against a database
        table to score records using the loaded model configuration. The
        table must contain the feature columns named in the model equations
        and a unique ``id_column``.

        Parameters
        ----------
        table_name : str
            Name of the database table to score (quoted automatically, so
            spaces and mixed case are safe).
        id_column : str, default="id"
            Name of the ID column in the table (quoted automatically).

        Returns
        -------
        str
            Complete SQL query string ready for execution.

        Examples
        --------
        >>> generator = SQLGenerator('model.json')
        >>> sql = generator.generate_sql('customers', 'customer_id')
        """
        if self.method == "intersect":
            return self._generate_intersect_sql(table_name, id_column)
        elif self.method == "venn":
            return self._generate_venn_sql(table_name, id_column)
        elif self.method == "ensemble":
            return self._generate_ensemble_sql(table_name, id_column)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _generate_logistic_regression_sql(
        self, equation_dict: Dict[str, Any], table_name: str, cte_name: str, id_column: str
    ) -> str:
        """
        Generate SQL CTE for a single logistic regression equation.

        Creates a Common Table Expression that applies the logistic regression
        formula: 1 / (1 + EXP(-(constant + feature1*coef1 + feature2*coef2 + ...)))

        Parameters
        ----------
        equation_dict : Dict[str, float]
            Dictionary mapping feature names to coefficients, with 'constant' key.
        table_name : str
            Name of the source table (already quoted).
        cte_name : str
            Name for the CTE.
        id_column : str
            Quoted ID column name.

        Returns
        -------
        str
            SQL CTE string.
        """
        # Extract constant (intercept)
        constant = equation_dict.get("constant", 0.0)

        feature_terms = []
        if "bins" in equation_dict:
            # Binned (WoE) equation: CASE ladder per feature mapping x to its
            # bin's WoE, times the feature coefficient. Mirrors the Python
            # rule woe[searchsorted(splits, x, side='right')].
            bins_spec: Dict[str, Any] = equation_dict["bins"]
            for feature_name, spec in bins_spec.items():
                col = _quote_ident(feature_name)
                splits = spec["splits"]
                woe = spec["woe"]
                coef = spec["coef"]
                whens = "".join(
                    f" WHEN {col} <= {float(s)} THEN {float(w)}" for s, w in zip(splits, woe[:-1])
                )
                ladder = f"CASE{whens} ELSE {float(woe[-1])} END"
                feature_terms.append(f"({ladder} * {coef})")
        else:
            # Plain linear equation
            for feature_name, coefficient in equation_dict.items():
                if feature_name != "constant":
                    feature_terms.append(f"({_quote_ident(feature_name)} * {coefficient})")

        # Build the linear combination
        if feature_terms:
            linear_combination = f"{constant} + " + " + ".join(feature_terms)
        else:
            linear_combination = str(constant)

        # Build the logistic regression formula
        probability_formula = f"1.0 / (1.0 + EXP(-({linear_combination})))"

        # Create CTE (select only the ID and the probability)
        cte_sql = f"""{cte_name} AS (
    SELECT
        {id_column},
        {probability_formula} AS predicted_probability
    FROM {table_name}
)"""

        return cte_sql

    def _decile_cte(self, model_index: int, id_column: str) -> str:
        """CTE keeping the top-d-decile IDs for one model, matching the
        Python rule of ceil(d * n / 10) highest-scoring rows."""
        return f"""model_{model_index}_top_deciles AS (
    SELECT {id_column}
    FROM (
        SELECT
            {id_column},
            ROW_NUMBER() OVER (ORDER BY predicted_probability DESC) AS row_num,
            CEIL({self.d} * COUNT(*) OVER () / 10.0) AS cutoff
        FROM model_{model_index}
    ) ranked
    WHERE row_num <= cutoff
)"""

    def _generate_intersect_sql(self, table_name: str, id_column: str) -> str:
        """
        Generate SQL for intersect method.

        Creates SQL that:
        1. Generates CTEs for each constituent model
        2. Keeps the top d deciles for each model (ceil(d*n/10) rows,
           the same rule as Python scoring)
        3. Aggregates IDs across models
        4. Counts occurrences per ID

        Parameters
        ----------
        table_name : str
            Name of the database table to score.
        id_column : str
            Name of the ID column.

        Returns
        -------
        str
            Complete SQL query.
        """
        table = _quote_ident(table_name)
        id_col = _quote_ident(id_column)

        # Generate scoring CTEs for each constituent model
        ctes = [
            self._generate_logistic_regression_sql(model_equation, table, f"model_{i}", id_col)
            for i, model_equation in enumerate(self.models)
        ]

        # Top d decile CTEs for each model
        decile_ctes = [self._decile_cte(i, id_col) for i in range(len(self.models))]

        # Union all top decile IDs, carrying each model's vote weight so
        # the weighted score can be aggregated alongside the set count
        weights = self.weights if self.weights is not None else [1.0] * len(self.models)
        union_parts = [
            f"    SELECT {id_col}, {weights[i]} AS contrib FROM model_{i}_top_deciles"
            for i in range(len(self.models))
        ]
        _NL = chr(10)
        union_cte = (
            "all_top_decile_ids AS (" + _NL + (_NL + "    UNION ALL" + _NL).join(union_parts) + ")"
        )

        # Final aggregation: sets = vote count, score = weighted vote total
        final_query = f"""SELECT
    {id_col} AS id,
    COUNT(*) AS sets,
    SUM(contrib) AS score
FROM all_top_decile_ids
GROUP BY {id_col}
ORDER BY sets DESC, score DESC, {id_col} ASC"""

        # Combine all CTEs and final query
        all_ctes = ctes + decile_ctes + [union_cte]
        full_query = f"WITH {',\n'.join(all_ctes)}\n{final_query};"

        return full_query

    def _generate_venn_sql(self, table_name: str, id_column: str) -> str:
        """
        Generate SQL for venn method.

        The venn method uses the same SQL structure as intersect for scoring:
        the difference between the two methods is which models were selected
        during training, not how scoring works.

        Parameters
        ----------
        table_name : str
            Name of the database table to score.
        id_column : str
            Name of the ID column.

        Returns
        -------
        str
            Complete SQL query.
        """
        return self._generate_intersect_sql(table_name, id_column)

    def _generate_ensemble_sql(self, table_name: str, id_column: str) -> str:
        """
        Generate SQL for ensemble method.

        Creates SQL that:
        1. Generates scoring CTEs for each constituent model
        2. Joins all constituent predictions on the ID column (which must
           be unique in the table)
        3. Applies meta-model equation to constituent probabilities
        4. Returns final probability scores

        Parameters
        ----------
        table_name : str
            Name of the database table to score.
        id_column : str
            Name of the ID column.

        Returns
        -------
        str
            Complete SQL query.
        """
        if self.meta_model is None:
            raise ValueError(
                "Meta-model is required for ensemble method but was not found " "in configuration."
            )

        table = _quote_ident(table_name)
        id_col = _quote_ident(id_column)

        # Generate scoring CTEs for each constituent model
        ctes = [
            self._generate_logistic_regression_sql(model_equation, table, f"model_{i}", id_col)
            for i, model_equation in enumerate(self.models)
        ]

        # Create a CTE that joins all constituent model predictions
        select_parts = [f"model_0.{id_col}"]
        for i in range(len(self.models)):
            select_parts.append(f'model_{i}.predicted_probability AS "model_{i}_prob"')

        # Build JOIN clause
        join_clause = "model_0"
        for i in range(1, len(self.models)):
            join_clause += f"\n    INNER JOIN model_{i} ON model_0.{id_col} = model_{i}.{id_col}"

        constituent_predictions_cte = f"""constituent_predictions AS (
    SELECT
        {',\n        '.join(select_parts)}
    FROM {join_clause}
)"""

        # Apply meta-model equation
        meta_constant = self.meta_model.get("constant", 0.0)
        meta_terms = []

        for feature_name, coefficient in self.meta_model.items():
            if feature_name != "constant":
                # Feature names in the meta-model are the generated
                # "model_i_prob" aliases of constituent_predictions
                meta_terms.append(f"({_quote_ident(feature_name)} * {coefficient})")

        # Build meta-model linear combination
        if meta_terms:
            meta_linear_combination = f"{meta_constant} + " + " + ".join(meta_terms)
        else:
            meta_linear_combination = str(meta_constant)

        # Build meta-model probability formula
        meta_probability_formula = f"1.0 / (1.0 + EXP(-({meta_linear_combination})))"

        # Final query
        final_query = f"""SELECT
    {id_col} AS id,
    {meta_probability_formula} AS predicted
FROM constituent_predictions
ORDER BY predicted DESC, {id_col} ASC"""

        # Combine all CTEs and final query
        all_ctes = ctes + [constituent_predictions_cte]
        full_query = f"WITH {',\n'.join(all_ctes)}\n{final_query};"

        return full_query
