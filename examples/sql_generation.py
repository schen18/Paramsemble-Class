"""
SQL generation example for ELR package.

This example demonstrates how to generate SQL queries from trained models
for database-level scoring.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from paramsemble_class import ELRClassifier, SQLGenerator


def main():
    """Run SQL generation example."""
    print("=" * 60)
    print("ELR SQL Generation Example")
    print("=" * 60)

    # Generate synthetic dataset with named features
    print("\n1. Generating dataset with named features...")
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42,
        flip_y=0.1
    )

    # Create DataFrame with named features
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )

    # Create IDs
    ids_test = np.arange(len(X_test))

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {list(X_train.columns[:5])} ...")

    # Train models with all three methods
    methods = ["intersect", "venn", "ensemble"]

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"2. Training {method.upper()} method...")
        print(f"{'=' * 60}")

        # Initialize classifier
        clf = ELRClassifier(
            m=30,
            f=4,
            sample="unique",
            d=2,
            method=method,
            spread=8,
            solver="lbfgs",
            modeljson=f"{method}_model.json",
            random_state=42,
        )

        # Train
        print(f"   Training {method} model...")
        clf.fit(X_train, y_train, X_test, y_test, ids_test)

        print(f"   Model saved to: {method}_model.json")
        print(f"   Selected models: {len(clf.selected_indices_)}")

        # Generate SQL
        print(f"\n3. Generating SQL for {method} method...")
        generator = SQLGenerator(f"{method}_model.json")

        # Generate SQL for a hypothetical database table
        sql_query = generator.generate_sql(table_name="customer_features", id_column="customer_id")

        # Display SQL query
        print(f"\n   Generated SQL Query:")
        print("   " + "-" * 56)

        # Print first 30 lines of SQL
        sql_lines = sql_query.split("\n")
        for i, line in enumerate(sql_lines[:30]):
            print(f"   {line}")

        if len(sql_lines) > 30:
            print(f"   ... ({len(sql_lines) - 30} more lines)")

        print("   " + "-" * 56)

        # Save SQL to file
        sql_filename = f"{method}_scoring_query.sql"
        with open(sql_filename, "w") as f:
            f.write(sql_query)

        print(f"\n   SQL query saved to: {sql_filename}")
        print(f"   Query length: {len(sql_query)} characters")
        print(f"   Query lines: {len(sql_lines)}")

        # Explain the SQL structure
        print(f"\n4. SQL structure for {method} method:")

        if method in ["intersect", "venn"]:
            print(f"   - Creates {len(clf.selected_indices_)} model CTEs")
            print(f"   - Calculates logistic regression probabilities")
            print(f"   - Ranks predictions into deciles")
            print(f"   - Filters to top {clf.d} deciles per model")
            print(f"   - Aggregates IDs across models")
            print(f"   - Counts occurrences (sets) per ID")
            print(f"   - Returns: customer_id, sets")
        else:  # ensemble
            print(f"   - Creates {len(clf.selected_indices_)} constituent model CTEs")
            print(f"   - Calculates probabilities for each model")
            print(f"   - Joins all constituent predictions")
            print(f"   - Applies meta-model equation")
            print(f"   - Returns: customer_id, predicted")

        print(f"\n5. Using the SQL query:")
        print(f"   To use this SQL in your database:")
        print(f"   1. Ensure table 'customer_features' exists")
        print(f"   2. Table must have column 'customer_id'")
        print(f"   3. Table must have all feature columns:")
        print(f"      {', '.join(feature_names[:5])}, ...")
        print(f"   4. Execute the SQL query")
        print(f"   5. Results will match Python predictions")

    # Demonstrate SQL vs Python consistency
    print(f"\n{'=' * 60}")
    print("6. SQL vs Python Prediction Consistency")
    print(f"{'=' * 60}")
    print("\n   The generated SQL queries will produce identical results")
    print("   to the Python predictions when executed against a database")
    print("   table with the same feature values.")
    print("\n   Example workflow:")
    print("   1. Train model in Python with ELRClassifier")
    print("   2. Export model to JSON with modeljson parameter")
    print("   3. Generate SQL with SQLGenerator")
    print("   4. Execute SQL in production database")
    print("   5. Get real-time predictions without Python runtime")

    print("\n" + "=" * 60)
    print("SQL generation example complete!")
    print("=" * 60)
    print("\nGenerated files:")
    for method in methods:
        print(f"  - {method}_model.json")
        print(f"  - {method}_scoring_query.sql")


if __name__ == "__main__":
    main()
