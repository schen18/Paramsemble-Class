"""
Intersect method example for ELR package.

This example demonstrates the intersect ensemble method, which identifies
high-confidence predictions that appear in multiple top-performing models.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from paramsemble_class import ELRClassifier


def main():
    """Run intersect method example."""
    print("=" * 60)
    print("ELR Intersect Method Example")
    print("=" * 60)

    # Generate synthetic imbalanced dataset
    print("\n1. Generating imbalanced classification dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=2,
        weights=[0.8, 0.2],  # Highly imbalanced
        flip_y=0.05,  # Add some noise
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Create IDs
    ids_test = np.arange(len(X_test))

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Positive class ratio: {y_train.mean():.2%}")

    # Initialize with intersect method
    print("\n2. Configuring intersect method...")
    clf = ELRClassifier(
        m=100,  # More combinations for better coverage
        f=7,  # Moderate feature subset size
        sample="unique",
        d=2,  # Top 20% of predictions
        method="intersect",  # Intersect method
        spread=15,  # Select top 15 models
        solver="lbfgs",  # Specific solver
        elr2json="intersect_all_models.json",  # Export all models
        modeljson="intersect_selected_models.json",  # Export selected models
        random_state=42,
    )

    print("   Intersect method configuration:")
    print(f"   - Feature combinations: {clf.m}")
    print(f"   - Features per combination: {clf.f}")
    print(f"   - Models to select: {clf.spread}")
    print(f"   - Top deciles: {clf.d}")

    # Train
    print("\n3. Training with intersect method...")
    clf.fit(X_train, y_train, X_test, y_test, ids_test)

    print(f"\n   Training metrics:")
    print(f"   - Baseline PLR: {clf.baseline_results_['plr']:.3f}")
    print(f"   - Baseline FNR: {clf.baseline_results_['fnr']:.3f}")
    print(f"   - Baseline DRP: {clf.baseline_results_['drp']:.3f}")

    # Analyze selected models
    print(f"\n4. Analyzing selected models...")
    print(f"   Selected {len(clf.selected_indices_)} models")

    # Show metrics for selected models
    print("\n   Top 5 selected model metrics:")
    for i, idx in enumerate(clf.selected_indices_[:5]):
        result = clf.constituent_results_[idx]
        print(
            f"   Model {i+1}: PLR={result['plr']:.3f}, "
            f"FNR={result['fnr']:.3f}, DRP={result['drp']:.3f}"
        )

    # Get predictions
    print("\n5. Generating intersect predictions...")
    predictions = clf.predict_detailed(X_test, ids_test)

    print(f"\n   Prediction statistics:")
    print(f"   - Total IDs in results: {len(predictions)}")
    print(f"   - Max sets (most models agree): {predictions['sets'].max()}")
    print(f"   - IDs in 10+ models: {(predictions['sets'] >= 10).sum()}")
    print(f"   - IDs in 5+ models: {(predictions['sets'] >= 5).sum()}")

    # Show high-confidence predictions
    print("\n   Top 15 high-confidence predictions:")
    print(predictions.head(15).to_string(index=False))

    # Analyze by confidence level
    print("\n6. Confidence level analysis...")
    for threshold in [15, 10, 5, 3]:
        high_conf = predictions[predictions["sets"] >= threshold]
        if len(high_conf) > 0:
            # Check actual positive rate for these IDs
            actual_positives = y_test[high_conf["id"].values].sum()
            precision = actual_positives / len(high_conf) if len(high_conf) > 0 else 0
            print(
                f"   Sets >= {threshold:2d}: {len(high_conf):4d} IDs, "
                f"Precision: {precision:.2%}"
            )

    print("\n7. Model export...")
    print(f"   All models exported to: {clf.elr2json}")
    print(f"   Selected models exported to: {clf.modeljson}")

    print("\n" + "=" * 60)
    print("Intersect method example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
