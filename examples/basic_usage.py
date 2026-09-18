"""
Basic usage example for ELR package.

This example demonstrates the fundamental workflow of training an ELR classifier
with the intersect method on synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from paramsemble_class import ELRClassifier


def main():
    """Run basic ELR example."""
    print("=" * 60)
    print("ELR Basic Usage Example")
    print("=" * 60)

    # Generate synthetic classification dataset
    print("\n1. Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced classes
        random_state=42,
        flip_y=0.1
    )

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Create IDs for test set
    ids_test = np.arange(len(X_test))

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Class distribution: {np.bincount(y_train)}")

    # Initialize ELR classifier with intersect method
    print("\n2. Initializing ELR classifier...")
    clf = ELRClassifier(
        m=50,  # Generate 50 feature combinations
        f=5,  # 5 features per combination
        sample="unique",  # No feature repetition within combinations
        d=2,  # Consider top 2 deciles
        method="intersect",  # Use intersect ensemble method
        spread=10,  # Select top 10 models
        solver="auto",  # Automatic solver selection
        random_state=42,
    )

    print("   Configuration:")
    print(f"   - Feature combinations (m): {clf.m}")
    print(f"   - Features per combination (f): {clf.f}")
    print(f"   - Ensemble method: {clf.method}")
    print(f"   - Top models to select (spread): {clf.spread}")
    print(f"   - Top deciles (d): {clf.d}")

    # Train the classifier
    print("\n3. Training ELR classifier...")
    print("   This may take a moment...")
    clf.fit(X_train, y_train, X_test, y_test, ids_test)

    print(f"\n   Training complete!")
    print(f"   - Baseline PLR: {clf.baseline_results_['plr']:.3f}")
    print(f"   - Baseline FNR: {clf.baseline_results_['fnr']:.3f}")
    print(f"   - Baseline DRP: {clf.baseline_results_['drp']:.3f}")
    print(f"   - Models trained: {len(clf.constituent_models_)}")
    print(f"   - Models selected: {len(clf.selected_indices_)}")

    # Make predictions
    print("\n4. Generating predictions...")
    predictions = clf.predict_detailed(X_test, ids_test)

    print(f"\n   Predictions shape: {predictions.shape}")
    print(f"   Columns: {list(predictions.columns)}")
    print("\n   Top 10 predictions:")
    print(predictions.head(10).to_string(index=False))

    # Analyze results
    print("\n5. Analyzing results...")
    print(f"   Total unique IDs predicted: {len(predictions)}")
    print(f"   Max sets count: {predictions['sets'].max()}")
    print(f"   Min sets count: {predictions['sets'].min()}")
    print(f"   Mean sets count: {predictions['sets'].mean():.2f}")

    # Show IDs appearing in most models
    high_confidence = predictions[predictions["sets"] >= 5]
    print(f"\n   High-confidence predictions (sets >= 5): {len(high_confidence)}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
