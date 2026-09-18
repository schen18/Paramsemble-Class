"""
Ensemble method example for ELR package.

This example demonstrates the ensemble method, which creates a meta-model
from top-performing logistic regressions to produce optimized predictions.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from paramsemble_class import ELRClassifier


def main():
    """Run ensemble method example."""
    print("=" * 60)
    print("ELR Ensemble Method Example")
    print("=" * 60)

    # Generate synthetic dataset
    print("\n1. Generating classification dataset...")
    X, y = make_classification(
        n_samples=2500,
        n_features=35,
        n_informative=25,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        flip_y=0.03,
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
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Positive class ratio: {y_train.mean():.2%}")

    # Initialize with ensemble method
    print("\n2. Configuring ensemble method...")
    clf = ELRClassifier(
        m=120,  # Many feature combinations
        f=8,  # Larger feature subsets
        sample="unique",
        d=2,  # Not used in ensemble method
        method="ensemble",  # Ensemble method with meta-model
        spread=20,  # Select top 20 models for meta-model
        solver="lbfgs",
        elr2json="ensemble_all_models.json",
        modeljson="ensemble_selected_models.json",
        random_state=42,
    )

    print("   Ensemble method configuration:")
    print(f"   - Feature combinations: {clf.m}")
    print(f"   - Features per combination: {clf.f}")
    print(f"   - Models for meta-model: {clf.spread}")
    print(f"   - Solver: {clf.solver}")

    # Train
    print("\n3. Training with ensemble method...")
    print("   Training constituent models and meta-model...")
    clf.fit(X_train, y_train, X_test, y_test, ids_test)

    print(f"\n   Training metrics:")
    print(f"   - Baseline PLR: {clf.baseline_results_['plr']:.3f}")
    print(f"   - Baseline FNR: {clf.baseline_results_['fnr']:.3f}")
    print(f"   - Baseline DRP: {clf.baseline_results_['drp']:.3f}")
    print(f"   - Constituent models trained: {len(clf.constituent_models_)}")
    print(f"   - Models selected for meta-model: {len(clf.selected_indices_)}")

    # Analyze selected models
    print(f"\n4. Analyzing selected constituent models...")
    print("\n   Top 5 selected model metrics:")
    for i, idx in enumerate(clf.selected_indices_[:5]):
        result = clf.constituent_results_[idx]
        print(
            f"   Model {i+1}: PLR={result['plr']:.3f}, "
            f"FNR={result['fnr']:.3f}, DRP={result['drp']:.3f}"
        )

    # Show meta-model equation
    print(f"\n5. Meta-model equation:")
    print(f"   Meta-model combines {len(clf.selected_indices_)} constituent models")
    if clf.meta_equation_:
        print(f"   Intercept: {clf.meta_equation_['constant']:.4f}")
        print(f"   Constituent model coefficients:")
        for i in range(min(5, len(clf.selected_indices_))):
            coef_key = f"model_{i}_prob"
            if coef_key in clf.meta_equation_:
                print(f"      {coef_key}: {clf.meta_equation_[coef_key]:.4f}")
        if len(clf.selected_indices_) > 5:
            print(f"      ... and {len(clf.selected_indices_) - 5} more")

    # Get predictions
    print("\n6. Generating ensemble predictions...")
    predictions = clf.predict_detailed(X_test, ids_test)

    print(f"\n   Prediction statistics:")
    print(f"   - Total predictions: {len(predictions)}")
    print(f"   - Min probability: {predictions['predicted'].min():.4f}")
    print(f"   - Max probability: {predictions['predicted'].max():.4f}")
    print(f"   - Mean probability: {predictions['predicted'].mean():.4f}")
    print(f"   - Median probability: {predictions['predicted'].median():.4f}")

    # Show top predictions
    print(f"\n   Top 15 predictions by probability:")
    top_preds = predictions.nlargest(15, "predicted")
    print(top_preds.to_string(index=False))

    # Evaluate performance
    print("\n7. Evaluating ensemble performance...")

    # Get predicted probabilities
    y_pred_proba = predictions.set_index("id").loc[ids_test, "predicted"].values

    # Calculate metrics at different thresholds
    thresholds = [0.3, 0.5, 0.7]
    print("\n   Performance at different thresholds:")
    for threshold in thresholds:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"   Threshold {threshold:.1f}:")
        print(f"      Precision: {precision:.3f}")
        print(f"      Recall: {recall:.3f}")
        print(f"      F1-Score: {f1:.3f}")

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n   ROC AUC Score: {auc:.3f}")

    # Use predict_proba method
    print("\n8. Using predict_proba method...")
    proba_array = clf.predict_proba(X_test)
    print(f"   Shape: {proba_array.shape}")
    print(
        f"   Class 0 probability range: [{proba_array[:, 0].min():.4f}, {proba_array[:, 0].max():.4f}]"
    )
    print(
        f"   Class 1 probability range: [{proba_array[:, 1].min():.4f}, {proba_array[:, 1].max():.4f}]"
    )

    print("\n9. Model export...")
    print(f"   All models exported to: {clf.elr2json}")
    print(f"   Selected models + meta-model exported to: {clf.modeljson}")

    print("\n" + "=" * 60)
    print("Ensemble method example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
