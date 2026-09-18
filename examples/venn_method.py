"""
Venn method example for ELR package.

This example demonstrates the venn ensemble method, which discovers unique
predictions not captured by the baseline model.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from paramsemble_class import ELRClassifier


def main():
    """Run venn method example."""
    print("=" * 60)
    print("ELR Venn Method Example")
    print("=" * 60)

    # Generate synthetic dataset with complex patterns
    print("\n1. Generating dataset with complex patterns...")
    X, y = make_classification(
        n_samples=1500,
        n_features=25,
        n_informative=18,
        n_redundant=4,
        n_classes=2,
        weights=[0.75, 0.25],
        flip_y=0.08,  # More noise to create opportunities for unique discoveries
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Create IDs
    ids_test = np.arange(len(X_test))

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Positive class ratio: {y_train.mean():.2%}")

    # Initialize with venn method
    print("\n2. Configuring venn method...")
    clf = ELRClassifier(
        m=80,  # Feature combinations
        f=6,  # Features per combination
        sample="unique",
        d=2,  # Top 2 deciles
        method="venn",  # Venn method for unique discoveries
        spread=12,  # Will initially select 2×12=24 models
        solver="auto",
        elr2json="venn_all_models.json",
        modeljson="venn_selected_models.json",
        random_state=42,
    )

    print("   Venn method configuration:")
    print(f"   - Feature combinations: {clf.m}")
    print(f"   - Features per combination: {clf.f}")
    print(f"   - Spread parameter: {clf.spread}")
    print(f"   - Initial selection: {2 * clf.spread} models")
    print(f"   - Top deciles: {clf.d}")

    # Train
    print("\n3. Training with venn method...")
    print("   Venn method filters models with unique predictions...")
    clf.fit(X_train, y_train, X_test, y_test, ids_test)

    print(f"\n   Training metrics:")
    print(f"   - Baseline PLR: {clf.baseline_results_['plr']:.3f}")
    print(f"   - Baseline FNR: {clf.baseline_results_['fnr']:.3f}")
    print(f"   - Baseline DRP: {clf.baseline_results_['drp']:.3f}")
    print(f"   - Baseline DPS size: {len(clf.baseline_results_['dps'])}")

    # Analyze model selection
    print(f"\n4. Analyzing venn model selection...")
    print(f"   Initially considered: {2 * clf.spread} models")
    print(f"   Models with unique predictions: {len(clf.selected_indices_)}")
    print(f"   Models discarded: {2 * clf.spread - len(clf.selected_indices_)}")

    # Show metrics for undiscarded models
    if len(clf.selected_indices_) > 0:
        print("\n   Top 5 undiscarded model metrics:")
        for i, idx in enumerate(clf.selected_indices_[:5]):
            result = clf.constituent_results_[idx]
            print(
                f"   Model {i+1}: PLR={result['plr']:.3f}, "
                f"FNR={result['fnr']:.3f}, DRP={result['drp']:.3f}"
            )

    # Get predictions
    print("\n5. Generating venn predictions...")
    predictions = clf.predict_detailed(X_test, ids_test)

    print(f"\n   Prediction statistics:")
    print(f"   - Total unique IDs: {len(predictions)}")
    print(f"   - Max sets count: {predictions['sets'].max()}")
    print(f"   - Mean sets count: {predictions['sets'].mean():.2f}")

    # Compare with baseline
    baseline_dps = clf.baseline_results_["dps"]
    venn_ids = set(predictions["id"].values)
    unique_to_venn = venn_ids - baseline_dps
    overlap_with_baseline = venn_ids & baseline_dps

    print(f"\n6. Comparison with baseline:")
    print(f"   - Baseline DPS size: {len(baseline_dps)}")
    print(f"   - Venn result size: {len(venn_ids)}")
    print(f"   - Unique to venn: {len(unique_to_venn)}")
    print(f"   - Overlap with baseline: {len(overlap_with_baseline)}")

    # Show some unique discoveries
    if len(unique_to_venn) > 0:
        unique_df = predictions[predictions["id"].isin(unique_to_venn)]
        print(f"\n   Sample of unique discoveries (not in baseline):")
        print(unique_df.head(10).to_string(index=False))

        # Check precision of unique discoveries
        unique_ids_list = list(unique_to_venn)
        if len(unique_ids_list) > 0:
            actual_positives = y_test[unique_ids_list].sum()
            precision = actual_positives / len(unique_ids_list)
            print(f"\n   Precision of unique discoveries: {precision:.2%}")

    # Show top predictions
    print(f"\n   Top 15 predictions by sets count:")
    print(predictions.head(15).to_string(index=False))

    print("\n7. Model export...")
    print(f"   All models exported to: {clf.elr2json}")
    print(f"   Selected models exported to: {clf.modeljson}")

    print("\n" + "=" * 60)
    print("Venn method example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
