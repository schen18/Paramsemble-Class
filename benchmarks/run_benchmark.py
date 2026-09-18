"""Head-to-head benchmark: ELR vs scikit-learn and XGBoost classifiers.

Protocol
--------
Each dataset is split three ways (stratified): train (50%), selection (25%),
test (25%).

- ELR fits its constituent models on *train* and uses *selection* for its
  internal model selection (its required workflow).
- Competitors (LogisticRegression, RandomForest, HistGradientBoosting,
  XGBoost) fit on *train + selection combined* -- i.e. they receive MORE
  labeled data than ELR's constituents, a deliberate handicap against ELR.
  Competitors get their standard imbalance handling (class_weight="balanced"
  / scale_pos_weight); ELR uses its default configuration, since metric-based
  selection is its own mechanism for imbalance.

Equal-budget Top-N comparison
-----------------------------
The budget N is defined by ELR-intersect's targeting output size on the test
set (the union of the selected models' top-d deciles). Every model is then
evaluated on its own top-N rows by score:

- capture@N : fraction of all test positives inside the top-N set (recall)
- precision@N : fraction of the top-N set that is truly positive
- lift@N     : precision@N / prevalence

Scores used for ranking: predicted probability for competitors and
ELR-ensemble; vote counts ("sets") for ELR-intersect/venn (absent ids rank
last). ROC-AUC is reported for reference.

Run:  python benchmarks/run_benchmark.py [--quick]
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from paramsemble_class import ELRClassifier

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results.csv")

ELR_PARAMS = dict(m=50, f=4, d=2, spread=10, sample="unique", n_jobs=1)

# ELR variants: (name, constructor overrides). The first entry defines the
# equal-budget N (same convention as the v1 benchmark).
ELR_VARIANTS = [
    ("elr_intersect", {}),
    ("elr_intersect_cov", {"selection": "coverage"}),
    ("elr_intersect_mw", {"selection": "coverage", "f_range": [4, 8, 12]}),
    ("elr_venn", {"method": "venn"}),
    ("elr_ensemble", {"method": "ensemble"}),
    ("elr_ensemble_mw", {"method": "ensemble", "f_range": [4, 8, 12]}),
]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def mixed_scales(X, rng, n_blocks=6):
    """Heterogeneous proxy: stretch random feature blocks to wild scales."""
    X = X.copy()
    for _ in range(n_blocks):
        cols = rng.choice(X.shape[1], size=max(1, X.shape[1] // 6), replace=False)
        X[:, cols] *= rng.choice([0.01, 10.0, 100.0])
    return X


def make_dataset(name, seed):
    rng = np.random.RandomState(seed)
    common = dict(n_informative=12, n_redundant=4, n_classes=2, random_state=seed)

    if name == "balanced":
        X, y = make_classification(n_samples=6000, n_features=20, class_sep=1.0, **common)
    elif name == "imbalanced_90_10":
        X, y = make_classification(
            n_samples=6000, n_features=20, class_sep=1.0, weights=[0.9, 0.1], **common
        )
    elif name == "rare_5pct":
        X, y = make_classification(
            n_samples=8000, n_features=20, class_sep=1.0, weights=[0.95, 0.05], **common
        )
    elif name == "rare_2pct":
        X, y = make_classification(
            n_samples=12000, n_features=20, class_sep=1.0, weights=[0.98, 0.02], **common
        )
    elif name == "rare_1pct":
        X, y = make_classification(
            n_samples=16000, n_features=20, class_sep=1.0, weights=[0.99, 0.01], **common
        )
    elif name == "heterogeneous":
        X, y = make_classification(
            n_samples=8000,
            n_features=30,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            weights=[0.9, 0.1],
            flip_y=0.1,
            class_sep=1.2,
            random_state=seed,
        )
        X = mixed_scales(X, rng)
    elif name == "drift_rare_2pct":
        # "Behavior change" regime at 2% prevalence: clean training period,
        # degraded-signal selection/test period (informative features
        # compressed toward the mean + extra noise). Prevalence unchanged.
        # shuffle=False keeps the first 12+4 columns informative/redundant
        # so the decay below targets the actual signal.
        X, y = make_classification(
            n_samples=12000,
            n_features=20,
            class_sep=1.0,
            weights=[0.98, 0.02],
            flip_y=0.02,
            shuffle=False,
            random_state=seed,
        )
    elif name == "drift_balanced":
        # Label-noise drift on a balanced base: training period clean (5%
        # flips conceptually), deployment period has 18% flipped labels.
        X, y = make_classification(
            n_samples=8000,
            n_features=20,
            class_sep=1.0,
            flip_y=0.05,
            n_informative=12,
            n_redundant=4,
            n_classes=2,
            random_state=seed,
        )
    else:
        raise ValueError(name)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )
    X_sel, X_test, y_sel, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )

    if name == "drift_rare_2pct":
        # Signal decay in the later period: compress the 16 signal columns
        # (12 informative + 4 redundant, unshuffled) toward their training
        # means and add noise -- prevalence is untouched.
        n_signal = 16
        mu = X_train[:, :n_signal].mean(axis=0)
        for block in (X_sel, X_test):
            block[:, :n_signal] = (
                0.45 * block[:, :n_signal] + 0.55 * mu + rng.randn(*block[:, :n_signal].shape) * 0.6
            )
    elif name == "drift_balanced":
        # Label-noise drift: 18% of later-period labels flipped.
        flip = rng.rand(len(y_sel) + len(y_test)) < 0.18
        y_sel = np.where(flip[: len(y_sel)], 1 - y_sel, y_sel)
        y_test = np.where(flip[len(y_sel) :], 1 - y_test, y_test)

    return X_train, y_train, X_sel, y_sel, X_test, y_test


DATASETS = [
    "balanced",
    "imbalanced_90_10",
    "rare_5pct",
    "rare_2pct",
    "rare_1pct",
    "heterogeneous",
    "drift_rare_2pct",
    "drift_balanced",
]


# ---------------------------------------------------------------------------
# Competitors
# ---------------------------------------------------------------------------


def fit_competitors(X_fit, y_fit, X_test, seed):
    """Fit sklearn/XGBoost models on train+selection; return {name: score}."""
    n_pos = max(int((y_fit == 1).sum()), 1)
    n_neg = max(int((y_fit == 0).sum()), 1)
    spw = n_neg / n_pos

    specs = {
        "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gb": lambda: HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            random_state=seed,
            class_weight="balanced",
        ),
        "xgboost": lambda: __import__("xgboost").XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=spw,
            tree_method="hist",
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        ),
    }

    scores, times = {}, {}
    for name, factory in specs.items():
        t0 = time.perf_counter()
        model = factory().fit(X_fit, y_fit)
        scores[name] = model.predict_proba(X_test)[:, 1]
        times[name] = time.perf_counter() - t0
    return scores, times


def fit_elr(X_train, y_train, X_sel, y_sel, seed, method="intersect", **overrides):
    t0 = time.perf_counter()
    params = dict(ELR_PARAMS)
    params.update(overrides)
    clf = ELRClassifier(method=method, random_state=seed, **params)
    clf.fit(X_train, y_train, X_sel, y_sel, np.arange(len(X_sel)))
    return clf, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def topn_metrics(y_true, score, n):
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    top = order[:n]
    n_pos_total = int((y_true == 1).sum())
    hits = int((y_true[top] == 1).sum())
    precision = hits / n if n else 0.0
    prevalence = n_pos_total / len(y_true)
    recall = hits / n_pos_total if n_pos_total else 0.0
    lift = precision / prevalence if prevalence > 0 else 0.0
    return recall, precision, lift


def evaluate(name, y_test, score, n, t, extra=""):
    recall, precision, lift = topn_metrics(y_test, score, n)
    auc = roc_auc_score(y_test, score) if len(np.unique(y_test)) == 2 else float("nan")
    # Fixed-budget capture, independent of ELR's emergent set size
    fixed = {}
    for frac in (0.10, 0.20):
        k = max(int(round(frac * len(y_test))), 1)
        fixed[f"capture_{int(frac * 100)}pct"] = topn_metrics(y_test, score, k)[0]
    return dict(
        model=name,
        capture_at_n=recall,
        precision_at_n=precision,
        lift_at_n=lift,
        auc=auc,
        fit_seconds=t,
        notes=extra,
        **fixed,
    )


def run_dataset(name, seed):
    X_train, y_train, X_sel, y_sel, X_test, y_test = make_dataset(name, seed)
    ids_test = np.arange(len(X_test))
    n_pos_test = int((y_test == 1).sum())

    # --- ELR (fits on train; selects on selection set) ---
    elr_rows = []
    intersect_budget = None

    for variant_name, overrides in ELR_VARIANTS:
        method = overrides.get("method", "intersect")
        clf, t = fit_elr(X_train, y_train, X_sel, y_sel, seed, **overrides)
        detailed = clf.predict_detailed(X_test, ids_test)
        notes = f"selected={len(clf.selected_indices_)}"
        if clf.fell_back_to_baseline_:
            notes += ";baseline_fallback"
        if overrides.get("selection"):
            notes += f";{overrides['selection']}"
        if overrides.get("f_range"):
            notes += ";multiwidth"

        if method == "ensemble":
            score = clf.predict_proba(X_test)[:, 1]
        else:
            # rank by the continuous weighted score (vote weights);
            # IDs outside every model's top-d deciles score zero
            score = np.zeros(len(X_test))
            if len(detailed):
                score[detailed["id"].values.astype(int)] = detailed["score"].values

        if variant_name == "elr_intersect":
            intersect_budget = max(len(detailed), 1)

        elr_rows.append(
            evaluate(
                variant_name,
                y_test,
                score,
                intersect_budget or max(len(detailed), 1),
                t,
                notes,
            )
        )

    # budget N from ELR-intersect's output set; every model uses the same N
    n_budget = intersect_budget

    # --- Competitors (fit on train + selection = more data than ELR) ---
    X_fit = np.vstack([X_train, X_sel])
    y_fit = np.concatenate([y_train, y_sel])
    comp_scores, comp_times = fit_competitors(X_fit, y_fit, X_test, seed)

    rows = []
    for row in elr_rows:
        row.update(n=n_budget)
        rows.append(row)
    for cname, score in comp_scores.items():
        rows.append(evaluate(cname, y_test, score, n_budget, comp_times[cname]))

    base = dict(
        dataset=name,
        seed=seed,
        n_test=len(y_test),
        n_pos_test=n_pos_test,
        prevalence=n_pos_test / len(y_test),
        n=n_budget,
        n_pct=n_budget / len(y_test),
    )
    return [dict(base, **r) for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--quick", action="store_true", help="1 seed, small datasets")
    args = parser.parse_args()

    datasets = DATASETS
    seeds = args.seeds
    if args.quick:
        datasets = ["balanced", "rare_2pct", "drift_rare_2pct"]
        seeds = seeds[:1]

    all_rows = []
    for name in datasets:
        for seed in seeds:
            t0 = time.perf_counter()
            rows = run_dataset(name, seed)
            all_rows.extend(rows)
            best = max(rows, key=lambda r: r["capture_at_n"])
            print(
                f"[{name} seed={seed}] n_test={rows[0]['n_test']} "
                f"pos={rows[0]['n_pos_test']} budget={rows[0]['n']} "
                f"({rows[0]['n_pct']:.0%}) | best capture@N: "
                f"{best['model']} ({best['capture_at_n']:.2f}) | "
                f"{time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
