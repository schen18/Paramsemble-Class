"""Spike: does global WoE preprocessing (optbinning) improve ELR performance?

Hypothesis test BEFORE building the production per-constituent binning:
fit one OptimalBinning per feature on the training split, WoE-transform
every split, then run the *existing* ELR unchanged on the transformed
features. Compare against raw-feature ELR (and raw/woe logreg for
reference) on the benchmark protocol's fixed budgets.
"""

import sys
import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)
sys.path.insert(0, "benchmarks")

import numpy as np
from optbinning import OptimalBinning

from run_benchmark import make_dataset, topn_metrics, ELR_PARAMS
from paramsemble_class import ELRClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def woe_transform(X_train, y_train, splits):
    """Fit one OptimalBinning per column on train; return transformed copies."""
    fitted = []
    for j in range(X_train.shape[1]):
        ob = OptimalBinning(dtype="numerical", max_n_bins=10, min_bin_size=0.05)
        ob.fit(X_train[:, j], y_train)
        fitted.append(ob)

    def transform(X):
        out = np.empty_like(X, dtype=float)
        for j, ob in enumerate(fitted):
            out[:, j] = ob.transform(X[:, j], metric="woe")
        return out, fitted

    return transform


def run_elr(X_train, y_train, X_sel, y_sel, X_test, y_test, seed, **over):
    clf = ELRClassifier(random_state=seed, **{**ELR_PARAMS, **over})
    clf.fit(X_train, y_train, X_sel, y_sel, np.arange(len(X_sel)))
    det = clf.predict_detailed(X_test, np.arange(len(X_test)))
    score = np.zeros(len(X_test))
    if len(det):
        score[det["id"].values.astype(int)] = det["score"].values
    cap10 = topn_metrics(y_test, score, max(int(0.10 * len(y_test)), 1))[0]
    cap20 = topn_metrics(y_test, score, max(int(0.20 * len(y_test)), 1))[0]
    auc = roc_auc_score(y_test, score)
    return cap10, cap20, auc, clf.fell_back_to_baseline_, len(clf.selected_indices_)


def main():
    datasets = ["heterogeneous", "rare_2pct", "drift_rare_2pct", "imbalanced_90_10"]
    print(
        f"{'dataset':18s} {'variant':22s} {'cap@10%':>8s} {'cap@20%':>8s} "
        f"{'AUC':>6s} {'fb':>4s} {'sel':>4s}"
    )
    for ds in datasets:
        rows = {}
        for seed in (42, 44):
            X_train, y_train, X_sel, y_sel, X_test, y_test = make_dataset(ds, seed)

            # raw ELR (f=4 rank, and mw) for the v2 baseline
            rows.setdefault("elr_raw", []).append(
                run_elr(X_train, y_train, X_sel, y_sel, X_test, y_test, seed)
            )
            rows.setdefault("elr_raw_mw", []).append(
                run_elr(
                    X_train,
                    y_train,
                    X_sel,
                    y_sel,
                    X_test,
                    y_test,
                    seed,
                    f_range=[4, 8, 12],
                )
            )

            # WoE-transformed (global bins from train) then the same ELR
            tr = woe_transform(X_train, y_train, None)
            Xtr_w, _ = tr(X_train)
            Xsel_w, fitted = tr(X_sel)

            def transform_with(X, fitted=fitted):
                out = np.empty_like(X, dtype=float)
                for j, ob in enumerate(fitted):
                    out[:, j] = ob.transform(X[:, j], metric="woe")
                return out

            Xte_w = transform_with(X_test)
            rows.setdefault("elr_woe", []).append(
                run_elr(Xtr_w, y_train, Xsel_w, y_sel, Xte_w, y_test, seed)
            )
            rows.setdefault("elr_woe_mw", []).append(
                run_elr(
                    Xtr_w,
                    y_train,
                    Xsel_w,
                    y_sel,
                    Xte_w,
                    y_test,
                    seed,
                    f_range=[4, 8, 12],
                )
            )

            # reference: full logreg raw vs woe
            lr_raw = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                np.vstack([X_train, X_sel]), np.concatenate([y_train, y_sel])
            )
            s = lr_raw.predict_proba(X_test)[:, 1]
            rows.setdefault("logreg_raw", []).append(
                (
                    topn_metrics(y_test, s, max(int(0.1 * len(y_test)), 1))[0],
                    topn_metrics(y_test, s, max(int(0.2 * len(y_test)), 1))[0],
                    roc_auc_score(y_test, s),
                    False,
                    0,
                )
            )
            lr_woe = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                np.vstack([Xtr_w, Xsel_w]), np.concatenate([y_train, y_sel])
            )
            s = lr_woe.predict_proba(Xte_w)[:, 1]
            rows.setdefault("logreg_woe", []).append(
                (
                    topn_metrics(y_test, s, max(int(0.1 * len(y_test)), 1))[0],
                    topn_metrics(y_test, s, max(int(0.2 * len(y_test)), 1))[0],
                    roc_auc_score(y_test, s),
                    False,
                    0,
                )
            )

        for variant, vals in rows.items():
            m = np.mean([[v[0], v[1], v[2]] for v in vals], axis=0)
            fb = np.mean([v[3] for v in vals])
            sel = np.mean([v[4] for v in vals])
            print(
                f"{ds:18s} {variant:22s} {m[0]:8.3f} {m[1]:8.3f} {m[2]:6.3f} "
                f"{fb:4.0%} {sel:4.1f}"
            )
        print()


if __name__ == "__main__":
    main()
