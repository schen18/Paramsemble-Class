"""Phase 3: WoE bins fit on the SELECTION regime (adaptive) vs TRAIN (stale)."""

import sys, warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)
sys.path.insert(0, "benchmarks")
import numpy as np
from optbinning import OptimalBinning
from run_benchmark import make_dataset, topn_metrics, ELR_PARAMS
from paramsemble_class import ELRClassifier
from sklearn.metrics import roc_auc_score


def transform_with(X, obs):
    Z = np.empty_like(X, dtype=float)
    for j, ob in enumerate(obs):
        Z[:, j] = ob.transform(X[:, j], metric="woe")
    return Z


def fit_bins(A, y):
    return [
        OptimalBinning(dtype="numerical", max_n_bins=10, min_bin_size=0.05).fit(A[:, j], y)
        for j in range(A.shape[1])
    ]


agg = {}
for seed in (42, 43, 44):
    Xtr, ytr, Xsel, ysel, Xte, yte = make_dataset("drift_rare_2pct", seed)

    for tag, bin_src in [
        ("bins=train(stale)", (Xtr, ytr)),
        ("bins=selection(adaptive)", (Xsel, ysel)),
    ]:
        obs = fit_bins(*bin_src)
        Ztr, Zsel, Zte = (
            transform_with(Xtr, obs),
            transform_with(Xsel, obs),
            transform_with(Xte, obs),
        )
        clf = ELRClassifier(random_state=seed, **{**ELR_PARAMS, "f_range": [4, 8, 12]})
        clf.fit(Ztr, ytr, Zsel, ysel, np.arange(len(Zsel)))
        det = clf.predict_detailed(Zte, np.arange(len(Zte)))
        s = np.zeros(len(Zte))
        if len(det):
            s[det["id"].values.astype(int)] = det["score"].values
        agg.setdefault(tag, []).append(
            (
                topn_metrics(yte, s, max(int(0.1 * len(yte)), 1))[0],
                topn_metrics(yte, s, max(int(0.2 * len(yte)), 1))[0],
                roc_auc_score(yte, s),
                clf.fell_back_to_baseline_,
                len(clf.selected_indices_),
            )
        )

    # raw baseline for reference (same seeds as main benchmark)
    clf = ELRClassifier(random_state=seed, **{**ELR_PARAMS, "f_range": [4, 8, 12]})
    clf.fit(Xtr, ytr, Xsel, ysel, np.arange(len(Xsel)))
    det = clf.predict_detailed(Xte, np.arange(len(Xte)))
    s = np.zeros(len(Xte))
    if len(det):
        s[det["id"].values.astype(int)] = det["score"].values
    agg.setdefault("raw(no binning)", []).append(
        (
            topn_metrics(yte, s, max(int(0.1 * len(yte)), 1))[0],
            topn_metrics(yte, s, max(int(0.2 * len(yte)), 1))[0],
            roc_auc_score(yte, s),
            clf.fell_back_to_baseline_,
            len(clf.selected_indices_),
        )
    )

print("drift_rare_2pct (3 seeds)          cap@10%  cap@20%   AUC    fb   sel")
for tag, vals in agg.items():
    m = np.mean([[v[0], v[1], v[2]] for v in vals], axis=0)
    fb = np.mean([v[3] for v in vals])
    sel = np.mean([v[4] for v in vals])
    print(f"{tag:28s} {m[0]:7.3f} {m[1]:8.3f} {m[2]:6.3f} {fb:4.0%} {sel:4.1f}")
