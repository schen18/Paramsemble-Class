"""Sensitivity of ELR performance to the features-per-model parameter f."""

import sys, warnings

warnings.filterwarnings("ignore")
import numpy as np, logging

logging.disable(logging.CRITICAL)
sys.path.insert(0, "benchmarks")
from run_benchmark import make_dataset, topn_metrics
from paramsemble_class import ELRClassifier


def run(ds, f, seed, method):
    X_train, y_train, X_sel, y_sel, X_test, y_test = make_dataset(ds, seed)
    clf = ELRClassifier(m=50, f=f, d=2, spread=10, method=method, random_state=seed)
    clf.fit(X_train, y_train, X_sel, y_sel, np.arange(len(X_sel)))
    det = clf.predict_detailed(X_test, np.arange(len(X_test)))
    if method == "ensemble":
        score = clf.predict_proba(X_test)[:, 1]
    else:
        score = np.zeros(len(X_test))
        score[det["id"].values.astype(int)] = det["sets"].values
    n = max(len(det), 1)
    cap_n, prec_n, _ = topn_metrics(y_test, score, n)
    cap10, _, _ = topn_metrics(y_test, score, max(int(0.10 * len(y_test)), 1))
    fb = clf.fell_back_to_baseline_
    return cap_n, cap10, n / len(y_test), len(clf.selected_indices_), fb


print(
    f"{'dataset':16s} {'f':>2s} {'method':10s} {'cap@N':>6s} {'cap@10%':>7s} {'N%':>4s} {'sel':>3s} {'fb':>4s}"
)
for ds in ["rare_2pct", "drift_rare_2pct", "heterogeneous"]:
    for f in [4, 8, 12]:
        for method in ["intersect", "ensemble"]:
            rows = [run(ds, f, s, method) for s in (42, 44)]
            cap_n = np.mean([r[0] for r in rows])
            cap10 = np.mean([r[1] for r in rows])
            n_pct = np.mean([r[2] for r in rows])
            sel = np.mean([r[3] for r in rows])
            fb = np.mean([r[4] for r in rows])
            print(
                f"{ds:16s} {f:2d} {method:10s} {cap_n:6.3f} {cap10:7.3f} {n_pct:4.0%} {sel:5.1f} {fb:4.0%}"
            )
