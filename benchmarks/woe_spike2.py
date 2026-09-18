"""Phase 2: does WoE help when features are messy (non-monotone / heavy-tail)?"""

import sys, warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)
sys.path.insert(0, "benchmarks")
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from optbinning import OptimalBinning
from run_benchmark import topn_metrics, ELR_PARAMS
from paramsemble_class import ELRClassifier


def make_messy(name, seed):
    rng = np.random.RandomState(seed)
    X, y = make_classification(
        n_samples=8000,
        n_features=20,
        n_informative=12,
        n_redundant=3,
        weights=[0.95, 0.05],
        random_state=seed,
    )
    if name == "nonmonotone":
        # U-shaped risk on first 6 features: risk grows with distance from 0
        for j in range(6):
            X[:, j] = np.abs(X[:, j]) * rng.choice([1.0, -1.0], size=len(X))
    elif name == "heavytail":
        # lognormal stretch + extreme outliers on informative features
        for j in range(12):
            X[:, j] = X[:, j] * np.exp(rng.randn(len(X)) * 0.8)
            out = rng.rand(len(X)) < 0.02
            X[out, j] *= 50.0
    X_tr, X_r, y_tr, y_r = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    X_sel, X_te, y_sel, y_te = train_test_split(
        X_r, y_r, test_size=0.5, random_state=seed, stratify=y_r
    )
    return X_tr, y_tr, X_sel, y_sel, X_te, y_te


def woe_fit_transform(X_tr, y_tr, *blocks):
    obs = [
        OptimalBinning(dtype="numerical", max_n_bins=10, min_bin_size=0.05).fit(X_tr[:, j], y_tr)
        for j in range(X_tr.shape[1])
    ]
    outs = []
    for X in blocks:
        Z = np.empty_like(X, dtype=float)
        for j, ob in enumerate(obs):
            Z[:, j] = ob.transform(X[:, j], metric="woe")
        outs.append(Z)
    return outs


def score_elr(Xtr, ytr, Xsel, ysel, Xte, yte, seed, **over):
    clf = ELRClassifier(random_state=seed, **{**ELR_PARAMS, **over})
    clf.fit(Xtr, ytr, Xsel, ysel, np.arange(len(Xsel)))
    det = clf.predict_detailed(Xte, np.arange(len(Xte)))
    s = np.zeros(len(Xte))
    if len(det):
        s[det["id"].values.astype(int)] = det["score"].values
    return s, clf.fell_back_to_baseline_, len(clf.selected_indices_)


def cap(s, yte, frac):
    return topn_metrics(yte, s, max(int(frac * len(yte)), 1))[0]


for name in ["nonmonotone", "heavytail"]:
    print(f"\n=== {name} (5% prevalence) ===")
    print(f"{'variant':16s} {'cap@10%':>8s} {'cap@20%':>8s} {'AUC':>6s} {'fb':>4s} {'sel':>4s}")
    agg = {}
    for seed in (42, 44):
        Xtr, ytr, Xsel, ysel, Xte, yte = make_messy(name, seed)
        Ztr, Zsel, Zte = woe_fit_transform(Xtr, ytr, Xtr, Xsel, Xte)

        for tag, args in [
            ("elr_raw_mw", (Xtr, ytr, Xsel, ysel, Xte, yte, dict(f_range=[4, 8, 12]))),
            ("elr_woe_mw", (Ztr, ytr, Zsel, ysel, Zte, yte, dict(f_range=[4, 8, 12]))),
        ]:
            s, fb, sel = score_elr(*args[:6], seed, **args[6])
            agg.setdefault(tag, []).append(
                (cap(s, yte, 0.1), cap(s, yte, 0.2), roc_auc_score(yte, s), fb, sel)
            )

        for tag, (A, B, C) in [("logreg_raw", (Xtr, Xsel, Xte)), ("logreg_woe", (Ztr, Zsel, Zte))]:
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                np.vstack([A, Xsel if A is Xtr else Zsel]), np.concatenate([ytr, ysel])
            )
            s = lr.predict_proba(C)[:, 1]
            agg.setdefault(tag, []).append(
                (cap(s, yte, 0.1), cap(s, yte, 0.2), roc_auc_score(yte, s), False, 0)
            )

        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        ).fit(np.vstack([Xtr, Xsel]), np.concatenate([ytr, ysel]))
        s = rf.predict_proba(Xte)[:, 1]
        agg.setdefault("random_forest", []).append(
            (cap(s, yte, 0.1), cap(s, yte, 0.2), roc_auc_score(yte, s), False, 0)
        )

    for tag, vals in agg.items():
        m = np.mean([[v[0], v[1], v[2]] for v in vals], axis=0)
        fb = np.mean([v[3] for v in vals])
        sel = np.mean([v[4] for v in vals])
        print(f"{tag:16s} {m[0]:8.3f} {m[1]:8.3f} {m[2]:6.3f} {fb:4.0%} {sel:4.1f}")
