"""Counterfactual: ELR with a WoE-binned full-feature LR baseline instead of RF.

Constituents are baseline-independent, so one fitted ELR per (dataset, seed)
serves both systems:
  A) current: gate/fallback vs Random Forest (as shipped)
  B) counterfactual: gate/fallback vs optbinning-WoE LogisticRegression
     on all features (bins fit on train)
Reference: external Random Forest trained on train+selection (the benchmark
competitor configuration).
"""

import sys, warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)
sys.path.insert(0, "benchmarks")
import numpy as np
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from run_benchmark import make_dataset, topn_metrics, ELR_PARAMS
from paramsemble_class import ELRClassifier
from paramsemble_class.ensemble.intersect import IntersectMethod
from paramsemble_class.ensemble._common import weights_for_selection
from paramsemble_class.metrics.performance import PerformanceMetrics
from paramsemble_class.metrics.performance import top_d_row_indices


def woe_fit(X, y):
    obs = [
        OptimalBinning(dtype="numerical", max_n_bins=10, min_bin_size=0.05).fit(X[:, j], y)
        for j in range(X.shape[1])
    ]
    return obs


def woe_apply(X, obs):
    Z = np.empty_like(X, dtype=float)
    for j, ob in enumerate(obs):
        Z[:, j] = ob.transform(X[:, j], metric="woe")
    return Z


def baseline_metrics(model, X_sel, y_sel, ids, d):
    pred = model.predict(X_sel)
    score = model.predict_proba(X_sel)[:, 1]
    return {
        "plr": PerformanceMetrics.positive_likelihood_ratio(y_sel, pred),
        "fnr": PerformanceMetrics.false_negative_rate(y_sel, pred),
        "drp": PerformanceMetrics.decile_ranked_performance(y_sel, score, d),
        "drs": PerformanceMetrics.extract_decile_ranked_set(ids, score, d),
        "dps": PerformanceMetrics.extract_decile_positive_set(ids, y_sel, score, d),
    }


def score_from_detailed(det, n):
    s = np.zeros(n)
    if len(det):
        s[det["id"].values.astype(int)] = det["score"].values
    return s


def evaluate(score, y_test):
    return (
        topn_metrics(y_test, score, max(int(0.10 * len(y_test)), 1))[0],
        topn_metrics(y_test, score, max(int(0.20 * len(y_test)), 1))[0],
        roc_auc_score(y_test, score),
    )


print(f"{'dataset':18s} {'system':22s} {'cap@10%':>8s} {'cap@20%':>8s} {'AUC':>6s} {'act':>5s}")
for ds in [
    "balanced",
    "imbalanced_90_10",
    "rare_2pct",
    "rare_1pct",
    "heterogeneous",
    "drift_rare_2pct",
    "drift_balanced",
]:
    agg = {}
    for seed in (42, 44):
        Xtr, ytr, Xsel, ysel, Xte, yte = make_dataset(ds, seed)
        ids_sel, ids_te = np.arange(len(Xsel)), np.arange(len(Xte))

        # one ELR fit, constituents shared by both systems
        clf = ELRClassifier(random_state=seed, **{**ELR_PARAMS, "f_range": [4, 8, 12]})
        clf.fit(Xtr, ytr, Xsel, ysel, ids_sel)

        # A) current behavior (RF baseline)
        agg.setdefault("A rf-baseline", []).append(
            evaluate(score_from_detailed(clf.predict_detailed(Xte, ids_te), len(Xte)), yte)
            + (not clf.fell_back_to_baseline_,)
        )

        # B) counterfactual: WoE LR baseline
        obs = woe_fit(Xtr, ytr)
        blr = LogisticRegression(max_iter=2000).fit(woe_apply(Xtr, obs), ytr)
        bl_sel = baseline_metrics(blr, woe_apply(Xsel, obs), ysel, ids_sel, clf.d)

        ranked = IntersectMethod._rank_models(clf.constituent_results_, bl_sel)
        sel = ranked[: clf.spread]
        if sel:
            clf.selected_indices_ = sel
            clf.selection_weights_ = weights_for_selection(sel, ranked)
            det = clf.predict_detailed(Xte, ids_te)
            agg.setdefault("B woelr-baseline", []).append(
                evaluate(score_from_detailed(det, len(Xte)), yte) + (True,)
            )
        else:
            # fallback to the binned LR itself: top-d deciles
            sc = blr.predict_proba(woe_apply(Xte, obs))[:, 1]
            agg.setdefault("B woelr-baseline", []).append(evaluate(sc, yte) + (False,))

        # reference: external RF on train+selection (competitor config)
        rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1).fit(
            np.vstack([Xtr, Xsel]), np.concatenate([ytr, ysel])
        )
        agg.setdefault("C ext-RF (ref)", []).append(
            evaluate(rf.predict_proba(Xte)[:, 1], yte) + (True,)
        )

    for tag, vals in agg.items():
        m = np.mean([[v[0], v[1], v[2]] for v in vals], axis=0)
        act = np.mean([v[3] for v in vals])
        print(f"{ds:18s} {tag:22s} {m[0]:8.3f} {m[1]:8.3f} {m[2]:6.3f} {act:5.0%}")
    print()
