"""Compare v1 (pre-selection-v2) and v2 benchmark results.

Attributes the change in ELR performance to each selection-v2 axis:
- weighted score ranking: v1 sets-ranked vs v2 score-ranked intersect
  (both classic 'rank' selection)
- coverage selection:     v2 rank vs v2 coverage (both f=4)
- multiwidth pool:        v2 coverage f=4 vs v2 coverage f_range=[4,8,12]
- ensemble multiwidth:    v2 ensemble vs v2 ensemble multiwidth

Competitor numbers are identical across runs (same seeds/data), so old
competitor rows are quoted directly for context.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = pd.read_csv(os.path.join(HERE, "results_v1.csv"))
V2 = pd.read_csv(os.path.join(HERE, "results.csv"))

METRICS = ["capture_at_n", "capture_10pct", "capture_20pct"]


def mean_metric(df, dataset, model, metric):
    sub = df[(df.dataset == dataset) & (df.model == model)]
    return sub[metric].mean(), sub[metric].std(), len(sub)


def main():
    datasets = sorted(V2.dataset.unique())
    pairs = [
        ("weighting (score vs sets)", "elr_intersect", V1, "elr_intersect", V2),
        ("coverage selection", "elr_intersect", V2, "elr_intersect_cov", V2),
        ("multiwidth pool", "elr_intersect_cov", V2, "elr_intersect_mw", V2),
        ("ensemble multiwidth", "elr_ensemble", V2, "elr_ensemble_mw", V2),
    ]

    print("## Selection v2 deltas (capture@N / capture@10%)\n")
    print(
        "| dataset | weighting | coverage | multiwidth | ens+multiwidth | v2 best vs best competitor |"
    )
    print("|---|---|---|---|---|---|")

    for ds in datasets:
        cells = []
        for label, m_old, df_old, m_new, df_new in pairs:
            old, _, n1 = mean_metric(df_old, ds, m_old, "capture_at_n")
            new, _, n2 = mean_metric(df_new, ds, m_new, "capture_at_n")
            old10, _, _ = mean_metric(df_old, ds, m_old, "capture_10pct")
            new10, _, _ = mean_metric(df_new, ds, m_new, "capture_10pct")
            if n1 == 0 or n2 == 0:
                cells.append("—")
            else:
                cells.append(f"{new - old:+.2f} / {new10 - old10:+.2f}")

        # v2 ELR best vs best competitor on this dataset
        elr_cols = [m for m in V2.model.unique() if m.startswith("elr_")]
        comp_cols = [m for m in V2.model.unique() if not m.startswith("elr_")]
        elr_best = max(
            (mean_metric(V2, ds, m, "capture_at_n")[0] for m in elr_cols),
        )
        comp_best = max(
            (mean_metric(V2, ds, m, "capture_at_n")[0] for m in comp_cols),
        )
        winner = "ELR" if elr_best >= comp_best else "comp."
        cells.append(f"{winner} ({elr_best:.2f} vs {comp_best:.2f})")
        print(f"| {ds} | " + " | ".join(cells) + " |")

    # Overall means on the drift dataset + all-dataset averages
    print("\n### drift_rare_2pct detail (mean over seeds)\n")
    sub = (
        V2[V2.dataset == "drift_rare_2pct"]
        .groupby("model")[["capture_at_n", "capture_10pct", "capture_20pct", "auc"]]
        .mean()
        .round(3)
    )
    print(sub.to_string())

    print("\n### fallback / selection rates (v2, per dataset)\n")
    for ds in datasets:
        row = V2[(V2.dataset == ds) & (V2.model == "elr_intersect_mw")]
        fb = row.notes.astype(str).str.contains("fallback").mean()
        sel = row.notes.astype(str).str.extract(r"selected=(\d+)")[0].astype(float).mean()
        print(f"{ds:20s} mw: fallback={fb:.0%} mean selected={sel:.1f}")


if __name__ == "__main__":
    main()
