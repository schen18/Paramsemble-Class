"""Aggregate benchmark results into markdown tables for the report.

Reads benchmarks/results.csv (written by run_benchmark.py) and produces
per-dataset mean +/- std tables plus cross-dataset summaries, including
fixed-budget capture (independent of ELR's emergent output size) and the
ELR baseline-fallback rate. Output is markdown for BENCHMARK_REPORT.md.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results.csv")

MODEL_ORDER = [
    "elr_intersect",
    "elr_intersect_cov",
    "elr_intersect_mw",
    "elr_venn",
    "elr_ensemble",
    "elr_ensemble_mw",
    "logreg",
    "random_forest",
    "hist_gb",
    "xgboost",
]


def fmt(mean, std, pct=False):
    if pct:
        return f"{mean * 100:.1f}±{std * 100:.1f}"
    return f"{mean:.2f}±{std:.2f}"


def dataset_table(df, name):
    sub = df[df.dataset == name]
    meta = sub.iloc[0]

    fallback = sub[sub.model == "elr_intersect"].notes.astype(str)
    fb_rate = fallback.str.contains("fallback").mean()

    lines = [
        f"### {name}",
        "",
        f"- test rows: {int(meta.n_test)}, positives: {int(meta.n_pos_test)} "
        f"(prevalence {meta.prevalence:.1%})",
        f"- budget N (ELR-intersect set size): {int(sub.n.mean())} rows "
        f"({sub.n_pct.mean():.0%} of test, range {sub.n_pct.min():.0%}-"
        f"{sub.n_pct.max():.0%}), identical for every model within a seed",
        f"- ELR baseline fallback (0 models selected): {fb_rate:.0%} of seeds",
        "",
        "| model | capture@N | capture@10% | capture@20% | precision@N | lift@N | AUC | fit (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in MODEL_ORDER:
        rows = sub[sub.model == model]
        if rows.empty:
            continue
        cells = [
            fmt(rows["capture_at_n"].mean(), rows["capture_at_n"].std(), pct=True),
            fmt(rows["capture_10pct"].mean(), rows["capture_10pct"].std(), pct=True),
            fmt(rows["capture_20pct"].mean(), rows["capture_20pct"].std(), pct=True),
            fmt(rows["precision_at_n"].mean(), rows["precision_at_n"].std(), pct=True),
            fmt(rows["lift_at_n"].mean(), rows["lift_at_n"].std()),
            fmt(rows["auc"].mean(), rows["auc"].std()),
            f"{rows.fit_seconds.mean():.1f}",
        ]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def rank_table(df, metric):
    ranks = []
    for _, sub in df.groupby("dataset"):
        means = sub.groupby("model")[metric].mean().reindex(MODEL_ORDER)
        ranks.append(means.rank(ascending=False))
    rank = pd.concat(ranks, axis=1).mean(axis=1).sort_values()
    lines = [
        f"Mean rank by {metric} across all datasets (lower is better):",
        "",
        "| model | mean rank |",
        "|---|---|",
    ]
    for model, r in rank.items():
        lines.append(f"| {model} | {r:.1f} |")
    return "\n".join(lines)


def winner_table(df, metric):
    lines = [
        f"Winner per dataset by mean {metric}:",
        "",
        "| dataset | winner | score | ELR best | ELR best score |",
        "|---|---|---|---|---|",
    ]
    for name, sub in df.groupby("dataset"):
        means = sub.groupby("model")[metric].mean()
        winner = means.idxmax()
        elr_best = means[[m for m in means.index if m.startswith("elr_")]].idxmax()
        lines.append(
            f"| {name} | **{winner}** | {means.max():.2f} | {elr_best} "
            f"| {means[elr_best]:.2f} |"
        )
    return "\n".join(lines)


def main():
    df = pd.read_csv(RESULTS_PATH)

    print(winner_table(df, "capture_at_n"))
    print()
    print(rank_table(df, "capture_at_n"))
    print()
    print(rank_table(df, "capture_10pct"))
    print()
    for name in df.dataset.unique():
        print(dataset_table(df, name))
        print()


if __name__ == "__main__":
    main()
