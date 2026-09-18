# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-09-15

Production-hardening release: correctness fixes for the scoring pipeline,
honest evaluation for the ensemble method, genuine scikit-learn
compatibility, and packaging cleanup.

### Fixed
- **Consistent decile rule everywhere.** Training, `predict_detailed`,
  `ModelScorer` and the generated SQL previously used three different
  top-d-decile cutoffs (`ceil`, `floor`, and `NTILE`), so the same model
  produced different targeting lists depending on the scoring path. All
  paths now use one rule: `ceil(d * n / 10)` highest-scoring rows,
  ties broken by row order (`paramsemble_class.metrics.performance.top_d_count`).
- **Ensemble method leakage.** The meta-model was previously fit *and*
  scored on the same selection rows, producing optimistically biased
  probabilities. It is now fit on out-of-fold predictions of the
  selection set (stratified 5-fold, reduced automatically for small data).
- **Ensemble method crash when no models pass the baseline gate.**
  All methods now fall back to the baseline Random Forest (with a
  warning) instead of raising or returning fabricated zeros.
- **Numpy round-trip scoring.** `fit` on numpy arrays followed by
  `ModelScorer.score` on numpy arrays now works: modeljson exports
  record the ordered `feature_names`.
- **`id_column` is now honored.** If a DataFrame contains the ID column
  it is excluded from features (previously a numeric ID silently became
  a feature) and used as the selection-set IDs when none are passed.
- **Ranking no longer dominated by PLR.** Model selection used the raw
  composite `plr - fnr + drp`, which unbounded PLR values dominated.
  Selection now uses a Borda count over ranks of the three metrics.
- **FeatureSampler "replace" mode** now samples multicombinations
  uniformly (stars-and-bars) instead of an ordered-draw distribution,
  matching the documented cap `C(n+f-1, f)`, and warns when fewer than
  the requested combinations can be produced.
- **DRP documentation.** The metric was documented as a "TPR ratio" but
  actually computes lift: precision in the top d deciles divided by
  prevalence. Docs and internals now say so; values are unchanged.
- Non-numeric DataFrame columns are dropped with a warning instead of
  silently; JSON exports are strict (non-finite values serialize as
  `null`, numpy ID types are preserved); `table_name`/`id_column` are
  quoted in generated SQL; SQL CTEs no longer rely on `SELECT *`.
- README examples called `fit` without the required selection-set
  arguments and would raise `TypeError`; all examples fixed.

### Added
- scikit-learn calling convention: `clf.fit(X, y)` splits off a
  stratified selection set internally, `predict` returns class labels,
  `predict_proba` returns `(n, 2)` probabilities for *all* methods
  (vote share for intersect/venn), and `classes_` is set.
  `cross_val_score` and `GridSearchCV` work.
- `predict_detailed(X, ids=None)` — the method-specific targeting
  tables (formerly the return value of `predict`).
- `class_weight` parameter (e.g. `"balanced"`) passed to constituents,
  the meta-model and the baseline forest.
- `n_jobs` parameter for parallel constituent training (joblib).
- Baseline fallback mode (`fell_back_to_baseline_`) when no constituent
  beats the baseline gate.
- CI workflow (pytest on 3.8-3.12 across OSes, black, flake8, mypy).
- **Weighted vote scores for intersect/venn**: selected models vote with
  Borda-position weights, adding a continuous `score` column to
  `predict_detailed`, `result_df_`, `ModelScorer` output and the generated
  SQL (`SUM(contrib)` alongside `COUNT(*)`); `predict_proba` now returns
  the weighted vote share. `sets` counts are unchanged.
- **Multiwidth constituent pools** (`f_range=[4, 8, 12]`): candidates
  trained at several feature widths in one pool; the gate and ranking
  select across all widths. Benchmarked: +7 to +20 points capture at
  fixed budgets on rare-event datasets, and far fewer baseline-fallback
  fits.
- **Coverage selection strategy** (`selection="coverage"`): greedy
  submodular selection by marginal true-positive coverage (diversity-aware
  alternative to Borda top-k; default remains "rank").
- modeljson now records per-model `weights` (backward compatible: absent
  weights score as 1.0).
- Benchmark suite (`benchmarks/`): equal-budget Top-N head-to-head vs
  LogisticRegression / RandomForest / HistGradientBoosting / XGBoost across 8
  datasets (balanced, imbalanced, rare 1-5%, heterogeneous, two drift regimes)
  with report (`BENCHMARK_REPORT.md`) and an `f`-sensitivity appendix. Evidence
  for the package's positioning: rare-event regimes under distribution drift.

### Changed
- **Default baseline is now a WoE-binned full-feature logistic regression**
  (`baseline="woe_logreg"`), replacing the Random Forest reference. The
  linear gate is the correct comparison class for linear constituents
  (+0.03 to +0.10 capture@10% on 5 of 7 benchmark datasets) and makes the
  fallback SQL-deployable: when the gate rejects every constituent, the
  baseline equation exports to modeljson (`"fallback": true`, WoE bins as
  CASE WHEN ladders in SQL). `baseline="random_forest"` restores the old
  behavior; `baseline="logreg"` is the unbinned linear reference. Without
  `optbinning` installed (optional extra `paramsemble-class[binning]`),
  `woe_logreg` degrades to `logreg` with a logged warning
  (`baseline_kind_` records what ran).
- `solver="auto"` now resolves to `lbfgs` (previously picked `saga` for
  larger data, which needs feature scaling and often failed to converge).
- Ranking gate remains "beats the baseline on any metric" per the
  original specification; a stricter majority gate is configurable via
  `ensemble._common.GATE_MIN_BETTER`.
- Library no longer calls `logging.basicConfig` on import (uses
  `NullHandler`).
- Packaging: version is single-sourced from `paramsemble_class.__version__`,
  `setup.py` removed, the `tests` package no longer ships in wheels.

### Deprecated
- `ELRClassifier.predict(X, ids)` two-argument form — emits a
  `DeprecationWarning` and returns `predict_detailed(X, ids)` (the 0.5.x
  behaviour) so existing code keeps working. Use `predict_detailed(X, ids)`
  for targeting tables or `predict(X)` for labels; the form will be
  removed in a future release.
