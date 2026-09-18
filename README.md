# Paramsemble-Class - Parametric Ensemble for Classification

Paramsemble-Class (Parametric Ensemble for Classification) is a Python library for advanced classification tasks using ensemble methods based on combinatorial feature selection and parametric logistic regression.

## What it is for (and what it is not for)

**ELR is not a general-purpose classifier.** It is built for a specific, hard
regime: **rare positive events under distribution drift** — when the relation
between features and the outcome changes over time (changing customer behavior,
new market conditions, epidemic dynamics) and you can obtain a labeled sample
from the *current* regime to drive model selection.

In a controlled head-to-head benchmark
([benchmarks/BENCHMARK_REPORT.md](benchmarks/BENCHMARK_REPORT.md): ELR vs
logistic regression, random forest, gradient boosting and XGBoost under an
equal-budget Top-N targeting protocol), on 2%-positive data with
training-to-deployment signal decay, ELR retained **87%** of its targeting
capture where random forest retained 71% and XGBoost/HistGB 65% — and it was
the only method to beat all competitors on every seed of that scenario. This
mirrors the package's motivating experience during COVID-19, where tree-ensemble
models collapsed while this approach maintained its performance despite some
declines. The mechanism is the design: constituents are sparse 4–8-feature
logistic views whose rankings survive feature-space drift, and the selection
layer re-picks the surviving views using *current-regime* labels — no refitting
required.

On stationary data with a stable feature-outcome relation, boosted trees and
forests win outright — and ELR **knows it**: its baseline gate detects this and
falls back to the internal Random Forest (see
[`fell_back_to_baseline_`](#quick-start)), so its worst case is bounded by the
baseline rather than by sparse linear models. The fallback doubles as a signal:
a persistently falling gate is the framework telling you the current problem is
not a drift problem.

Beyond its niche, two properties hold on any dataset: every selected model is a
**transparent linear equation**, and deployment can run as **pure SQL inside
your database** — neither of which tree ensembles offer.

## Features

- **Multiple Ensemble Strategies**: Choose from three distinct ensemble methods:
  - **Intersect**: Identify high-confidence predictions supported by diverse feature combinations
  - **Venn**: Discover unique predictions not captured by baseline models
  - **Ensemble**: Create meta-models from top-performing logistic regressions

- **Scikit-learn Compatible**: Works with `cross_val_score`, `GridSearchCV` and other sklearn tooling via `fit(X, y)` / `predict` / `predict_proba`, while retaining an ELR-native API for targeting-table workflows

- **Specialized Metrics**: Tailored for imbalanced classification tasks:
  - Positive Likelihood Ratio (PLR)
  - False Negative Rate (FNR)
  - Decile Ranked Performance (DRP) — lift at the top d deciles (precision in the top d deciles divided by prevalence)

- **SQL Export**: Convert trained models to SQL queries for efficient database-level scoring. Python and SQL scoring produce identical results (same decile rule, `ceil(d * n / 10)` top rows)

- **Model Persistence**: Export and import models via JSON for production deployment

## How the method works

Given a training set and a **selection set** (a labeled dataset held out from training):

1. **Feature combinations** — `m` feature subsets of size `f` are sampled (`sample="unique"` without intra-set repetition, `"replace"` for uniform multicombinations).
2. **Baseline** — a reference model on all features is evaluated on the selection set (PLR, FNR, DRP plus decile ID sets DRS/DPS). The default is a full-feature logistic regression on WoE-binned features (`baseline="woe_logreg"`, requires `pip install paramsemble-class[binning]`; without optbinning it degrades to an unbinned logistic regression, logged and visible in `baseline_kind_`). `baseline="random_forest"` restores the classic reference.
3. **Constituents** — `m` logistic regressions are trained on the feature subsets and evaluated the same way.
4. **Selection & combination** — models that beat the baseline on any metric are ranked by a Borda count over PLR/FNR/DRP, then combined by the chosen `method`.
5. **Deployment** — selected equations export to JSON, scoreable in Python or SQL.

> **Note on honest evaluation.** The selection set is used to *choose* models, so
> metrics measured on it are optimistically biased by the selection itself. The
> ensemble method's meta-model is trained on out-of-fold predictions of the
> selection set to avoid leakage, but for final reporting, evaluate on a third,
> untouched holdout (e.g. via `predict_proba`).

## Installation

### From PyPI (when published)

```bash
pip install paramsemble-class
```

### From Source

```bash
git clone https://github.com/schen18/Paramsemble-Class.git
cd Paramsemble-Class
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### ELR-native workflow (targeting tables)

```python
from paramsemble_class import ELRClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, weights=[0.7, 0.3], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
ids_test = range(len(X_test))  # IDs for the selection set

clf = ELRClassifier(
    m=100,              # Number of feature combinations
    f=5,                # Features per combination
    method="intersect", # Ensemble method
    spread=10,          # Number of top models to select
    d=2,                # Top deciles to consider
    random_state=42
)

clf.fit(X_train, y_train, X_test, y_test, ids_test)

# Targeting table: which IDs land in the top deciles of how many selected models
results = clf.predict_detailed(X_test, ids_test)  # DataFrame with [id, sets, score]
```

### scikit-learn workflow (labels and probabilities)

```python
from paramsemble_class import ELRClassifier
from sklearn.model_selection import cross_val_score

clf = ELRClassifier(m=50, f=5, method="ensemble", random_state=42)

# fit(X, y) splits off a stratified selection set internally
clf.fit(X, y)
labels = clf.predict(X)          # class labels (ndarray)
proba = clf.predict_proba(X)     # (n, 2) probabilities, available for all methods

scores = cross_val_score(ELRClassifier(m=20, f=4, random_state=0), X, y, cv=3)
```

For intersect/venn, `predict` labels an instance positive when a majority of
the *weighted* votes places it in the top-d deciles, and `predict_proba`
returns the weighted vote share (each selected model votes with its
Borda-position weight) — a continuous score usable for ranking and AUC.

**Baseline fallback (honest abstinence).** If no constituent beats the baseline
gate, the classifier sets `fell_back_to_baseline_ = True`, logs a warning, and
produces all outputs from the baseline model instead (its top-d deciles with
`sets = 1` for intersect/venn; its probabilities for ensemble). With the default
linear baseline the fallback equation exports to `modeljson` (flagged
`"fallback": true`, weight 1.0), so **even a fallback fit is SQL-deployable** —
WoE bins become `CASE WHEN` ladders in the generated SQL. A gate that keeps
rejecting every constituent across seeds is a signal that one full linear model
already solves the problem — not a failure.

## Ensemble Methods

### Intersect Method

Identifies IDs that appear in multiple top-performing models, providing high-confidence predictions:

```python
clf = ELRClassifier(method="intersect", spread=10, d=2)
clf.fit(X_train, y_train, X_test, y_test, ids_test)
results = clf.predict_detailed(X_test, ids_test)  # DataFrame with [id, sets, score]
```

### Venn Method

Discovers unique predictions not captured by the baseline model:

```python
clf = ELRClassifier(method="venn", spread=10, d=2)
clf.fit(X_train, y_train, X_test, y_test, ids_test)
results = clf.predict_detailed(X_test, ids_test)  # DataFrame with [id, sets, score]
```

### Ensemble Method

Creates a meta-model combining predictions from top-performing models. The
meta-model is trained on **out-of-fold** predictions of the selection set, so
its coefficients are not fit on the rows it scores:

```python
clf = ELRClassifier(method="ensemble", spread=10)
clf.fit(X_train, y_train, X_test, y_test, ids_test)
predictions = clf.predict_detailed(X_test, ids_test)  # DataFrame with [id, predicted]
```

## SQL Generation

Export trained models to SQL for database-level scoring. The generated query
uses the same decile rule as Python scoring (`ceil(d * n / 10)` top rows by
score), so both paths return the same targeting lists (up to tie-breaking at
the cutoff, which databases resolve arbitrarily):

```python
from paramsemble_class import ELRClassifier, SQLGenerator

clf = ELRClassifier(method="intersect", modeljson="selected_models.json", ...)
clf.fit(X_train, y_train, X_test, y_test, ids_test)

generator = SQLGenerator("selected_models.json")
sql_query = generator.generate_sql("my_table", id_column="customer_id")

# Execute the SQL query in your database (table needs the feature columns
# and a unique customer_id)
```

## Model Persistence

Export models for production use:

```python
# Export all model metrics (analysis)
clf = ELRClassifier(m=50, elr2json="all_models.json")
clf.fit(X_train, y_train, X_test, y_test, ids_test)

# Export selected models for scoring
clf = ELRClassifier(m=50, modeljson="selected_models.json")
clf.fit(X_train, y_train, X_test, y_test, ids_test)

# Score new data using saved models (numpy arrays or named DataFrames)
from paramsemble_class import ModelScorer
scorer = ModelScorer("selected_models.json")
predictions = scorer.score(X_new, ids_new)
```

## Parameters

- `m` (int): Number of feature combinations to generate (default: 100)
- `f` (int): Features per constituent model when `f_range` is not set
  (default: 5). Controls the sparsity/diversity trade-off: small `f` (≈4–8)
  maximizes decorrelated views — the drift-robustness mechanism — but each
  model is weak and may never pass the baseline gate on easy data; larger `f`
  (roughly 20–60% of n_features) makes constituents individually competitive
  but less diverse. Prefer `f_range` (below), which lets selection choose the
  width; see [benchmarks/BENCHMARK_REPORT.md](benchmarks/BENCHMARK_REPORT.md)
- `f_range` (list of int, optional): train constituents at several feature
  widths (e.g. `[4, 8, 12]`) in one candidate pool — `m` is split across the
  widths and the gate/ranking selects across all of them. Benchmarks show
  this is the single best selection-layer improvement: +7 to +20 points of
  capture at fixed budgets on rare-event datasets, and far fewer
  baseline-fallback fits (the gate finds admissible candidates at some width)
- `selection` (str): "rank" (default) selects the top `spread` models by
  Borda count. "coverage" greedily selects models maximizing marginal
  true-positive coverage on the selection set (diversity-aware); benchmarks
  showed no average gain over "rank", so it remains an option rather than
  the default
- `baseline` (str): the gate/fallback reference model. `"woe_logreg"`
  (default) — full-feature logistic regression on WoE-binned features, the
  correct comparison class for linear constituents and a SQL-deployable
  fallback (uses `optbinning` when installed, else degrades to `"logreg"`,
  logged, see `baseline_kind_`). `"logreg"` — unbinned full-feature LR.
  `"random_forest"` — the classic 100-tree reference (Python-only fallback,
  slightly better gate behavior under heavy drift)
- `sample` (str): Feature sampling method - "unique" or "replace" (default: "unique")
- `d` (int): Number of top deciles to consider (1-10) (default: 2)
- `method` (str): Ensemble method - "intersect", "venn", or "ensemble" (default: "intersect")
- `spread` (int): Number of top models to select (default: 10)
- `solver` (str): Logistic regression solver or "auto" (resolves to "lbfgs") (default: "auto")
- `id_column` (str): Name of ID column in DataFrames; excluded from features (default: "id")
- `elr2json` (str): Path to export all model metrics (optional)
- `modeljson` (str): Path to export selected model equations (optional)
- `class_weight` (dict or "balanced"): Class weights for imbalanced targets (optional)
- `n_jobs` (int): Parallel jobs for constituent training, -1 for all cores (default: 1)
- `random_state` (int): Random seed for reproducibility (optional)

## Requirements

- Python >= 3.8
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- joblib >= 1.0.0

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black paramsemble_class/ tests/

# Lint code
flake8 paramsemble_class/ tests/ --max-line-length=100 --extend-ignore=E203

# Type checking
mypy paramsemble_class/
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use Paramsemble-Class in your research, please cite:

```
@software{paramsemble_class,
  title = {Paramsemble-Class: Ensemble Logistic Regression},
  author = {Stephen Chen},
  year = {2024},
  url = {https://github.com/schen18/Paramsemble-Class}
}
```
