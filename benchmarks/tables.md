Winner per dataset by mean capture_at_n:

| dataset | winner | score | ELR best | ELR best score |
|---|---|---|---|---|
| balanced | **random_forest** | 0.40 | elr_ensemble | 0.38 |
| drift_balanced | **xgboost** | 0.33 | elr_intersect_mw | 0.32 |
| drift_rare_2pct | **elr_ensemble** | 0.66 | elr_ensemble | 0.66 |
| heterogeneous | **hist_gb** | 0.81 | elr_ensemble_mw | 0.71 |
| imbalanced_90_10 | **hist_gb** | 0.98 | elr_intersect_mw | 0.93 |
| rare_1pct | **random_forest** | 0.82 | elr_ensemble_mw | 0.75 |
| rare_2pct | **hist_gb** | 0.88 | elr_intersect_mw | 0.80 |
| rare_5pct | **hist_gb** | 0.97 | elr_ensemble | 0.93 |

Mean rank by capture_at_n across all datasets (lower is better):

| model | mean rank |
|---|---|
| hist_gb | 2.5 |
| random_forest | 2.9 |
| xgboost | 3.3 |
| logreg | 5.1 |
| elr_ensemble_mw | 6.0 |
| elr_intersect_mw | 6.1 |
| elr_ensemble | 6.6 |
| elr_intersect | 6.7 |
| elr_intersect_cov | 7.7 |
| elr_venn | 8.1 |

Mean rank by capture_10pct across all datasets (lower is better):

| model | mean rank |
|---|---|
| random_forest | 2.3 |
| hist_gb | 2.6 |
| xgboost | 3.2 |
| logreg | 4.6 |
| elr_ensemble_mw | 4.9 |
| elr_intersect_mw | 6.2 |
| elr_ensemble | 6.4 |
| elr_venn | 8.1 |
| elr_intersect | 8.2 |
| elr_intersect_cov | 8.4 |

### balanced

- test rows: 1500, positives: 752 (prevalence 50.1%)
- budget N (ELR-intersect set size): 300 rows (20% of test, range 20%-20%), identical for every model within a seed
- ELR baseline fallback (0 models selected): 100% of seeds

| model | capture@N | capture@10% | capture@20% | precision@N | lift@N | AUC | fit (s) |
|---|---|---|---|---|---|---|---|
| elr_intersect | 37.6±1.6 | 18.8±1.0 | 37.6±1.6 | 94.1±3.8 | 1.88±0.08 | 0.68±0.02 | 2.1 |
| elr_intersect_cov | 37.6±1.6 | 18.8±1.0 | 37.6±1.6 | 94.1±3.8 | 1.88±0.08 | 0.68±0.02 | 2.2 |
| elr_intersect_mw | 37.5±1.5 | 18.7±0.9 | 37.5±1.5 | 93.9±3.5 | 1.88±0.07 | 0.70±0.05 | 2.2 |
| elr_venn | 37.6±1.6 | 18.8±1.0 | 37.6±1.6 | 94.1±3.8 | 1.88±0.08 | 0.68±0.02 | 1.8 |
| elr_ensemble | 37.6±1.6 | 19.3±0.7 | 37.6±1.6 | 94.1±3.8 | 1.88±0.08 | 0.90±0.05 | 1.4 |
| elr_ensemble_mw | 37.5±1.5 | 19.3±0.7 | 37.5±1.5 | 93.9±3.5 | 1.88±0.07 | 0.90±0.05 | 1.8 |
| logreg | 36.9±1.9 | 19.1±0.4 | 36.9±1.9 | 92.3±4.6 | 1.85±0.10 | 0.89±0.06 | 0.0 |
| random_forest | 39.9±0.1 | 19.9±0.1 | 39.9±0.1 | 99.8±0.2 | 1.99±0.00 | 0.98±0.01 | 1.3 |
| hist_gb | 39.8±0.1 | 19.9±0.1 | 39.8±0.1 | 99.7±0.0 | 1.99±0.00 | 0.99±0.01 | 5.0 |
| xgboost | 39.9±0.1 | 19.9±0.1 | 39.9±0.1 | 99.8±0.2 | 1.99±0.00 | 0.99±0.01 | 0.6 |

### imbalanced_90_10

- test rows: 1500, positives: 159 (prevalence 10.6%)
- budget N (ELR-intersect set size): 908 rows (61% of test, range 53%-66%), identical for every model within a seed
- ELR baseline fallback (0 models selected): 0% of seeds

| model | capture@N | capture@10% | capture@20% | precision@N | lift@N | AUC | fit (s) |
|---|---|---|---|---|---|---|---|
| elr_intersect | 92.6±1.9 | 43.5±5.0 | 61.4±5.2 | 16.1±2.0 | 1.54±0.19 | 0.82±0.02 | 2.4 |
| elr_intersect_cov | 91.7±2.2 | 41.8±1.6 | 63.1±2.2 | 16.0±2.0 | 1.53±0.20 | 0.82±0.02 | 2.4 |
| elr_intersect_mw | 93.4±2.8 | 43.1±10.2 | 72.5±13.9 | 16.3±1.9 | 1.56±0.19 | 0.85±0.04 | 2.6 |
| elr_venn | 90.9±1.8 | 40.3±3.7 | 62.9±4.0 | 15.9±2.2 | 1.52±0.22 | 0.82±0.02 | 2.1 |
| elr_ensemble | 92.4±2.7 | 49.1±8.8 | 68.8±6.0 | 16.1±1.7 | 1.54±0.17 | 0.85±0.03 | 1.6 |
| elr_ensemble_mw | 92.6±2.2 | 54.4±9.9 | 73.5±14.5 | 16.1±1.8 | 1.54±0.18 | 0.87±0.05 | 2.3 |
| logreg | 93.7±3.7 | 53.7±13.6 | 72.9±14.7 | 16.3±1.9 | 1.56±0.19 | 0.87±0.06 | 0.0 |
| random_forest | 97.3±1.4 | 79.9±5.3 | 92.8±4.1 | 17.0±2.1 | 1.62±0.21 | 0.96±0.02 | 1.3 |
| hist_gb | 98.1±1.1 | 83.0±3.3 | 92.0±4.5 | 17.1±2.1 | 1.64±0.21 | 0.96±0.02 | 3.9 |
| xgboost | 97.9±1.4 | 81.1±3.6 | 92.6±5.1 | 17.1±2.2 | 1.63±0.21 | 0.96±0.02 | 0.6 |

### rare_5pct

- test rows: 2000, positives: 108 (prevalence 5.4%)
- budget N (ELR-intersect set size): 1208 rows (60% of test, range 57%-65%), identical for every model within a seed
- ELR baseline fallback (0 models selected): 0% of seeds

| model | capture@N | capture@10% | capture@20% | precision@N | lift@N | AUC | fit (s) |
|---|---|---|---|---|---|---|---|
| elr_intersect | 92.9±3.5 | 43.8±12.7 | 62.5±12.7 | 8.4±0.5 | 1.54±0.09 | 0.80±0.06 | 2.4 |
| elr_intersect_cov | 91.7±2.8 | 45.1±11.0 | 62.2±14.1 | 8.3±0.3 | 1.52±0.06 | 0.80±0.05 | 2.6 |
| elr_intersect_mw | 92.3±4.2 | 53.0±10.5 | 73.3±5.8 | 8.3±0.6 | 1.53±0.11 | 0.84±0.04 | 2.2 |
| elr_venn | 89.5±6.2 | 47.8±10.4 | 61.9±12.6 | 8.1±0.5 | 1.48±0.08 | 0.79±0.07 | 1.8 |
| elr_ensemble | 92.9±3.5 | 48.4±12.4 | 66.2±16.7 | 8.4±0.5 | 1.54±0.09 | 0.83±0.06 | 2.2 |
| elr_ensemble_mw | 92.0±1.4 | 61.9±6.7 | 75.4±8.1 | 8.3±0.6 | 1.53±0.11 | 0.86±0.03 | 2.7 |
| logreg | 95.4±1.8 | 60.1±7.1 | 77.0±6.8 | 8.6±0.7 | 1.58±0.13 | 0.87±0.02 | 0.0 |
| random_forest | 96.6±2.1 | 85.9±3.5 | 90.5±2.3 | 8.7±0.6 | 1.60±0.12 | 0.94±0.02 | 1.4 |
| hist_gb | 96.6±1.4 | 86.2±1.9 | 89.6±1.9 | 8.7±0.6 | 1.60±0.12 | 0.94±0.02 | 4.4 |
| xgboost | 95.4±2.8 | 86.2±3.2 | 89.3±3.3 | 8.6±0.7 | 1.58±0.14 | 0.94±0.02 | 0.8 |

### rare_2pct

- test rows: 3000, positives: 72 (prevalence 2.4%)
- budget N (ELR-intersect set size): 1282 rows (43% of test, range 37%-54%), identical for every model within a seed
- ELR baseline fallback (0 models selected): 0% of seeds

| model | capture@N | capture@10% | capture@20% | precision@N | lift@N | AUC | fit (s) |
|---|---|---|---|---|---|---|---|
| elr_intersect | 78.6±10.3 | 49.9±10.2 | 63.8±12.3 | 4.6±0.7 | 1.88±0.32 | 0.77±0.07 | 2.4 |
| elr_intersect_cov | 78.2±12.3 | 52.5±10.3 | 65.1±14.9 | 4.6±0.7 | 1.86±0.33 | 0.77±0.07 | 2.1 |
| elr_intersect_mw | 79.6±11.2 | 61.1±14.1 | 72.5±15.0 | 4.7±1.0 | 1.91±0.46 | 0.80±0.08 | 2.3 |
| elr_venn | 77.7±11.8 | 50.7±10.0 | 64.2±14.3 | 4.5±0.7 | 1.85±0.34 | 0.77±0.08 | 2.2 |
| elr_ensemble | 78.6±11.6 | 52.0±9.4 | 66.5±10.2 | 4.6±0.7 | 1.87±0.32 | 0.79±0.07 | 2.3 |
| elr_ensemble_mw | 79.6±13.5 | 63.8±14.5 | 72.0±15.9 | 4.7±1.1 | 1.91±0.49 | 0.81±0.09 | 2.1 |
| logreg | 82.3±9.7 | 65.6±18.0 | 72.9±15.8 | 4.8±0.9 | 1.97±0.41 | 0.83±0.08 | 0.0 |
| random_forest | 85.5±8.2 | 75.1±11.6 | 81.5±11.0 | 5.0±1.0 | 2.06±0.46 | 0.88±0.05 | 1.2 |
| hist_gb | 87.8±6.0 | 77.4±8.2 | 80.1±6.8 | 5.2±0.9 | 2.11±0.41 | 0.88±0.04 | 2.3 |
| xgboost | 86.9±8.9 | 73.8±12.2 | 78.4±9.7 | 5.1±0.9 | 2.09±0.43 | 0.88±0.05 | 0.4 |

### rare_1pct

- test rows: 4000, positives: 59 (prevalence 1.5%)
- budget N (ELR-intersect set size): 1524 rows (38% of test, range 20%-53%), identical for every model within a seed
- ELR baseline fallback (0 models selected): 0% of seeds

| model | capture@N | capture@10% | capture@20% | precision@N | lift@N | AUC | fit (s) |
|---|---|---|---|---|---|---|---|
| elr_intersect | 69.6±18.0 | 38.8±12.4 | 52.9±2.8 | 2.9±0.8 | 1.97±0.53 | 0.72±0.06 | 2.1 |
| elr_intersect_cov | 69.6±18.0 | 38.8±12.1 | 54.6±2.5 | 2.9±0.8 | 1.97±0.53 | 0.71±0.06 | 2.0 |
| elr_intersect_mw | 74.1±15.1 | 46.1±4.4 | 58.6±2.5 | 3.1±1.0 | 2.13±0.64 | 0.74±0.02 | 2.1 |
| elr_venn | 69.1±18.1 | 40.0±13.4 | 53.4±2.4 | 2.9±0.9 | 1.96±0.54 | 0.71±0.06 | 2.1 |
| elr_ensemble | 69.6±16.0 | 45.0±8.1 | 55.2±6.9 | 2.9±0.9 | 1.98±0.54 | 0.73±0.06 | 2.0 |
| elr_ensemble_mw | 74.6±12.7 | 48.9±5.7 | 60.2±1.1 | 3.2±1.2 | 2.17±0.74 | 0.76±0.03 | 2.1 |
| logreg | 75.1±9.1 | 46.6±6.7 | 66.5±2.6 | 3.3±1.4 | 2.23±0.90 | 0.77±0.03 | 0.0 |
| random_forest | 82.0±13.3 | 65.3±2.3 | 74.0±6.5 | 3.5±1.3 | 2.40±0.84 | 0.84±0.04 | 1.4 |
| hist_gb | 76.9±12.2 | 51.2±1.8 | 64.8±3.7 | 3.3±1.3 | 2.26±0.86 | 0.77±0.04 | 1.6 |
| xgboost | 76.9±13.3 | 54.6±1.8 | 66.5±4.4 | 3.3±1.2 | 2.24±0.77 | 0.80±0.02 | 0.5 |

### heterogeneous

- test rows: 2000, positives: 281 (prevalence 14.1%)
- budget N (ELR-intersect set size): 816 rows (41% of test, range 20%-61%), identical for every model within a seed
- ELR baseline fallback (0 models selected): 0% of seeds

| model | capture@N | capture@10% | capture@20% | precision@N | lift@N | AUC | fit (s) |
|---|---|---|---|---|---|---|---|
| elr_intersect | 60.5±24.9 | 32.0±12.2 | 45.9±11.7 | 21.4±2.7 | 1.53±0.18 | 0.68±0.09 | 3.3 |
| elr_intersect_cov | 60.7±25.2 | 33.2±13.6 | 46.6±12.4 | 21.5±2.6 | 1.53±0.18 | 0.68±0.10 | 3.2 |
| elr_intersect_mw | 69.0±21.2 | 36.8±15.4 | 56.3±9.9 | 25.7±6.0 | 1.83±0.42 | 0.76±0.05 | 7.0 |
| elr_venn | 60.6±25.0 | 32.2±12.4 | 45.8±11.6 | 21.5±2.6 | 1.53±0.18 | 0.68±0.09 | 3.3 |
| elr_ensemble | 62.0±26.3 | 35.9±13.9 | 49.7±15.5 | 21.9±2.5 | 1.56±0.17 | 0.71±0.10 | 3.3 |
| elr_ensemble_mw | 71.4±18.3 | 45.2±11.8 | 59.9±8.3 | 27.2±8.0 | 1.94±0.57 | 0.79±0.04 | 7.1 |
| logreg | 71.3±19.4 | 41.8±10.6 | 58.8±8.4 | 26.9±7.4 | 1.92±0.52 | 0.78±0.04 | 0.7 |
| random_forest | 79.4±6.9 | 60.8±2.1 | 71.6±3.1 | 32.5±16.0 | 2.31±1.13 | 0.83±0.02 | 1.2 |
| hist_gb | 80.6±8.2 | 61.1±2.5 | 71.3±3.3 | 32.8±15.6 | 2.34±1.10 | 0.84±0.02 | 3.0 |
| xgboost | 78.7±8.2 | 59.8±1.9 | 70.3±4.1 | 31.9±15.1 | 2.28±1.07 | 0.83±0.02 | 0.6 |

### drift_rare_2pct

- test rows: 3000, positives: 83 (prevalence 2.8%)
- budget N (ELR-intersect set size): 1069 rows (36% of test, range 29%-46%), identical for every model within a seed
- ELR baseline fallback (0 models selected): 0% of seeds

| model | capture@N | capture@10% | capture@20% | precision@N | lift@N | AUC | fit (s) |
|---|---|---|---|---|---|---|---|
| elr_intersect | 65.2±2.1 | 34.6±6.4 | 52.9±2.3 | 5.4±0.9 | 1.89±0.39 | 0.69±0.02 | 2.1 |
| elr_intersect_cov | 64.1±3.8 | 32.6±6.0 | 48.1±8.1 | 5.3±1.0 | 1.86±0.41 | 0.68±0.04 | 2.1 |
| elr_intersect_mw | 63.3±3.6 | 32.8±4.6 | 51.4±0.7 | 5.2±0.8 | 1.83±0.32 | 0.69±0.02 | 2.1 |
| elr_venn | 66.1±3.5 | 33.7±8.9 | 44.7±15.4 | 5.5±1.1 | 1.92±0.44 | 0.69±0.05 | 2.1 |
| elr_ensemble | 66.4±1.2 | 32.8±3.5 | 51.4±4.6 | 5.5±1.0 | 1.93±0.40 | 0.71±0.04 | 2.1 |
| elr_ensemble_mw | 65.2±4.0 | 32.4±3.9 | 47.6±5.9 | 5.4±0.8 | 1.88±0.34 | 0.70±0.04 | 2.1 |
| logreg | 63.7±4.6 | 35.1±2.7 | 49.5±3.5 | 5.3±0.9 | 1.84±0.38 | 0.71±0.03 | 0.0 |
| random_forest | 62.5±3.5 | 34.6±5.0 | 48.3±1.9 | 5.2±0.9 | 1.81±0.38 | 0.68±0.02 | 1.4 |
| hist_gb | 54.4±3.6 | 30.4±7.8 | 42.8±3.8 | 4.5±0.6 | 1.57±0.27 | 0.64±0.01 | 2.6 |
| xgboost | 54.2±8.6 | 29.7±2.7 | 41.3±1.4 | 4.4±0.4 | 1.54±0.16 | 0.64±0.01 | 0.5 |

### drift_balanced

- test rows: 2000, positives: 1030 (prevalence 51.5%)
- budget N (ELR-intersect set size): 400 rows (20% of test, range 20%-20%), identical for every model within a seed
- ELR baseline fallback (0 models selected): 33% of seeds

| model | capture@N | capture@10% | capture@20% | precision@N | lift@N | AUC | fit (s) |
|---|---|---|---|---|---|---|---|
| elr_intersect | 29.2±2.8 | 15.0±1.1 | 29.2±2.8 | 74.4±6.6 | 1.46±0.14 | 0.59±0.03 | 1.6 |
| elr_intersect_cov | 29.2±2.8 | 15.0±1.1 | 29.2±2.8 | 74.4±6.6 | 1.46±0.14 | 0.59±0.03 | 1.5 |
| elr_intersect_mw | 31.6±0.8 | 16.3±0.6 | 31.6±0.8 | 80.4±2.0 | 1.58±0.04 | 0.68±0.02 | 1.6 |
| elr_venn | 29.2±2.8 | 15.0±1.1 | 29.2±2.8 | 74.4±6.6 | 1.46±0.14 | 0.59±0.03 | 1.5 |
| elr_ensemble | 29.2±2.8 | 15.2±2.1 | 29.2±2.8 | 74.4±6.6 | 1.46±0.14 | 0.73±0.03 | 1.6 |
| elr_ensemble_mw | 31.2±0.8 | 15.8±0.4 | 31.2±0.8 | 79.4±2.5 | 1.56±0.04 | 0.76±0.01 | 1.6 |
| logreg | 31.2±0.6 | 15.7±0.5 | 31.2±0.6 | 79.6±1.7 | 1.56±0.03 | 0.76±0.01 | 0.0 |
| random_forest | 31.9±0.6 | 15.9±0.3 | 31.9±0.6 | 81.2±0.7 | 1.59±0.03 | 0.81±0.01 | 1.2 |
| hist_gb | 32.2±0.2 | 16.2±0.3 | 32.2±0.2 | 81.9±0.5 | 1.61±0.01 | 0.81±0.01 | 2.4 |
| xgboost | 32.5±0.5 | 16.4±0.1 | 32.5±0.5 | 82.9±1.8 | 1.63±0.02 | 0.81±0.01 | 0.4 |

