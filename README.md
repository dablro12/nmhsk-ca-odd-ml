# ODD Random Forest — Code Repository

Training, inference, and interpretation code for the Random Forest models described in
*Adolescent-Level and Parent-Level Correlates of Oppositional Defiant Disorder in Korean
Adolescents: A Machine Learning Analysis of National Survey Data* (submitted to
*Psychiatry Investigation*). This is the code referenced by the manuscript's Code
Availability statement.

> **Not a diagnostic or screening tool.** At the Youden threshold the combined-feature-set
> model's positive predictive value is ≈0.18 (Results 3.2; Discussion, Limitations) — most
> adolescents flagged positive do not have ODD. This repository is for research
> reproducibility only.

## What this reproduces

| Script | Reproduces |
|---|---|
| `src/train.py --evaluate` | The stratified 5-fold × 3-seed cross-validation in Table 2 / Supplementary Table S9 (AUC, recall, specificity, accuracy, F1, each with 95% CI) |
| `src/train.py` (final fit) | The locked, deployable Random Forest (500 trees, Gini, no class weighting or resampling — Methods 2.2–2.3) |
| `src/infer.py` | Scoring new records with a trained model, at the Youden threshold |
| `src/analyze.py` | Permutation importance (ΔAUC) and SHAP, per encoded feature (Methods 2.5; Figures 4–5) |

All three feature sets from the manuscript are supported: `combined`, `adolescent`, `parent`.

## Folder layout

```
6_CodeRepo/
├── README.md              this file
├── requirements.txt       pinned package versions
├── src/
│   ├── metadata.py         feature schema: column names, types, feature-set membership, plain-language labels
│   ├── preprocessing.py    raw NMHSK-CA 2022 item-level export -> engineered feature table (Methods 2.1-2.2)
│   ├── features.py         feature-set selection + the sklearn ColumnTransformer (impute/scale/one-hot)
│   ├── train.py            training system (CV evaluation + final model fit) — CLI
│   ├── infer.py            inference system (score new records with a saved model) — CLI
│   └── analyze.py          analysis system (permutation importance + SHAP) — CLI
├── examples/
│   └── synthetic_example_input.csv   12 synthetic rows, correct column schema, for a format smoke test only — not real survey data, not fit for any inference
└── artifacts/              trained models are saved here (gitignored / empty in this package — see Data Availability)
```

## Data

**The NMHSK-CA 2022 data are not included in this repository.** Per the manuscript's
Availability of Data and Material statement, the dataset analyzed in this study is not
publicly available due to restrictions imposed to protect participant privacy, but is
available from the corresponding author on reasonable request, subject to the data
custodian's (National Center for Mental Health) approval.

To run this code you need either:
- the engineered feature table (`ODD_CP` outcome column + the columns listed in
  `src/metadata.py`), e.g. by requesting access and running `src/preprocessing.py` on the
  raw item-level export yourself, or
- your own dataset with the same column names and coding, for methodological reuse on a
  different sample.

`examples/synthetic_example_input.csv` has the correct column schema with randomly
generated values, so you can confirm the pipeline runs end-to-end before you have real
data access.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10, scikit-learn 1.7.2, pandas 2.3.3, shap 0.51.0.

## Quickstart

```bash
# 1. Train (reproduces Table 2's Combined-feature-set row, then fits the final model)
python -m src.train --data /path/to/engineered.csv --feature-set combined \
    --out artifacts/odd_rf_combined.joblib --evaluate

# 2. Score new records (format check against the bundled synthetic example)
python -m src.infer --model artifacts/odd_rf_combined.joblib \
    --data examples/synthetic_example_input.csv --out predictions.csv

# 3. Permutation importance + SHAP (Figures 4-5 style tables)
python -m src.analyze --model artifacts/odd_rf_combined.joblib \
    --data /path/to/engineered.csv --out-prefix results/combined
```

Repeat step 1 with `--feature-set adolescent` or `--feature-set parent` for the other two
models in the paper.

## Reproducibility notes

- **Random seeds.** Cross-validation uses seeds 42/43/44 (`StratifiedKFold`), matching the
  manuscript. The final deployable model is fit with `random_state=42`.
- **No leakage.** The preprocessor (imputer, scaler, one-hot encoder) is fit on the
  training fold only in `--evaluate` mode, consistent with Methods 2.2.
- **Threshold.** The Youden index is computed from out-of-bag probabilities on the
  training data in each fold (and on the full training set for the final model), never
  from the test/new data being scored.
- **One-hot levels.** Permutation importance and SHAP are reported per encoded column —
  each level of a nominal feature (e.g., paternal occupation category) is ranked
  separately, exactly as in Methods 2.5, not pooled into one importance per original
  variable.
- **Verified against the manuscript.** Running `src/train.py --evaluate --feature-set
  combined` on the analysis dataset reproduces Table 2's combined-feature-set row exactly:
  AUC 0.892 (95% CI 0.886–0.897), recall 0.829 (95% CI 0.797–0.860), specificity 0.787
  (95% CI 0.771–0.803).
- **Item-level missingness in composite scores.** `src/preprocessing.py`'s `_sum_existing()`
  (PHQ-9/GAD-7/SAS-SV totals, the 18-item ADHD score) and the binary symptom flags
  (PS/CS/SV/CV codes) silently treat missing source items as 0 rather than propagating
  NaN for imputation — this matches the original analysis pipeline exactly and had no
  effect on the published results (Methods 2.1: no participant in the analytic sample
  had missing data), but check for item-level missingness explicitly before trusting
  these composites on a dataset that has any.

## What is intentionally not reproduced here

Supplementary ablations (classifier comparison, upsampling, loss functions, hyperparameter
grid, the design-based weighted sensitivity analysis, and the DICCD replication) use the
same `src/features.py` preprocessing but were run as one-off analysis scripts, not part of
this packaged pipeline. The numbers themselves are in `5_supplementary/tables/`.
