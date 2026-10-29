#!/usr/bin/env python3
"""Analysis system: permutation importance and SHAP for a trained model.

Reproduces the interpretation pipeline in Methods 2.5 (Figure 4 / Figure 5):
permutation importance (delta-AUC) and mean |SHAP| are computed per encoded
column -- one-hot levels of a nominal feature are ranked separately, exactly
as in the manuscript (Methods 2.5: "computed ... for every encoded feature,
one-hot levels separately").

Usage
-----
    python -m src.analyze --model artifacts/odd_rf_combined.joblib \
        --data path/to/engineered.csv --out-prefix results/combined

Produces <out-prefix>_permutation.csv and <out-prefix>_shap.csv, and prints
the top 10 of each to stdout.

`--data` should contain the outcome column (ODD_CP); ideally a held-out
split, not the data the model was fit on (the manuscript computes this on
cross-validation test folds). SHAP values are computed on up to
`--shap-sample` rows (default 1000) for speed; permutation importance uses
the full input.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import fit_transform  # noqa: E402
from src.metadata import FEATURE_LABELS, TARGET_COL  # noqa: E402

_PREFIX_RE = re.compile(r"^(continuous|binary|ordinal|ordinal_parent|nominal)__")


def pretty_name(encoded_col: str) -> str:
    base = _PREFIX_RE.sub("", encoded_col)
    # nominal one-hot columns look like "F3_2.0" -> base feature "F3", level "2.0"
    for feat, label in FEATURE_LABELS.items():
        if base == feat:
            return label
        if base.startswith(feat + "_"):
            level = base[len(feat) + 1 :]
            return f"{label} = {level}"
    return base


def load_scored_data(model_path: str, data_path: str):
    import joblib

    artifact = joblib.load(model_path)
    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"--data must contain the outcome column {TARGET_COL!r} for analysis.")
    cols = artifact["raw_feature_columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Input data is missing required columns: {missing}")
    X = df[cols].copy()
    y = df[TARGET_COL].astype(int)
    X_ml, _, _ = fit_transform(X, None, preprocessor=artifact["preprocessor"])
    return artifact, X_ml, y


def run_permutation_importance(artifact: dict, X_ml: pd.DataFrame, y: pd.Series, n_repeats: int = 10, seed: int = 42) -> pd.DataFrame:
    result = permutation_importance(
        artifact["model"], X_ml, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1, scoring="roc_auc"
    )
    out = pd.DataFrame(
        {
            "feature": X_ml.columns,
            "feature_label": [pretty_name(c) for c in X_ml.columns],
            "delta_auc_mean": result.importances_mean,
            "delta_auc_std": result.importances_std,
        }
    ).sort_values("delta_auc_mean", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def run_shap(artifact: dict, X_ml: pd.DataFrame, sample_n: int = 1000, seed: int = 42) -> pd.DataFrame:
    import shap

    X_sample = X_ml.sample(n=min(sample_n, len(X_ml)), random_state=seed) if len(X_ml) > sample_n else X_ml
    explainer = shap.TreeExplainer(artifact["model"])
    sv = explainer.shap_values(X_sample)
    # Binary classifiers: take the positive-class SHAP values if shap_values
    # returns a (n_samples, n_features, n_classes) array or a list of arrays.
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    mean_abs_shap = np.abs(sv).mean(axis=0)
    out = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "feature_label": [pretty_name(c) for c in X_sample.columns],
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--n-repeats", type=int, default=10, help="Permutation-importance repeats (manuscript used 1 repeat per fold, averaged over folds; use more here on a single split for stability)")
    ap.add_argument("--shap-sample", type=int, default=1000)
    ap.add_argument("--skip-shap", action="store_true", help="Skip SHAP (permutation importance only; SHAP needs the shap package and is slower)")
    args = ap.parse_args()

    artifact, X_ml, y = load_scored_data(args.model, args.data)
    print(f"[analyze] feature_set={artifact['feature_set']} n={len(y)} cases={int(y.sum())} n_features={X_ml.shape[1]}")

    perm = run_permutation_importance(artifact, X_ml, y, n_repeats=args.n_repeats)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    perm_path = out_prefix.with_name(out_prefix.name + "_permutation.csv")
    perm.to_csv(perm_path, index=False)
    print(f"\n[analyze] permutation importance (top 10) -> {perm_path}")
    print(perm.head(10).to_string(index=False))

    if not args.skip_shap:
        shap_df = run_shap(artifact, X_ml, sample_n=args.shap_sample)
        shap_path = out_prefix.with_name(out_prefix.name + "_shap.csv")
        shap_df.to_csv(shap_path, index=False)
        print(f"\n[analyze] mean |SHAP| (top 10) -> {shap_path}")
        print(shap_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
