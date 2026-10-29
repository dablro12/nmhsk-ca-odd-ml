#!/usr/bin/env python3
"""Training system.

Fits the locked Random Forest configuration from Methods 2.2-2.3
(500 trees, Gini, no class weighting, no resampling) and saves a
deployable model artifact for src/infer.py.

Usage
-----
    python -m src.train --data path/to/engineered.csv --feature-set combined \
        --out artifacts/odd_rf_combined.joblib

    # Also reproduce the paper's stratified 5-fold x 3-seed CV metrics
    # (Table 2 / Supplementary Table S1) before fitting the final model:
    python -m src.train --data path/to/engineered.csv --feature-set combined \
        --out artifacts/odd_rf_combined.joblib --evaluate

`--data` must already be the engineered feature table (see
src/preprocessing.py / README.md) with column ODD_CP as the outcome.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import fit_transform, select_xy  # noqa: E402
from src.metadata import TARGET_COL  # noqa: E402

N_ESTIMATORS = 500
SEEDS = (42, 43, 44)
N_SPLITS = 5


def youden_threshold(y_true, y_score) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.isfinite(thresholds)
    if not np.any(valid):
        return 0.5
    j = tpr[valid] - fpr[valid]
    return float(thresholds[valid][int(np.argmax(j))])


def _metrics_at_threshold(y_true, scores, th: float) -> dict:
    pred = (np.asarray(scores) >= th).astype(int)
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "Specificity": float(spec),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, scores)),
        "threshold": float(th),
    }


def _ci(x) -> str:
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return f"{m:.3f}"
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return f"{m:.3f} (95% CI {m - t * se:.3f}-{m + t * se:.3f})"


def run_evaluation(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Stratified 5-fold x 3-seed cross-validation, as in Table 2."""
    X, y = select_xy(df, feature_set, TARGET_COL)
    rows = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()
            X_tr_ml, X_te_ml, _ = fit_transform(X_tr, X_te)
            clf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=seed + fold,
                n_jobs=-1,
                oob_score=True,
                bootstrap=True,
            )
            clf.fit(X_tr_ml, y_tr)
            oob = clf.oob_decision_function_[:, 1]
            oob = np.where(np.isnan(oob), np.nanmean(oob), oob)
            th = youden_threshold(y_tr, oob)
            scores = clf.predict_proba(X_te_ml)[:, 1]
            m = _metrics_at_threshold(y_te, scores, th)
            m.update({"seed": seed, "fold": fold})
            rows.append(m)
            print(
                f"[evaluate] {feature_set} seed={seed} fold={fold} "
                f"AUC={m['AUC']:.3f} Recall={m['Recall']:.3f} Specificity={m['Specificity']:.3f}"
            )
    out = pd.DataFrame(rows)
    print(f"\n[evaluate] {feature_set} — 15-fold summary (mean, 95% CI):")
    for metric in ("Accuracy", "Recall", "Specificity", "F1", "AUC"):
        print(f"  {metric:12s} {_ci(out[metric])}")
    return out


def train_final_model(df: pd.DataFrame, feature_set: str) -> dict:
    """Fit the deployable model on the full dataset (no held-out split)."""
    X, y = select_xy(df, feature_set, TARGET_COL)
    X_ml, _, preprocessor = fit_transform(X, None)
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    clf.fit(X_ml, y)
    oob = clf.oob_decision_function_[:, 1]
    oob = np.where(np.isnan(oob), np.nanmean(oob), oob)
    threshold = youden_threshold(y, oob)
    oob_pred = (oob >= threshold).astype(int)
    oob_metrics = _metrics_at_threshold(y, oob, threshold)
    print(
        f"[train] {feature_set}: fit on N={len(y)} (cases={int(y.sum())}); "
        f"OOB AUC={oob_metrics['AUC']:.3f}, OOB threshold={threshold:.4f}"
    )
    return {
        "model": clf,
        "preprocessor": preprocessor,
        "feature_set": feature_set,
        "raw_feature_columns": list(X.columns),
        "ml_feature_names": list(preprocessor.get_feature_names_out()),
        "target_col": TARGET_COL,
        "threshold": threshold,
        "oob_metrics": oob_metrics,
        "n_estimators": N_ESTIMATORS,
        "trained_n": int(len(y)),
        "trained_n_cases": int(y.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Engineered feature CSV (see src/preprocessing.py)")
    ap.add_argument("--feature-set", choices=["combined", "adolescent", "parent"], default="combined")
    ap.add_argument("--out", required=True, help="Output path for the model artifact (.joblib)")
    ap.add_argument("--evaluate", action="store_true", help="Also run the 5-fold x 3-seed CV before fitting the final model")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"--data must contain the outcome column {TARGET_COL!r} for training.")

    if args.evaluate:
        run_evaluation(df, args.feature_set)
        print()

    artifact = train_final_model(df, args.feature_set)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "feature_set": artifact["feature_set"],
                "raw_feature_columns": artifact["raw_feature_columns"],
                "target_col": artifact["target_col"],
                "threshold": artifact["threshold"],
                "oob_metrics": artifact["oob_metrics"],
                "n_estimators": artifact["n_estimators"],
                "trained_n": artifact["trained_n"],
                "trained_n_cases": artifact["trained_n_cases"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[train] saved model -> {out_path}")
    print(f"[train] saved summary -> {sidecar}")


if __name__ == "__main__":
    main()
