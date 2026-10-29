#!/usr/bin/env python3
"""Inference system.

Loads a model artifact saved by src/train.py and scores new records.

Usage
-----
    python -m src.infer --model artifacts/odd_rf_combined.joblib \
        --data examples/example_input.csv --out predictions.csv

Input data must contain the raw (engineered) feature columns the model
was trained on (see the artifact's .json sidecar, "raw_feature_columns").
Extra columns are ignored; missing columns raise an error. The outcome
column (ODD_CP) is optional -- if present it is carried through to the
output for convenience but is never used for scoring.

IMPORTANT: this is a research tool, not a diagnostic or screening
instrument. See README.md and manuscript Discussion ("Limitations") --
at the Youden threshold, most adolescents flagged positive do not have
ODD (PPV ~0.18 for the combined feature set).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import fit_transform  # noqa: E402


def load_model(path: str | Path) -> dict:
    return joblib.load(path)


def predict(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    cols = artifact["raw_feature_columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Input data is missing required columns for this model "
            f"({artifact['feature_set']}): {missing}"
        )
    X = df[cols].copy()
    X_ml, _, _ = fit_transform(X, None, preprocessor=artifact["preprocessor"])
    proba = artifact["model"].predict_proba(X_ml)[:, 1]
    pred = (proba >= artifact["threshold"]).astype(int)

    out = pd.DataFrame(index=df.index)
    if artifact["target_col"] in df.columns:
        out[artifact["target_col"]] = df[artifact["target_col"]]
    out["odd_probability"] = proba
    out["odd_predicted_label"] = pred
    out["threshold_used"] = artifact["threshold"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to a .joblib artifact from src/train.py")
    ap.add_argument("--data", required=True, help="CSV of new records (engineered feature schema)")
    ap.add_argument("--out", default=None, help="Where to write predictions.csv (default: stdout only)")
    args = ap.parse_args()

    artifact = load_model(args.model)
    df = pd.read_csv(args.data)
    preds = predict(artifact, df)

    print(
        f"[infer] feature_set={artifact['feature_set']} n={len(preds)} "
        f"threshold={artifact['threshold']:.4f} "
        f"flagged_positive={int(preds['odd_predicted_label'].sum())}/{len(preds)}"
    )
    print(preds.head(10).to_string(index=False))

    if args.out:
        preds.to_csv(args.out, index=False)
        print(f"[infer] saved -> {args.out}")

    print(
        "\nNOTE: this model is a research tool, not a diagnostic or "
        "screening instrument (see manuscript Discussion, Limitations)."
    )


if __name__ == "__main__":
    main()
