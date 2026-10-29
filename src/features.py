"""Feature-set selection and the sklearn preprocessing pipeline.

Mirrors Methods 2.1-2.2: continuous features are median-imputed and
standardized; binary and most ordinal features are most-frequent imputed;
the six parental ordinal items (which can be structurally absent -- see
mother_missing/father_missing) are median-imputed instead; nominal
features are most-frequent imputed and one-hot encoded.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metadata import (
    FEATURE_GROUPS,
    PARENT_ORDINAL_MEDIAN_IMPUTE,
    select_feature_cols,
)


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    cols = set(columns)
    binary_cols = [c for c in FEATURE_GROUPS["binary"] if c in cols]
    ordinal_cols = [c for c in FEATURE_GROUPS["ordinal"] if c in cols]
    nominal_cols = [c for c in FEATURE_GROUPS["nominal"] if c in cols]
    continuous_cols = [c for c in FEATURE_GROUPS["continuous"] if c in cols]

    parent_ordinal = [c for c in ordinal_cols if c in PARENT_ORDINAL_MEDIAN_IMPUTE]
    other_ordinal = [c for c in ordinal_cols if c not in PARENT_ORDINAL_MEDIAN_IMPUTE]

    continuous_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [
        ("continuous", continuous_pipe, continuous_cols),
        ("binary", binary_pipe, binary_cols),
    ]
    if other_ordinal:
        transformers.append(
            ("ordinal", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), other_ordinal)
        )
    if parent_ordinal:
        transformers.append(
            ("ordinal_parent", Pipeline([("imputer", SimpleImputer(strategy="median"))]), parent_ordinal)
        )
    transformers.append(("nominal", nominal_pipe, nominal_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def select_xy(df: pd.DataFrame, feature_set: str, target_col: str | None = "ODD_CP"):
    """Subset df to one feature set's predictor columns (+ y if present)."""
    feature_cols = select_feature_cols(feature_set)
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    if target_col is not None and target_col in df.columns:
        y = df[target_col].astype(int)
        return X, y
    return X, None


def fit_transform(X_train: pd.DataFrame, X_test: pd.DataFrame | None, preprocessor: ColumnTransformer | None = None):
    """Fit (or reuse) the ColumnTransformer and return dense feature frames.

    Fit only on X_train, exactly as in cross-validation (Methods 2.2): no
    information from X_test/new records leaks into the imputer or scaler.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor(list(X_train.columns))
        X_train_arr = preprocessor.fit_transform(X_train)
    else:
        X_train_arr = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    X_train_ml = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    if X_test is None:
        return X_train_ml, None, preprocessor
    X_test_arr = preprocessor.transform(X_test)
    X_test_ml = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)
    return X_train_ml, X_test_ml, preprocessor
