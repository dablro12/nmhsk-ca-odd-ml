"""Raw NMHSK-CA 2022 item-level export -> engineered feature table.

This reproduces the preprocessing described in Methods 2.1-2.2: it builds
the derived binary/sum-score items (bullying/cyberbullying codes, PHQ-9 and
GAD-7 totals, the 18-item ADHD symptom score with the ODD/CD items removed
to avoid label leakage, the Smartphone Addiction Scale total, and the two
parent-absence indicators) and returns a table restricted to the locked
analysis columns in metadata.py.

Requires the raw NMHSK-CA 2022 item-level file, which is NOT distributed
with this repository (see README.md / Data Availability).
"""
from __future__ import annotations

import pandas as pd

from .metadata import ADOLESCENT_FEATURES, OTHER_FEATURES, PARENT_FEATURES, TARGET_COL

KDBDRS_ODD_ITEMS = [
    "KDBDRS_01_3",
    "KDBDRS_02_1",
    "KDBDRS_02_2",
    "KDBDRS_02_4",
    "KDBDRS_02_10",
    "KDBDRS_03_2",
    "KDBDRS_03_4",
    "KDBDRS_04_5",
]
KDBDRS_CD_ITEMS = [
    "KDBDRS_01_2",
    "KDBDRS_01_4",
    "KDBDRS_01_5",
    "KDBDRS_01_7",
    "KDBDRS_01_9",
    "KDBDRS_02_3",
    "KDBDRS_02_7",
    "KDBDRS_03_7",
    "KDBDRS_03_8",
    "KDBDRS_04_2",
    "KDBDRS_04_4",
    "KDBDRS_04_6",
    "KDBDRS_04_7",
    "KDBDRS_04_9",
    "KDBDRS_04_11",
]
KDBDRS_LEAKAGE = set(KDBDRS_ODD_ITEMS + KDBDRS_CD_ITEMS)
KDBDRS_ALL = (
    [f"KDBDRS_01_{i}" for i in range(1, 11)]
    + [f"KDBDRS_02_{i}" for i in range(1, 11)]
    + [f"KDBDRS_03_{i}" for i in range(1, 11)]
    + [f"KDBDRS_04_{i}" for i in range(1, 12)]
)


def _sum_existing(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    # pandas .sum(axis=1) defaults to skipna=True, min_count=0: a row with some
    # or all of `cols` missing gets a deflated (not NaN) total, so it bypasses
    # the imputer in features.py. This matches the original analysis pipeline
    # exactly (Notebooks_ODD/01_preprocess_odd.py), and the manuscript's own
    # analytic sample had no missing data at all (Methods 2.1: "No participant
    # was excluded because of missing data"), so it did not affect the
    # published results. If reusing this code on a dataset with item-level
    # missingness in PHQ/GAD/SAS/KDBDRS or in the PS/CS/SV/CV source items,
    # check for it explicitly before trusting these composite scores -- as
    # written, missing items silently lower a sum score or binary flag (e.g. a
    # respondent missing every SV3 item is coded SV3_code=0, "no perpetration",
    # not missing) rather than propagating NaN for imputation.
    use = [c for c in cols if c in df.columns]
    if not use:
        raise KeyError(f"None of columns found: {cols}")
    return df[use].sum(axis=1)


def preprocess_raw(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Transform a raw NMHSK-CA 2022 item-level export into the engineered
    feature table used throughout training/inference/analysis.

    Parameters
    ----------
    df:
        Raw item-level dataframe with the original NMHSK-CA 2022 column
        names (KDBDRS_*, SV2_*/SV3_*, CV1_*/CV2_*, PHQ_*, GAD_*, SAS1_*,
        F3/F4_M/F4_F/F5M/F5F/F6, FA_M/FA_F/FT_M/FT_F, IGD_P, PHA1, ST1_*,
        ksQ15_t/ksQ16_t/ksQ22_t/ksQ24_t, dong, sex/gender, and ODD_CP).
    require_target:
        If True (default), the outcome column ODD_CP must be present and
        is cast to int. Set False to preprocess new records with an
        unknown outcome (the inference use case) -- ODD_CP is then
        omitted from the output.

    Returns
    -------
    A dataframe restricted to the locked analysis columns (see
    metadata.py), ready for features.build_preprocessor().
    """
    out = df.copy()
    if "Unnamed: 0" in out.columns:
        out = out.drop(columns=["Unnamed: 0"])
    if "gender" in out.columns and "sex" not in out.columns:
        out = out.rename(columns={"gender": "sex"})

    drop_suicide = [c for c in out.columns if c.upper().startswith("SUICIDE")]
    if drop_suicide:
        out = out.drop(columns=drop_suicide)

    out["PS_code"] = ((out["PS1"] >= 4) | (out["PS2"] <= 2) | (out["PS3"].isin([1, 7]))).astype(int)
    out["CS_code"] = ((out["CS1"] >= 4) | (out["CS2"] <= 2) | (out["CS3"].isin([1, 7]))).astype(int)

    out["PHQ_total"] = _sum_existing(out, [f"PHQ_{i}" for i in range(1, 10)]).astype(float)
    out["GAD_total"] = _sum_existing(out, [f"GAD_{i}" for i in range(1, 8)]).astype(float)

    sv2 = [c for c in ["SV2_1", "SV2_2", "SV2_3", "SV2_4"] if c in out.columns]
    sv3 = [c for c in ["SV3_1", "SV3_2", "SV3_3", "SV3_4"] if c in out.columns]
    out["SV2_code"] = (out[sv2] >= 2).any(axis=1).astype(int)
    out["SV3_code"] = (out[sv3] >= 2).any(axis=1).astype(int)

    cv1 = [f"CV1_{i}" for i in range(1, 6)]
    cv2 = [f"CV2_{i}" for i in range(1, 6)]
    out["CV1_code"] = (out[cv1] >= 2).any(axis=1).astype(int)
    out["CV2_code"] = (out[cv2] >= 2).any(axis=1).astype(int)

    out["SAS1_total"] = _sum_existing(out, [f"SAS1_{i}" for i in range(1, 11)]).astype(float)

    kdb_keep = [c for c in KDBDRS_ALL if c in out.columns and c not in KDBDRS_LEAKAGE]
    if len(kdb_keep) != 18:
        raise RuntimeError(f"Expected 18 ADHD items in KDBDRS_ADHD, got {len(kdb_keep)}: {kdb_keep}")
    out["KDBDRS_ADHD"] = out[kdb_keep].sum(axis=1).astype(float)

    out["mother_missing"] = out["F4_M"].isna().astype(int)
    out["father_missing"] = out["F4_F"].isna().astype(int)
    if "IGD_P" in out.columns:
        out["IGD_P"] = out["IGD_P"].fillna(9999.0)

    keep = list(OTHER_FEATURES) + ["sex"] + list(ADOLESCENT_FEATURES) + list(PARENT_FEATURES)
    if require_target:
        keep = keep + [TARGET_COL]

    missing = [c for c in keep if c not in out.columns]
    if missing:
        raise RuntimeError(f"Missing required columns after preprocessing: {missing}")

    out = out[keep].copy()
    if require_target:
        out[TARGET_COL] = out[TARGET_COL].astype(int)
    out["sex"] = out["sex"].astype(int)
    return out
