"""Feature schema for the ODD Random Forest pipeline.

This mirrors the locked feature list used throughout the manuscript
(Methods 2.1-2.3). It is the single source of truth for which engineered
columns exist, what type each one is, and which feature set (Adolescent /
Parent / Combined) each one belongs to.
"""
from __future__ import annotations

TARGET_COL = "ODD_CP"

OTHER_FEATURES = ["dong"]

ADOLESCENT_FEATURES = [
    "PHA1",
    "ST1_1",
    "ST1_2",
    "ST1_3",
    "ST1_4",
    "CS_code",
    "SV2_code",
    "SV3_code",
    "CV1_code",
    "CV2_code",
    "KDBDRS_ADHD",
    "SAS1_total",
    "IGD_P",
    "ksQ15_t",
    "ksQ16_t",
    "ksQ22_t",
    "ksQ24_t",
]

PARENT_FEATURES = [
    "F3",
    "F4_M",
    "F4_F",
    "F5M",
    "F5F",
    "F6",
    "PS_code",
    "PHQ_total",
    "GAD_total",
    "FA_M",
    "FA_F",
    "FT_M",
    "FT_F",
    "mother_missing",
    "father_missing",
]

FEATURE_TYPES = {
    "dong": "nominal",
    "PHA1": "ordinal",
    "ST1_1": "ordinal",
    "ST1_2": "ordinal",
    "ST1_3": "ordinal",
    "ST1_4": "ordinal",
    "CS_code": "binary",
    "SV2_code": "binary",
    "SV3_code": "binary",
    "CV1_code": "binary",
    "CV2_code": "binary",
    "KDBDRS_ADHD": "continuous",
    "SAS1_total": "continuous",
    "IGD_P": "nominal",
    "ksQ15_t": "ordinal",
    "ksQ16_t": "binary",
    "ksQ22_t": "binary",
    "ksQ24_t": "binary",
    "F3": "nominal",
    "F4_M": "ordinal",
    "F4_F": "ordinal",
    "F5M": "nominal",
    "F5F": "nominal",
    "F6": "continuous",
    "PS_code": "binary",
    "PHQ_total": "continuous",
    "GAD_total": "continuous",
    "FA_M": "ordinal",
    "FA_F": "ordinal",
    "FT_M": "ordinal",
    "FT_F": "ordinal",
    "mother_missing": "binary",
    "father_missing": "binary",
}

# Human-readable labels, used by analyze.py so permutation/SHAP tables read
# like Figure 4 / Figure 5 in the manuscript instead of raw column codes.
FEATURE_LABELS = {
    "dong": "Residential district",
    "PHA1": "Physical activity (days/week >=60 min)",
    "ST1_1": "Video screen time",
    "ST1_2": "Social media screen time",
    "ST1_3": "Game screen time",
    "ST1_4": "Educational screen time",
    "CS_code": "Child sleep problem",
    "SV2_code": "School bullying victimization",
    "SV3_code": "School bullying perpetration",
    "CV1_code": "Cyberbullying victimization",
    "CV2_code": "Cyberbullying perpetration",
    "KDBDRS_ADHD": "ADHD symptoms (18-item parent-rated score)",
    "SAS1_total": "Smartphone addiction (SAS-SV total)",
    "IGD_P": "Game addiction risk (IGD)",
    "ksQ15_t": "Best friend (has a close friend)",
    "ksQ16_t": "Academic decline",
    "ksQ22_t": "Depressive disorder (adolescent, KSADS)",
    "ksQ24_t": "Anxiety disorder (adolescent, KSADS)",
    "F3": "Parental marital status",
    "F4_M": "Maternal education",
    "F4_F": "Paternal education",
    "F5M": "Maternal occupation",
    "F5F": "Paternal occupation",
    "F6": "Household monthly income",
    "PS_code": "Parental sleep problem",
    "PHQ_total": "Parental depression (PHQ-9 total)",
    "GAD_total": "Parental anxiety (GAD-7 total)",
    "FA_M": "Maternal alcohol use",
    "FA_F": "Paternal alcohol use",
    "FT_M": "Maternal smoking",
    "FT_F": "Paternal smoking",
    "mother_missing": "Mother absent from household",
    "father_missing": "Father absent from household",
}

FEATURE_GROUPS = {
    "binary": [c for c, t in FEATURE_TYPES.items() if t == "binary"],
    "ordinal": [c for c, t in FEATURE_TYPES.items() if t == "ordinal"],
    "nominal": [c for c, t in FEATURE_TYPES.items() if t == "nominal"],
    "continuous": [c for c, t in FEATURE_TYPES.items() if t == "continuous"],
}

# Parental ordinal items are median-imputed (numeric, not most-frequent),
# because they are the only ordinals where a parent can be structurally
# absent (see mother_missing / father_missing) rather than just missing
# at random. See Methods 2.2.
PARENT_ORDINAL_MEDIAN_IMPUTE = {"F4_M", "F4_F", "FA_M", "FA_F", "FT_M", "FT_F"}

FEATURE_SET_GROUPS = {
    "combined": ["other", "adolescent", "parent"],
    "adolescent": ["other", "adolescent"],
    "parent": ["parent"],
}


def select_feature_cols(feature_set: str) -> list[str]:
    """Column list for one of 'combined' / 'adolescent' / 'parent'."""
    if feature_set not in FEATURE_SET_GROUPS:
        raise ValueError(
            f"Unknown feature_set {feature_set!r}; choose from {list(FEATURE_SET_GROUPS)}"
        )
    group_map = {
        "other": OTHER_FEATURES,
        "adolescent": ADOLESCENT_FEATURES,
        "parent": PARENT_FEATURES,
    }
    cols: list[str] = []
    for group_name in FEATURE_SET_GROUPS[feature_set]:
        cols.extend(group_map[group_name])
    return cols
