# =============================================================================
# cleaning.py — Data Cleaning & Validation Stage
# =============================================================================
"""
Responsibilities
----------------
1. Schema normalisation  – rename columns to a consistent internal name.
2. Type coercion         – ensure numeric / datetime types are correct.
3. Null handling         – report and impute or drop missing values.
4. Range validation      – flag / remove physically impossible values.
5. Duplicate removal     – keep the most recent record per event id.
6. Outlier detection     – IQR-based flagging on magnitude and depth.

All steps emit INFO-level log lines with row counts so the pipeline
operator can see exactly what was removed at each gate.
"""

import logging
import numpy as np
import pandas as pd
from config import (
    MAGNITUDE_MIN, MAGNITUDE_MAX,
    DEPTH_MIN_KM, DEPTH_MAX_KM,
    LAT_RANGE, LON_RANGE,
    IQR_MULTIPLIER,
)

logger = logging.getLogger(__name__)


# ── 1. Schema Normalisation ───────────────────────────────────────────────

# Map whatever column names your cron job uses → internal pipeline names.
# Adjust the left-hand side if your raw table uses different names.
COLUMN_RENAME_MAP = {
    "mag":        "magnitude",
    "magnitude":  "magnitude",   # already correct
    "lat":        "latitude",
    "latitude":   "latitude",
    "lon":        "longitude",
    "longitude":  "longitude",
    "depth":      "depth_km",
    "time":       "event_time",
    "updated":    "updated_time",
    "place":      "place",
    "type":       "event_type",
    "magtype":    "mag_type",
    "magType":    "mag_type",
    "status":     "status",
    "id":         "raw_id",
}


def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to the pipeline's internal naming convention."""
    df = df.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items()
                             if k in df.columns})
    logger.info(f"[schema]     {len(df):,} rows after column rename.")
    return df


# ── 2. Type Coercion ──────────────────────────────────────────────────────

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to the correct Python / pandas dtype.
    Rows that cannot be coerced are set to NaN (errors='coerce').
    """
    numeric_cols = ["magnitude", "latitude", "longitude", "depth_km"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"],
                                          utc=True, errors="coerce")

    # Normalise free-text fields
    for col in ["place", "event_type", "mag_type", "status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "": np.nan})

    logger.info(f"[types]      {len(df):,} rows after type coercion.")
    return df


# ── 3. Null Handling ──────────────────────────────────────────────────────

# Columns that MUST be present for a record to be useful
REQUIRED_COLS = ["magnitude", "latitude", "longitude", "depth_km", "event_time"]


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Log the null-count per required column.
    2. Drop rows missing any required value.
    3. For optional text columns, fill with 'unknown'.
    """
    # ── report
    null_counts = df[REQUIRED_COLS].isnull().sum()
    for col, n in null_counts.items():
        if n:
            logger.warning(f"[nulls]      '{col}' has {n:,} null values — rows will be dropped.")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    after  = len(df)
    dropped = before - after
    if dropped:
        logger.info(f"[nulls]      Dropped {dropped:,} rows with null required fields. "
                    f"{after:,} rows remain.")
    else:
        logger.info(f"[nulls]      No null required-field rows found.")

    # Fill optional text fields
    for col in ["place", "event_type", "mag_type", "status"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    return df


# ── 4. Range Validation ───────────────────────────────────────────────────

def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows whose core numeric fields fall outside physically
    plausible ranges (defined in config.py).
    """
    before = len(df)

    mask = (
        df["magnitude"].between(MAGNITUDE_MIN, MAGNITUDE_MAX) &
        df["latitude"].between(*LAT_RANGE) &
        df["longitude"].between(*LON_RANGE) &
        df["depth_km"].between(DEPTH_MIN_KM, DEPTH_MAX_KM)
    )

    df = df[mask].copy()
    dropped = before - len(df)
    if dropped:
        logger.info(f"[ranges]     Dropped {dropped:,} rows outside valid physical ranges. "
                    f"{len(df):,} rows remain.")
    else:
        logger.info(f"[ranges]     All rows pass range validation.")
    return df


# ── 5. Duplicate Removal ──────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows.
    If an 'updated_time' column exists, keep the most recently updated
    record per (latitude, longitude, event_time) triplet.
    """
    before = len(df)

    # Hard duplicates (identical rows)
    df = df.drop_duplicates()

    # Soft duplicates — same event, multiple ingestion records
    if "updated_time" in df.columns and "raw_id" in df.columns:
        df = (df.sort_values("updated_time", ascending=False)
                .drop_duplicates(subset=["latitude", "longitude", "event_time"])
                .sort_values("raw_id"))
    else:
        df = df.drop_duplicates(subset=["latitude", "longitude", "event_time"])

    dropped = before - len(df)
    if dropped:
        logger.info(f"[duplicates] Removed {dropped:,} duplicate rows. "
                    f"{len(df):,} rows remain.")
    else:
        logger.info(f"[duplicates] No duplicate rows found.")
    return df


# ── 6. Outlier Detection ──────────────────────────────────────────────────

def _iqr_bounds(series: pd.Series, multiplier: float):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - multiplier * IQR, Q3 + multiplier * IQR


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier removal on magnitude and depth.

    Note: earthquake magnitude follows a power-law (Gutenberg–Richter),
    so IQR here acts as a loose sanity check rather than strict exclusion.
    Rows flagged as outliers are logged but *not* dropped — instead a
    boolean column 'is_outlier' is added for downstream handling.
    """
    df = df.copy()
    df["is_outlier"] = False

    for col in ["magnitude", "depth_km"]:
        lo, hi = _iqr_bounds(df[col], IQR_MULTIPLIER)
        flag = ~df[col].between(lo, hi)
        count = flag.sum()
        if count:
            logger.info(f"[outliers]   '{col}' — {count:,} rows outside "
                        f"[{lo:.2f}, {hi:.2f}] (IQR×{IQR_MULTIPLIER}) — "
                        f"flagged as is_outlier=True (not dropped).")
        df.loc[flag, "is_outlier"] = True

    return df


# ── Orchestrator ──────────────────────────────────────────────────────────

def run_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute all cleaning steps in order and return the cleaned DataFrame.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1 — Data Cleaning & Validation")
    logger.info(f"Input: {len(df):,} rows")
    logger.info("=" * 60)

    df = normalise_schema(df)
    df = coerce_types(df)
    df = handle_nulls(df)
    df = validate_ranges(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)

    logger.info(f"Cleaning complete. Output: {len(df):,} rows.")
    return df
