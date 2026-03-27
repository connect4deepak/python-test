# =============================================================================
# transforms.py — Transformation Stage (Scaling & Encoding)
# =============================================================================
"""
Transformations applied
-----------------------
NUMERIC SCALING (Min-Max normalisation → [0, 1])
  magnitude_scaled   – normalised magnitude
  depth_scaled       – normalised depth
  latitude_scaled    – normalised latitude
  longitude_scaled   – normalised longitude

  Why Min-Max?
    • Bounded output range is friendly for heatmaps and scatter charts.
    • Preserves zero-values (no earthquakes have negative magnitude in practice).
    • Easily interpretable: 1.0 = dataset maximum.

CATEGORICAL ENCODING
  mag_type is one-hot encoded (mc, md, ml, mw, mwb, mwc, mwr, mww …)
  Only the top-N most frequent types are kept; remainder → 'other'.
  Columns are named  magtype_<value>  (boolean / int8).

  mag_category and depth_category are stored as ordered categoricals
  (useful for colour-mapped visualisations).

SCALER PERSISTENCE
  The fitted MinMaxScaler is saved as a pickle file so:
    • The same scaler is reused on incremental batches.
    • Visualisation code can inverse-transform scaled values.
"""

import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

SCALER_PATH = "scaler.pkl"  # saved beside the pipeline scripts


# ── Numeric Scaling ───────────────────────────────────────────────────────

SCALE_COLS = ["magnitude", "depth_km", "latitude", "longitude"]
SCALE_OUTPUT = {
    "magnitude":  "magnitude_scaled",
    "depth_km":   "depth_scaled",
    "latitude":   "latitude_scaled",
    "longitude":  "longitude_scaled",
}


def _load_or_fit_scaler(df: pd.DataFrame) -> MinMaxScaler:
    """
    If a persisted scaler exists on disk, load it (incremental mode).
    Otherwise fit a new one and save it (first-run mode).
    """
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"[transforms] Loaded existing scaler from '{SCALER_PATH}'.")
    else:
        scaler = MinMaxScaler()
        scaler.fit(df[SCALE_COLS].dropna())
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"[transforms] Fitted new MinMaxScaler and saved to '{SCALER_PATH}'.")
    return scaler


def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Min-Max scaling and append *_scaled columns."""
    df = df.copy()
    scaler = _load_or_fit_scaler(df)

    scaled_values = scaler.transform(df[SCALE_COLS].fillna(0))
    scaled_df = pd.DataFrame(
        scaled_values,
        columns=SCALE_COLS,
        index=df.index,
    )

    for raw_col, out_col in SCALE_OUTPUT.items():
        df[out_col] = scaled_df[raw_col].round(6)

    logger.info(
        f"[transforms] Min-Max scaling applied to: {SCALE_COLS}."
    )
    return df


# ── Categorical Encoding ──────────────────────────────────────────────────

TOP_N_MAGTYPES = 8   # keep the N most common mag_type values


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Consolidate rare mag_type values into 'other'.
    2. One-hot encode mag_type → magtype_<value> boolean columns.
    3. Convert mag_category / depth_category to pandas Categorical with
       a defined order (useful for ordered colour scales in Grafana / Plotly).
    """
    df = df.copy()

    # ── mag_type one-hot ────────────────────────────────────────────────
    if "mag_type" in df.columns:
        top_types = (
            df["mag_type"].value_counts().nlargest(TOP_N_MAGTYPES).index.tolist()
        )
        df["mag_type_clean"] = df["mag_type"].where(
            df["mag_type"].isin(top_types), other="other"
        )

        dummies = pd.get_dummies(
            df["mag_type_clean"],
            prefix="magtype",
            dtype=np.int8,
        )
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=["mag_type_clean"], inplace=True)
        logger.info(
            f"[transforms] mag_type one-hot encoded: {list(dummies.columns)}"
        )

    # ── mag_category ordered categorical ───────────────────────────────
    if "mag_category" in df.columns:
        mag_order = ["micro", "minor", "light", "moderate", "strong", "major", "great"]
        df["mag_category"] = pd.Categorical(
            df["mag_category"], categories=mag_order, ordered=True
        )

    # ── depth_category ordered categorical ─────────────────────────────
    if "depth_category" in df.columns:
        depth_order = ["shallow", "intermediate", "deep"]
        df["depth_category"] = pd.Categorical(
            df["depth_category"], categories=depth_order, ordered=True
        )

    logger.info("[transforms] Categorical encoding complete.")
    return df


# ── Select & Order Final Columns ──────────────────────────────────────────

FINAL_COLUMNS = [
    # identifiers / raw fields
    "raw_id", "magnitude", "latitude", "longitude", "depth_km",
    "event_time", "place", "mag_type", "event_type", "status",
    # time features
    "year", "month", "day_of_week", "hour", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # engineered features
    "mag_category", "depth_category", "distance_from_ref_km",
    # scaled numerics
    "magnitude_scaled", "depth_scaled", "latitude_scaled", "longitude_scaled",
]


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns that belong in the processed table.
    One-hot magtype_* columns are dynamically appended.
    """
    magtype_cols = []  # one-hot cols excluded from DB; mag_type text col is sufficient
    keep = [c for c in FINAL_COLUMNS if c in df.columns]
    df = df[keep].copy()

    # Convert ordered categoricals to string for PostgreSQL compatibility
    for col in ["mag_category", "depth_category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    logger.info(f"[transforms] Final column set ({len(df.columns)} cols): {list(df.columns)}")
    return df


# ── Orchestrator ──────────────────────────────────────────────────────────

def run_transforms(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STAGE 3 — Transformations (Scaling & Encoding)")
    logger.info("=" * 60)

    df = scale_numeric(df)
    df = encode_categorical(df)
    df = select_final_columns(df)

    logger.info(f"Transformations complete. Shape: {df.shape}")
    return df