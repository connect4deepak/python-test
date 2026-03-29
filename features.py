# =============================================================================
# features.py — Feature Engineering Stage
# =============================================================================
"""
Features added
--------------
TIME FEATURES
  year, month, day_of_week, hour          – Calendar decomposition
  is_weekend                              – Boolean flag
  hour_sin / hour_cos                     – Cyclic encoding of hour-of-day
  month_sin / month_cos                   – Cyclic encoding of month

CATEGORICAL FEATURES
  mag_category     – Human-readable magnitude scale (Richter-based labels)
  depth_category   – Shallow / Intermediate / Deep (USGS classification)

DISTANCE FEATURE
  distance_from_ref_km – Great-circle distance from a reference point
                         (configured in config.py, default: Dublin, Ireland)
"""

import logging
import numpy as np
import pandas as pd
from config import REFERENCE_LAT, REFERENCE_LON, REFERENCE_LABEL

logger = logging.getLogger(__name__)


# ── Time Features ─────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose event_time into calendar components and cyclic encodings.
    Cyclic encoding preserves the circular nature of time:
      hour 23 and hour 0 are numerically close.
    """
    df = df.copy()
    t = df["event_time"].dt

    df["year"]        = t.year.astype("Int16")
    df["month"]       = t.month.astype("Int8")       # 1–12
    df["day_of_week"] = t.dayofweek.astype("Int8")   # 0=Mon, 6=Sun
    df["hour"]        = t.hour.astype("Int8")         # 0–23
    df["is_weekend"]  = df["day_of_week"].isin([5, 6])

    # Cyclic encoding — map to unit circle so ML models see no boundary jump
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.info("[features]   Time features added (year, month, dow, hour, cyclic).")
    return df


# ── Magnitude Category ────────────────────────────────────────────────────
#
# Richter / moment-magnitude labels used by USGS:
#   < 2.0   micro    – not felt by people
#   2–3.9   minor    – felt slightly by some
#   4–4.9   light    – felt by most, minor damage
#   5–5.9   moderate – slight damage to buildings
#   6–6.9   strong   – destructive in populated areas
#   7–7.9   major    – serious damage over large areas
#   ≥ 8     great    – total destruction, tsunamis possible

_MAG_BINS   = [-np.inf, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.inf]
_MAG_LABELS = ["micro", "minor", "light", "moderate", "strong", "major", "great"]


def add_magnitude_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mag_category"] = pd.cut(
        df["magnitude"],
        bins=_MAG_BINS,
        labels=_MAG_LABELS,
        right=False,
    ).astype(str)
    counts = df["mag_category"].value_counts().to_dict()
    logger.info(f"[features]   mag_category distribution: {counts}")
    return df


# ── Depth Category ────────────────────────────────────────────────────────
#
# Standard USGS seismological classification:
#   0  –  70 km   Shallow      – cause most damage (closest to surface)
#   70 – 300 km   Intermediate – reduced but still significant surface impact
#   > 300 km      Deep         – rarely cause surface damage

_DEPTH_BINS   = [0, 70, 300, 700]
_DEPTH_LABELS = ["shallow", "intermediate", "deep"]


def add_depth_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["depth_category"] = pd.cut(
        df["depth_km"],
        bins=_DEPTH_BINS,
        labels=_DEPTH_LABELS,
        include_lowest=True,
    ).astype(str)
    counts = df["depth_category"].value_counts().to_dict()
    logger.info(f"[features]   depth_category distribution: {counts}")
    return df


# ── Distance Feature ──────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorised Haversine formula.
    Returns great-circle distance in kilometres between two points.
    """
    R = 6371.0  # Earth's mean radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute great-circle distance between each earthquake epicentre and
    the reference point defined in config.py.
    """
    df = df.copy()
    df["distance_from_ref_km"] = _haversine_km(
        df["latitude"].values,
        df["longitude"].values,
        REFERENCE_LAT,
        REFERENCE_LON,
    ).round(3)
    logger.info(
        f"[features]   distance_from_ref_km added "
        f"(ref = {REFERENCE_LABEL}; "
        f"min={df['distance_from_ref_km'].min():.0f} km, "
        f"max={df['distance_from_ref_km'].max():.0f} km)."
    )
    return df


# ── Orchestrator ──────────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STAGE 2 — Feature Engineering")
    logger.info("=" * 60)

    df = add_time_features(df)
    df = add_magnitude_category(df)
    df = add_depth_category(df)
    df = add_distance_feature(df)

    logger.info(f"Feature engineering complete. Columns: {list(df.columns)}")
    return df