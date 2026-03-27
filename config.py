# =============================================================================
# config.py — Central configuration for the Earthquake Pipeline
# =============================================================================

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "earthquake",
    "user":     "postgres",
    "password": "your_password_here",   # ← update this
}

# Raw table written by your cron job
RAW_TABLE = "earthquakes"

# Output table for the processed / feature-engineered data
PROCESSED_TABLE = "earthquakes_processed"

# ── Cleaning thresholds ────────────────────────────────────────────────────
MAGNITUDE_MIN   = -2.0   # below this is likely noise
MAGNITUDE_MAX   = 10.0   # physical upper bound
DEPTH_MIN_KM    = 0.0
DEPTH_MAX_KM    = 700.0  # deepest recorded earthquakes ~670 km
LAT_RANGE       = (-90.0,  90.0)
LON_RANGE       = (-180.0, 180.0)

# IQR multiplier for outlier detection (1.5 = standard, 3.0 = conservative)
IQR_MULTIPLIER  = 1.5

# ── Reference point for distance feature (e.g. Dublin, Ireland) ───────────
REFERENCE_LAT   = 53.3498
REFERENCE_LON   = -6.2603
REFERENCE_LABEL = "Dublin, Ireland"
