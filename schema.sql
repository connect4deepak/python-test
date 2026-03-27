-- =============================================================================
-- schema.sql — Processed Earthquake Table
-- =============================================================================
-- Run once to create the destination table:
--   psql -U postgres -d earthquake -f schema.sql
-- (The pipeline also auto-creates this on first run via db.py)
-- =============================================================================

CREATE TABLE IF NOT EXISTS earthquakes_processed (

    -- ── Identity ──────────────────────────────────────────────────────────
    raw_id                  BIGINT          PRIMARY KEY,   -- FK to earthquakes.id

    -- ── Cleaned core fields ───────────────────────────────────────────────
    magnitude               NUMERIC(5,2)    NOT NULL,
    latitude                NUMERIC(9,5)    NOT NULL,
    longitude               NUMERIC(9,5)    NOT NULL,
    depth_km                NUMERIC(8,3)    NOT NULL,
    event_time              TIMESTAMPTZ     NOT NULL,
    place                   TEXT,
    mag_type                TEXT,
    event_type              TEXT,
    status                  TEXT,

    -- ── Time features ─────────────────────────────────────────────────────
    year                    SMALLINT,
    month                   SMALLINT,       -- 1–12
    day_of_week             SMALLINT,       -- 0=Mon, 6=Sun
    hour                    SMALLINT,       -- 0–23
    is_weekend              BOOLEAN,
    hour_sin                NUMERIC(10,8),
    hour_cos                NUMERIC(10,8),
    month_sin               NUMERIC(10,8),
    month_cos               NUMERIC(10,8),

    -- ── Engineered features ───────────────────────────────────────────────
    mag_category            TEXT,
        -- CHECK (mag_category IN ('micro','minor','light','moderate','strong','major','great')),
    depth_category          TEXT,
        -- CHECK (depth_category IN ('shallow','intermediate','deep')),
    distance_from_ref_km    NUMERIC(10,3),
    is_outlier              BOOLEAN         DEFAULT FALSE,

    -- ── Scaled numerics ───────────────────────────────────────────────────
    magnitude_scaled        NUMERIC(10,6),
    depth_scaled            NUMERIC(10,6),
    latitude_scaled         NUMERIC(10,6),
    longitude_scaled        NUMERIC(10,6),

    -- ── One-hot encoded mag_type (most common types) ──────────────────────
    magtype_md              SMALLINT        DEFAULT 0,
    magtype_ml              SMALLINT        DEFAULT 0,
    magtype_mw              SMALLINT        DEFAULT 0,
    magtype_mwb             SMALLINT        DEFAULT 0,
    magtype_mwc             SMALLINT        DEFAULT 0,
    magtype_mwr             SMALLINT        DEFAULT 0,
    magtype_mww             SMALLINT        DEFAULT 0,
    magtype_other           SMALLINT        DEFAULT 0,

    -- ── Pipeline metadata ─────────────────────────────────────────────────
    pipeline_version        TEXT,
    processed_at            TIMESTAMPTZ     DEFAULT NOW()
);

-- Indexes for common analytics queries
CREATE INDEX IF NOT EXISTS idx_ep_event_time     ON earthquakes_processed (event_time);
CREATE INDEX IF NOT EXISTS idx_ep_magnitude      ON earthquakes_processed (magnitude);
CREATE INDEX IF NOT EXISTS idx_ep_mag_category   ON earthquakes_processed (mag_category);
CREATE INDEX IF NOT EXISTS idx_ep_depth_category ON earthquakes_processed (depth_category);
CREATE INDEX IF NOT EXISTS idx_ep_location       ON earthquakes_processed (latitude, longitude);

COMMENT ON TABLE earthquakes_processed IS
    'Cleaned, feature-engineered and normalised earthquake data.
     Populated by the Python preprocessing pipeline (pipeline.py).';
