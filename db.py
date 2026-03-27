# =============================================================================
# db.py — PostgreSQL I/O helpers
# =============================================================================

import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from config import DB_CONFIG, RAW_TABLE, PROCESSED_TABLE

logger = logging.getLogger(__name__)


def get_connection():
    """Return a live psycopg2 connection using config.py credentials."""
    return psycopg2.connect(**DB_CONFIG)


# ── READ ──────────────────────────────────────────────────────────────────

def load_raw_earthquakes() -> pd.DataFrame:
    """
    Load the full raw earthquake table into a DataFrame.
    Assumes columns: id, magnitude, latitude, longitude, depth,
                     time, place, magtype, type, status, ...
    """
    query = f"SELECT * FROM {RAW_TABLE};"
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    logger.info(f"Loaded {len(df):,} rows from '{RAW_TABLE}'.")
    return df


def load_new_earthquakes(since_id: int = 0) -> pd.DataFrame:
    """
    Incremental load — fetch only rows with id > since_id.
    Useful when the pipeline runs on a schedule alongside the cron job.
    """
    query = f"SELECT * FROM {RAW_TABLE} WHERE id > %s ORDER BY id;"
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(since_id,))
    logger.info(f"Loaded {len(df):,} new rows (id > {since_id}).")
    return df


def get_last_processed_id() -> int:
    """Return the highest raw_id already written to the processed table."""
    query = f"SELECT COALESCE(MAX(raw_id), 0) FROM {PROCESSED_TABLE};"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchone()[0]


# ── WRITE ─────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame) -> None:
    """
    Upsert the processed DataFrame into earthquakes_processed.
    Uses ON CONFLICT (raw_id) DO UPDATE so re-runs are idempotent.
    """
    if df.empty:
        logger.warning("save_processed called with empty DataFrame — skipping.")
        return

    cols = list(df.columns)
    rows = [tuple(r) for r in df.itertuples(index=False)]

    # Build the SET clause for the upsert (skip the conflict key itself)
    update_cols = [c for c in cols if c != "raw_id"]
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    insert_sql = f"""
        INSERT INTO {PROCESSED_TABLE} ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (raw_id) DO UPDATE SET {update_clause};
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
        conn.commit()

    logger.info(f"Upserted {len(df):,} rows into '{PROCESSED_TABLE}'.")


def create_processed_table() -> None:
    """
    Create the earthquakes_processed table if it does not exist.
    Run this once at setup (or call from pipeline.py on first run).
    """
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {PROCESSED_TABLE} (
        raw_id              BIGINT PRIMARY KEY,

        -- Cleaned core fields
        magnitude           NUMERIC(5,2),
        latitude            NUMERIC(9,5),
        longitude           NUMERIC(9,5),
        depth_km            NUMERIC(8,3),
        event_time          TIMESTAMP WITH TIME ZONE,
        place               TEXT,
        mag_type            TEXT,
        event_type          TEXT,
        status              TEXT,

        -- Time features
        year                SMALLINT,
        month               SMALLINT,
        day_of_week         SMALLINT,   -- 0=Mon … 6=Sun
        hour                SMALLINT,
        is_weekend          BOOLEAN,
        hour_sin            NUMERIC(10,8),
        hour_cos            NUMERIC(10,8),
        month_sin           NUMERIC(10,8),
        month_cos           NUMERIC(10,8),

        -- Engineered features
        mag_category        TEXT,       -- micro/minor/light/moderate/strong/major/great
        depth_category      TEXT,       -- shallow/intermediate/deep
        distance_from_ref_km NUMERIC(10,3),

        -- Scaled / encoded numerics
        magnitude_scaled    NUMERIC(10,6),
        depth_scaled        NUMERIC(10,6),
        latitude_scaled     NUMERIC(10,6),
        longitude_scaled    NUMERIC(10,6),

        -- Pipeline metadata
        processed_at        TIMESTAMP DEFAULT NOW(),
        pipeline_version    TEXT
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    logger.info(f"Table '{PROCESSED_TABLE}' is ready.")
