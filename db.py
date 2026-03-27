# =============================================================================
# db.py — PostgreSQL I/O helpers
# =============================================================================

import logging
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
from config import DB_CONFIG, RAW_TABLE, PROCESSED_TABLE

logger = logging.getLogger(__name__)


def get_connection():
    """Return a live psycopg2 connection using config.py credentials."""
    return psycopg2.connect(**DB_CONFIG)


def get_engine():
    """SQLAlchemy engine — used by pd.read_sql to suppress DBAPI2 warning."""
    c = DB_CONFIG
    url = (f"postgresql+psycopg2://{c['user']}:{c['password']}"
           f"@{c['host']}:{c['port']}/{c['dbname']}")
    return create_engine(url)


# ── READ ──────────────────────────────────────────────────────────────────

def load_raw_earthquakes() -> pd.DataFrame:
    """Load the full raw earthquake table into a DataFrame."""
    query = f"SELECT * FROM {RAW_TABLE};"
    df = pd.read_sql(query, get_engine())
    logger.info(f"Loaded {len(df):,} rows from '{RAW_TABLE}'.")
    return df


def load_new_earthquakes(since_id: int = 0) -> pd.DataFrame:
    """Incremental load — fetch only rows with id > since_id."""
    query = f"SELECT * FROM {RAW_TABLE} WHERE id > %(sid)s ORDER BY id;"
    df = pd.read_sql(query, get_engine(), params={"sid": since_id})
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

def _to_native(val):
    """
    Convert numpy scalar types to native Python equivalents.
    psycopg2 cannot adapt numpy.int16, numpy.float32, etc.
    """
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return None if np.isnan(val) else float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def save_processed(df: pd.DataFrame) -> None:
    """
    Upsert the processed DataFrame into earthquakes_processed.
    Uses ON CONFLICT (raw_id) DO UPDATE so re-runs are idempotent.
    """
    if df.empty:
        logger.warning("save_processed called with empty DataFrame — skipping.")
        return

    # Convert all numpy scalar types → native Python (psycopg2 requirement)
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(_to_native)

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
    """
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {PROCESSED_TABLE} (
        raw_id                  BIGINT          PRIMARY KEY,
        magnitude               NUMERIC(5,2),
        latitude                NUMERIC(9,5),
        longitude               NUMERIC(9,5),
        depth_km                NUMERIC(8,3),
        event_time              TIMESTAMPTZ,
        place                   TEXT,
        mag_type                TEXT,
        event_type              TEXT,
        status                  TEXT,
        year                    SMALLINT,
        month                   SMALLINT,
        day_of_week             SMALLINT,
        hour                    SMALLINT,
        is_weekend              BOOLEAN,
        hour_sin                NUMERIC(10,8),
        hour_cos                NUMERIC(10,8),
        month_sin               NUMERIC(10,8),
        month_cos               NUMERIC(10,8),
        mag_category            TEXT,
        depth_category          TEXT,
        distance_from_ref_km    NUMERIC(10,3),
        is_outlier              BOOLEAN         DEFAULT FALSE,
        magnitude_scaled        NUMERIC(10,6),
        depth_scaled            NUMERIC(10,6),
        latitude_scaled         NUMERIC(10,6),
        longitude_scaled        NUMERIC(10,6),
        pipeline_version        TEXT,
        processed_at            TIMESTAMPTZ     DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_ep_event_time     ON {PROCESSED_TABLE} (event_time);
    CREATE INDEX IF NOT EXISTS idx_ep_magnitude      ON {PROCESSED_TABLE} (magnitude);
    CREATE INDEX IF NOT EXISTS idx_ep_mag_category   ON {PROCESSED_TABLE} (mag_category);
    CREATE INDEX IF NOT EXISTS idx_ep_depth_category ON {PROCESSED_TABLE} (depth_category);
    CREATE INDEX IF NOT EXISTS idx_ep_location       ON {PROCESSED_TABLE} (latitude, longitude);
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    logger.info(f"Table '{PROCESSED_TABLE}' is ready.")