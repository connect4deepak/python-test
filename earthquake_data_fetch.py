#cronjob 
#*/5 * * * * /home/ubuntu/venv/bin/python3 /home/ubuntu/earthquake_fetch.py >> /home/ubuntu/logs/cron.log 2>&1

#!/usr/bin/env python3
"""
earthquake_fetch.py
────────────────────────────────────────────────────────────────────────────
Real-time USGS Earthquake Data Pipeline
  • Fetches last 5 minutes of earthquake data from USGS FDSN API
  • Stores results in PostgreSQL (deduplicates on event_id)
  • Exports a timestamped CSV per run
  • Designed to run as a cron job every 5 minutes

Author  : Deepak Kumar
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import json
import logging
import requests
import psycopg2
import pandas as pd

from datetime import datetime, timezone, timedelta
from pathlib import Path
#from dotenv import load_dotenv

# ─────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────

# Load .env from the same directory as this script
SCRIPT_DIR = Path(__file__).parent.resolve()
#load_dotenv(SCRIPT_DIR / ".env")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "earthquake_db"),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "root"),    
}

CSV_OUTPUT_DIR = Path(os.getenv("CSV_OUTPUT_DIR", SCRIPT_DIR / "csv_exports"))
LOG_FILE       = Path(os.getenv("LOG_FILE",       SCRIPT_DIR / "logs" / "earthquake.log"))

USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# How far back to fetch on each run (slightly overlapping to avoid gaps)
LOOKBACK_MINUTES = 10

# ─────────────────────────────────────────
# 1. Logging setup
# ─────────────────────────────────────────

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 2. Database helpers
# ─────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS earthquakes (
    id             SERIAL PRIMARY KEY,
    event_id       VARCHAR(30)  NOT NULL UNIQUE,   -- USGS unique ID  e.g. us7000pxyz
    magnitude      FLOAT,
    magnitude_type VARCHAR(10),
    place          TEXT,
    event_time     TIMESTAMPTZ,
    depth_km       FLOAT,
    latitude       FLOAT,
    longitude      FLOAT,
    alert          VARCHAR(10),
    tsunami        SMALLINT,
    sig            INT,                             -- USGS significance score
    url            TEXT,
    status         VARCHAR(20),
    net            VARCHAR(10),                     -- contributing network
    fetched_at     TIMESTAMPTZ  DEFAULT NOW()
);

-- Index for time-range queries
CREATE INDEX IF NOT EXISTS idx_earthquakes_event_time ON earthquakes (event_time DESC);
CREATE INDEX IF NOT EXISTS idx_earthquakes_magnitude  ON earthquakes (magnitude DESC);
"""

INSERT_SQL = """
INSERT INTO earthquakes (
    event_id, magnitude, magnitude_type, place, event_time,
    depth_km, latitude, longitude, alert, tsunami,
    sig, url, status, net
) VALUES (
    %(event_id)s, %(magnitude)s, %(magnitude_type)s, %(place)s, %(event_time)s,
    %(depth_km)s, %(latitude)s, %(longitude)s, %(alert)s, %(tsunami)s,
    %(sig)s, %(url)s, %(status)s, %(net)s
)
ON CONFLICT (event_id) DO NOTHING;
"""


def get_connection():
    """Return a psycopg2 connection."""
    return psycopg2.connect(**DB_CONFIG)


def ensure_table(conn):
    """Create table + indexes if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    log.info("Table created/verified")


def insert_events(conn, events: list[dict]) -> tuple[int, int]:
    """
    Bulk-insert earthquake events.
    Returns (inserted_count, skipped_count).
    """
    inserted = skipped = 0
    with conn.cursor() as cur:
        for ev in events:
            cur.execute(INSERT_SQL, ev)
            if cur.rowcount == 1:
                inserted += 1
            else:
                skipped += 1
    conn.commit()
    return inserted, skipped


# ─────────────────────────────────────────
# 3. USGS API fetch
# ─────────────────────────────────────────

def fetch_usgs_events(lookback_minutes: int = LOOKBACK_MINUTES) -> list[dict]:
    """
    Query USGS FDSN API for earthquakes in the past `lookback_minutes`.
    Returns a list of normalised dicts ready for DB insertion.
    """
    now       = datetime.now(timezone.utc)
    starttime = now - timedelta(minutes=lookback_minutes)

    params = {
        "format":    "geojson",
        "starttime": starttime.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime":   now.strftime("%Y-%m-%dT%H:%M:%S"),
        "orderby":   "time",
        "limit":     500,                  # max per call
    }

    log.info("Fetching data from USGS API (%d-minute window)…", lookback_minutes)

    try:
        resp = requests.get(USGS_API_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("USGS API request failed: %s", exc)
        return []

    data     = resp.json()
    features = data.get("features", [])
    log.info("Fetched %d events from USGS", len(features))

    events = []
    for feature in features:
        props = feature.get("properties", {})
        geom  = feature.get("geometry",   {})
        coords = geom.get("coordinates", [None, None, None])   # [lon, lat, depth_km]

        # Convert epoch milliseconds → timezone-aware datetime
        epoch_ms = props.get("time")
        event_time = (
            datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
            if epoch_ms else None
        )

        events.append({
            "event_id":       feature.get("id"),
            "magnitude":      props.get("mag"),
            "magnitude_type": props.get("magType"),
            "place":          props.get("place"),
            "event_time":     event_time,
            "depth_km":       coords[2],
            "latitude":       coords[1],
            "longitude":      coords[0],
            "alert":          props.get("alert"),
            "tsunami":        int(props.get("tsunami", 0) or 0),
            "sig":            props.get("sig"),
            "url":            props.get("url"),
            "status":         props.get("status"),
            "net":            props.get("net"),
        })

    return events


# ─────────────────────────────────────────
# 4. CSV export
# ─────────────────────────────────────────

def export_csv(events: list[dict]) -> Path:
    """
    Write the fetched events to a timestamped CSV file.
    Returns the path of the created file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = CSV_OUTPUT_DIR / f"earthquakes_{timestamp}.csv"

    if not events:
        log.warning("No events to export — CSV not written")
        return csv_path

    df = pd.DataFrame(events)
    df.to_csv(csv_path, index=False)
    log.info("CSV exported → %s  (%d rows)", csv_path, len(df))
    return csv_path


# ─────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Earthquake pipeline run started")

    # -- Fetch --
    events = fetch_usgs_events()

    # -- Export CSV (even if DB fails) --
    export_csv(events)

    if not events:
        log.warning("No events fetched — nothing to insert")
        return

    # -- Store in DB --
    try:
        conn = get_connection()
        ensure_table(conn)
        inserted, skipped = insert_events(conn, events)
        log.info(
            "DB: %d new record(s) inserted, %d duplicate(s) skipped",
            inserted, skipped
        )
        conn.close()
    except psycopg2.OperationalError as exc:
        log.error("Database connection failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected DB error: %s", exc)
        sys.exit(1)

    log.info("Pipeline run complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()