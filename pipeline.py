#!/usr/bin/env python3
# =============================================================================
# pipeline.py — Main Orchestrator
# =============================================================================
"""
Earthquake Data Acquisition & Preprocessing Pipeline
=====================================================
Execution modes
---------------
  Full run (default):
    python pipeline.py

  Incremental run (only new rows since last run):
    python pipeline.py --incremental

  Schema setup only (create processed table):
    python pipeline.py --setup

Flow
----
  Raw PostgreSQL table  (earthquakes)
        │
        ▼
  [Stage 1]  Cleaning & Validation     (cleaning.py)
        │
        ▼
  [Stage 2]  Feature Engineering       (features.py)
        │
        ▼
  [Stage 3]  Transformations           (transforms.py)
        │
        ▼
  Processed PostgreSQL table  (earthquakes_processed)
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from db import (
    create_processed_table,
    load_raw_earthquakes,
    load_new_earthquakes,
    get_last_processed_id,
    save_processed,
)
from cleaning import run_cleaning
from features import run_feature_engineering
from transforms import run_transforms

# ── Logging setup ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

PIPELINE_VERSION = "1.0.0"


# ── Pipeline Stages ───────────────────────────────────────────────────────

def generate_report(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    elapsed: float,
) -> None:
    """Print a concise summary report after the pipeline completes."""

    # Compute cleaning stats
    rows_in   = len(raw_df)
    rows_out  = len(processed_df)
    dropped   = rows_in - rows_out
    pct_kept  = rows_out / rows_in * 100 if rows_in else 0

    print()
    print("=" * 60)
    print("  PIPELINE RUN SUMMARY")
    print("=" * 60)
    print(f"  Completed at   : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Elapsed time   : {elapsed:.2f} s")
    print(f"  Pipeline ver.  : {PIPELINE_VERSION}")
    print()
    print(f"  Input rows     : {rows_in:,}")
    print(f"  Output rows    : {rows_out:,}  ({pct_kept:.1f}% retained)")
    print(f"  Dropped rows   : {dropped:,}")
    print()

    if not processed_df.empty:
        print(f"  Date range     : {processed_df['event_time'].min()} → "
              f"{processed_df['event_time'].max()}")
        print(f"  Magnitude      : {processed_df['magnitude'].min():.1f} – "
              f"{processed_df['magnitude'].max():.1f}  "
              f"(mean {processed_df['magnitude'].mean():.2f})")
        print(f"  Depth (km)     : {processed_df['depth_km'].min():.1f} – "
              f"{processed_df['depth_km'].max():.1f}  "
              f"(mean {processed_df['depth_km'].mean():.2f})")
        print()

        if "mag_category" in processed_df.columns:
            print("  Mag category breakdown:")
            cats = processed_df["mag_category"].value_counts().sort_index()
            for cat, cnt in cats.items():
                bar = "█" * min(int(cnt / max(cats) * 30), 30)
                print(f"    {cat:<12} {cnt:>6,}  {bar}")
        print()

        if "depth_category" in processed_df.columns:
            print("  Depth category breakdown:")
            for cat, cnt in processed_df["depth_category"].value_counts().items():
                print(f"    {cat:<14} {cnt:>6,}")

    print("=" * 60)
    print()


# ── Main Entry Point ──────────────────────────────────────────────────────

def run_pipeline(incremental: bool = False) -> pd.DataFrame:
    """
    Execute the full pipeline.

    Parameters
    ----------
    incremental : bool
        If True, only process rows not yet in the processed table.

    Returns
    -------
    pd.DataFrame
        The final processed DataFrame (also saved to PostgreSQL).
    """
    start = time.time()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     EARTHQUAKE DATA PREPROCESSING PIPELINE               ║")
    logger.info(f"║     Mode: {'INCREMENTAL' if incremental else 'FULL':10}  Version: {PIPELINE_VERSION:10}              ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # ── 0. Ensure destination table exists ───────────────────────────────
    create_processed_table()

    # ── 0. Load raw data ──────────────────────────────────────────────────
    if incremental:
        last_id = get_last_processed_id()
        logger.info(f"Incremental mode: loading rows with raw_id > {last_id}")
        raw_df = load_new_earthquakes(since_id=last_id)
    else:
        raw_df = load_raw_earthquakes()

    if raw_df.empty:
        logger.info("No new data to process. Pipeline exiting.")
        return pd.DataFrame()

    # ── 1. Cleaning & Validation ──────────────────────────────────────────
    cleaned_df = run_cleaning(raw_df.copy())

    if cleaned_df.empty:
        logger.warning("All rows were dropped during cleaning. Nothing to save.")
        return pd.DataFrame()

    # ── 2. Feature Engineering ────────────────────────────────────────────
    featured_df = run_feature_engineering(cleaned_df)

    # ── 3. Transformations ────────────────────────────────────────────────
    processed_df = run_transforms(featured_df)

    # ── Tag with pipeline version ─────────────────────────────────────────
    processed_df["pipeline_version"] = PIPELINE_VERSION

    # ── 4. Save to PostgreSQL ─────────────────────────────────────────────
    save_processed(processed_df)

    # ── 5. Summary report ─────────────────────────────────────────────────
    elapsed = time.time() - start
    generate_report(raw_df, processed_df, elapsed)

    return processed_df


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Earthquake Data Acquisition & Preprocessing Pipeline"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only process rows not yet in the processed table.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create the processed table and exit (no data processing).",
    )
    args = parser.parse_args()

    if args.setup:
        create_processed_table()
        print("Schema setup complete.")
        sys.exit(0)

    run_pipeline(incremental=args.incremental)
