# Earthquake Data Acquisition & Preprocessing Pipeline

A modular Python pipeline that reads raw earthquake data collected from the
USGS FDSNWS API, cleans it, engineers analytical features, applies numeric
transformations, and writes the result to a PostgreSQL table ready for
analytics and visualisation.

---

## Architecture

```
earthquakes  (raw table, filled by cron job)
      │
      ▼
┌─────────────────────────────────────────────┐
│  Stage 1 — Cleaning & Validation            │  cleaning.py
│  • Schema normalisation                     │
│  • Type coercion                            │
│  • Null handling & required-field checks    │
│  • Physical range validation                │
│  • Duplicate removal                        │
│  • IQR outlier flagging                     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Stage 2 — Feature Engineering              │  features.py
│  • Calendar decomposition (year/month/…)    │
│  • Cyclic encoding (hour_sin/cos, …)        │
│  • Magnitude category (micro → great)       │
│  • Depth category (shallow/inter/deep)      │
│  • Great-circle distance from reference pt  │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Stage 3 — Transformations                  │  transforms.py
│  • Min-Max scaling (magnitude, depth, …)    │
│  • One-hot encoding (mag_type)              │
│  • Ordered categorical (mag_/depth_cat)     │
│  • Scaler persistence (scaler.pkl)          │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
earthquakes_processed  (analytics-ready table)
```

---

## File Structure

```
earthquake_pipeline/
├── config.py          # DB credentials & thresholds
├── db.py              # PostgreSQL read / write helpers
├── cleaning.py        # Stage 1 — Cleaning & Validation
├── features.py        # Stage 2 — Feature Engineering
├── transforms.py      # Stage 3 — Scaling & Encoding
├── pipeline.py        # Main orchestrator (entry point)
├── schema.sql         # DDL for earthquakes_processed table
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Setup

```bash
# 1. Activate your virtual environment
source /path/to/venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Edit config.py — update DB_CONFIG password and REFERENCE_LAT/LON

# 4. Create the processed table (one-time)
python pipeline.py --setup
# or directly in psql:
# psql -U postgres -d earthquake -f schema.sql
```

---

## Running the Pipeline

```bash
# Full run — process every row in the raw table
python pipeline.py

# Incremental run — only rows added since the last pipeline run
python pipeline.py --incremental
```

### Add to crontab (run 5 minutes after the data-fetch cron job)

```cron
# Data fetch  — every hour at :00
0 * * * * /path/to/venv/bin/python /path/to/fetch_earthquakes.py

# Pipeline    — every hour at :05 (gives the fetch job time to finish)
5 * * * * /path/to/venv/bin/python /path/to/earthquake_pipeline/pipeline.py --incremental >> /var/log/eq_pipeline.log 2>&1
```

---

## Features Produced

| Column | Type | Description |
|---|---|---|
| `year` / `month` / `hour` | int | Calendar decomposition |
| `day_of_week` | int | 0 = Monday, 6 = Sunday |
| `is_weekend` | bool | Saturday or Sunday |
| `hour_sin` / `hour_cos` | float | Cyclic hour encoding |
| `month_sin` / `month_cos` | float | Cyclic month encoding |
| `mag_category` | text | micro / minor / light / moderate / strong / major / great |
| `depth_category` | text | shallow / intermediate / deep |
| `distance_from_ref_km` | float | Great-circle km from reference point |
| `magnitude_scaled` | float | Min-Max normalised magnitude [0,1] |
| `depth_scaled` | float | Min-Max normalised depth [0,1] |
| `latitude_scaled` | float | Min-Max normalised latitude [0,1] |
| `longitude_scaled` | float | Min-Max normalised longitude [0,1] |
| `magtype_ml` … | int | One-hot encoded magnitude type |
| `is_outlier` | bool | IQR outlier flag (row kept, not dropped) |

---

## Configuration (config.py)

| Setting | Default | Purpose |
|---|---|---|
| `MAGNITUDE_MIN/MAX` | -2 / 10 | Physical validity range |
| `DEPTH_MIN/MAX_KM` | 0 / 700 | Physical validity range |
| `IQR_MULTIPLIER` | 1.5 | Outlier sensitivity (higher = looser) |
| `REFERENCE_LAT/LON` | Dublin, Ireland | Distance feature anchor point |
| `TOP_N_MAGTYPES` | 8 | One-hot encode top N mag types |
