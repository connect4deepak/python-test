"""
app.py — Earthquake Pipeline Flask Dashboard
============================================
Routes
  GET  /              → main dashboard (stats + charts)
  GET  /api/stats     → JSON summary stats
  GET  /api/charts    → JSON data for all charts
  GET  /api/table     → JSON paginated data table
  GET  /api/map       → JSON lat/lon points for map
  POST /api/run       → trigger pipeline run (full or incremental)
"""

import sys
import os
import json
import subprocess
from datetime import datetime, timezone

from flask import Flask, render_template, jsonify, request
import psycopg2
import psycopg2.extras

# ── Allow importing pipeline modules from parent dir ─────────────────────
PIPELINE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PIPELINE_DIR)

from config import DB_CONFIG, PROCESSED_TABLE, RAW_TABLE

app = Flask(__name__)


# ── DB helper ─────────────────────────────────────────────────────────────

def query_db(sql: str, params=None) -> list[dict]:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or ())
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def query_one(sql: str, params=None) -> dict:
    rows = query_db(sql, params)
    return rows[0] if rows else {}


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    stats = query_one(f"""
        SELECT
            COUNT(*)                                        AS total_processed,
            ROUND(AVG(magnitude)::numeric, 2)               AS avg_magnitude,
            ROUND(MAX(magnitude)::numeric, 2)               AS max_magnitude,
            ROUND(MIN(magnitude)::numeric, 2)               AS min_magnitude,
            ROUND(AVG(depth_km)::numeric, 2)                AS avg_depth_km,
            ROUND(MAX(depth_km)::numeric, 2)                AS max_depth_km,
            ROUND(AVG(distance_from_ref_km)::numeric, 0)    AS avg_distance_km,
            SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END)     AS outlier_count,
            MIN(event_time)                                 AS earliest_event,
            MAX(event_time)                                 AS latest_event
        FROM {PROCESSED_TABLE};
    """)

    raw = query_one(f"SELECT COUNT(*) AS raw_count FROM {RAW_TABLE};")
    stats["raw_count"] = raw.get("raw_count", 0)

    # Serialise datetimes
    for k, v in stats.items():
        if isinstance(v, datetime):
            stats[k] = v.isoformat()

    return jsonify(stats)


@app.route("/api/charts")
def api_charts():

    # 1. Magnitude category counts
    mag_cats = query_db(f"""
        SELECT mag_category, COUNT(*) AS count
        FROM {PROCESSED_TABLE}
        GROUP BY mag_category
        ORDER BY CASE mag_category
            WHEN 'micro'    THEN 1 WHEN 'minor'    THEN 2 WHEN 'light'   THEN 3
            WHEN 'moderate' THEN 4 WHEN 'strong'   THEN 5 WHEN 'major'   THEN 6
            WHEN 'great'    THEN 7 ELSE 8 END;
    """)

    # 2. Depth category counts
    depth_cats = query_db(f"""
        SELECT depth_category, COUNT(*) AS count
        FROM {PROCESSED_TABLE}
        GROUP BY depth_category
        ORDER BY CASE depth_category
            WHEN 'shallow' THEN 1 WHEN 'intermediate' THEN 2 WHEN 'deep' THEN 3 END;
    """)

    # 3. Events per hour (activity pattern)
    hourly = query_db(f"""
        SELECT hour, COUNT(*) AS count
        FROM {PROCESSED_TABLE}
        GROUP BY hour ORDER BY hour;
    """)

    # 4. Events per day of week
    dow_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    dow_raw = query_db(f"""
        SELECT day_of_week, COUNT(*) AS count
        FROM {PROCESSED_TABLE}
        GROUP BY day_of_week ORDER BY day_of_week;
    """)
    dow = [{"day": dow_map.get(r["day_of_week"], r["day_of_week"]), "count": r["count"]}
           for r in dow_raw]

    # 5. Magnitude distribution histogram (bins of 0.5)
    mag_hist = query_db(f"""
        SELECT
            ROUND((magnitude / 0.5)::numeric, 0) * 0.5 AS bin,
            COUNT(*) AS count
        FROM {PROCESSED_TABLE}
        GROUP BY bin ORDER BY bin;
    """)

    # 6. Events over time (daily)
    timeline = query_db(f"""
        SELECT
            DATE(event_time AT TIME ZONE 'UTC') AS day,
            COUNT(*) AS count,
            ROUND(AVG(magnitude)::numeric, 2)   AS avg_mag
        FROM {PROCESSED_TABLE}
        GROUP BY day ORDER BY day;
    """)
    for r in timeline:
        if isinstance(r.get("day"), datetime) or hasattr(r.get("day"), "isoformat"):
            r["day"] = r["day"].isoformat()

    return jsonify({
        "mag_categories": mag_cats,
        "depth_categories": depth_cats,
        "hourly": hourly,
        "dow": dow,
        "mag_histogram": mag_hist,
        "timeline": timeline,
    })


@app.route("/api/table")
def api_table():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    offset   = (page - 1) * per_page

    mag_filter   = request.args.get("mag_category", "")
    depth_filter = request.args.get("depth_category", "")

    where_clauses, params = [], []
    if mag_filter:
        where_clauses.append("mag_category = %s")
        params.append(mag_filter)
    if depth_filter:
        where_clauses.append("depth_category = %s")
        params.append(depth_filter)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    total = query_one(f"SELECT COUNT(*) AS n FROM {PROCESSED_TABLE} {where};", params)["n"]

    rows = query_db(f"""
        SELECT
            raw_id, magnitude, mag_category, depth_km, depth_category,
            latitude, longitude, place, event_time, status,
            distance_from_ref_km, is_outlier, magnitude_scaled
        FROM {PROCESSED_TABLE}
        {where}
        ORDER BY event_time DESC
        LIMIT %s OFFSET %s;
    """, params + [per_page, offset])

    for r in rows:
        if isinstance(r.get("event_time"), datetime):
            r["event_time"] = r["event_time"].strftime("%Y-%m-%d %H:%M")

    return jsonify({"total": total, "page": page, "per_page": per_page, "rows": rows})


@app.route("/api/map")
def api_map():
    points = query_db(f"""
        SELECT
            latitude, longitude, magnitude, mag_category,
            place, depth_km, event_time
        FROM {PROCESSED_TABLE}
        ORDER BY event_time DESC
        LIMIT 1000;
    """)
    for r in points:
        if isinstance(r.get("event_time"), datetime):
            r["event_time"] = r["event_time"].strftime("%Y-%m-%d %H:%M")
    return jsonify(points)


@app.route("/api/run", methods=["POST"])
def api_run():
    mode = request.json.get("mode", "incremental")
    flag = "--incremental" if mode == "incremental" else ""
    pipeline_path = os.path.join(PIPELINE_DIR, "pipeline.py")
    cmd = [sys.executable, pipeline_path]
    if flag:
        cmd.append(flag)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        return jsonify({
            "status": "ok" if result.returncode == 0 else "error",
            "stdout": result.stdout[-3000:],   # last 3000 chars
            "stderr": result.stderr[-1000:],
        })
    except subprocess.TimeoutExpired:
        return jsonify({"status": "timeout", "stdout": "", "stderr": "Pipeline timed out after 120s"})


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


# ── Test Runner ───────────────────────────────────────────────────────────

import re as _re

TESTS_DIR = os.path.join(PIPELINE_DIR, "tests")


def _parse_test_output(stdout: str, stderr: str, returncode: int) -> dict:
    """
    Parse unittest verbose output into structured results per test.
    Returns a dict with summary stats and per-test list.
    """
    tests = []
    # Match lines like:  test_foo (module.Class.test_foo) ... ok
    #                    test_foo (module.Class.test_foo) ... FAIL
    #                    test_foo (module.Class.test_foo) ... skipped 'reason'
    pattern = _re.compile(
        r'^(test_\S+)\s+\(([^)]+)\)\s+.*\.\.\.\s*(ok|FAIL|ERROR|skipped.*)',
        _re.MULTILINE
    )
    for m in pattern.finditer(stdout):
        name, cls, result = m.group(1), m.group(2), m.group(3).strip()
        status = "pass" if result == "ok" else \
                 "skip" if result.startswith("skipped") else \
                 "fail" if result == "FAIL" else "error"
        # Extract class name from dotted path
        parts   = cls.split(".")
        suite   = parts[-1] if len(parts) > 1 else cls
        tests.append({
            "name":   name,
            "suite":  suite,
            "status": status,
            "detail": result if status not in ("pass", "skip") else "",
        })

    # Parse FAIL/ERROR detail blocks
    fail_blocks = _re.findall(
        r'(FAIL|ERROR): (test_\S+).*?\n-{60,}\n(.*?)\n(?=-{60,}|\Z)',
        stdout, _re.DOTALL
    )
    fail_map = {name: detail.strip() for _, name, detail in fail_blocks}
    for t in tests:
        if t["name"] in fail_map:
            t["detail"] = fail_map[t["name"]]

    # Summary line: "Ran N test(s) in X.XXs"
    summary_m = _re.search(r'Ran (\d+) test.*?in ([\d.]+)s', stdout)
    ran     = int(summary_m.group(1))  if summary_m else len(tests)
    elapsed = float(summary_m.group(2)) if summary_m else 0.0

    passed  = sum(1 for t in tests if t["status"] == "pass")
    failed  = sum(1 for t in tests if t["status"] in ("fail", "error"))
    skipped = sum(1 for t in tests if t["status"] == "skip")

    return {
        "status":  "pass" if returncode == 0 else "fail",
        "ran":     ran,
        "passed":  passed,
        "failed":  failed,
        "skipped": skipped,
        "elapsed": elapsed,
        "tests":   tests,
        "raw":     stdout[-5000:],
    }


@app.route("/api/tests/run", methods=["POST"])
def api_run_tests():
    """
    Run the test suite and return structured JSON results.
    Body: { "suite": "all" | "unit" | "integration" }
    """
    suite = request.json.get("suite", "all")

    run_script = os.path.join(TESTS_DIR, "run_tests.py")
    if not os.path.exists(run_script):
        return jsonify({"status": "error", "raw": f"run_tests.py not found at {TESTS_DIR}"}), 404

    cmd = [sys.executable, run_script]
    if suite == "unit":
        cmd.append("--unit")
    elif suite == "integration":
        cmd.append("--integration")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=TESTS_DIR,
        )
        parsed = _parse_test_output(result.stdout, result.stderr, result.returncode)
        return jsonify(parsed)
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "raw": "Tests timed out after 180s"})
    except Exception as e:
        return jsonify({"status": "error", "raw": str(e)})