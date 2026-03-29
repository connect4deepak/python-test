# =============================================================================
# test_integration.py — Integration Test: Flask API ↔ PostgreSQL
# =============================================================================
"""
Integration test: verifies that the Flask frontend (app.py) correctly
interacts with the PostgreSQL backend (earthquakes_processed table).

Tests use Flask's built-in test client so no live HTTP server is needed,
but the real PostgreSQL database IS used — this is a true integration test
that exercises the full frontend → backend → database → response chain.

Prerequisites
-------------
  • PostgreSQL running with earthquakes_processed populated
  • config.py has correct DB credentials
  • Run after at least one full pipeline.py execution

Skip behaviour
--------------
  If the DB is unreachable or the table is empty, tests are skipped
  gracefully rather than failing with a confusing error.
"""

import sys
import os
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python-test'))


# ── Try to connect to the DB — skip all tests if unavailable ──────────────

DB_AVAILABLE   = False
TABLE_HAS_DATA = False

try:
    import psycopg2
    from config import DB_CONFIG, PROCESSED_TABLE
    _conn = psycopg2.connect(**DB_CONFIG)
    _cur  = _conn.cursor()
    _cur.execute(f"SELECT COUNT(*) FROM {PROCESSED_TABLE}")
    _count = _cur.fetchone()[0]
    _conn.close()
    DB_AVAILABLE   = True
    TABLE_HAS_DATA = _count > 0
except Exception as _e:
    print(f"[integration] DB not available: {_e} — tests will be skipped.")


# ── Import the Flask app ───────────────────────────────────────────────────

try:
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    FLASK_AVAILABLE = True
except Exception as _fe:
    print(f"[integration] Flask app import failed: {_fe}")
    FLASK_AVAILABLE = False


@unittest.skipUnless(DB_AVAILABLE and FLASK_AVAILABLE,
                     "DB or Flask app not available")
class TestFlaskAPIIntegration(unittest.TestCase):
    """
    Integration tests: Flask test client → real PostgreSQL DB.
    Each test makes a request and validates the response structure,
    HTTP status, Content-Type, and data correctness.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = flask_app.test_client()

    # ── /api/stats ────────────────────────────────────────────────────────

    def test_stats_returns_200(self):
        resp = self.client.get("/api/stats")
        self.assertEqual(resp.status_code, 200,
                         msg=f"Expected 200, got {resp.status_code}")

    def test_stats_content_type_is_json(self):
        resp = self.client.get("/api/stats")
        self.assertIn("application/json", resp.content_type)

    def test_stats_has_required_keys(self):
        resp = self.client.get("/api/stats")
        data = json.loads(resp.data)
        required_keys = [
            "total_processed", "avg_magnitude", "max_magnitude",
            "min_magnitude",   "avg_depth_km",  "outlier_count",
            "raw_count",
        ]
        for key in required_keys:
            self.assertIn(key, data, msg=f"Missing key in /api/stats: '{key}'")

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_stats_processed_count_positive(self):
        resp = self.client.get("/api/stats")
        data = json.loads(resp.data)
        self.assertGreater(int(data["total_processed"]), 0)

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_stats_avg_magnitude_in_valid_range(self):
        resp = self.client.get("/api/stats")
        data = json.loads(resp.data)
        avg = float(data["avg_magnitude"])
        self.assertGreaterEqual(avg, -2.0)
        self.assertLessEqual(avg,   10.0)

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_stats_max_gte_min_magnitude(self):
        resp = self.client.get("/api/stats")
        data = json.loads(resp.data)
        self.assertGreaterEqual(
            float(data["max_magnitude"]),
            float(data["min_magnitude"])
        )

    # ── /api/charts ───────────────────────────────────────────────────────

    def test_charts_returns_200(self):
        resp = self.client.get("/api/charts")
        self.assertEqual(resp.status_code, 200)

    def test_charts_content_type_is_json(self):
        resp = self.client.get("/api/charts")
        self.assertIn("application/json", resp.content_type)

    def test_charts_has_required_sections(self):
        resp = self.client.get("/api/charts")
        data = json.loads(resp.data)
        required = ["mag_categories", "depth_categories",
                    "hourly", "dow", "mag_histogram", "timeline"]
        for key in required:
            self.assertIn(key, data, msg=f"Missing chart section: '{key}'")

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_charts_mag_categories_structure(self):
        resp = self.client.get("/api/charts")
        data = json.loads(resp.data)
        cats = data["mag_categories"]
        self.assertIsInstance(cats, list)
        self.assertGreater(len(cats), 0)
        # Each item must have mag_category and count
        for item in cats:
            self.assertIn("mag_category", item)
            self.assertIn("count",        item)
            self.assertGreater(int(item["count"]), 0)

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_charts_depth_categories_valid_values(self):
        resp = self.client.get("/api/charts")
        data = json.loads(resp.data)
        valid_depths = {"shallow", "intermediate", "deep"}
        for item in data["depth_categories"]:
            self.assertIn(item["depth_category"], valid_depths)

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_charts_hourly_has_24_or_fewer_entries(self):
        resp = self.client.get("/api/charts")
        data = json.loads(resp.data)
        self.assertLessEqual(len(data["hourly"]), 24)

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_charts_timeline_dates_are_strings(self):
        resp = self.client.get("/api/charts")
        data = json.loads(resp.data)
        for item in data["timeline"]:
            self.assertIsInstance(item["day"], str)

    # ── /api/table ────────────────────────────────────────────────────────

    def test_table_returns_200(self):
        resp = self.client.get("/api/table")
        self.assertEqual(resp.status_code, 200)

    def test_table_content_type_is_json(self):
        resp = self.client.get("/api/table")
        self.assertIn("application/json", resp.content_type)

    def test_table_has_pagination_fields(self):
        resp = self.client.get("/api/table")
        data = json.loads(resp.data)
        for key in ["total", "page", "per_page", "rows"]:
            self.assertIn(key, data, msg=f"Missing pagination field: '{key}'")

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_table_default_page_is_1(self):
        resp = self.client.get("/api/table")
        data = json.loads(resp.data)
        self.assertEqual(data["page"], 1)

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_table_rows_not_exceed_per_page(self):
        resp = self.client.get("/api/table?per_page=5")
        data = json.loads(resp.data)
        self.assertLessEqual(len(data["rows"]), 5)

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_table_row_has_required_fields(self):
        resp = self.client.get("/api/table?per_page=1")
        data = json.loads(resp.data)
        row = data["rows"][0]
        required = ["raw_id", "magnitude", "mag_category",
                    "depth_km", "depth_category", "event_time",
                    "distance_from_ref_km", "magnitude_scaled"]
        for field in required:
            self.assertIn(field, row, msg=f"Row missing field: '{field}'")

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_table_magnitude_filter_returns_only_matching_rows(self):
        resp = self.client.get("/api/table?mag_category=micro&per_page=50")
        data = json.loads(resp.data)
        for row in data["rows"]:
            self.assertEqual(row["mag_category"], "micro",
                             msg="Filter returned wrong mag_category")

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_table_depth_filter_works(self):
        resp = self.client.get("/api/table?depth_category=shallow&per_page=50")
        data = json.loads(resp.data)
        for row in data["rows"]:
            self.assertEqual(row["depth_category"], "shallow")

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_table_combined_filter(self):
        resp = self.client.get(
            "/api/table?mag_category=minor&depth_category=shallow&per_page=20"
        )
        data = json.loads(resp.data)
        for row in data["rows"]:
            self.assertEqual(row["mag_category"],   "minor")
            self.assertEqual(row["depth_category"], "shallow")

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_table_pagination_page_2(self):
        resp1 = self.client.get("/api/table?page=1&per_page=5")
        resp2 = self.client.get("/api/table?page=2&per_page=5")
        d1 = json.loads(resp1.data)
        d2 = json.loads(resp2.data)
        if len(d1["rows"]) > 0 and len(d2["rows"]) > 0:
            # Pages should return different rows
            ids1 = {r["raw_id"] for r in d1["rows"]}
            ids2 = {r["raw_id"] for r in d2["rows"]}
            self.assertEqual(ids1 & ids2, set(),
                             msg="Pages 1 and 2 returned overlapping rows")

    # ── /api/map ──────────────────────────────────────────────────────────

    def test_map_returns_200(self):
        resp = self.client.get("/api/map")
        self.assertEqual(resp.status_code, 200)

    def test_map_returns_list(self):
        resp = self.client.get("/api/map")
        data = json.loads(resp.data)
        self.assertIsInstance(data, list)

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_map_points_have_lat_lon(self):
        resp = self.client.get("/api/map")
        data = json.loads(resp.data)
        for point in data[:10]:
            self.assertIn("latitude",  point)
            self.assertIn("longitude", point)

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_map_lat_lon_in_valid_range(self):
        resp = self.client.get("/api/map")
        data = json.loads(resp.data)
        for point in data:
            lat = float(point["latitude"])
            lon = float(point["longitude"])
            self.assertGreaterEqual(lat, -90.0)
            self.assertLessEqual(lat,     90.0)
            self.assertGreaterEqual(lon, -180.0)
            self.assertLessEqual(lon,     180.0)

    @unittest.skipUnless(TABLE_HAS_DATA, "Table is empty")
    def test_map_max_1000_points(self):
        resp = self.client.get("/api/map")
        data = json.loads(resp.data)
        self.assertLessEqual(len(data), 1000)

    # ── / (HTML page) ────────────────────────────────────────────────────

    def test_index_returns_200(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_index_content_type_is_html(self):
        resp = self.client.get("/")
        self.assertIn("text/html", resp.content_type)

    def test_index_contains_dashboard_title(self):
        resp = self.client.get("/")
        # Title may vary — check for "Pipeline" which is always present
        self.assertTrue(
            b"Pipeline" in resp.data or b"Earthquake" in resp.data,
            msg="Dashboard HTML does not contain expected title keyword"
        )

    def test_index_loads_chartjs(self):
        resp = self.client.get("/")
        self.assertIn(b"chart.js", resp.data.lower())

    def test_index_loads_leaflet(self):
        resp = self.client.get("/")
        self.assertIn(b"leaflet", resp.data.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)