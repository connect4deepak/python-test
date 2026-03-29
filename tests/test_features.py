# =============================================================================
# test_features.py — Unit Tests for Stage 2: Feature Engineering
# =============================================================================
"""
Tests cover every function in features.py:
  - add_time_features
  - add_magnitude_category
  - add_depth_category
  - add_distance_feature
  - run_feature_engineering (orchestrator)
"""

import sys
import os
import math
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python-test'))

from features import (
    add_time_features,
    add_magnitude_category,
    add_depth_category,
    add_distance_feature,
    run_feature_engineering,
    _haversine_km,
)


# ── Fixture ───────────────────────────────────────────────────────────────

def make_df(**overrides):
    base = {
        "raw_id":     [1],
        "magnitude":  [3.5],
        "latitude":   [37.5],
        "longitude":  [-118.2],
        "depth_km":   [10.0],
        "event_time": [pd.Timestamp("2025-03-15 14:30:00", tz="UTC")],
        "place":      ["Test location"],
        "mag_type":   ["ml"],
        "event_type": ["earthquake"],
        "status":     ["reviewed"],
        "is_outlier": [False],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ═════════════════════════════════════════════════════════════════════════
# 1. add_time_features
# ═════════════════════════════════════════════════════════════════════════

class TestAddTimeFeatures(unittest.TestCase):

    def setUp(self):
        # 2025-03-15 14:30 UTC = Saturday, month=3, hour=14
        self.df = make_df()
        self.out = add_time_features(self.df)

    def test_year_correct(self):
        self.assertEqual(self.out["year"].iloc[0], 2025)

    def test_month_correct(self):
        self.assertEqual(self.out["month"].iloc[0], 3)

    def test_hour_correct(self):
        self.assertEqual(self.out["hour"].iloc[0], 14)

    def test_day_of_week_saturday(self):
        # 2025-03-15 is a Saturday → dayofweek = 5
        self.assertEqual(self.out["day_of_week"].iloc[0], 5)

    def test_is_weekend_true_on_saturday(self):
        self.assertTrue(self.out["is_weekend"].iloc[0])

    def test_is_weekend_false_on_weekday(self):
        # 2025-03-17 is a Monday
        df = make_df(event_time=[pd.Timestamp("2025-03-17 08:00:00", tz="UTC")])
        out = add_time_features(df)
        self.assertFalse(out["is_weekend"].iloc[0])

    def test_hour_sin_cos_on_unit_circle(self):
        """sin² + cos² must equal 1 (Pythagorean identity)."""
        s = self.out["hour_sin"].iloc[0]
        c = self.out["hour_cos"].iloc[0]
        self.assertAlmostEqual(s**2 + c**2, 1.0, places=6)

    def test_month_sin_cos_on_unit_circle(self):
        s = self.out["month_sin"].iloc[0]
        c = self.out["month_cos"].iloc[0]
        self.assertAlmostEqual(s**2 + c**2, 1.0, places=6)

    def test_cyclic_hour_midnight_near_hour23(self):
        """
        Cyclic encoding: hour 0 and hour 23 should be close on the unit circle.
        cos(0) ≈ 1 and cos(23 * 2π/24) ≈ 0.966 — both near 1.
        """
        df0  = make_df(event_time=[pd.Timestamp("2025-01-01 00:00:00", tz="UTC")])
        df23 = make_df(event_time=[pd.Timestamp("2025-01-01 23:00:00", tz="UTC")])
        out0  = add_time_features(df0)
        out23 = add_time_features(df23)
        cos0  = out0["hour_cos"].iloc[0]
        cos23 = out23["hour_cos"].iloc[0]
        # Both should be close to 1.0 (top of the circle)
        self.assertGreater(cos0,  0.95)
        self.assertGreater(cos23, 0.95)

    def test_all_time_columns_present(self):
        expected = ["year", "month", "day_of_week", "hour",
                    "is_weekend", "hour_sin", "hour_cos",
                    "month_sin", "month_cos"]
        for col in expected:
            self.assertIn(col, self.out.columns, msg=f"Missing column: {col}")


# ═════════════════════════════════════════════════════════════════════════
# 2. add_magnitude_category
# ═════════════════════════════════════════════════════════════════════════

class TestAddMagnitudeCategory(unittest.TestCase):

    def _cat(self, mag):
        df = make_df(magnitude=[mag])
        return add_magnitude_category(df)["mag_category"].iloc[0]

    def test_micro(self):    self.assertEqual(self._cat(0.5),  "micro")
    def test_minor(self):    self.assertEqual(self._cat(2.5),  "minor")
    def test_light(self):    self.assertEqual(self._cat(4.5),  "light")
    def test_moderate(self): self.assertEqual(self._cat(5.5),  "moderate")
    def test_strong(self):   self.assertEqual(self._cat(6.5),  "strong")
    def test_major(self):    self.assertEqual(self._cat(7.5),  "major")
    def test_great(self):    self.assertEqual(self._cat(8.5),  "great")

    def test_boundary_exactly_2_is_minor(self):
        # bins are right=False so 2.0 → minor bucket [2, 4)
        self.assertEqual(self._cat(2.0), "minor")

    def test_boundary_exactly_8_is_great(self):
        self.assertEqual(self._cat(8.0), "great")

    def test_column_present(self):
        df = make_df()
        out = add_magnitude_category(df)
        self.assertIn("mag_category", out.columns)

    def test_all_valid_categories(self):
        valid = {"micro", "minor", "light", "moderate", "strong", "major", "great"}
        mags = [0.5, 2.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        df = pd.concat([make_df(magnitude=[m]) for m in mags], ignore_index=True)
        out = add_magnitude_category(df)
        self.assertTrue(set(out["mag_category"].unique()).issubset(valid))


# ═════════════════════════════════════════════════════════════════════════
# 3. add_depth_category
# ═════════════════════════════════════════════════════════════════════════

class TestAddDepthCategory(unittest.TestCase):

    def _cat(self, depth):
        df = make_df(depth_km=[depth])
        return add_depth_category(df)["depth_category"].iloc[0]

    def test_shallow(self):      self.assertEqual(self._cat(10.0),  "shallow")
    def test_intermediate(self): self.assertEqual(self._cat(150.0), "intermediate")
    def test_deep(self):         self.assertEqual(self._cat(500.0), "deep")

    def test_boundary_70km_is_intermediate(self):
        self.assertEqual(self._cat(70.0), "intermediate")

    def test_boundary_300km_is_deep(self):
        self.assertEqual(self._cat(300.0), "deep")

    def test_column_present(self):
        df = make_df()
        out = add_depth_category(df)
        self.assertIn("depth_category", out.columns)


# ═════════════════════════════════════════════════════════════════════════
# 4. Haversine distance
# ═════════════════════════════════════════════════════════════════════════

class TestHaversineAndDistanceFeature(unittest.TestCase):

    def test_same_point_is_zero(self):
        d = _haversine_km(51.5, -0.12, 51.5, -0.12)
        self.assertAlmostEqual(d, 0.0, places=3)

    def test_london_to_paris_approx(self):
        # London (51.5, -0.12) to Paris (48.85, 2.35) ≈ 340 km
        d = _haversine_km(51.5, -0.12, 48.85, 2.35)
        self.assertGreater(d, 300)
        self.assertLess(d, 380)

    def test_dublin_to_new_york_approx(self):
        # Dublin (53.35, -6.26) to New York (40.71, -74.01) ≈ 5100 km
        d = _haversine_km(53.35, -6.26, 40.71, -74.01)
        self.assertGreater(d, 4800)
        self.assertLess(d, 5400)

    def test_add_distance_feature_column_present(self):
        df = make_df()
        out = add_distance_feature(df)
        self.assertIn("distance_from_ref_km", out.columns)

    def test_add_distance_feature_is_positive(self):
        df = make_df(latitude=[35.0], longitude=[-120.0])
        out = add_distance_feature(df)
        self.assertGreater(out["distance_from_ref_km"].iloc[0], 0)

    def test_add_distance_feature_vectorised_multiple_rows(self):
        df = pd.concat([
            make_df(raw_id=[1], latitude=[35.0], longitude=[-120.0]),
            make_df(raw_id=[2], latitude=[35.7], longitude=[139.7]),
        ], ignore_index=True)
        out = add_distance_feature(df)
        self.assertEqual(len(out), 2)
        # Japan should be farther from Dublin than California
        self.assertGreater(
            out["distance_from_ref_km"].iloc[1],
            out["distance_from_ref_km"].iloc[0]
        )


# ═════════════════════════════════════════════════════════════════════════
# 5. run_feature_engineering (orchestrator)
# ═════════════════════════════════════════════════════════════════════════

class TestRunFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.df = make_df()
        self.out = run_feature_engineering(self.df)

    def test_all_feature_columns_present(self):
        expected = [
            "year", "month", "day_of_week", "hour", "is_weekend",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "mag_category", "depth_category", "distance_from_ref_km",
        ]
        for col in expected:
            self.assertIn(col, self.out.columns, msg=f"Missing: {col}")

    def test_row_count_unchanged(self):
        self.assertEqual(len(self.out), len(self.df))

    def test_original_columns_preserved(self):
        for col in ["magnitude", "latitude", "longitude", "depth_km", "event_time"]:
            self.assertIn(col, self.out.columns)

    def test_multiple_rows_processed(self):
        df = pd.concat([make_df(raw_id=[i]) for i in range(5)], ignore_index=True)
        out = run_feature_engineering(df)
        self.assertEqual(len(out), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
