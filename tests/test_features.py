# =============================================================================
# test_features.py — Unit Tests for Stage 2: Feature Engineering
# =============================================================================

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python-test'))

from features import (
    add_time_features, add_magnitude_category,
    add_depth_category, add_distance_feature,
    run_feature_engineering, _haversine_km,
)


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


# ── Time Features ─────────────────────────────────────────────────────────

class TestAddTimeFeatures(unittest.TestCase):
    def setUp(self):
        self.out = add_time_features(make_df())

    def test_year_correct(self):   self.assertEqual(self.out["year"].iloc[0],  2025)
    def test_month_correct(self):  self.assertEqual(self.out["month"].iloc[0], 3)
    def test_hour_correct(self):   self.assertEqual(self.out["hour"].iloc[0],  14)

    def test_day_of_week_saturday(self):
        self.assertEqual(self.out["day_of_week"].iloc[0], 5)

    def test_is_weekend_true_on_saturday(self):
        self.assertTrue(self.out["is_weekend"].iloc[0])

    def test_is_weekend_false_on_weekday(self):
        out = add_time_features(make_df(event_time=[pd.Timestamp("2025-03-17 08:00:00", tz="UTC")]))
        self.assertFalse(out["is_weekend"].iloc[0])

    def test_hour_sin_cos_on_unit_circle(self):
        s, c = self.out["hour_sin"].iloc[0], self.out["hour_cos"].iloc[0]
        self.assertAlmostEqual(s**2 + c**2, 1.0, places=6)

    def test_month_sin_cos_on_unit_circle(self):
        s, c = self.out["month_sin"].iloc[0], self.out["month_cos"].iloc[0]
        self.assertAlmostEqual(s**2 + c**2, 1.0, places=6)

    def test_cyclic_hour_midnight_near_hour23(self):
        df0  = make_df(event_time=[pd.Timestamp("2025-01-01 00:00:00", tz="UTC")])
        df23 = make_df(event_time=[pd.Timestamp("2025-01-01 23:00:00", tz="UTC")])
        cos0  = add_time_features(df0)["hour_cos"].iloc[0]
        cos23 = add_time_features(df23)["hour_cos"].iloc[0]
        self.assertGreater(cos0,  0.95)
        self.assertGreater(cos23, 0.95)

    def test_all_time_columns_present(self):
        for col in ["year","month","day_of_week","hour","is_weekend",
                    "hour_sin","hour_cos","month_sin","month_cos"]:
            self.assertIn(col, self.out.columns)


# ── Magnitude Category ────────────────────────────────────────────────────

class TestAddMagnitudeCategory(unittest.TestCase):
    def _cat(self, mag):
        return add_magnitude_category(make_df(magnitude=[mag]))["mag_category"].iloc[0]

    def test_micro(self):    self.assertEqual(self._cat(0.5),  "micro")
    def test_minor(self):    self.assertEqual(self._cat(2.5),  "minor")
    def test_light(self):    self.assertEqual(self._cat(4.5),  "light")
    def test_moderate(self): self.assertEqual(self._cat(5.5),  "moderate")
    def test_strong(self):   self.assertEqual(self._cat(6.5),  "strong")
    def test_major(self):    self.assertEqual(self._cat(7.5),  "major")
    def test_great(self):    self.assertEqual(self._cat(8.5),  "great")

    def test_boundary_exactly_2_is_minor(self):
        self.assertEqual(self._cat(2.0), "minor")

    def test_boundary_exactly_8_is_great(self):
        self.assertEqual(self._cat(8.0), "great")

    def test_column_present(self):
        self.assertIn("mag_category", add_magnitude_category(make_df()).columns)

    def test_all_valid_categories(self):
        valid = {"micro","minor","light","moderate","strong","major","great"}
        df = pd.concat([make_df(magnitude=[m]) for m in [0.5,2.5,4.5,5.5,6.5,7.5,8.5]],
                       ignore_index=True)
        self.assertTrue(set(add_magnitude_category(df)["mag_category"].unique()).issubset(valid))


# ── Depth Category ────────────────────────────────────────────────────────
#
# pd.cut bins=[0, 70, 300, 700], include_lowest=True
#   [0, 70]   → shallow      (includes 0 and 70)
#   (70, 300] → intermediate (includes 300, excludes 70)
#   (300,700] → deep         (includes 700, excludes 300)

class TestAddDepthCategory(unittest.TestCase):
    def _cat(self, depth):
        return add_depth_category(make_df(depth_km=[depth]))["depth_category"].iloc[0]

    def test_shallow_10km(self):     self.assertEqual(self._cat(10.0),  "shallow")
    def test_intermediate_150km(self): self.assertEqual(self._cat(150.0), "intermediate")
    def test_deep_500km(self):       self.assertEqual(self._cat(500.0), "deep")

    def test_boundary_70km_is_shallow(self):
        # 70 is the right edge of the first bin [0,70] → shallow
        self.assertEqual(self._cat(70.0), "shallow")

    def test_boundary_71km_is_intermediate(self):
        # Just above 70 falls in (70,300] → intermediate
        self.assertEqual(self._cat(71.0), "intermediate")

    def test_boundary_300km_is_intermediate(self):
        # 300 is the right edge of (70,300] → intermediate
        self.assertEqual(self._cat(300.0), "intermediate")

    def test_boundary_301km_is_deep(self):
        # Just above 300 falls in (300,700] → deep
        self.assertEqual(self._cat(301.0), "deep")

    def test_column_present(self):
        self.assertIn("depth_category", add_depth_category(make_df()).columns)


# ── Haversine & Distance ──────────────────────────────────────────────────

class TestHaversineAndDistanceFeature(unittest.TestCase):
    def test_same_point_is_zero(self):
        self.assertAlmostEqual(_haversine_km(51.5,-0.12, 51.5,-0.12), 0.0, places=3)

    def test_london_to_paris_approx(self):
        d = _haversine_km(51.5,-0.12, 48.85, 2.35)
        self.assertGreater(d, 300); self.assertLess(d, 380)

    def test_dublin_to_new_york_approx(self):
        d = _haversine_km(53.35,-6.26, 40.71,-74.01)
        self.assertGreater(d, 4800); self.assertLess(d, 5400)

    def test_add_distance_feature_column_present(self):
        self.assertIn("distance_from_ref_km", add_distance_feature(make_df()).columns)

    def test_add_distance_feature_is_positive(self):
        out = add_distance_feature(make_df(latitude=[35.0], longitude=[-120.0]))
        self.assertGreater(out["distance_from_ref_km"].iloc[0], 0)

    def test_add_distance_feature_vectorised_multiple_rows(self):
        df = pd.concat([
            make_df(raw_id=[1], latitude=[35.0], longitude=[-120.0]),
            make_df(raw_id=[2], latitude=[35.7], longitude=[139.7]),
        ], ignore_index=True)
        out = add_distance_feature(df)
        self.assertEqual(len(out), 2)
        # Japan farther from Dublin than California
        self.assertGreater(out["distance_from_ref_km"].iloc[1],
                           out["distance_from_ref_km"].iloc[0])


# ── Orchestrator ──────────────────────────────────────────────────────────

class TestRunFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.out = run_feature_engineering(make_df())

    def test_all_feature_columns_present(self):
        for col in ["year","month","day_of_week","hour","is_weekend",
                    "hour_sin","hour_cos","month_sin","month_cos",
                    "mag_category","depth_category","distance_from_ref_km"]:
            self.assertIn(col, self.out.columns)

    def test_row_count_unchanged(self):
        self.assertEqual(len(self.out), 1)

    def test_original_columns_preserved(self):
        for col in ["magnitude","latitude","longitude","depth_km","event_time"]:
            self.assertIn(col, self.out.columns)

    def test_multiple_rows_processed(self):
        df = pd.concat([make_df(raw_id=[i]) for i in range(5)], ignore_index=True)
        self.assertEqual(len(run_feature_engineering(df)), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)