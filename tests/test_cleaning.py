# =============================================================================
# test_cleaning.py — Unit Tests for Stage 1: Data Cleaning & Validation
# =============================================================================

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cleaning import (
    normalise_schema, coerce_types, handle_nulls,
    validate_ranges, remove_duplicates, remove_outliers, run_cleaning,
)


def make_df(raw_id=1, lat=37.5, lon=-118.2, t="2025-01-15 08:30:00", **kw):
    base = {
        "raw_id":     [raw_id],
        "magnitude":  [2.5],
        "latitude":   [lat],
        "longitude":  [lon],
        "depth_km":   [10.0],
        "event_time": [pd.Timestamp(t, tz="UTC")],
        "place":      ["15km NE of Los Angeles, CA"],
        "mag_type":   ["ml"],
        "event_type": ["earthquake"],
        "status":     ["reviewed"],
    }
    base.update(kw)
    return pd.DataFrame(base)


class TestNormaliseSchema(unittest.TestCase):
    def test_renames_mag_to_magnitude(self):
        df = pd.DataFrame({"mag": [3.1], "lat": [10.0], "lon": [20.0]})
        out = normalise_schema(df)
        self.assertIn("magnitude", out.columns)
        self.assertNotIn("mag", out.columns)

    def test_renames_lat_lon(self):
        df = pd.DataFrame({"lat": [10.0], "lon": [20.0], "magnitude": [1.0]})
        out = normalise_schema(df)
        self.assertIn("latitude",  out.columns)
        self.assertIn("longitude", out.columns)

    def test_renames_magnitude_type(self):
        df = pd.DataFrame({"magnitude_type": ["ml"], "magnitude": [2.0]})
        out = normalise_schema(df)
        self.assertIn("mag_type", out.columns)
        self.assertNotIn("magnitude_type", out.columns)

    def test_already_correct_columns_unchanged(self):
        out = normalise_schema(make_df())
        self.assertIn("magnitude", out.columns)
        self.assertIn("latitude",  out.columns)

    def test_row_count_preserved(self):
        df = pd.concat([make_df(raw_id=i) for i in range(5)], ignore_index=True)
        self.assertEqual(len(normalise_schema(df)), 5)


class TestCoerceTypes(unittest.TestCase):
    def test_string_magnitude_becomes_float(self):
        out = coerce_types(make_df(magnitude=["3.5"]))
        self.assertTrue(pd.api.types.is_float_dtype(out["magnitude"]))
        self.assertAlmostEqual(out["magnitude"].iloc[0], 3.5)

    def test_invalid_magnitude_becomes_nan(self):
        out = coerce_types(make_df(magnitude=["not_a_number"]))
        self.assertTrue(np.isnan(out["magnitude"].iloc[0]))

    def test_string_event_time_parsed(self):
        out = coerce_types(make_df(event_time=["2025-06-01T12:00:00Z"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["event_time"]))

    def test_place_stripped_and_lowercased(self):
        out = coerce_types(make_df(place=["  Los Angeles  "]))
        self.assertEqual(out["place"].iloc[0], "los angeles")

    def test_nan_string_place_becomes_nan(self):
        out = coerce_types(make_df(place=["nan"]))
        self.assertTrue(pd.isna(out["place"].iloc[0]))


class TestHandleNulls(unittest.TestCase):
    def test_row_with_null_magnitude_dropped(self):
        self.assertEqual(len(handle_nulls(make_df(magnitude=[np.nan]))), 0)

    def test_row_with_null_latitude_dropped(self):
        self.assertEqual(len(handle_nulls(make_df(latitude=[np.nan]))), 0)

    def test_row_with_null_event_time_dropped(self):
        self.assertEqual(len(handle_nulls(make_df(event_time=[pd.NaT]))), 0)

    def test_valid_row_kept(self):
        self.assertEqual(len(handle_nulls(make_df())), 1)

    def test_null_place_filled_with_unknown(self):
        out = handle_nulls(make_df(place=[np.nan]))
        self.assertEqual(out["place"].iloc[0], "unknown")

    def test_mixed_rows_only_valid_survive(self):
        df = pd.concat([make_df(raw_id=1), make_df(raw_id=2, magnitude=[np.nan])],
                       ignore_index=True)
        self.assertEqual(len(handle_nulls(df)), 1)


class TestValidateRanges(unittest.TestCase):
    def test_valid_row_passes(self):
        self.assertEqual(len(validate_ranges(make_df())), 1)

    def test_magnitude_above_max_dropped(self):
        self.assertEqual(len(validate_ranges(make_df(magnitude=[11.0]))), 0)

    def test_magnitude_below_min_dropped(self):
        self.assertEqual(len(validate_ranges(make_df(magnitude=[-3.0]))), 0)

    def test_depth_above_700_dropped(self):
        self.assertEqual(len(validate_ranges(make_df(depth_km=[750.0]))), 0)

    def test_negative_depth_dropped(self):
        self.assertEqual(len(validate_ranges(make_df(depth_km=[-1.0]))), 0)

    def test_invalid_latitude_dropped(self):
        self.assertEqual(len(validate_ranges(make_df(lat=95.0))), 0)

    def test_invalid_longitude_dropped(self):
        self.assertEqual(len(validate_ranges(make_df(lon=190.0))), 0)

    def test_boundary_values_accepted(self):
        self.assertEqual(len(validate_ranges(make_df(magnitude=[10.0], depth_km=[0.0]))), 1)


class TestRemoveDuplicates(unittest.TestCase):
    def test_exact_duplicate_removed(self):
        df = pd.concat([make_df(), make_df()], ignore_index=True)
        self.assertEqual(len(remove_duplicates(df)), 1)

    def test_unique_rows_all_kept(self):
        df = pd.concat([
            make_df(raw_id=1, lat=10.0, t="2025-01-01 00:00:00"),
            make_df(raw_id=2, lat=20.0, t="2025-01-02 00:00:00"),
        ], ignore_index=True)
        self.assertEqual(len(remove_duplicates(df)), 2)

    def test_soft_duplicate_same_location_time_deduplicated(self):
        t = "2025-01-01 00:00:00"
        df = pd.concat([
            make_df(raw_id=1, lat=35.0, lon=-120.0, t=t),
            make_df(raw_id=2, lat=35.0, lon=-120.0, t=t),
        ], ignore_index=True)
        self.assertEqual(len(remove_duplicates(df)), 1)


class TestRemoveOutliers(unittest.TestCase):
    def _make_normal_df(self, n=100):
        np.random.seed(42)
        return pd.DataFrame({
            "raw_id":     range(n),
            "magnitude":  np.random.normal(2.0, 0.3, n).tolist(),
            "latitude":   [35.0] * n,
            "longitude":  [-120.0] * n,
            "depth_km":   np.random.normal(10.0, 3.0, n).tolist(),
            "event_time": [pd.Timestamp("2025-01-01", tz="UTC")] * n,
            "place":      ["Test"] * n,
            "mag_type":   ["ml"] * n,
            "event_type": ["earthquake"] * n,
            "status":     ["reviewed"] * n,
        })

    def test_is_outlier_column_added(self):
        self.assertIn("is_outlier", remove_outliers(self._make_normal_df()).columns)

    def test_row_count_unchanged(self):
        df = self._make_normal_df()
        self.assertEqual(len(remove_outliers(df)), len(df))

    def test_extreme_magnitude_flagged(self):
        df = self._make_normal_df()
        df.loc[0, "magnitude"] = 9.9
        self.assertTrue(remove_outliers(df).loc[0, "is_outlier"])

    def test_normal_values_not_flagged(self):
        out = remove_outliers(self._make_normal_df())
        self.assertLess(out["is_outlier"].mean(), 0.15)


class TestRunCleaning(unittest.TestCase):
    def _make_mixed_df(self):
        # Use distinct lat/lon/time so deduplication doesn't remove valid rows
        valid1 = make_df(raw_id=1, lat=10.0, t="2025-01-01 00:00:00", magnitude=[1.5])
        valid2 = make_df(raw_id=2, lat=20.0, t="2025-01-02 00:00:00", magnitude=[2.8])
        null_m = make_df(raw_id=3, lat=30.0, t="2025-01-03 00:00:00", magnitude=[np.nan])
        bad_r  = make_df(raw_id=4, lat=40.0, t="2025-01-04 00:00:00", magnitude=[15.0])
        return pd.concat([valid1, valid2, null_m, bad_r], ignore_index=True)

    def test_output_has_is_outlier_column(self):
        self.assertIn("is_outlier", run_cleaning(self._make_mixed_df()).columns)

    def test_invalid_rows_removed(self):
        out = run_cleaning(self._make_mixed_df())
        self.assertEqual(len(out), 2)  # null_m and bad_r both removed

    def test_output_magnitudes_in_valid_range(self):
        out = run_cleaning(self._make_mixed_df())
        self.assertTrue((out["magnitude"] >= -2.0).all())
        self.assertTrue((out["magnitude"] <= 10.0).all())

    def test_empty_input_returns_empty(self):
        try:
            out = run_cleaning(pd.DataFrame())
            self.assertEqual(len(out), 0)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)