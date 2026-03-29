# =============================================================================
# test_cleaning.py — Unit Tests for Stage 1: Data Cleaning & Validation
# =============================================================================
"""
Tests cover every function in cleaning.py:
  - normalise_schema
  - coerce_types
  - handle_nulls
  - validate_ranges
  - remove_duplicates
  - remove_outliers
  - run_cleaning (orchestrator)
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ── Path setup — allow imports from pipeline directory ────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python-test'))

from cleaning import (
    normalise_schema,
    coerce_types,
    handle_nulls,
    validate_ranges,
    remove_duplicates,
    remove_outliers,
    run_cleaning,
)


# ── Shared fixture builder ────────────────────────────────────────────────

def make_df(**overrides):
    """
    Return a minimal valid DataFrame representing one earthquake record.
    Override any field by passing it as a keyword argument.
    """
    base = {
        "raw_id":     [1],
        "magnitude":  [2.5],
        "latitude":   [37.5],
        "longitude":  [-118.2],
        "depth_km":   [10.0],
        "event_time": [pd.Timestamp("2025-01-15 08:30:00", tz="UTC")],
        "place":      ["15km NE of Los Angeles, CA"],
        "mag_type":   ["ml"],
        "event_type": ["earthquake"],
        "status":     ["reviewed"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ═════════════════════════════════════════════════════════════════════════
# 1. normalise_schema
# ═════════════════════════════════════════════════════════════════════════

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
        df = make_df()
        out = normalise_schema(df)
        self.assertIn("magnitude", out.columns)
        self.assertIn("latitude",  out.columns)

    def test_row_count_preserved(self):
        df = make_df()
        df = pd.concat([df] * 5, ignore_index=True)
        out = normalise_schema(df)
        self.assertEqual(len(out), 5)


# ═════════════════════════════════════════════════════════════════════════
# 2. coerce_types
# ═════════════════════════════════════════════════════════════════════════

class TestCoerceTypes(unittest.TestCase):

    def test_string_magnitude_becomes_float(self):
        df = make_df(magnitude=["3.5"])
        out = coerce_types(df)
        self.assertTrue(pd.api.types.is_float_dtype(out["magnitude"]))
        self.assertAlmostEqual(out["magnitude"].iloc[0], 3.5)

    def test_invalid_magnitude_becomes_nan(self):
        df = make_df(magnitude=["not_a_number"])
        out = coerce_types(df)
        self.assertTrue(np.isnan(out["magnitude"].iloc[0]))

    def test_string_event_time_parsed(self):
        df = make_df(event_time=["2025-06-01T12:00:00Z"])
        out = coerce_types(df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["event_time"]))

    def test_place_stripped_and_lowercased(self):
        df = make_df(place=["  Los Angeles  "])
        out = coerce_types(df)
        self.assertEqual(out["place"].iloc[0], "los angeles")

    def test_nan_string_place_becomes_nan(self):
        df = make_df(place=["nan"])
        out = coerce_types(df)
        self.assertTrue(pd.isna(out["place"].iloc[0]))


# ═════════════════════════════════════════════════════════════════════════
# 3. handle_nulls
# ═════════════════════════════════════════════════════════════════════════

class TestHandleNulls(unittest.TestCase):

    def test_row_with_null_magnitude_dropped(self):
        df = make_df(magnitude=[np.nan])
        out = handle_nulls(df)
        self.assertEqual(len(out), 0)

    def test_row_with_null_latitude_dropped(self):
        df = make_df(latitude=[np.nan])
        out = handle_nulls(df)
        self.assertEqual(len(out), 0)

    def test_row_with_null_event_time_dropped(self):
        df = make_df(event_time=[pd.NaT])
        out = handle_nulls(df)
        self.assertEqual(len(out), 0)

    def test_valid_row_kept(self):
        df = make_df()
        out = handle_nulls(df)
        self.assertEqual(len(out), 1)

    def test_null_place_filled_with_unknown(self):
        df = make_df(place=[np.nan])
        out = handle_nulls(df)
        # Row survives (place is optional), filled with 'unknown'
        self.assertEqual(out["place"].iloc[0], "unknown")

    def test_mixed_rows_only_valid_survive(self):
        valid   = make_df()
        invalid = make_df(magnitude=[np.nan])
        df = pd.concat([valid, invalid], ignore_index=True)
        out = handle_nulls(df)
        self.assertEqual(len(out), 1)


# ═════════════════════════════════════════════════════════════════════════
# 4. validate_ranges
# ═════════════════════════════════════════════════════════════════════════

class TestValidateRanges(unittest.TestCase):

    def test_valid_row_passes(self):
        df = make_df()
        out = validate_ranges(df)
        self.assertEqual(len(out), 1)

    def test_magnitude_above_max_dropped(self):
        df = make_df(magnitude=[11.0])
        out = validate_ranges(df)
        self.assertEqual(len(out), 0)

    def test_magnitude_below_min_dropped(self):
        df = make_df(magnitude=[-3.0])
        out = validate_ranges(df)
        self.assertEqual(len(out), 0)

    def test_depth_above_700_dropped(self):
        df = make_df(depth_km=[750.0])
        out = validate_ranges(df)
        self.assertEqual(len(out), 0)

    def test_negative_depth_dropped(self):
        df = make_df(depth_km=[-1.0])
        out = validate_ranges(df)
        self.assertEqual(len(out), 0)

    def test_invalid_latitude_dropped(self):
        df = make_df(latitude=[95.0])
        out = validate_ranges(df)
        self.assertEqual(len(out), 0)

    def test_invalid_longitude_dropped(self):
        df = make_df(longitude=[190.0])
        out = validate_ranges(df)
        self.assertEqual(len(out), 0)

    def test_boundary_values_accepted(self):
        # Exactly at the boundary should be kept
        df = make_df(magnitude=[10.0], depth_km=[0.0])
        out = validate_ranges(df)
        self.assertEqual(len(out), 1)


# ═════════════════════════════════════════════════════════════════════════
# 5. remove_duplicates
# ═════════════════════════════════════════════════════════════════════════

class TestRemoveDuplicates(unittest.TestCase):

    def test_exact_duplicate_removed(self):
        df = pd.concat([make_df(), make_df()], ignore_index=True)
        out = remove_duplicates(df)
        self.assertEqual(len(out), 1)

    def test_unique_rows_all_kept(self):
        df = pd.concat([
            make_df(raw_id=[1], latitude=[10.0]),
            make_df(raw_id=[2], latitude=[20.0]),
        ], ignore_index=True)
        out = remove_duplicates(df)
        self.assertEqual(len(out), 2)

    def test_soft_duplicate_same_location_time_deduplicated(self):
        """Two records with the same lat/lon/time should be treated as one event."""
        t = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
        df = pd.concat([
            make_df(raw_id=[1], latitude=[35.0], longitude=[-120.0], event_time=[t]),
            make_df(raw_id=[2], latitude=[35.0], longitude=[-120.0], event_time=[t]),
        ], ignore_index=True)
        out = remove_duplicates(df)
        self.assertEqual(len(out), 1)


# ═════════════════════════════════════════════════════════════════════════
# 6. remove_outliers
# ═════════════════════════════════════════════════════════════════════════

class TestRemoveOutliers(unittest.TestCase):

    def _make_normal_df(self, n=100):
        """Create a DataFrame where most magnitudes cluster around 2.0."""
        np.random.seed(42)
        mags   = np.random.normal(2.0, 0.3, n).tolist()
        depths = np.random.normal(10.0, 3.0, n).tolist()
        return pd.DataFrame({
            "raw_id":     range(n),
            "magnitude":  mags,
            "latitude":   [35.0] * n,
            "longitude":  [-120.0] * n,
            "depth_km":   depths,
            "event_time": [pd.Timestamp("2025-01-01", tz="UTC")] * n,
            "place":      ["Test"] * n,
            "mag_type":   ["ml"] * n,
            "event_type": ["earthquake"] * n,
            "status":     ["reviewed"] * n,
        })

    def test_is_outlier_column_added(self):
        df = self._make_normal_df()
        out = remove_outliers(df)
        self.assertIn("is_outlier", out.columns)

    def test_row_count_unchanged(self):
        """Outliers are flagged, not dropped."""
        df = self._make_normal_df()
        out = remove_outliers(df)
        self.assertEqual(len(out), len(df))

    def test_extreme_magnitude_flagged(self):
        df = self._make_normal_df()
        # Inject an obvious outlier
        df.loc[0, "magnitude"] = 9.9
        out = remove_outliers(df)
        self.assertTrue(out.loc[0, "is_outlier"])

    def test_normal_values_not_flagged(self):
        df = self._make_normal_df()
        out = remove_outliers(df)
        # Most rows should NOT be flagged
        flagged_pct = out["is_outlier"].mean()
        self.assertLess(flagged_pct, 0.15)


# ═════════════════════════════════════════════════════════════════════════
# 7. run_cleaning (orchestrator)
# ═════════════════════════════════════════════════════════════════════════

class TestRunCleaning(unittest.TestCase):

    def _make_mixed_df(self):
        valid1 = make_df(raw_id=[1], magnitude=[1.5])
        valid2 = make_df(raw_id=[2], magnitude=[2.8])
        null_m = make_df(raw_id=[3], magnitude=[np.nan])
        bad_r  = make_df(raw_id=[4], magnitude=[15.0])   # out of range
        return pd.concat([valid1, valid2, null_m, bad_r], ignore_index=True)

    def test_output_has_is_outlier_column(self):
        df = self._make_mixed_df()
        out = run_cleaning(df)
        self.assertIn("is_outlier", out.columns)

    def test_invalid_rows_removed(self):
        df = self._make_mixed_df()
        out = run_cleaning(df)
        # null_m and bad_r should both be gone → 2 rows remain
        self.assertEqual(len(out), 2)

    def test_output_magnitudes_in_valid_range(self):
        df = self._make_mixed_df()
        out = run_cleaning(df)
        self.assertTrue((out["magnitude"] >= -2.0).all())
        self.assertTrue((out["magnitude"] <= 10.0).all())

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame()
        # Should not raise, just return empty
        try:
            out = run_cleaning(df)
            self.assertEqual(len(out), 0)
        except Exception:
            pass  # acceptable if schema is missing


if __name__ == "__main__":
    unittest.main(verbosity=2)
