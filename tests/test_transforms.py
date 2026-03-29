# =============================================================================
# test_transforms.py — Unit Tests for Stage 3: Transformations
# =============================================================================
"""
Tests cover:
  - scale_numeric          (Min-Max scaling)
  - encode_categorical      (one-hot + ordered categorical)
  - select_final_columns    (column selection & ordering)
  - run_transforms          (orchestrator)
"""

import sys
import os
import unittest
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python-test'))

# Remove any existing scaler so tests always fit a fresh one
SCALER_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'python-test', 'scaler.pkl'
)


def _remove_scaler():
    if os.path.exists(SCALER_PATH):
        os.remove(SCALER_PATH)


from transforms import (
    scale_numeric,
    encode_categorical,
    select_final_columns,
    run_transforms,
    FINAL_COLUMNS,
)


# ── Fixture ───────────────────────────────────────────────────────────────

def make_df(n=10, **overrides):
    """
    Return a DataFrame with n rows of valid feature-engineered data,
    simulating the output of Stage 2 (features.py).
    """
    np.random.seed(0)
    base = {
        "raw_id":              list(range(1, n + 1)),
        "magnitude":           np.random.uniform(0.5, 5.0, n).tolist(),
        "latitude":            np.random.uniform(-60, 60, n).tolist(),
        "longitude":           np.random.uniform(-180, 180, n).tolist(),
        "depth_km":            np.random.uniform(0, 100, n).tolist(),
        "event_time":          [pd.Timestamp("2025-01-01", tz="UTC")] * n,
        "place":               ["Test"] * n,
        "mag_type":            (["ml"] * (n // 2)) + (["md"] * (n - n // 2)),
        "event_type":          ["earthquake"] * n,
        "status":              ["reviewed"] * n,
        "year":                [2025] * n,
        "month":               [3] * n,
        "day_of_week":         [1] * n,
        "hour":                [12] * n,
        "is_weekend":          [False] * n,
        "hour_sin":            [0.0] * n,
        "hour_cos":            [1.0] * n,
        "month_sin":           [0.5] * n,
        "month_cos":           [0.866] * n,
        "mag_category":        ["minor"] * n,
        "depth_category":      ["shallow"] * n,
        "distance_from_ref_km": np.random.uniform(1000, 15000, n).tolist(),
        "is_outlier":          [False] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ═════════════════════════════════════════════════════════════════════════
# 1. scale_numeric
# ═════════════════════════════════════════════════════════════════════════

class TestScaleNumeric(unittest.TestCase):

    def setUp(self):
        _remove_scaler()
        self.df = make_df(n=50)
        self.out = scale_numeric(self.df)

    def test_scaled_columns_exist(self):
        for col in ["magnitude_scaled", "depth_scaled",
                    "latitude_scaled", "longitude_scaled"]:
            self.assertIn(col, self.out.columns, msg=f"Missing: {col}")

    def test_scaled_values_between_0_and_1(self):
        for col in ["magnitude_scaled", "depth_scaled",
                    "latitude_scaled", "longitude_scaled"]:
            vals = self.out[col].dropna()
            self.assertGreaterEqual(vals.min(), 0.0 - 1e-6,
                                    msg=f"{col} below 0")
            self.assertLessEqual(vals.max(), 1.0 + 1e-6,
                                 msg=f"{col} above 1")

    def test_min_value_is_zero(self):
        """The minimum of the fitted range should map to 0."""
        self.assertAlmostEqual(self.out["magnitude_scaled"].min(), 0.0, places=4)

    def test_max_value_is_one(self):
        """The maximum of the fitted range should map to 1."""
        self.assertAlmostEqual(self.out["magnitude_scaled"].max(), 1.0, places=4)

    def test_original_columns_preserved(self):
        """Raw columns should still be present after scaling."""
        for col in ["magnitude", "depth_km", "latitude", "longitude"]:
            self.assertIn(col, self.out.columns)

    def test_scaler_pickle_created(self):
        self.assertTrue(os.path.exists(SCALER_PATH))

    def test_scaler_reused_on_second_call(self):
        """Second call should load the same scaler, not refit."""
        out2 = scale_numeric(self.df)
        pd.testing.assert_frame_equal(
            self.out[["magnitude_scaled"]],
            out2[["magnitude_scaled"]],
        )

    def test_row_count_unchanged(self):
        self.assertEqual(len(self.out), len(self.df))


# ═════════════════════════════════════════════════════════════════════════
# 2. encode_categorical
# ═════════════════════════════════════════════════════════════════════════

class TestEncodeCategorical(unittest.TestCase):

    def setUp(self):
        self.df = make_df(n=20)
        self.out = encode_categorical(self.df)

    def test_magtype_dummies_created(self):
        """At least one magtype_* column should be present."""
        magtype_cols = [c for c in self.out.columns if c.startswith("magtype_")]
        self.assertGreater(len(magtype_cols), 0)

    def test_magtype_values_are_0_or_1(self):
        magtype_cols = [c for c in self.out.columns if c.startswith("magtype_")]
        for col in magtype_cols:
            unique_vals = set(self.out[col].unique())
            self.assertTrue(unique_vals.issubset({0, 1}),
                            msg=f"{col} has non-binary values: {unique_vals}")

    def test_mag_category_column_preserved(self):
        self.assertIn("mag_category", self.out.columns)

    def test_depth_category_column_preserved(self):
        self.assertIn("depth_category", self.out.columns)

    def test_row_count_unchanged(self):
        self.assertEqual(len(self.out), len(self.df))

    def test_rare_magtypes_consolidated_to_other(self):
        """Mag types outside the top-N should appear as magtype_other."""
        # Give each row a unique mag type to force 'other' consolidation
        rare_types = [f"mx{i}" for i in range(20)]
        df = make_df(n=20, mag_type=rare_types)
        out = encode_categorical(df)
        self.assertIn("magtype_other", out.columns)

    def test_mag_category_values_are_strings(self):
        vals = self.out["mag_category"].dropna().unique()
        for v in vals:
            self.assertIsInstance(v, str)


# ═════════════════════════════════════════════════════════════════════════
# 3. select_final_columns
# ═════════════════════════════════════════════════════════════════════════

class TestSelectFinalColumns(unittest.TestCase):

    def setUp(self):
        df = make_df(n=5)
        # Add some extra junk columns that should be dropped
        df["junk_col_a"] = 99
        df["junk_col_b"] = "noise"
        self.out = select_final_columns(df)

    def test_junk_columns_dropped(self):
        self.assertNotIn("junk_col_a", self.out.columns)
        self.assertNotIn("junk_col_b", self.out.columns)

    def test_required_columns_present(self):
        required = ["raw_id", "magnitude", "latitude", "longitude",
                    "depth_km", "event_time"]
        for col in required:
            self.assertIn(col, self.out.columns)

    def test_categorical_cols_are_strings(self):
        for col in ["mag_category", "depth_category"]:
            if col in self.out.columns:
                self.assertEqual(self.out[col].dtype, object)

    def test_row_count_unchanged(self):
        df = make_df(n=8)
        out = select_final_columns(df)
        self.assertEqual(len(out), 8)


# ═════════════════════════════════════════════════════════════════════════
# 4. run_transforms (orchestrator)
# ═════════════════════════════════════════════════════════════════════════

class TestRunTransforms(unittest.TestCase):

    def setUp(self):
        _remove_scaler()
        self.df = make_df(n=30)
        self.out = run_transforms(self.df)

    def test_output_is_dataframe(self):
        self.assertIsInstance(self.out, pd.DataFrame)

    def test_scaled_columns_in_output(self):
        for col in ["magnitude_scaled", "depth_scaled",
                    "latitude_scaled", "longitude_scaled"]:
            self.assertIn(col, self.out.columns)

    def test_scaled_values_bounded(self):
        for col in ["magnitude_scaled", "depth_scaled"]:
            vals = self.out[col].dropna()
            self.assertGreaterEqual(vals.min(), 0.0 - 1e-6)
            self.assertLessEqual(vals.max(),    1.0 + 1e-6)

    def test_categorical_columns_are_strings(self):
        for col in ["mag_category", "depth_category"]:
            if col in self.out.columns:
                self.assertEqual(self.out[col].dtype, object,
                                 msg=f"{col} should be str/object for PostgreSQL")

    def test_no_extra_columns(self):
        """Output should only contain whitelisted + magtype_* columns."""
        allowed = set(FINAL_COLUMNS) | {c for c in self.out.columns
                                         if c.startswith("magtype_")}
        extra = set(self.out.columns) - allowed
        self.assertEqual(extra, set(), msg=f"Unexpected columns: {extra}")

    def test_row_count_unchanged(self):
        self.assertEqual(len(self.out), len(self.df))

    def tearDown(self):
        _remove_scaler()


if __name__ == "__main__":
    unittest.main(verbosity=2)
