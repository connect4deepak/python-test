# =============================================================================
# test_transforms.py — Unit Tests for Stage 3: Transformations
# =============================================================================

import sys
import os
import unittest
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Scaler is saved in CWD of the running process (python-test/ when run from there,
# or the tests/ dir when run via run_tests.py). We patch the path to tests/
import transforms as _transforms_mod

SCALER_PATH = os.path.join(os.path.dirname(__file__), 'test_scaler.pkl')
_transforms_mod.SCALER_PATH = SCALER_PATH  # redirect scaler to a test-only file


def _remove_scaler():
    if os.path.exists(SCALER_PATH):
        os.remove(SCALER_PATH)


from transforms import scale_numeric, encode_categorical, select_final_columns, run_transforms, FINAL_COLUMNS


def make_df(n=20, **overrides):
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


# ── scale_numeric ─────────────────────────────────────────────────────────

class TestScaleNumeric(unittest.TestCase):
    def setUp(self):
        _remove_scaler()
        self.df  = make_df(n=50)
        self.out = scale_numeric(self.df)

    def test_scaled_columns_exist(self):
        for col in ["magnitude_scaled","depth_scaled","latitude_scaled","longitude_scaled"]:
            self.assertIn(col, self.out.columns)

    def test_scaled_values_between_0_and_1(self):
        for col in ["magnitude_scaled","depth_scaled","latitude_scaled","longitude_scaled"]:
            vals = self.out[col].dropna()
            self.assertGreaterEqual(vals.min(), 0.0 - 1e-4, msg=f"{col} below 0")
            self.assertLessEqual(vals.max(),    1.0 + 1e-4, msg=f"{col} above 1")

    def test_min_value_is_zero(self):
        # The min of the fitted dataset should map to ~0
        self.assertAlmostEqual(self.out["magnitude_scaled"].min(), 0.0, places=3)

    def test_max_value_is_one(self):
        self.assertAlmostEqual(self.out["magnitude_scaled"].max(), 1.0, places=3)

    def test_original_columns_preserved(self):
        for col in ["magnitude","depth_km","latitude","longitude"]:
            self.assertIn(col, self.out.columns)

    def test_scaler_pickle_created(self):
        self.assertTrue(os.path.exists(SCALER_PATH))

    def test_scaler_reused_on_second_call(self):
        out2 = scale_numeric(self.df)
        pd.testing.assert_frame_equal(
            self.out[["magnitude_scaled"]],
            out2[["magnitude_scaled"]],
        )

    def test_row_count_unchanged(self):
        self.assertEqual(len(self.out), len(self.df))

    def tearDown(self):
        _remove_scaler()


# ── encode_categorical ────────────────────────────────────────────────────

class TestEncodeCategorical(unittest.TestCase):
    def setUp(self):
        self.out = encode_categorical(make_df(n=20))

    def test_magtype_dummies_created(self):
        cols = [c for c in self.out.columns if c.startswith("magtype_")]
        self.assertGreater(len(cols), 0)

    def test_magtype_values_are_0_or_1(self):
        for col in [c for c in self.out.columns if c.startswith("magtype_")]:
            self.assertTrue(set(self.out[col].unique()).issubset({0, 1}))

    def test_mag_category_column_preserved(self):
        self.assertIn("mag_category", self.out.columns)

    def test_depth_category_column_preserved(self):
        self.assertIn("depth_category", self.out.columns)

    def test_row_count_unchanged(self):
        self.assertEqual(len(self.out), 20)

    def test_rare_magtypes_consolidated_to_other(self):
        rare_types = [f"mx{i}" for i in range(20)]
        out = encode_categorical(make_df(n=20, mag_type=rare_types))
        self.assertIn("magtype_other", out.columns)

    def test_mag_category_values_are_strings(self):
        for v in self.out["mag_category"].dropna().unique():
            self.assertIsInstance(v, str)


# ── select_final_columns ──────────────────────────────────────────────────

class TestSelectFinalColumns(unittest.TestCase):
    def setUp(self):
        df = make_df(n=5)
        df["junk_col_a"] = 99
        df["junk_col_b"] = "noise"
        self.out = select_final_columns(df)

    def test_junk_columns_dropped(self):
        self.assertNotIn("junk_col_a", self.out.columns)
        self.assertNotIn("junk_col_b", self.out.columns)

    def test_required_columns_present(self):
        for col in ["raw_id","magnitude","latitude","longitude","depth_km","event_time"]:
            self.assertIn(col, self.out.columns)

    def test_categorical_cols_are_string_dtype(self):
        # Accept both legacy 'object' dtype and newer pandas StringDtype
        for col in ["mag_category","depth_category"]:
            if col in self.out.columns:
                dtype = self.out[col].dtype
                self.assertTrue(
                    pd.api.types.is_string_dtype(dtype) or
                    pd.api.types.is_object_dtype(dtype),
                    msg=f"{col} dtype {dtype} is not string-like"
                )

    def test_row_count_unchanged(self):
        self.assertEqual(len(select_final_columns(make_df(n=8))), 8)


# ── run_transforms ────────────────────────────────────────────────────────

class TestRunTransforms(unittest.TestCase):
    def setUp(self):
        _remove_scaler()
        self.df  = make_df(n=30)
        self.out = run_transforms(self.df)

    def test_output_is_dataframe(self):
        self.assertIsInstance(self.out, pd.DataFrame)

    def test_scaled_columns_in_output(self):
        for col in ["magnitude_scaled","depth_scaled","latitude_scaled","longitude_scaled"]:
            self.assertIn(col, self.out.columns)

    def test_scaled_values_bounded(self):
        for col in ["magnitude_scaled","depth_scaled"]:
            vals = self.out[col].dropna()
            self.assertGreaterEqual(vals.min(), 0.0 - 1e-4)
            self.assertLessEqual(vals.max(),    1.0 + 1e-4)

    def test_categorical_columns_are_string_dtype(self):
        for col in ["mag_category","depth_category"]:
            if col in self.out.columns:
                dtype = self.out[col].dtype
                self.assertTrue(
                    pd.api.types.is_string_dtype(dtype) or
                    pd.api.types.is_object_dtype(dtype),
                    msg=f"{col} dtype {dtype} is not string-like for PostgreSQL"
                )

    def test_no_extra_columns(self):
        allowed = set(FINAL_COLUMNS) | {c for c in self.out.columns if c.startswith("magtype_")}
        extra   = set(self.out.columns) - allowed
        self.assertEqual(extra, set(), msg=f"Unexpected columns: {extra}")

    def test_row_count_unchanged(self):
        self.assertEqual(len(self.out), len(self.df))

    def tearDown(self):
        _remove_scaler()


if __name__ == "__main__":
    unittest.main(verbosity=2)