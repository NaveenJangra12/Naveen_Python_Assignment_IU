"""
test_pipeline.py
Unit tests for DLMDSPWP01 – Selection of Ideal Functions

Tests cover every class and critical method:
    - BaseFunction.deviation()
    - TrainingFunction.max_deviation()
    - IdealFunction.squared_error()
    - TestPoint.is_assignable()   (sqrt(2) rule)
    - DataManager.load_csv()      (valid + missing file)
    - DataManager.save_to_db()    (in-memory SQLite)

Run with:
    python -m pytest test_pipeline.py -v
    -- or --
    python test_pipeline.py
"""

import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

# Import all classes from the main module
from main import (
    BaseFunction,
    TrainingFunction,
    IdealFunction,
    TestPoint,
    DataManager,
    DataValidationError,
    MappingError,
)


# ─────────────────────────────────────────────────────────────────────────────
# BaseFunction
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseFunction(unittest.TestCase):
    """Tests for BaseFunction.deviation()"""

    def test_deviation_correct_values(self):
        """Absolute deviation should equal |self.y - other_y| element-wise."""
        bf     = BaseFunction([0, 1, 2], [3.0, 4.0, 5.0])
        result = bf.deviation([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, [2.0, 2.0, 2.0])

    def test_deviation_identical_arrays(self):
        """Deviation of two identical arrays must be all zeros."""
        bf = BaseFunction([0, 1], [5.0, 6.0])
        np.testing.assert_array_equal(bf.deviation([5.0, 6.0]), [0.0, 0.0])

    def test_deviation_negative_values(self):
        """Deviation must be non-negative even for negative y-values."""
        bf     = BaseFunction([0], [-3.0])
        result = bf.deviation([2.0])
        self.assertGreaterEqual(float(result[0]), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TrainingFunction
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingFunction(unittest.TestCase):
    """Tests for TrainingFunction.max_deviation()"""

    def test_max_deviation_returns_largest(self):
        """max_deviation must return the single largest absolute difference."""
        tf = TrainingFunction([0, 1, 2], [1.0, 2.0, 10.0])
        self.assertAlmostEqual(tf.max_deviation([1.0, 2.0, 3.0]), 7.0)

    def test_max_deviation_identical(self):
        """max_deviation must be 0 when training and ideal are identical."""
        tf = TrainingFunction([0, 1], [3.0, 4.0])
        self.assertAlmostEqual(tf.max_deviation([3.0, 4.0]), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# IdealFunction
# ─────────────────────────────────────────────────────────────────────────────

class TestIdealFunction(unittest.TestCase):
    """Tests for IdealFunction.squared_error()"""

    def test_squared_error_zero_for_perfect_match(self):
        """SSE must be 0 when ideal function perfectly matches training data."""
        idf = IdealFunction([0, 1, 2], [2.0, 4.0, 6.0])
        self.assertAlmostEqual(idf.squared_error([2.0, 4.0, 6.0]), 0.0)

    def test_squared_error_known_value(self):
        """SSE = (1-0)^2 + (2-1)^2 = 2.0 for a unit shift of two points."""
        idf = IdealFunction([0, 1], [0.0, 1.0])
        self.assertAlmostEqual(idf.squared_error([1.0, 2.0]), 2.0)

    def test_squared_error_always_non_negative(self):
        """SSE must never be negative."""
        idf = IdealFunction([0, 1, 2], [5.0, -3.0, 2.0])
        self.assertGreaterEqual(idf.squared_error([1.0, 1.0, 1.0]), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestPoint
# ─────────────────────────────────────────────────────────────────────────────

class TestTestPoint(unittest.TestCase):
    """Tests for TestPoint.is_assignable() – the sqrt(2) rule."""

    def test_assignable_when_delta_below_threshold(self):
        """
        delta = 1.0, max_dev = 1.0 -> threshold = sqrt(2) ~= 1.414
        1.0 <= 1.414, so the point should be assigned.
        """
        tp = TestPoint(0.0, 5.0)
        self.assertTrue(tp.is_assignable(4.0, 1.0))

    def test_not_assignable_when_delta_exceeds_threshold(self):
        """
        delta = 6.0, max_dev = 1.0 -> threshold = sqrt(2) ~= 1.414
        6.0 > 1.414, so the point must not be assigned.
        """
        tp = TestPoint(0.0, 10.0)
        self.assertFalse(tp.is_assignable(4.0, 1.0))

    def test_assignable_at_exact_boundary(self):
        """
        A delta exactly equal to max_dev * sqrt(2) should be accepted
        (boundary is inclusive per the assignment specification).
        """
        threshold = 1.0 * math.sqrt(2)
        tp = TestPoint(0.0, threshold)          # y_test = sqrt(2), ideal_y = 0
        self.assertTrue(tp.is_assignable(0.0, 1.0))

    def test_not_assignable_just_above_boundary(self):
        """A delta just above max_dev * sqrt(2) must be rejected."""
        threshold = 1.0 * math.sqrt(2) + 1e-9
        tp = TestPoint(0.0, threshold)
        self.assertFalse(tp.is_assignable(0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# DataManager
# ─────────────────────────────────────────────────────────────────────────────

class TestDataManager(unittest.TestCase):
    """Tests for DataManager.load_csv() and save_to_db()."""

    def setUp(self):
        """Create a temporary directory and a valid CSV for testing."""
        self.tmp_dir = tempfile.mkdtemp()
        self.valid_csv = os.path.join(self.tmp_dir, "valid.csv")
        pd.DataFrame({"x": [1.0, 2.0], "y1": [3.0, 4.0]}).to_csv(
            self.valid_csv, index=False
        )
        self.dm = DataManager(db_path=os.path.join(self.tmp_dir, "test.db"))

    def test_load_valid_csv_succeeds(self):
        """A valid CSV file must load without raising any exception."""
        df = self.dm.load_csv(self.valid_csv, name="test")
        self.assertFalse(df.empty)
        self.assertIn("x", df.columns)

    def test_load_missing_file_raises(self):
        """Loading a non-existent file must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.dm.load_csv(os.path.join(self.tmp_dir, "ghost.csv"))

    def test_load_empty_csv_raises(self):
        """Loading an empty CSV (headers only, no rows) must raise DataValidationError."""
        empty_path = os.path.join(self.tmp_dir, "empty.csv")
        pd.DataFrame({"x": [], "y": []}).to_csv(empty_path, index=False)
        with self.assertRaises(DataValidationError):
            self.dm.load_csv(empty_path, name="empty")

    def test_load_csv_with_nulls_raises(self):
        """A CSV containing null values must raise DataValidationError."""
        null_path = os.path.join(self.tmp_dir, "nulls.csv")
        pd.DataFrame({"x": [1.0, None], "y": [2.0, 3.0]}).to_csv(
            null_path, index=False
        )
        with self.assertRaises(DataValidationError):
            self.dm.load_csv(null_path, name="nulls")

    def test_save_and_read_roundtrip(self):
        """Data saved to the database must be retrievable and identical."""
        df_original = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        self.dm.save_to_db(df_original, "roundtrip")
        df_retrieved = self.dm.read_from_db("roundtrip")
        pd.testing.assert_frame_equal(
            df_original.reset_index(drop=True),
            df_retrieved.reset_index(drop=True),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
