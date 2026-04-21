# DLMDSPWP01 – Selection of Ideal Functions and Test Data Mapping

**Author:** Naveen Kumar | **Matric No.:** 10244074  
**Module:** Programming with Python – DLMDSPWP01  
**Institution:** IU International University of Applied Sciences, Berlin

---

## Overview

A Python-based pipeline that:

1. Loads three CSV datasets (training data, 50 ideal functions, test data)
2. Persists all raw data to a **SQLite** database via **SQLAlchemy**
3. Selects the 4 best-fitting ideal functions using the **Least Squares Method**
4. Maps test data points to the selected functions using the **√2 deviation rule**
5. Saves all results back to the database
6. Generates an interactive **Bokeh** HTML plot and three static **Matplotlib** PNG plots

---

## Repository Structure

```
DLMDSPWP01/
│
├── data/
│   ├── train.csv          # 4 training functions (x, y1–y4) with Gaussian noise
│   ├── ideal.csv          # 50 candidate ideal functions (x, y1–y50)
│   └── test.csv           # Test data points (x, y)
│
├── main.py                # Full pipeline + all classes (BaseFunction, etc.)
├── test_pipeline.py       # 12 unit tests (unittest + pytest compatible)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
python main.py
```

Expected output:
```
=== DLMDSPWP01 Pipeline Starting ===

[Step 1] Loading CSV datasets...
  Loaded Training Data: 400 rows x 5 columns
  Loaded Ideal Functions: 400 rows x 51 columns
  Loaded Test Data: 100 rows x 2 columns

[Step 2] Persisting datasets to SQLite...
  Saved 'training_data' (400 rows) to functions.db
  Saved 'ideal_functions' (400 rows) to functions.db

[Step 3] Selecting ideal functions via least squares...
  Selection Results:
   Training Function Best Ideal Function    SSE (min)  Delta_max
                  y1                 y11      367.1879     3.7912
                  y2                 y22      916.2043     4.5901
                  y3                 y33       98.2049     1.4928
                  y4                 y40      260.7610     2.3941

[Step 4] Mapping test data using sqrt(2) threshold...
  Assigned: 100 / 100 test points

[Step 5] Saving mapping results to database...
  Saved 'test_mapping' (100 rows) to functions.db

[Step 6] Generating visualisations...
  Saved outputs/plot_training_vs_ideal.png
  Saved outputs/plot_sse_maxdev.png
  Saved outputs/plot_mapping_summary.png
  Saved outputs/bokeh_plot.html

=== Pipeline Complete ===
  Database : functions.db
  Plots    : outputs/
```

---

## Running Unit Tests

```bash
# Using pytest (recommended)
pytest test_pipeline.py -v

# Using unittest directly
python test_pipeline.py
```

All 12 tests should pass:

| Test Class | Method | What It Verifies |
|---|---|---|
| TestBaseFunction | test_deviation_correct_values | Element-wise absolute deviation |
| TestBaseFunction | test_deviation_identical_arrays | Zero deviation for identical arrays |
| TestBaseFunction | test_deviation_negative_values | Non-negative result always |
| TestTrainingFunction | test_max_deviation_returns_largest | Returns the single largest difference |
| TestTrainingFunction | test_max_deviation_identical | Zero when train = ideal |
| TestIdealFunction | test_squared_error_zero_for_perfect_match | SSE = 0 for perfect fit |
| TestIdealFunction | test_squared_error_known_value | SSE = 2.0 for unit shift |
| TestIdealFunction | test_squared_error_always_non_negative | SSE >= 0 always |
| TestTestPoint | test_assignable_when_delta_below_threshold | delta <= sqrt(2)*max_dev → True |
| TestTestPoint | test_not_assignable_when_delta_exceeds_threshold | delta > sqrt(2)*max_dev → False |
| TestTestPoint | test_assignable_at_exact_boundary | Boundary is inclusive |
| TestTestPoint | test_not_assignable_just_above_boundary | Just above boundary → False |
| TestDataManager | test_load_valid_csv_succeeds | Valid CSV loads cleanly |
| TestDataManager | test_load_missing_file_raises | FileNotFoundError raised |
| TestDataManager | test_load_empty_csv_raises | DataValidationError raised |
| TestDataManager | test_load_csv_with_nulls_raises | DataValidationError raised |
| TestDataManager | test_save_and_read_roundtrip | DB save/load preserves data |

---

## Class Architecture

```
BaseFunction
│   Attributes : x (ndarray), y (ndarray)
│   Methods    : deviation(other_y) → |self.y – other_y|
│
├── TrainingFunction(BaseFunction)
│       max_deviation(ideal_y) → max |train_y – ideal_y|
│
├── IdealFunction(BaseFunction)
│       squared_error(training_y) → Σ(training_y – ideal_y)²
│
└── TestPoint(BaseFunction)
        is_assignable(ideal_y_at_x, max_dev) → bool (√2 rule)

DataManager       (I/O: CSV loading, SQLite persistence)
FunctionSelector  (Algorithm: least-squares selection + √2 mapping)
Visualiser        (Outputs: Bokeh HTML + Matplotlib PNGs)
```

---

## Mathematical Basis

**Ideal Function Selection — Least Squares:**

```
SSE(j, k) = Σᵢ ( y_train(i) − y_ideal(i) )²
j* = argmin_j SSE(j, k)
```

**Test Data Mapping — √2 Rule:**

```
|y_test − y_ideal(x_test)| ≤ Δ_max × √2
```

where `Δ_max = max |y_train(i) − y_ideal(i)|` computed during training.

---

## Database Schema

| Table | Key Columns | Purpose |
|---|---|---|
| `training_data` | x, y1, y2, y3, y4 | Raw training functions |
| `ideal_functions` | x, y1 … y50 | All 50 candidate ideal functions |
| `test_mapping` | x, y_test, ideal_function, deviation, assigned | Mapping results |

---

## References

- Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). *Introduction to linear regression analysis* (5th ed.). Wiley.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An introduction to statistical learning* (2nd ed.). Springer.
- Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design patterns*. Addison-Wesley.
- McKinney, W. (2018). *Python for data analysis* (2nd ed.). O'Reilly Media.
- Bayer, M. (2023). SQLAlchemy documentation. https://www.sqlalchemy.org
- Bokeh Development Team. (2023). Bokeh library. https://bokeh.org
- Van Rossum, G., & Drake, F. L. (2009). *Python 3 reference manual*. CreateSpace.
