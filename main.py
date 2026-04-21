"""
DLMDSPWP01 – Programming with Python
Selection of Ideal Functions and Test Data Mapping Using Least Squares Method

Author      : Naveen Kumar
Matric No.  : 10244074
Institution : IU International University of Applied Sciences, Berlin

Entry point for the full pipeline:
    1. Load and validate CSV datasets
    2. Persist to SQLite database
    3. Select ideal functions via least squares
    4. Map test data using sqrt(2) deviation threshold
    5. Save results to database
    6. Generate Bokeh interactive visualisation + Matplotlib static plots

Usage:
    python main.py
"""

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, Legend, ColumnDataSource
from bokeh.palettes import Category10
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class DataValidationError(Exception):
    """Raised when a loaded dataset fails a validation check (empty or nulls)."""
    pass


class MappingError(Exception):
    """Raised when no test points can be mapped to any selected ideal function."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BaseFunction:
    """
    Represents any mathematical function as discrete (x, y) value pairs.

    This base class provides the shared attributes and the deviation method
    used by all specialised subclasses.

    Parameters
    ----------
    x : array-like
        x-values shared across all datasets.
    y : array-like
        Corresponding y-values for this function instance.
    """

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def deviation(self, other_y):
        """
        Compute element-wise absolute deviation from another y-array.

        Parameters
        ----------
        other_y : array-like
            y-values to compare against.

        Returns
        -------
        np.ndarray
            Array of |self.y - other_y| values.
        """
        return np.abs(self.y - np.asarray(other_y, dtype=float))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

class TrainingFunction(BaseFunction):
    """
    Represents one of the four training datasets.

    Extends BaseFunction with the ability to compute the maximum absolute
    deviation against a candidate ideal function, which is used as the
    threshold reference for test data mapping.
    """

    def max_deviation(self, ideal_y):
        """
        Compute the maximum absolute difference between this training function
        and a candidate ideal function across all x-values.

        Parameters
        ----------
        ideal_y : array-like
            y-values of the candidate ideal function.

        Returns
        -------
        float
            Maximum |train_y - ideal_y| across all data points.
        """
        return float(np.max(self.deviation(ideal_y)))


# ─────────────────────────────────────────────────────────────────────────────
# IDEAL FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

class IdealFunction(BaseFunction):
    """
    Represents one of the fifty candidate ideal functions.

    Extends BaseFunction with the squared_error method, which implements
    the least-squares criterion used for ideal function selection.
    """

    def squared_error(self, training_y):
        """
        Compute the sum of squared differences (least-squares criterion)
        between this ideal function and a training dataset.

        Parameters
        ----------
        training_y : array-like
            y-values of the training function to compare against.

        Returns
        -------
        float
            SSE = sum((training_y - ideal_y)^2) across all data points.
        """
        diff = np.asarray(training_y, dtype=float) - self.y
        return float(np.sum(diff ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# TEST POINT
# ─────────────────────────────────────────────────────────────────────────────

class TestPoint(BaseFunction):
    """
    Represents a single test data point (scalar x, scalar y).

    Extends BaseFunction with the sqrt(2) assignment rule used to determine
    whether this point can be mapped to a selected ideal function.
    """

    def is_assignable(self, ideal_y_at_x, max_dev):
        """
        Determine whether this test point satisfies the sqrt(2) deviation rule.

        A test point is assignable to an ideal function if:
            |test_y - ideal_y| <= max_dev * sqrt(2)

        Parameters
        ----------
        ideal_y_at_x : float
            The ideal function's y-value at the x-position of this test point.
        max_dev : float
            The maximum deviation (Delta_max) computed during the training phase
            for the corresponding ideal function.

        Returns
        -------
        bool
            True if the test point satisfies the threshold condition.
        """
        delta = abs(float(self.y) - float(ideal_y_at_x))
        return delta <= max_dev * math.sqrt(2)


# ─────────────────────────────────────────────────────────────────────────────
# DATA MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class DataManager:
    """
    Handles all data input/output operations: CSV loading, validation,
    SQLite database creation, and result persistence.

    Separation of I/O concerns from domain logic follows the Single
    Responsibility Principle (Gamma et al., 1994).

    Parameters
    ----------
    db_path : str
        File path for the SQLite database. Defaults to 'functions.db'.
    """

    def __init__(self, db_path="functions.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")

    def load_csv(self, filepath, name="dataset"):
        """
        Load a CSV file into a Pandas DataFrame and validate its contents.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        name : str, optional
            Descriptive label used in error messages.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        DataValidationError
            If the loaded DataFrame is empty or contains null values.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[{name}] File not found: {filepath}")
        df = pd.read_csv(filepath)
        if df.empty:
            raise DataValidationError(f"[{name}] Dataset is empty.")
        if df.isnull().any().any():
            raise DataValidationError(f"[{name}] Dataset contains null values.")
        print(f"  Loaded {name}: {df.shape[0]} rows x {df.shape[1]} columns")
        return df

    def save_to_db(self, df, table_name):
        """
        Persist a DataFrame as a table in the SQLite database.

        Parameters
        ----------
        df : pd.DataFrame
            Data to store.
        table_name : str
            Name of the target database table.

        Raises
        ------
        RuntimeError
            If the database write operation fails.
        """
        try:
            df.to_sql(table_name, self.engine, if_exists="replace", index=False)
            print(f"  Saved '{table_name}' ({len(df)} rows) to {self.db_path}")
        except Exception as exc:
            raise RuntimeError(f"Database write failed for '{table_name}': {exc}")

    def read_from_db(self, table_name):
        """
        Read an entire table from the SQLite database.

        Parameters
        ----------
        table_name : str
            Name of the table to retrieve.

        Returns
        -------
        pd.DataFrame
        """
        with self.engine.connect() as conn:
            return pd.read_sql(f'SELECT * FROM "{table_name}"', conn)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

class FunctionSelector:
    """
    Implements the least-squares ideal function selection and the sqrt(2)
    test data mapping algorithm.

    Parameters
    ----------
    training_df : pd.DataFrame
        Training dataset with columns: x, y1, y2, y3, y4.
    ideal_df : pd.DataFrame
        Ideal functions dataset with columns: x, y1 ... y50.
    """

    def __init__(self, training_df, ideal_df):
        self.training_df     = training_df
        self.ideal_df        = ideal_df
        self.x               = training_df["x"].values
        self.chosen_ideals   = []   # ideal column names, one per training function
        self.max_deviations  = []   # Delta_max values, one per chosen ideal
        self.results_df      = None

    def select_ideal_functions(self):
        """
        For each of the four training functions, identify the ideal function
        from fifty candidates that minimises the sum of squared errors.

        Populates self.chosen_ideals and self.max_deviations.

        Returns
        -------
        pd.DataFrame
            Summary table with training function, best ideal, SSE, and Delta_max.
        """
        train_cols = ["y1", "y2", "y3", "y4"]
        ideal_cols = [c for c in self.ideal_df.columns if c != "x"]
        records    = []

        for t_col in train_cols:
            train_y    = self.training_df[t_col].values
            train_func = TrainingFunction(self.x, train_y)
            min_sse    = float("inf")
            best_col   = None
            best_y     = None

            for i_col in ideal_cols:
                ideal_y = self.ideal_df[i_col].values
                sse     = IdealFunction(self.x, ideal_y).squared_error(train_y)
                if sse < min_sse:
                    min_sse  = sse
                    best_col = i_col
                    best_y   = ideal_y

            max_dev = train_func.max_deviation(best_y)
            self.chosen_ideals.append(best_col)
            self.max_deviations.append(max_dev)
            records.append({
                "Training Function"  : t_col,
                "Best Ideal Function": best_col,
                "SSE (min)"          : round(min_sse, 4),
                "Delta_max"          : round(max_dev, 4),
            })

        self.results_df = pd.DataFrame(records)
        return self.results_df

    def map_test_data(self, test_df):
        """
        Assign each test data point to one of the four selected ideal functions
        using the sqrt(2) deviation threshold.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test dataset with columns: x, y.

        Returns
        -------
        pd.DataFrame
            Mapping results with columns:
            x, y_test, ideal_function, deviation, assigned.

        Raises
        ------
        MappingError
            If no test point could be assigned to any ideal function.
        """
        mapped_rows = []

        for _, row in test_df.iterrows():
            x_val    = float(row["x"])
            y_val    = float(row["y"])
            tp       = TestPoint(x_val, y_val)
            assigned = False

            for idx, i_col in enumerate(self.chosen_ideals):
                nearest_idx  = int(np.argmin(np.abs(self.ideal_df["x"].values - x_val)))
                ideal_y_at_x = float(self.ideal_df[i_col].iloc[nearest_idx])
                max_dev      = self.max_deviations[idx]
                delta        = abs(y_val - ideal_y_at_x)

                if tp.is_assignable(ideal_y_at_x, max_dev):
                    mapped_rows.append({
                        "x"              : x_val,
                        "y_test"         : y_val,
                        "ideal_function" : i_col,
                        "deviation"      : round(delta, 6),
                        "assigned"       : True,
                    })
                    assigned = True
                    break

            if not assigned:
                mapped_rows.append({
                    "x"              : x_val,
                    "y_test"         : y_val,
                    "ideal_function" : None,
                    "deviation"      : None,
                    "assigned"       : False,
                })

        result = pd.DataFrame(mapped_rows)
        if result["assigned"].sum() == 0:
            raise MappingError("No test data points could be mapped to any ideal function.")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

class Visualiser:
    """
    Generates all visualisation outputs: Bokeh interactive HTML plot and
    Matplotlib static PNG plots.

    Parameters
    ----------
    training_df : pd.DataFrame
    ideal_df    : pd.DataFrame
    mapping_df  : pd.DataFrame
    selector    : FunctionSelector  (provides chosen_ideals, max_deviations, results_df)
    output_dir  : str               (directory for output files)
    """

    TRAIN_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    IDEAL_COLORS = ["#F44336", "#009688", "#FF5722", "#3F51B5"]

    def __init__(self, training_df, ideal_df, mapping_df, selector, output_dir="."):
        self.training_df = training_df
        self.ideal_df    = ideal_df
        self.mapping_df  = mapping_df
        self.selector    = selector
        self.output_dir  = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_training_vs_ideal(self):
        """
        Produce a 2x2 Matplotlib figure comparing each training function
        with its selected ideal function, including the sqrt(2)*Delta_max
        tolerance band and mapped test points.

        Output saved to: <output_dir>/plot_training_vs_ideal.png
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Training Data vs Selected Ideal Functions",
            fontsize=14, fontweight="bold"
        )
        train_cols = ["y1", "y2", "y3", "y4"]

        for i, (ax, t_col) in enumerate(zip(axes.flat, train_cols)):
            i_col   = self.selector.chosen_ideals[i]
            max_dev = self.selector.max_deviations[i]
            x_vals  = self.training_df["x"].values
            y_train = self.training_df[t_col].values
            y_ideal = self.ideal_df[i_col].values
            sse_val = self.selector.results_df.loc[i, "SSE (min)"]

            ax.scatter(x_vals, y_train, s=6, alpha=0.5,
                       color=self.TRAIN_COLORS[i], label=f"Training {t_col}")
            ax.plot(x_vals, y_ideal, color=self.IDEAL_COLORS[i],
                    linewidth=1.8, label=f"Ideal {i_col}")
            ax.fill_between(
                x_vals,
                y_ideal - max_dev * math.sqrt(2),
                y_ideal + max_dev * math.sqrt(2),
                alpha=0.12, color=self.IDEAL_COLORS[i],
                label=f"sqrt(2)*Delta_max = {max_dev*math.sqrt(2):.2f}"
            )
            mapped_here = self.mapping_df[
                (self.mapping_df["ideal_function"] == i_col) &
                self.mapping_df["assigned"]
            ]
            if not mapped_here.empty:
                ax.scatter(
                    mapped_here["x"], mapped_here["y_test"],
                    s=40, marker="+", color="black",
                    linewidths=1.2, zorder=5, label="Mapped test pts"
                )
            ax.set_title(
                f"{t_col} -> Ideal {i_col}  |  SSE={sse_val}",
                fontsize=10
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(self.output_dir, "plot_training_vs_ideal.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")

    def plot_sse_and_maxdev(self):
        """
        Produce a side-by-side bar chart of minimum SSE and Delta_max values
        for each training-ideal function pair.

        Output saved to: <output_dir>/plot_sse_maxdev.png
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        labels = [
            f"{r['Training Function']}\n->{r['Best Ideal Function']}"
            for _, r in self.selector.results_df.iterrows()
        ]
        sses  = self.selector.results_df["SSE (min)"].values
        mdevs = self.selector.results_df["Delta_max"].values

        bars1 = ax1.bar(labels, sses, color=self.TRAIN_COLORS,
                        edgecolor="white", width=0.5)
        ax1.set_title("Minimum SSE per Training Function", fontweight="bold")
        ax1.set_ylabel("Sum of Squared Errors")
        ax1.set_xlabel("Training -> Selected Ideal")
        for bar, val in zip(bars1, sses):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9
            )
        ax1.grid(axis="y", alpha=0.3)

        bars2 = ax2.bar(labels, mdevs, color=self.IDEAL_COLORS,
                        edgecolor="white", width=0.5)
        ax2.set_title("Max Deviation (Delta_max) per Training Function",
                      fontweight="bold")
        ax2.set_ylabel("Max Absolute Deviation")
        ax2.set_xlabel("Training -> Selected Ideal")
        for bar, val in zip(bars2, mdevs):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9
            )
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = os.path.join(self.output_dir, "plot_sse_maxdev.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")

    def plot_mapping_summary(self):
        """
        Produce a pie chart (assigned vs. unassigned) and a bar chart
        (assignments per ideal function) summarising the test mapping results.

        Output saved to: <output_dir>/plot_mapping_summary.png
        """
        assigned   = int(self.mapping_df["assigned"].sum())
        unassigned = len(self.mapping_df) - assigned

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.pie(
            [assigned, unassigned],
            labels=[f"Assigned ({assigned})", f"Not assigned ({unassigned})"],
            autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax1.set_title("Test Data Assignment Rate", fontweight="bold")

        assigned_df = self.mapping_df[self.mapping_df["assigned"]]
        counts      = assigned_df.groupby("ideal_function").size()
        ax2.bar(counts.index, counts.values,
                color=self.IDEAL_COLORS[:len(counts)], edgecolor="white")
        ax2.set_title("Test Points Assigned per Ideal Function",
                      fontweight="bold")
        ax2.set_xlabel("Ideal Function")
        ax2.set_ylabel("Number of Test Points")
        for i, v in enumerate(counts.values):
            ax2.text(i, v + 0.3, str(v), ha="center", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = os.path.join(self.output_dir, "plot_mapping_summary.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")

    def plot_bokeh_interactive(self):
        """
        Produce an interactive Bokeh HTML plot showing training scatter,
        ideal function lines, mapped test points, and unassigned points.
        Hover tooltips display type, x, and y for every data point.

        Output saved to: <output_dir>/bokeh_plot.html
        """
        out_path = os.path.join(self.output_dir, "bokeh_plot.html")
        output_file(out_path)

        p = figure(
            title="Training Data, Ideal Functions & Mapped Test Points",
            width=1000, height=550,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        hover = HoverTool(tooltips=[
            ("Type", "@label"),
            ("x",    "@x{0.00}"),
            ("y",    "@y{0.00}"),
        ])
        p.add_tools(hover)
        p.title.text_font_size = "13pt"

        palette      = Category10[8]
        train_cols   = ["y1", "y2", "y3", "y4"]
        legend_items = []

        for i, t_col in enumerate(train_cols):
            src = ColumnDataSource(dict(
                x=self.training_df["x"].values,
                y=self.training_df[t_col].values,
                label=[f"Training {t_col}"] * len(self.training_df),
            ))
            r = p.scatter("x", "y", source=src, size=4, alpha=0.4,
                          color=palette[i], marker="circle")
            legend_items.append((f"Training {t_col}", [r]))

        for i, i_col in enumerate(self.selector.chosen_ideals):
            src = ColumnDataSource(dict(
                x=self.ideal_df["x"].values,
                y=self.ideal_df[i_col].values,
                label=[f"Ideal {i_col}"] * len(self.ideal_df),
            ))
            r = p.line("x", "y", source=src, line_width=2,
                       color=palette[i + 4])
            legend_items.append((f"Ideal {i_col}", [r]))

        mapped = self.mapping_df[self.mapping_df["assigned"]]
        if not mapped.empty:
            src_m = ColumnDataSource(dict(
                x=mapped["x"].values,
                y=mapped["y_test"].values,
                label=[f"Test->{f}" for f in mapped["ideal_function"]],
            ))
            r_m = p.scatter("x", "y", source=src_m, size=10,
                            marker="cross", color="black", line_width=2)
            legend_items.append(("Mapped Test Data", [r_m]))

        unmap = self.mapping_df[~self.mapping_df["assigned"]]
        if not unmap.empty:
            src_u = ColumnDataSource(dict(
                x=unmap["x"].values,
                y=unmap["y_test"].values,
                label=["Unassigned"] * len(unmap),
            ))
            r_u = p.scatter("x", "y", source=src_u, size=8,
                            marker="x", color="red", line_width=1.5)
            legend_items.append(("Unassigned", [r_u]))

        from bokeh.models import Legend as BokehLegend
        legend = BokehLegend(
            items=legend_items,
            location="top_left",
            click_policy="hide",
            label_text_font_size="8pt",
        )
        p.add_layout(legend, "right")
        p.xaxis.axis_label = "x"
        p.yaxis.axis_label = "y"
        p.grid.grid_line_alpha = 0.3

        save(p)
        print(f"  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    train_path="train.csv",
    ideal_path="ideal.csv",
    test_path="test.csv",
    db_path="functions.db",
    output_dir="outputs",
):
    """
    Execute the full data pipeline from CSV loading to visualisation.

    Parameters
    ----------
    train_path : str  – path to training data CSV
    ideal_path : str  – path to ideal functions CSV
    test_path  : str  – path to test data CSV
    db_path    : str  – path for SQLite database output
    output_dir : str  – directory for plot outputs
    """
    print("\n=== DLMDSPWP01 Pipeline Starting ===\n")

    # Step 1 — Load and validate
    print("[Step 1] Loading CSV datasets...")
    dm          = DataManager(db_path=db_path)
    training_df = dm.load_csv(train_path, name="Training Data")
    ideal_df    = dm.load_csv(ideal_path, name="Ideal Functions")
    test_df     = dm.load_csv(test_path,  name="Test Data")

    # Step 2 — Persist raw data to database
    print("\n[Step 2] Persisting datasets to SQLite...")
    dm.save_to_db(training_df, "training_data")
    dm.save_to_db(ideal_df,    "ideal_functions")

    # Step 3 — Least-squares ideal function selection
    print("\n[Step 3] Selecting ideal functions via least squares...")
    selector = FunctionSelector(training_df, ideal_df)
    results  = selector.select_ideal_functions()
    print("\n  Selection Results:")
    print(results.to_string(index=False))

    # Step 4 — Test data mapping
    print("\n[Step 4] Mapping test data using sqrt(2) threshold...")
    mapping_df = selector.map_test_data(test_df)
    assigned   = int(mapping_df["assigned"].sum())
    print(f"  Assigned: {assigned} / {len(mapping_df)} test points")

    # Step 5 — Persist mapping results
    print("\n[Step 5] Saving mapping results to database...")
    dm.save_to_db(mapping_df, "test_mapping")

    # Step 6 — Visualisation
    print("\n[Step 6] Generating visualisations...")
    vis = Visualiser(training_df, ideal_df, mapping_df, selector,
                     output_dir=output_dir)
    vis.plot_training_vs_ideal()
    vis.plot_sse_and_maxdev()
    vis.plot_mapping_summary()
    vis.plot_bokeh_interactive()

    print("\n=== Pipeline Complete ===")
    print(f"  Database : {db_path}")
    print(f"  Plots    : {output_dir}/")
    return selector, mapping_df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(
        train_path="data/train.csv",
        ideal_path="data/ideal.csv",
        test_path ="data/test.csv",
        db_path   ="functions.db",
        output_dir="outputs",
    )
