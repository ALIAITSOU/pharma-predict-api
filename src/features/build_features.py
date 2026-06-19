"""
=================================================================
 FEATURE ENGINEERING — Build the ML-ready training dataset
=================================================================
Input  : data/processed/sales_monthly.csv
         data/processed/stock_monthly.csv
Output : data/processed/training_dataset.csv

Transforms raw monthly aggregates into a feature-rich table that
machine learning models can learn from: calendar features, lag
features, rolling statistics, and stock-related ratios.
=================================================================
"""
import pandas as pd
import numpy as np
import os

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add month/quarter/year + cyclical encodings (sin/cos) to avoid
    artificial discontinuities between December and January."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["year"] = df[date_col].dt.year
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_quarter"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["cos_quarter"] = np.cos(2 * np.pi * df["quarter"] / 4)
    start = df[date_col].min()
    df["months_since_start"] = ((df[date_col] - start).dt.days / 30.44).round().astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target: str, lags=LAGS) -> pd.DataFrame:
    """Add lagged values of the target (demand N months ago)."""
    df = df.copy().sort_values("date")
    for lag in lags:
        df[f"{target}_lag{lag}"] = df.groupby("Product Name")[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str, windows=ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Add rolling mean/std of the target.
    IMPORTANT: shift(1) is applied first so that the current month's value
    is never included in its own rolling statistics (avoids data leakage).
    """
    df = df.copy().sort_values("date")
    for w in windows:
        df[f"{target}_ma{w}"] = df.groupby("Product Name")[target].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"{target}_std{w}"] = df.groupby("Product Name")[target].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0)
        )
    return df


def build_training_dataset() -> pd.DataFrame:
    print("Loading processed sales & stock data...")
    sales = pd.read_csv(os.path.join(PROC_DIR, "sales_monthly.csv"), parse_dates=["date"])
    stock = pd.read_csv(os.path.join(PROC_DIR, "stock_monthly.csv"), parse_dates=["date"])

    df = sales.merge(
        stock[["date", "product_id", "stock_end_of_month", "safety_stock",
               "lead_time_days", "stockout_flag"]],
        on=["date", "product_id"], how="left"
    )

    df["class_code"] = df["Product Class"].astype("category").cat.codes

    print("Adding calendar features...")
    df = add_calendar_features(df, "date")

    print("Adding lag features...")
    df = add_lag_features(df, "quantity_sold")

    print("Adding rolling features...")
    df = add_rolling_features(df, "quantity_sold")

    # Business features
    df["stock_to_demand_ratio"] = df["stock_end_of_month"] / (df["quantity_sold"] + 1)
    df["coverage_months"] = df["stock_end_of_month"] / (df["quantity_sold_ma3"] + 1)
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())

    df = df.dropna().sort_values(["Product Name", "date"]).reset_index(drop=True)
    return df


def run():
    df = build_training_dataset()
    out_path = os.path.join(PROC_DIR, "training_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\nDone. training_dataset.csv -> {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Products: {df['Product Name'].nunique()} | "
          f"Period: {df['date'].min().date()} -> {df['date'].max().date()}")


if __name__ == "__main__":
    run()
