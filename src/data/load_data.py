"""
=================================================================
 DATA LOADER — Loading & cleaning of historical pharma sales data
=================================================================
Input  : data/raw/pharma_sales_history.csv  (real transactional data)
Output : data/processed/sales_monthly.csv
         data/processed/stock_monthly.csv
         data/processed/product_stats.csv

This module corresponds to deliverable:
  "Preparation and structuring of historical stock and sales data"
=================================================================
"""
import pandas as pd
import numpy as np
import os

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Average delivery lead time per therapeutic class (days) — used later for
# reorder point calculations. These are reasonable industry-style defaults.
LEAD_TIME_BY_CLASS = {
    'Analgesics': 4, 'Antibiotics': 5, 'Antiseptics': 3,
    'Antipiretics': 3, 'Antimalarial': 7, 'Mood Stabilizers': 5,
}

SAFETY_STOCK_PCT = 0.30  # 30% of average monthly demand


def load_raw(filename: str = "pharma_sales_history.csv") -> pd.DataFrame:
    """Load the raw transactional CSV file."""
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid rows (negative quantity/sales = returns or corrections)
    and build a proper datetime column from Month + Year.
    """
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Sales"] > 0)].copy()
    removed = before - len(df)
    print(f"  Removed {removed:,} invalid rows (returns/corrections) "
          f"({before:,} -> {len(df):,})")

    df["month_num"] = df["Month"].map(MONTH_MAP)
    df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["month_num"], day=1))
    df["Product Name"] = df["Product Name"].str.strip()
    df["Distributor"] = df["Distributor"].str.strip()
    return df


def build_product_ids(df: pd.DataFrame) -> dict:
    """Assign a short, stable product_id (P001, P002...) to each product name."""
    products = sorted(df["Product Name"].unique())
    return {name: f"P{i+1:03d}" for i, name in enumerate(products)}


def aggregate_monthly_sales(df: pd.DataFrame, product_to_id: dict) -> pd.DataFrame:
    """Aggregate raw transactions into one row per (product, month)."""
    sales = (
        df.groupby(["date", "Product Name", "Product Class"])
        .agg(
            quantity_sold=("Quantity", "sum"),
            revenue=("Sales", "sum"),
            unit_price=("Price", "mean"),
            n_transactions=("Sales", "count"),
            n_cities=("City", "nunique"),
            n_countries=("Country", "nunique"),
        )
        .reset_index()
        .sort_values(["Product Name", "date"])
    )
    sales["unit_price"] = sales["unit_price"].round(2)
    sales["product_id"] = sales["Product Name"].map(product_to_id)
    return sales


def simulate_stock_levels(sales: pd.DataFrame, product_to_id: dict) -> pd.DataFrame:
    """
    The source dataset contains sales but no stock/purchase ledger.
    We reconstruct a plausible monthly stock level per product using a
    classic inventory simulation:
        stock(t) = stock(t-1) + inbound(t) - outbound(t)
    where inbound ~ demand x (1.05-1.25) (a realistic replenishment buffer)
    and outbound = actual sold quantity.
    """
    rows = []
    rng = np.random.default_rng(42)

    demand_stats = (
        sales.groupby("Product Name")["quantity_sold"]
        .agg(["mean", "std"]).rename(columns={"mean": "avg", "std": "std"}).fillna(0)
    )

    for product, pid in product_to_id.items():
        product_class = sales.loc[sales["Product Name"] == product, "Product Class"].iloc[0]
        lead_time = LEAD_TIME_BY_CLASS.get(product_class, 5)
        avg_demand = demand_stats.loc[product, "avg"]
        safety_stock = max(50, int(avg_demand * SAFETY_STOCK_PCT))
        stock = int(avg_demand * 2)  # initial stock = ~2 months of demand

        history = sales[sales["Product Name"] == product].sort_values("date")
        for _, row in history.iterrows():
            inbound = int(row["quantity_sold"] * rng.uniform(1.05, 1.25))
            outbound = int(row["quantity_sold"])
            stock = max(0, stock + inbound - outbound)
            rows.append({
                "date": row["date"],
                "product_id": pid,
                "product_name": product,
                "product_class": product_class,
                "stock_end_of_month": stock,
                "safety_stock": safety_stock,
                "lead_time_days": lead_time,
                "inbound": inbound,
                "outbound": outbound,
                "stockout_flag": int(stock < safety_stock),
            })
    return pd.DataFrame(rows)


def build_product_stats(sales: pd.DataFrame, product_to_id: dict) -> pd.DataFrame:
    """Per-product summary statistics, used later by the recommender."""
    stats = (
        sales.groupby(["product_id", "Product Name", "Product Class"])
        .agg(
            avg_monthly_demand=("quantity_sold", "mean"),
            std_monthly_demand=("quantity_sold", "std"),
            unit_price=("unit_price", "mean"),
            total_revenue=("revenue", "sum"),
        )
        .reset_index()
        .fillna(0)
    )
    stats["avg_weekly_demand"] = stats["avg_monthly_demand"] / 4.33
    stats["std_weekly_demand"] = stats["std_monthly_demand"] / 4.33
    stats["safety_stock"] = (stats["avg_monthly_demand"] * SAFETY_STOCK_PCT).astype(int).clip(lower=50)
    stats["lead_time_days"] = stats["Product Class"].map(LEAD_TIME_BY_CLASS).fillna(5)
    return stats.round(2)


def run():
    os.makedirs(PROC_DIR, exist_ok=True)
    print("Loading raw transactional data...")
    df = load_raw()
    print(f"  {len(df):,} raw rows loaded")

    df = clean_transactions(df)
    product_to_id = build_product_ids(df)

    print("Aggregating monthly sales per product...")
    sales = aggregate_monthly_sales(df, product_to_id)

    print("Simulating monthly stock levels...")
    stock = simulate_stock_levels(sales, product_to_id)

    print("Computing per-product statistics...")
    stats = build_product_stats(sales, product_to_id)

    sales.to_csv(os.path.join(PROC_DIR, "sales_monthly.csv"), index=False)
    stock.to_csv(os.path.join(PROC_DIR, "stock_monthly.csv"), index=False)
    stats.to_csv(os.path.join(PROC_DIR, "product_stats.csv"), index=False)

    print("\nDone.")
    print(f"  sales_monthly.csv   -> {len(sales):,} rows | {sales['Product Name'].nunique()} products")
    print(f"  stock_monthly.csv   -> {len(stock):,} rows")
    print(f"  product_stats.csv   -> {len(stats)} products")
    print(f"  Period: {sales['date'].min().date()} -> {sales['date'].max().date()}")


if __name__ == "__main__":
    run()
