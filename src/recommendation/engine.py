"""
=================================================================
 RECOMMENDATION ENGINE — Restocking decision support
=================================================================
Converts demand forecasts into actionable restocking decisions
using classic inventory-management formulas:
  - Reorder Point (ROP)
  - Economic Order Quantity (EOQ)
  - ABC classification (Pareto analysis)
=================================================================
"""
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass, fields
from typing import List

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

LEAD_TIME_BY_CLASS = {
    "Analgesics": 4, "Antibiotics": 5, "Antiseptics": 3,
    "Antipiretics": 3, "Antimalarial": 7, "Mood Stabilizers": 5,
}

SERVICE_LEVEL_Z = 1.65       # ~95% service level
ORDER_COST = 100.0           # fixed cost per purchase order (currency units)
HOLDING_COST_PCT = 0.25      # annual holding cost as % of unit price


@dataclass
class Recommendation:
    product_name: str
    product_class: str
    product_id: str
    urgency: str                  # CRITICAL / HIGH / NORMAL / SURPLUS
    priority: int                 # 1 (most urgent) to 4
    action: str
    reason: str
    abc_class: str                # A / B / C
    predicted_monthly_demand: float
    current_stock: float
    safety_stock: int
    reorder_point: float
    recommended_order_qty: float
    days_of_coverage: float


def load_product_stats() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROC_DIR, "product_stats.csv")).set_index("Product Name")


def load_current_stock() -> dict:
    df = pd.read_csv(os.path.join(PROC_DIR, "stock_monthly.csv"), parse_dates=["date"])
    return df.sort_values("date").groupby("product_name")["stock_end_of_month"].last().to_dict()


def economic_order_quantity(annual_demand: float, order_cost: float = ORDER_COST,
                             holding_cost_pct: float = HOLDING_COST_PCT,
                             unit_price: float = 400.0) -> float:
    """EOQ = sqrt(2 * D * S / H)"""
    holding_cost = holding_cost_pct * unit_price
    if annual_demand <= 0 or holding_cost <= 0:
        return 0
    return round(np.sqrt(2 * annual_demand * order_cost / holding_cost))


def reorder_point(monthly_demand: float, lead_time_days: int, safety_stock: int,
                   variability: float = 0.2) -> float:
    """ROP = demand during lead time + safety buffer + safety stock."""
    daily_demand = monthly_demand / 30
    sigma = variability * daily_demand * np.sqrt(lead_time_days)
    return round(daily_demand * lead_time_days + SERVICE_LEVEL_Z * sigma + safety_stock)


def classify_abc(annual_demand_series: pd.Series) -> pd.Series:
    """Classic Pareto ABC classification based on cumulative demand share."""
    sorted_series = annual_demand_series.sort_values(ascending=False)
    cumulative_pct = sorted_series.cumsum() / sorted_series.sum() * 100
    classes = pd.Series(index=annual_demand_series.index, dtype=str)
    for idx in annual_demand_series.index:
        pct = cumulative_pct[idx] if idx in cumulative_pct.index else 100
        classes[idx] = "A" if pct <= 70 else ("B" if pct <= 90 else "C")
    return classes


def generate_recommendations(predictions: dict, current_stocks: dict) -> List[Recommendation]:
    """
    predictions     : {product_name: predicted_avg_monthly_demand}
    current_stocks  : {product_name: current_stock_level}
    """
    stats = load_product_stats()
    annual_demand = pd.Series({k: v * 12 for k, v in predictions.items()})
    abc_classes = classify_abc(annual_demand)

    recommendations = []
    for product, monthly_demand in predictions.items():
        if product not in stats.index:
            continue
        row = stats.loc[product]
        stock = float(current_stocks.get(product, 0))
        safety_stock = int(row.get("safety_stock", 100))
        unit_price = float(row.get("unit_price", 400))
        product_class = str(row.get("Product Class", "Unknown"))
        product_id = str(row.get("product_id", "?")) if "product_id" in row.index else "?"
        lead_time = LEAD_TIME_BY_CLASS.get(product_class, 5)

        rop = reorder_point(monthly_demand, lead_time, safety_stock)
        eoq = economic_order_quantity(monthly_demand * 12, unit_price=unit_price)
        days_coverage = (stock / (monthly_demand / 30)) if monthly_demand > 0 else 999

        if stock < safety_stock * 0.5:
            urgency, priority, action = "CRITICAL", 1, "ORDER_IMMEDIATELY"
            reason = f"Stock {stock:.0f} is below 50% of safety stock ({safety_stock})"
        elif stock < safety_stock or stock < rop:
            urgency, priority, action = "HIGH", 2, "ORDER_WITHIN_48H"
            reason = f"Stock {stock:.0f} is at or below the reorder point ({rop:.0f})"
        elif days_coverage < 60:
            urgency, priority, action = "NORMAL", 3, "PLAN_ORDER"
            reason = f"Coverage of {days_coverage:.0f} days - plan ahead of lead time"
        elif stock > monthly_demand * 4:
            urgency, priority, action = "SURPLUS", 4, "DO_NOT_ORDER"
            reason = f"Stock covers {days_coverage:.0f} days - slow-moving item"
        else:
            urgency, priority, action = "NORMAL", 3, "MONITOR"
            reason = f"Stock level adequate - {days_coverage:.0f} days of coverage"

        recommendations.append(Recommendation(
            product_name=product, product_class=product_class, product_id=product_id,
            urgency=urgency, priority=priority, action=action, reason=reason,
            abc_class=abc_classes.get(product, "C"),
            predicted_monthly_demand=round(monthly_demand, 0),
            current_stock=round(stock, 0),
            safety_stock=safety_stock,
            reorder_point=rop,
            recommended_order_qty=eoq if action != "DO_NOT_ORDER" else 0,
            days_of_coverage=round(min(days_coverage, 999), 0),
        ))

    return sorted(recommendations, key=lambda r: (r.priority, -r.predicted_monthly_demand))


if __name__ == "__main__":
    stats = pd.read_csv(os.path.join(PROC_DIR, "product_stats.csv"))
    predictions = {row["Product Name"]: float(row["avg_monthly_demand"]) for _, row in stats.iterrows()}
    stocks = load_current_stock()
    recs = generate_recommendations(predictions, stocks)
    print(f"Generated {len(recs)} recommendations\n")
    for r in recs[:10]:
        print(f"[{r.urgency:<10}][{r.abc_class}] {r.product_name:<35} "
              f"stock={r.current_stock:>8.0f} -> {r.action}")
