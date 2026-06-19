"""
=================================================================
 DEMO SCRIPT — Exercises all API business logic without a server
=================================================================
Useful to validate the prediction + recommendation pipeline before
starting the actual FastAPI server, and to run inside CI without
needing network access or a live HTTP server.

Run with:
    python -m src.api.demo
=================================================================
"""
import json
import os
import pandas as pd
from datetime import datetime

from src.models.predictor import DemandPredictor
from src.recommendation.engine import generate_recommendations, load_current_stock

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")


def main():
    print("Initialising predictor...")
    predictor = DemandPredictor()

    stats_df = pd.read_csv(os.path.join(PROC_DIR, "product_stats.csv"))
    products_info = {
        row["Product Name"]: {
            "product_id": row.get("product_id", "?"),
            "product_class": row.get("Product Class", "?"),
            "unit_price": round(float(row.get("unit_price", 0)), 2),
            "avg_monthly_demand": round(float(row.get("avg_monthly_demand", 0)), 0),
        }
        for _, row in stats_df.iterrows()
    }
    current_stock = load_current_stock()
    sep = "=" * 65

    print(f"\n{sep}\nGET /health\n{sep}")
    print(json.dumps({
        "status": "healthy",
        "n_products": len(products_info),
        "timestamp": datetime.now().isoformat(),
    }, indent=2))

    print(f"\n{sep}\nGET /classes\n{sep}")
    print(sorted(stats_df["Product Class"].unique().tolist()))

    sample_product = sorted(products_info.keys())[0]
    print(f"\n{sep}\nPOST /predict/{sample_product}?horizon=3\n{sep}")
    forecast = predictor.predict(sample_product, horizon=3)
    print(json.dumps(forecast, indent=2, ensure_ascii=False))

    print(f"\n{sep}\nGET /recommend (summary)\n{sep}")
    predictions = predictor.predict_all(horizon=3)
    recs = generate_recommendations(predictions, current_stock)
    summary = {
        "critical": sum(1 for r in recs if r.urgency == "CRITICAL"),
        "high": sum(1 for r in recs if r.urgency == "HIGH"),
        "normal": sum(1 for r in recs if r.urgency == "NORMAL"),
        "surplus": sum(1 for r in recs if r.urgency == "SURPLUS"),
    }
    print("Summary:", summary)
    for r in recs[:6]:
        print(f"  [{r.urgency:<10}][{r.abc_class}] {r.product_name:<35} -> {r.action}")

    print(f"\n{sep}\nGET /metrics\n{sep}")
    print(json.dumps(predictor.metrics(), indent=2))


if __name__ == "__main__":
    main()
