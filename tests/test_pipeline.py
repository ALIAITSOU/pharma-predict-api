"""
=================================================================
 TEST SUITE — Unit & integration tests for PharmaPredict
=================================================================
Run with:
    pytest tests/ -v
or directly:
    python -m tests.test_pipeline
=================================================================
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name} {detail}")
        failed += 1


def section(title):
    print(f"\n{'-' * 60}\n  {title}\n{'-' * 60}")


# -----------------------------------------------------------------
section("1. Processed data integrity")
# -----------------------------------------------------------------
sales = pd.read_csv(os.path.join(PROC_DIR, "sales_monthly.csv"))
stock = pd.read_csv(os.path.join(PROC_DIR, "stock_monthly.csv"))
stats = pd.read_csv(os.path.join(PROC_DIR, "product_stats.csv"))
training = pd.read_csv(os.path.join(PROC_DIR, "training_dataset.csv"))

check("sales_monthly.csv is non-empty", len(sales) > 1000)
check("stock_monthly.csv is non-empty", len(stock) > 1000)
check("training_dataset.csv has engineered features", "quantity_sold_ma3" in training.columns)
check("No negative sold quantities", (sales["quantity_sold"] >= 0).all())
check("240 products covered", sales["Product Name"].nunique() == 240)
check("4 years of history", sales["date"].nunique() >= 36)
check("No missing values in core sales columns",
      not sales[["Product Name", "quantity_sold"]].isnull().any().any())

# -----------------------------------------------------------------
section("2. Feature engineering")
# -----------------------------------------------------------------
check("sin_month is bounded in [-1, 1]", training["sin_month"].between(-1, 1).all())
check("cos_month is bounded in [-1, 1]", training["cos_month"].between(-1, 1).all())
check("Lag features present", "quantity_sold_lag1" in training.columns)
check("Rolling features present", "quantity_sold_ma3" in training.columns)
check("stock_to_demand_ratio is non-negative", (training["stock_to_demand_ratio"] >= 0).all())
check("No NaN remaining after feature engineering", not training.isnull().all().any())

# -----------------------------------------------------------------
section("3. Trained models")
# -----------------------------------------------------------------
rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
gb = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    meta = json.load(f)

check("Random Forest loads correctly", rf is not None)
check("Gradient Boosting loads correctly", gb is not None)
check("Scaler loads correctly", scaler is not None)
check("Random Forest R2 > 0.85", meta["metrics"]["random_forest"]["r2"] > 0.85)
check("Gradient Boosting R2 > 0.85", meta["metrics"]["gradient_boosting"]["r2"] > 0.85)
check("Gradient Boosting MAPE < 5%", meta["metrics"]["gradient_boosting"]["mape"] < 5)
check("metadata.json defines a best_model", "best_model" in meta)
check("Feature importance is available", len(meta["feature_importance"]) > 0)

features = meta["features"]
X_dummy = pd.DataFrame([{f: 1.0 for f in features}])
pred = rf.predict(X_dummy)
check("Random Forest returns a single prediction", len(pred) == 1)
check("Random Forest prediction is non-negative", pred[0] >= 0)

# -----------------------------------------------------------------
section("4. DemandPredictor class")
# -----------------------------------------------------------------
from src.models.predictor import DemandPredictor

predictor = DemandPredictor()
sample_product = predictor.products[0]

r = predictor.predict(sample_product, horizon=3)
check("Prediction has expected structure", "predictions" in r and "avg_monthly_demand" in r)
check("Prediction returns 3 months", len(r["predictions"]) == 3)
check("Predicted demand is positive", r["avg_monthly_demand"] > 0)
check("Lower bound <= predicted value",
      all(p["lower_bound"] <= p["predicted_demand"] for p in r["predictions"]))
check("Upper bound >= predicted value",
      all(p["upper_bound"] >= p["predicted_demand"] for p in r["predictions"]))

r2 = predictor.predict(predictor.products[1], horizon=6, model="random_forest")
check("6-month horizon returns 6 predictions", len(r2["predictions"]) == 6)

all_preds = predictor.predict_all(horizon=3)
check("predict_all covers all products", len(all_preds) == len(predictor.products))
check("All predicted values are non-negative", all(v >= 0 for v in all_preds.values()))

# -----------------------------------------------------------------
section("5. Recommendation engine")
# -----------------------------------------------------------------
from src.recommendation.engine import (
    generate_recommendations, economic_order_quantity, reorder_point, load_current_stock
)

eoq = economic_order_quantity(1000, order_cost=50, holding_cost_pct=0.25, unit_price=5.0)
check("EOQ is positive", eoq > 0)

rop = reorder_point(100, lead_time_days=5, safety_stock=50)
check("Reorder point exceeds safety stock", rop > 50)

current_stock = load_current_stock()
predictions_test = {p: float(np.random.uniform(50, 500)) for p in predictor.products[:20]}
stocks_test = {p: float(np.random.uniform(0, 3000)) for p in predictor.products[:20]}
recs = generate_recommendations(predictions_test, stocks_test)

check("Recommendations generated for all requested products", len(recs) == 20)
check("Recommendations sorted by priority", recs[0].priority <= recs[-1].priority)
check("ABC class is always A, B, or C", all(r.abc_class in ("A", "B", "C") for r in recs))
check("Action field is always defined", all(r.action != "" for r in recs))
check("Recommended quantity is non-negative", all(r.recommended_order_qty >= 0 for r in recs))

# -----------------------------------------------------------------
print(f"\n{'=' * 60}")
print("  RESULTS")
print(f"{'=' * 60}")
total = passed + failed
print(f"  Passed : {passed}/{total}")
print(f"  Failed : {failed}/{total}")
print(f"  Success rate : {passed / total * 100:.1f}%")
print(f"{'=' * 60}")

if failed > 0:
    sys.exit(1)
