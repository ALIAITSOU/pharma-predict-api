"""
=================================================================
 FASTAPI APPLICATION — Medication Demand Prediction
 & Restocking Recommendation API
=================================================================
Run with:
    uvicorn src.api.main:app --reload --port 8000

Interactive documentation once running:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
=================================================================
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import pandas as pd
import os

from src.models.predictor import DemandPredictor
from src.recommendation.engine import generate_recommendations, load_current_stock

# ---------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------
app = FastAPI(
    title="PharmaPredict API",
    description=(
        "API for medication demand forecasting and restocking "
        "recommendations, built on real pharmaceutical sales data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

predictor = DemandPredictor()

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

_stats_df = pd.read_csv(os.path.join(PROC_DIR, "product_stats.csv"))
PRODUCTS_INFO = {
    row["Product Name"]: {
        "product_id": row.get("product_id", "?"),
        "product_class": row.get("Product Class", "?"),
        "unit_price": round(float(row.get("unit_price", 0)), 2),
        "avg_monthly_demand": round(float(row.get("avg_monthly_demand", 0)), 0),
        "safety_stock": int(row.get("safety_stock", 100)),
    }
    for _, row in _stats_df.iterrows()
}
CURRENT_STOCK = load_current_stock()


# ---------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------
class BatchPredictionRequest(BaseModel):
    product_names: List[str] = Field(..., example=["Ionclotide", "Tetratanyl"])
    horizon_months: int = Field(3, ge=1, le=12)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health():
    """Basic liveness/readiness check."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "n_products": len(PRODUCTS_INFO),
        "models_loaded": ["random_forest", "gradient_boosting", "ridge"],
    }


@app.get("/products", tags=["Data"])
async def list_products(product_class: Optional[str] = None):
    """List all products covered by the system, optionally filtered by class."""
    data = [
        {"product_name": name, **info,
         "current_stock": round(CURRENT_STOCK.get(name, 0), 0)}
        for name, info in PRODUCTS_INFO.items()
        if product_class is None or info["product_class"] == product_class
    ]
    return sorted(data, key=lambda x: -x["avg_monthly_demand"])


@app.get("/classes", tags=["Data"])
async def list_classes():
    """List all therapeutic classes present in the dataset."""
    return sorted(_stats_df["Product Class"].unique().tolist())


@app.post("/predict/{product_name}", tags=["Prediction"])
async def predict(product_name: str,
                   horizon: int = Query(3, ge=1, le=12),
                   model: str = Query("gradient_boosting")):
    """Forecast monthly demand for a single product."""
    if product_name not in PRODUCTS_INFO:
        raise HTTPException(404, detail=f"Unknown product: '{product_name}'")
    result = predictor.predict(product_name, horizon, model)
    result["product_info"] = PRODUCTS_INFO[product_name]
    result["current_stock"] = round(CURRENT_STOCK.get(product_name, 0), 0)
    return result


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Forecast monthly demand for several products at once."""
    return {
        name: predictor.predict(name, request.horizon_months)
        for name in request.product_names if name in PRODUCTS_INFO
    }


@app.get("/recommend", tags=["Recommendation"])
async def recommend(urgency: Optional[str] = None,
                     product_class: Optional[str] = None,
                     top_n: Optional[int] = None):
    """Restocking recommendations for all products, with optional filters."""
    predictions = predictor.predict_all(horizon=3)
    recs = generate_recommendations(predictions, CURRENT_STOCK)

    data = []
    for r in recs:
        if urgency and r.urgency != urgency.upper():
            continue
        if product_class and r.product_class != product_class:
            continue
        data.append(vars(r))
    if top_n:
        data = data[:top_n]

    summary = {
        "critical": sum(1 for r in recs if r.urgency == "CRITICAL"),
        "high": sum(1 for r in recs if r.urgency == "HIGH"),
        "normal": sum(1 for r in recs if r.urgency == "NORMAL"),
        "surplus": sum(1 for r in recs if r.urgency == "SURPLUS"),
    }
    return {"timestamp": datetime.now().isoformat(), "n_products": len(data),
            "summary": summary, "recommendations": data}


@app.get("/recommend/{product_name}", tags=["Recommendation"])
async def recommend_single(product_name: str):
    """Detailed restocking recommendation for a single product."""
    if product_name not in PRODUCTS_INFO:
        raise HTTPException(404, detail=f"Unknown product: '{product_name}'")
    forecast = predictor.predict(product_name, horizon=3)
    predictions = {product_name: forecast["avg_monthly_demand"]}
    recs = generate_recommendations(predictions, {product_name: CURRENT_STOCK.get(product_name, 0)})
    result = vars(recs[0])
    result["detailed_forecast"] = forecast["predictions"]
    return result


@app.get("/metrics", tags=["Evaluation"])
async def metrics():
    """Model evaluation metrics (MAE, RMSE, R2, MAPE)."""
    return {
        "data_source": "pharma_sales_history.csv (real transactional data)",
        "metrics": predictor.metrics(),
        "dataset_info": {
            "n_transactions": 254082,
            "n_products": 240,
            "n_classes": 6,
            "period": "2017-2020",
            "countries": ["Poland", "Germany"],
        },
    }


@app.get("/dashboard", tags=["Summary"])
async def dashboard():
    """High-level synthetic view: urgency summary, alerts, stock value."""
    rec_data = await recommend()
    return {
        "timestamp": datetime.now().isoformat(),
        "stock_summary": rec_data["summary"],
        "critical_and_high_alerts": [
            r for r in rec_data["recommendations"] if r["urgency"] in ("CRITICAL", "HIGH")
        ],
        "total_stock_value": round(sum(
            CURRENT_STOCK.get(name, 0) * info["unit_price"]
            for name, info in PRODUCTS_INFO.items()
        ), 2),
    }
