"""
=================================================================
 PREDICTOR — Bridge between trained ML models and the API layer
=================================================================
Loads the trained models and produces forward-looking demand
forecasts for any product, for a configurable horizon (in months).
=================================================================
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from dateutil.relativedelta import relativedelta

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    METADATA = json.load(f)
FEATURES = METADATA["features"]


class DemandPredictor:
    """Loads trained models once and serves demand forecasts."""

    def __init__(self):
        self.rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
        self.gb = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
        self.ridge = joblib.load(os.path.join(MODEL_DIR, "ridge.pkl"))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.stats = pd.read_csv(
            os.path.join(PROC_DIR, "product_stats.csv")
        ).set_index("Product Name")
        self.df = pd.read_csv(
            os.path.join(PROC_DIR, "training_dataset.csv"), parse_dates=["date"]
        )
        self.products = sorted(self.df["Product Name"].unique())
        print(f"DemandPredictor ready - {len(self.products)} products loaded")

    def _build_future_features(self, product_name: str, horizon: int) -> pd.DataFrame:
        history = self.df[self.df["Product Name"] == product_name].sort_values("date")
        if history.empty:
            raise ValueError(f"Unknown product: '{product_name}'")

        last = history.iloc[-1]
        ref_date = pd.Timestamp(last["date"])

        rows = []
        for i in range(horizon):
            d = ref_date + relativedelta(months=i + 1)
            rows.append({
                "month": d.month,
                "quarter": (d.month - 1) // 3 + 1,
                "year": d.year,
                "months_since_start": int(last.get("months_since_start", 0)) + i + 1,
                "sin_month": np.sin(2 * np.pi * d.month / 12),
                "cos_month": np.cos(2 * np.pi * d.month / 12),
                "sin_quarter": np.sin(2 * np.pi * ((d.month - 1) // 3 + 1) / 4),
                "cos_quarter": np.cos(2 * np.pi * ((d.month - 1) // 3 + 1) / 4),
                "quantity_sold_lag1": float(last.get("quantity_sold", 0)),
                "quantity_sold_lag2": float(last.get("quantity_sold_lag1", 0)),
                "quantity_sold_lag3": float(last.get("quantity_sold_lag2", 0)),
                "quantity_sold_lag6": float(last.get("quantity_sold_lag3", 0)),
                "quantity_sold_lag12": float(last.get("quantity_sold_lag6", 0)),
                "quantity_sold_ma3": float(last.get("quantity_sold_ma3", 0)),
                "quantity_sold_ma6": float(last.get("quantity_sold_ma6", 0)),
                "quantity_sold_ma12": float(last.get("quantity_sold_ma12", 0)),
                "quantity_sold_std3": float(last.get("quantity_sold_std3", 0)),
                "quantity_sold_std6": float(last.get("quantity_sold_std6", 0)),
                "quantity_sold_std12": float(last.get("quantity_sold_std12", 0)),
                "stock_end_of_month": float(last.get("stock_end_of_month", 0)),
                "stock_to_demand_ratio": float(last.get("stock_to_demand_ratio", 1)),
                "coverage_months": float(last.get("coverage_months", 1)),
                "class_code": float(last.get("class_code", 0)),
                "unit_price": float(last.get("unit_price", 400)),
                "n_transactions": float(last.get("n_transactions", 10)),
                "n_cities": float(last.get("n_cities", 5)),
                "_date": d,
            })
        return pd.DataFrame(rows)

    def predict(self, product_name: str, horizon: int = 3,
                model: str = "gradient_boosting") -> dict:
        """Predict monthly demand for `product_name` over `horizon` months."""
        future = self._build_future_features(product_name, horizon)
        dates = future["_date"].tolist()
        X = future[[f for f in FEATURES if f in future.columns]].fillna(0)

        if model == "random_forest":
            preds = np.maximum(0, self.rf.predict(X))
        elif model == "gradient_boosting":
            preds = np.maximum(0, self.gb.predict(X))
        elif model == "ensemble":
            preds = 0.5 * np.maximum(0, self.rf.predict(X)) + \
                     0.5 * np.maximum(0, self.gb.predict(X))
        else:
            raise ValueError(f"Unknown model: '{model}'")

        std = float(self.stats.loc[product_name, "std_monthly_demand"]) \
            if product_name in self.stats.index else float(np.std(preds)) * 0.3

        return {
            "product_name": product_name,
            "product_class": self.df.loc[
                self.df["Product Name"] == product_name, "Product Class"
            ].iloc[0],
            "model_used": model,
            "horizon_months": horizon,
            "predictions": [
                {
                    "month_index": i + 1,
                    "period": d.strftime("%Y-%m"),
                    "predicted_demand": round(float(p), 0),
                    "lower_bound": round(max(0, float(p) - 1.5 * std), 0),
                    "upper_bound": round(float(p) + 1.5 * std, 0),
                }
                for i, (p, d) in enumerate(zip(preds, dates))
            ],
            "total_demand_period": round(float(preds.sum()), 0),
            "avg_monthly_demand": round(float(preds.mean()), 0),
        }

    def predict_all(self, horizon: int = 3) -> dict:
        """Predict average monthly demand for every known product."""
        results = {}
        for product in self.products:
            try:
                r = self.predict(product, horizon)
                results[product] = r["avg_monthly_demand"]
            except Exception:
                if product in self.stats.index:
                    results[product] = float(self.stats.loc[product, "avg_monthly_demand"])
        return results

    def metrics(self) -> dict:
        return METADATA["metrics"]


if __name__ == "__main__":
    predictor = DemandPredictor()
    sample = predictor.predict(predictor.products[0], horizon=3)
    print(json.dumps(sample, indent=2, ensure_ascii=False))
