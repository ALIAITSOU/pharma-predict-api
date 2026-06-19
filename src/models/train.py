"""
=================================================================
 MODEL TRAINING — Train and evaluate demand forecasting models
=================================================================
Input  : data/processed/training_dataset.csv
Output : models/random_forest.pkl
         models/gradient_boosting.pkl
         models/ridge.pkl
         models/scaler.pkl
         models/metadata.json

Trains three regression models, evaluates them on a strict
time-based train/test split, and saves the best-performing
artifacts for use by the prediction API.
=================================================================
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURES = [
    "month", "quarter", "year", "months_since_start",
    "sin_month", "cos_month", "sin_quarter", "cos_quarter",
    "quantity_sold_lag1", "quantity_sold_lag2", "quantity_sold_lag3",
    "quantity_sold_lag6", "quantity_sold_lag12",
    "quantity_sold_ma3", "quantity_sold_ma6", "quantity_sold_ma12",
    "quantity_sold_std3", "quantity_sold_std6", "quantity_sold_std12",
    "stock_end_of_month", "stock_to_demand_ratio", "coverage_months",
    "class_code", "unit_price", "n_transactions", "n_cities",
]
TARGET = "quantity_sold"

TRAIN_YEARS_BEFORE = 2020   # train on 2017-2019, test on 2020
TEST_YEAR = 2020


def evaluate(model, X_test, y_test, name: str) -> dict:
    y_pred = np.maximum(0, model.predict(X_test))
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
    print(f"\n  -- {name} --")
    print(f"     MAE={mae:.1f}  RMSE={rmse:.1f}  R2={r2:.4f}  MAPE={mape:.2f}%")
    return {"mae": round(mae, 1), "rmse": round(rmse, 1),
            "r2": round(r2, 4), "mape": round(mape, 2)}


def run():
    print("Loading training dataset...")
    df = pd.read_csv(os.path.join(PROC_DIR, "training_dataset.csv"), parse_dates=["date"])
    print(f"  {len(df):,} rows | {df['Product Name'].nunique()} products")

    df_train = df[df["year"] < TEST_YEAR]
    df_test = df[df["year"] == TEST_YEAR]
    print(f"  Train: {len(df_train):,} (2017-2019) | Test: {len(df_test):,} ({TEST_YEAR})")

    feat_ok = [f for f in FEATURES if f in df.columns]
    missing = set(FEATURES) - set(feat_ok)
    if missing:
        print(f"  Warning - missing features ignored: {missing}")

    X_train, y_train = df_train[feat_ok].fillna(0), df_train[TARGET]
    X_test, y_test = df_test[feat_ok].fillna(0), df_test[TARGET]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                                min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    metrics_rf = evaluate(rf, X_test, y_test, "Random Forest")

    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                    max_depth=6, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    metrics_gb = evaluate(gb, X_test, y_test, "Gradient Boosting")

    print("\nTraining Ridge Regression (baseline)...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, y_train)
    metrics_ridge = evaluate(ridge, X_test_sc, y_test, "Ridge")

    importance = pd.Series(rf.feature_importances_, index=feat_ok).sort_values(ascending=False)
    print("\nTop 10 most important features (Random Forest):")
    for feat, val in importance.head(10).items():
        print(f"   {feat:<30} {val:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    joblib.dump(gb, os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
    joblib.dump(ridge, os.path.join(MODEL_DIR, "ridge.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    best_model = "gradient_boosting" if metrics_gb["r2"] >= metrics_rf["r2"] else "random_forest"
    metadata = {
        "data_source": "pharma_sales_history.csv (real transactional data)",
        "features": feat_ok,
        "target": TARGET,
        "train_period": "2017-2019",
        "test_period": str(TEST_YEAR),
        "train_size": len(df_train),
        "test_size": len(df_test),
        "n_products": int(df["Product Name"].nunique()),
        "metrics": {
            "random_forest": metrics_rf,
            "gradient_boosting": metrics_gb,
            "ridge": metrics_ridge,
        },
        "best_model": best_model,
        "feature_importance": importance.head(15).round(4).to_dict(),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nModels saved to {MODEL_DIR}/")
    print(f"Best model: {best_model} "
          f"(R2={max(metrics_rf['r2'], metrics_gb['r2'])})")


if __name__ == "__main__":
    run()
