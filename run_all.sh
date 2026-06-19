#!/bin/bash
# =================================================================
#  PharmaPredict API - Full pipeline execution (Linux / macOS)
# =================================================================
set -e
cd "$(dirname "$0")"

echo "==============================================================="
echo "  PHARMAPREDICT API - FULL PIPELINE"
echo "==============================================================="

echo
echo "[1/6] Installing dependencies..."
echo "---------------------------------------------------------------"
pip install -r requirements.txt

echo
echo "[2/6] Loading and cleaning historical data..."
echo "---------------------------------------------------------------"
python -m src.data.load_data

echo
echo "[3/6] Building the ML training dataset..."
echo "---------------------------------------------------------------"
python -m src.features.build_features

echo
echo "[4/6] Training and evaluating models..."
echo "---------------------------------------------------------------"
python -m src.models.train

echo
echo "[5/6] Running the test suite..."
echo "---------------------------------------------------------------"
python -m tests.test_pipeline

echo
echo "[6/6] Generating the evaluation dashboard..."
echo "---------------------------------------------------------------"
python notebooks/evaluation.py

echo
echo "==============================================================="
echo "  PIPELINE COMPLETE"
echo "==============================================================="
echo
echo "  Results available at:"
echo "    data/processed/training_dataset.csv"
echo "    models/*.pkl + metadata.json"
echo "    docs/evaluation_dashboard.png"
echo
echo "  To start the API server, run:"
echo "    uvicorn src.api.main:app --reload --port 8000"
echo
echo "  Then open http://localhost:8000/docs"
echo
