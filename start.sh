#!/bin/bash
# ICU-HemoPredict v3 — Startup Script
# Run from project root: ./start.sh

echo ""
echo "======================================"
echo "  ICU-HemoPredict v3"
echo "  Haemodynamic Instability Prediction"
echo "  NEDUET + SIUT — Group 17"
echo "======================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install from python.org"
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "ERROR: pip not found."
    exit 1
fi

# Install dependencies if needed
cd backend
echo "Checking dependencies..."
if ! python3 -c "import fastapi, uvicorn, xgboost, lightgbm, shap, sklearn" 2>/dev/null; then
    echo "Installing dependencies (first run only)..."
    pip3 install -r requirements.txt
    echo ""
fi

echo "Starting API server at http://localhost:8000"
echo ""
echo "Open frontend/index.html in Chrome or Firefox."
echo "The dashboard will show 'API Online' when connected."
echo ""
echo "Press Ctrl+C to stop."
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
