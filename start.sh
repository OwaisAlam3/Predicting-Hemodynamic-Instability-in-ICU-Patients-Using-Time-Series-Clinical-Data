#!/bin/bash
# ICU-HemoPredict — startup script
# Run this from the project root directory

echo ""
echo "ICU-HemoPredict"
echo "Haemodynamic Instability Risk Prediction System"
echo "SIUT / NEDUET Group 17 — CS Batch 2022"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install from python.org"
    exit 1
fi

# Install dependencies if needed
cd backend
if ! python3 -c "import fastapi, uvicorn, xgboost, shap" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting API server at http://localhost:8000"
echo "Open frontend/index.html in your browser"
echo ""
echo "Press Ctrl+C to stop."
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
