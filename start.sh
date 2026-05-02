#!/bin/bash
# ICU HemoPredict — Quick Start
# Run from the hemopredict/ root directory

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   ICU HemoPredict v3 — Full Stack    ║"
echo "║   NEDUET × SIUT Karachi — Group 17   ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check if model artifacts exist
MISSING=()
for f in xgb_final_v3.pkl lgb_final_v3.pkl lr_final_v3.pkl scaler_v3.pkl \
          selected_features_v3.npy ensemble_weights_v3.npy ensemble_threshold_v3.npy; do
  if [ ! -f "backend/$f" ]; then
    MISSING+=($f)
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "⚠ Missing model artifacts in backend/:"
  for f in "${MISSING[@]}"; do echo "  - $f"; done
  echo ""
  echo "Copy .pkl and .npy files from the original deploy/backend/ into backend/"
  echo "Then run this script again."
  exit 1
fi

echo "✓ Model artifacts found"

# Start backend
echo "Starting backend on http://localhost:8000 ..."
cd backend
python3 -m pip install -r requirements.txt -q
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 4
if curl -s http://localhost:8000/health > /dev/null; then
  echo "✓ Backend online"
else
  echo "✗ Backend failed to start. Check logs."
  kill $BACKEND_PID 2>/dev/null
  exit 1
fi

# Start frontend
echo "Starting frontend on http://localhost:3000 ..."
cd frontend
npm install -q
REACT_APP_API_URL=http://localhost:8000 npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "══════════════════════════════════════"
echo "  Dashboard: http://localhost:3000"
echo "  API Docs:  http://localhost:8000/docs"
echo "══════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop all services."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
