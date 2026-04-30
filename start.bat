@echo off
echo.
echo ======================================
echo   ICU-HemoPredict v3
echo   Haemodynamic Instability Prediction
echo   NEDUET + SIUT - Group 17
echo ======================================
echo.

cd backend

echo Checking dependencies...
python -c "import fastapi, uvicorn, xgboost, lightgbm, shap, sklearn" 2>nul
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

echo Starting API at http://localhost:8000
echo Open frontend\index.html in Chrome or Firefox.
echo Press Ctrl+C to stop.
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000
