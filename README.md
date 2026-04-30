# ICU-HemoPredict v3

**Haemodynamic Instability Risk Prediction for ICU Patients**

NED University of Engineering & Technology — Group 17, CS Batch 2022  
Industrial Partner: Sindh Institute of Urology & Transplantation (SIUT), Karachi  
Supervisor: Dr. Syed Zaffar Qasim | Industrial Advisor: Prof. Dr. Fakhir Raza Haidri

---

## Validated Performance (SIUT cohort, n=253 patients)

| Metric | 5-Fold CV | Held-out Test |
|---|---|---|
| AUROC | 0.9941 ± 0.0064 | 0.9928 |
| Sensitivity | 96.67% ± 1.86% | 97.22% |
| Specificity | 95.59% ± 4.87% | 95.24% |
| F1 Score | 0.9588 ± 0.0198 | 0.9589 |
| Brier Score | 0.0316 ± 0.0130 | 0.0346 |

---

## Project structure

```
icu_hemopredict/
  backend/
    main.py                      FastAPI backend — v3 ensemble
    xgb_final_v3.pkl             XGBoost model (weight=0.2)
    lgb_final_v3.pkl             LightGBM model (weight=0.7)
    lr_final_v3.pkl              Logistic Regression (weight=0.1)
    scaler_v3.pkl                StandardScaler for LR
    selected_features_v3.npy    50 SHAP-selected feature names
    ensemble_weights_v3.npy     [0.2, 0.7, 0.1]
    ensemble_threshold_v3.npy   0.5 (decision threshold)
    requirements.txt
  frontend/
    index.html                   Clinical dashboard (standalone HTML)
  start.sh                       Linux/macOS startup script
  start.bat                      Windows startup script
  README.md
```

---

## Running locally

**Requirements:** Python 3.9+

### Linux / macOS
```bash
chmod +x start.sh
./start.sh
```

### Windows
```
Double-click start.bat
OR
cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `frontend/index.html` in Chrome or Firefox.  
The dashboard shows **API Online** in green when connected.

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | System info and model metadata |
| `/health` | GET | Health check |
| `/predict/stay` | POST | Full patient stay prediction |

### Example request
```bash
curl -X POST http://localhost:8000/predict/stay \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      {"sbp":115,"dbp":72,"hr":88,"rr":18,"temp":37.2,"spo2":97,
       "fio2":21,"uop":65,"avpu":0,"lact":1.2,"ph":7.40,"hco3":24,
       "k":4.0,"na":137,"cre":0.9,"hct":34,"tlc":10,"plt":210,
       "bili":0.9,"crt":2},
      ...
    ],
    "age":62, "gender":1, "dm":1, "htn":1, "ckd":1,
    "ihd":0, "copd":0, "diagnosis":"sepsis_shock"
  }'
```

---

## Model architecture

**Ensemble:** XGBoost (w=0.2) + LightGBM (w=0.7) + Logistic Regression (w=0.1)  
**Features:** 50 selected from 125 engineered features via SHAP importance ranking  
**Input:** 2–24 time-point readings (2-hour intervals over 48 hours)  
**Decision threshold:** 0.50  

Feature groups:
- Vital sign rolling statistics (8-hour window): mean, std, min, max
- Rate-of-change delta features (1-step and 2-step)
- Cumulative trend slopes
- Composite clinical signals (Shock Index, SpO2/FiO2 ratio, HCO3-Lactate gap etc.)
- Patient-level expanding aggregates (no future leakage — computed from slots 0..i only)
- Static lab values at admission
- Patient metadata (age, gender, comorbidities, diagnosis category)

---

## Team

| Name | Roll No. |
|---|---|
| Muhammad Owais Alam | CS-22070 |
| Hussain Raza | CS-22082 |
| Mudasir Shaikh | CS-22135 |

---

## Disclaimer

Research prototype validated on a single-centre retrospective cohort.  
Clinical decision support tool only — does not replace clinical judgment.  
Prospective multi-centre validation required before clinical deployment.
