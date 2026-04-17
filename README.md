# ICU-HemoPredict

**Haemodynamic Instability Risk Prediction for ICU Patients**

Final Year Design Project — Group 17, CS Batch 2022  
NED University of Engineering & Technology  
Industrial Partner: Sindh Institute of Urology & Transplantation (SIUT), Karachi

---

## Overview

ICU-HemoPredict is a clinical decision support system that predicts hemodynamic instability in ICU patients using 48-hour time-series vital signs and laboratory data. The system uses a trained XGBoost classifier with SHAP-based explainability to generate per-slot risk scores and a full clinical interpretation report.

**Validated performance (SIUT cohort, n=253 patients):**

| Metric | Value |
|---|---|
| AUROC | 0.9891 |
| Sensitivity | 98.38% |
| Specificity | 95.24% |
| PPV | 94.65% |
| NPV | 98.56% |
| Decision threshold | 47.74% |

---

## Project structure

```
icu_hemopredict/
  backend/
    main.py                  FastAPI backend with exact feature engineering
    xgb_final.pkl            Trained XGBoost model
    feature_cols.npy         Feature column order (140 features)
    clinical_threshold.npy   Decision threshold (0.4774)
    requirements.txt
  frontend/
    index.html               Standalone dashboard (no build step required)
  start.sh                   One-command startup script
  README.md
```

---

## Running locally (demo / poster presentation)

**Prerequisites:** Python 3.9+, pip

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Open the dashboard
# Open frontend/index.html in any browser
# The dashboard connects to http://localhost:8000 automatically
```

Or use the startup script (macOS/Linux):
```bash
chmod +x start.sh
./start.sh
```

**The dashboard will show "API connected" in green when the backend is running.**

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | System info and model metadata |
| `/health` | GET | Health check |
| `/predict/stay` | POST | Full 48-hour patient stay prediction |

### Sample request

```bash
curl -X POST http://localhost:8000/predict/stay \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      {"sbp":115,"dbp":72,"hr":88,"rr":18,"temp":37.2,"spo2":97,"fio2":21,
       "uop":65,"avpu":0,"lact":1.2,"ph":7.40,"hco3":24,"k":4.0,"na":137,
       "cre":0.9,"hct":34,"tlc":10,"plt":210,"bili":0.9,"crt":2},
      ...
    ],
    "age":62,"gender":1,"dm":1,"htn":1,"ckd":1,"ihd":0,"copd":0,
    "diagnosis":"sepsis_shock"
  }'
```

---

## Feature engineering

The model uses 140 features derived from 20 raw parameters:

- **Raw vitals & labs (22):** AVPU, Temp, HR, RR, SBP, DBP, MAP, PP, SpO2, FiO2, CRT, UOP, pH, HCO3, K, Na, Creatinine, HCT, TLC, Platelets, Bilirubin, Lactate
- **Rolling statistics (48):** 8-hour rolling mean, std, min, max for 12 parameters
- **Rate-of-change features (24):** 1-step and 2-step deltas for 12 parameters
- **Trend slopes (6):** Expanding linear slope for SBP, MAP, HR, Lactate, Creatinine, UOP
- **Composite signals (7):** Shock Index, SpO2/FiO2 ratio, HCO3-Lactate gap, PP variation, Renal stress index, MAP deficit, Respiratory load
- **Patient-level aggregates (14):** Worst-case values across the full stay
- **Metadata (17):** Age, gender, comorbidities (DM/HTN/CKD/IHD/COPD), diagnosis category

---

## Team

| Name | Roll No. |
|---|---|
| Muhammad Owais Alam | CS-22070 |
| Hussain Raza | CS-22082 |
| Mudasir Shaikh | CS-22135 |

**Supervisor:** Dr. Syed Zaffar Qasim  
**Industrial Advisor:** Prof. Dr. Fakhir Raza Haidri (SIUT)

---

## Disclaimer

This system is a research prototype validated on a single-centre cohort. It is intended as a clinical decision support tool and does not replace clinical judgment. Prospective multi-centre validation is required before clinical deployment.
