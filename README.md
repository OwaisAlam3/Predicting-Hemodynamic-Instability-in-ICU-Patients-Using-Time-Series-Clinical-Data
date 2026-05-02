# ICU HemoPredict — Full SaaS Dashboard

**Haemodynamic Instability Risk Prediction for ICU Patients**  
NEDUET × SIUT Karachi | CS Batch 2022 — Group 17  
Supervisor: Dr. Syed Zaffar Qasim | Industrial Advisor: Prof. Dr. Fakhir Raza Haidri

---

## Validated Performance (SIUT Karachi, n=253)

| Metric | 5-Fold CV | Held-out Test |
|---|---|---|
| AUROC | 0.9941 ± 0.0064 | 0.9928 |
| Sensitivity | 96.67% ± 1.86% | 97.22% |
| Specificity | 95.59% ± 4.87% | 95.24% |
| F1 Score | 0.9588 ± 0.0198 | 0.9589 |
| Brier Score | 0.0316 ± 0.0130 | 0.0346 |

---

## Project Structure

```
hemopredict/
  backend/
    main.py                     FastAPI backend — ensemble v3 + batch CSV endpoint
    Dockerfile
    requirements.txt
    *.pkl / *.npy               Model artifacts (copy from original deploy/)
  frontend/
    src/
      App.jsx                   Main app shell with routing
      components/Sidebar.jsx    Navigation + session history
      pages/Dashboard.jsx       Overview with stats and session table
      pages/PatientInput.jsx    Multi-slot vitals entry + CSV import
      pages/Results.jsx         Risk trajectory chart, SHAP, flags, reasoning
      pages/BatchAnalysis.jsx   CSV batch upload and results table
    Dockerfile                  Multi-stage: Node build + nginx serve
    nginx.conf
    package.json
  docker-compose.yml            Full stack: backend + frontend
  .env.example
```

---

## Option 1: Docker (Recommended — one command)

```bash
# 1. Copy model artifacts into backend/
cp /path/to/original/deploy/backend/*.pkl  backend/
cp /path/to/original/deploy/backend/*.npy  backend/

# 2. Start everything
docker-compose up --build

# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## Option 2: Local Dev (No Docker)

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install
REACT_APP_API_URL=http://localhost:8000 npm start
# Opens http://localhost:3000
```

---

## Option 3: Cloud Deploy (Free Tier)

### Backend → Render.com
1. Push repo to GitHub
2. New Web Service → select backend/ folder
3. Runtime: Python 3.11 | Build: `pip install -r requirements.txt`
4. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Copy the service URL (e.g. `https://hemopredict-api.onrender.com`)

### Frontend → Vercel / Netlify
1. New project → select frontend/ folder
2. Build command: `npm run build` | Output: `build`
3. Environment variable: `REACT_APP_API_URL=https://hemopredict-api.onrender.com`
4. Deploy

### Backend → Railway.app (Alternative)
```bash
cd backend
railway init
railway up
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | System metadata |
| `/health` | GET | Health check |
| `/predict/stay` | POST | Single patient multi-slot prediction |
| `/predict/batch-csv` | POST | Batch CSV upload — multiple patients |
| `/docs` | GET | Auto-generated Swagger UI |

---

## Features

- **React SaaS Dashboard** — dark, clinical aesthetic with Syne + DM Mono typography
- **Session History** — sidebar tracks up to 20 analyses per session
- **Risk Trajectory Chart** — AreaChart showing probability over time with threshold lines
- **Per-Slot Timeline** — individual slot risk cards with colour coding
- **SHAP Feature Attribution** — diverging bar chart with XGBoost TreeSHAP
- **Clinical Flags** — automated alerts for MAP, lactate, SpO₂, pH, etc.
- **Clinical Reasoning Panel** — plain-language interpretation of model output
- **Derived Haemodynamic Indices** — MAP, Shock Index, SF ratio, lactate, etc.
- **CSV Import** — import time-series vitals directly into the input form
- **Batch CSV Analysis** — upload multi-patient CSV, get risk table
- **Export** — download individual result as JSON

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
