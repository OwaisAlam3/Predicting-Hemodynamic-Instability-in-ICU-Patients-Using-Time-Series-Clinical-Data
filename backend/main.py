"""
ICU-HemoPredict — Backend API v3
Ensemble: XGBoost (0.2) + LightGBM (0.7) + Logistic Regression (0.1)
50 SHAP-selected features | Vitals time-series + Static labs at admission
Validated AUROC: 0.9941 ± 0.0064 (5-fold CV) | Sensitivity: 96.67%
SIUT Karachi cohort | n=253 patients

Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import numpy as np, pickle, shap, os, warnings
warnings.filterwarnings("ignore")

# ── Load artifacts ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    return pickle.load(open(os.path.join(BASE, name), "rb"))

XGB_MODEL       = load("xgb_final_v3.pkl")
LGB_MODEL       = load("lgb_final_v3.pkl")
LR_MODEL        = load("lr_final_v3.pkl")
SCALER          = load("scaler_v3.pkl")
SELECTED_FEATS  = np.load(os.path.join(BASE, "selected_features_v3.npy"), allow_pickle=True).tolist()
W_XGB, W_LGB, W_LR = np.load(os.path.join(BASE, "ensemble_weights_v3.npy"))
THRESHOLD       = float(np.load(os.path.join(BASE, "ensemble_threshold_v3.npy")))
EXPLAINER       = shap.TreeExplainer(XGB_MODEL)

print(f"[ICU-HemoPredict] Models loaded | Features: {len(SELECTED_FEATS)} | Threshold: {THRESHOLD}")
print(f"[ICU-HemoPredict] Weights: XGB={W_XGB}, LGB={W_LGB}, LR={W_LR}")

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="ICU-HemoPredict API",
    description="Haemodynamic instability risk prediction — Ensemble v3",
    version="3.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ────────────────────────────────────────────────────
class Reading(BaseModel):
    sbp:  float = Field(..., description="Systolic BP (mmHg)")
    dbp:  float = Field(..., description="Diastolic BP (mmHg)")
    hr:   float = Field(..., description="Heart rate (bpm)")
    rr:   float = Field(..., description="Respiratory rate (br/min)")
    temp: float = Field(..., description="Temperature (°C)")
    spo2: float = Field(..., description="SpO2 (%)")
    fio2: float = Field(..., description="FiO2 (%)")
    uop:  float = Field(..., description="Urine output (ml/hr)")
    avpu: int   = Field(..., description="AVPU score: A=0 V=1 P/S=2 U=3")
    lact: float = Field(..., description="Lactate (mmol/L)")
    ph:   float = Field(..., description="Arterial pH")
    hco3: float = Field(..., description="Serum HCO3 (mEq/L)")
    k:    float = Field(..., description="Serum K (mEq/L)")
    na:   float = Field(..., description="Serum Na (mEq/L)")
    cre:  float = Field(..., description="Serum creatinine (mg/dL)")
    hct:  float = Field(..., description="Haematocrit (%)")
    tlc:  float = Field(..., description="TLC (×10³/µL)")
    plt:  float = Field(..., description="Platelets (×10³/µL)")
    bili: float = Field(..., description="Total bilirubin (mg/dL)")
    crt:  float = Field(..., description="CRT (seconds)")

class PatientRequest(BaseModel):
    readings:   List[Reading] = Field(..., min_length=2, max_length=24)
    age:        float = Field(50,    description="Patient age")
    gender:     int   = Field(1,     description="1=male, 0=female")
    dm:         int   = Field(0,     description="Diabetes mellitus")
    htn:        int   = Field(0,     description="Hypertension")
    ckd:        int   = Field(0,     description="Chronic kidney disease")
    ihd:        int   = Field(0,     description="Ischaemic heart disease")
    copd:       int   = Field(0,     description="COPD")
    diagnosis:  str   = Field("other",
        description="respiratory | renal | sepsis_shock | gi_liver | cardiac | neuro | infection_other | other")

# ── Feature Engineering ────────────────────────────────────────
DIAG_CATS = ["cardiac","gi_liver","infection_other","neuro","other",
             "renal","respiratory","sepsis_shock"]
W = 4  # rolling window = 4 slots = 8 hours

def build_feature_matrix(req: PatientRequest) -> np.ndarray:
    """
    Builds the exact 50-feature matrix used during training.
    Patient-level aggregates are computed as expanding aggregates
    (only slots 0..i are used at slot i — no future leakage).
    Labs are static admission-baseline features (same value for all slots).
    """
    readings = req.readings
    n = len(readings)

    # Pre-compute per-slot derived values
    maps   = [(r.sbp + 2*r.dbp)/3 for r in readings]
    shocks = [min(r.hr / max(r.sbp, 1), 5) for r in readings]

    def h(attr):       return [getattr(r, attr) for r in readings]
    hr_h  = h("hr");   rr_h  = h("rr");   sbp_h = h("sbp");  dbp_h = h("dbp")
    spo2_h= h("spo2"); fio2_h= h("fio2"); uop_h = h("uop");  temp_h= h("temp")
    lact_h= h("lact"); ph_h  = h("ph");   cre_h = h("cre");  hco3_h= h("hco3")
    avpu_h= h("avpu")

    comorbid_count = req.dm + req.htn + req.ckd + req.ihd + req.copd
    no_comorbid    = int(comorbid_count == 0)
    diag_enc = {f"diag_{d}": int(req.diagnosis == d) for d in DIAG_CATS}
    if req.diagnosis not in DIAG_CATS:
        diag_enc["diag_other"] = 1

    rows = []
    for i, r in enumerate(readings):
        map_v = maps[i]
        shock = shocks[i]
        sf    = min(r.spo2 / max(r.fio2, 1), 10)
        hl    = r.hco3 - r.lact * 3
        map_def  = map_v - 65
        resp_load= min(r.fio2 / max(r.spo2, 1), 2.0)

        # ── Rolling window stats (no future — uses slots max(0,i-W+1)..i) ──
        ws = max(0, i + 1 - W)
        we = i + 1

        def rmean(arr): return float(np.mean(arr[ws:we]))
        def rstd(arr):  w=arr[ws:we]; return float(np.std(w)) if len(w)>1 else 0.0
        def rmin(arr):  return float(np.min(arr[ws:we]))
        def rmax(arr):  return float(np.max(arr[ws:we]))

        map_w = maps[ws:we]

        # ── Delta (rate of change) ─────────────────────────────
        def d1(arr): return float(arr[i]-arr[i-1]) if i>=1 else 0.0
        def d2(arr): return float(arr[i]-arr[i-2]) if i>=2 else 0.0

        # ── Expanding slope (uses all slots 0..i) ─────────────
        def slope(arr):
            vals = arr[:i+1]
            if len(vals) < 3: return 0.0
            x = np.arange(len(vals), dtype=float)
            return float(np.polyfit(x, vals, 1)[0])

        # ── Patient aggregates — EXPANDING (slots 0..i only) ──
        # This is the key: [:i+1] not the full array
        sbp_so  = sbp_h[:i+1];  map_so  = maps[:i+1]
        hr_so   = hr_h[:i+1];   lact_so = lact_h[:i+1]
        cre_so  = cre_h[:i+1];  fio2_so = fio2_h[:i+1]
        uop_so  = uop_h[:i+1];  rr_so   = rr_h[:i+1]
        avpu_so = avpu_h[:i+1]; shock_so= shocks[:i+1]
        spo2_so = spo2_h[:i+1]

        # ── Renal stress and pp_variation ─────────────────────
        renal_stress = d1(cre_h) - (d1(uop_h) / 50.0)
        pp_hist  = [readings[j].sbp - readings[j].dbp for j in range(ws, we)]
        pp_mean_w= float(np.mean(pp_hist)) if pp_hist else (r.sbp - r.dbp)
        pp_std_w = float(np.std(pp_hist))  if len(pp_hist) > 1 else 0.0
        pp_var   = pp_std_w / max(pp_mean_w, 1.0)

        lookup = {
            # ── Raw vitals / labs ──────────────────────────────
            "AVPU_score":        r.avpu,    "Temp":             r.temp,
            "HR (bpm)":          r.hr,      "RR (breaths/min)": r.rr,
            "SBP (mmHg)":        r.sbp,     "DBP (mmHg)":       r.dbp,
            "MAP":               map_v,      "Pulse pressure":   r.sbp - r.dbp,
            "SPO₂ (%)":          r.spo2,    "FiO2 (%)":         r.fio2,
            "CRT (seconds)":     r.crt,     "UOP":              r.uop,
            "Arterial pH":       r.ph,      "Serum HCO3":       r.hco3,
            "Serum K":           r.k,       "Serum Na":         r.na,
            "Serum Creatinine":  r.cre,     "HCT":              r.hct,
            "TLC":               r.tlc,     "Platelets":        r.plt,
            "Total bilirubin":   r.bili,    "Lactate":          r.lact,
            # ── Time ──────────────────────────────────────────
            "time_position":     i / 23.0,
            "day2_flag":         int(i >= 12),
            # ── Rolling stats ──────────────────────────────────
            "HR_roll_mean":   rmean(hr_h),   "HR_roll_std":   rstd(hr_h),
            "HR_roll_min":    rmin(hr_h),    "HR_roll_max":   rmax(hr_h),
            "RR_roll_mean":   rmean(rr_h),   "RR_roll_std":   rstd(rr_h),
            "RR_roll_min":    rmin(rr_h),    "RR_roll_max":   rmax(rr_h),
            "SBP_roll_mean":  rmean(sbp_h),  "SBP_roll_std":  rstd(sbp_h),
            "SBP_roll_min":   rmin(sbp_h),   "SBP_roll_max":  rmax(sbp_h),
            "DBP_roll_mean":  rmean(dbp_h),  "DBP_roll_std":  rstd(dbp_h),
            "DBP_roll_min":   rmin(dbp_h),   "DBP_roll_max":  rmax(dbp_h),
            "MAP_roll_mean":  float(np.mean(map_w)), "MAP_roll_std": float(np.std(map_w)) if len(map_w)>1 else 0.0,
            "MAP_roll_min":   float(np.min(map_w)),  "MAP_roll_max": float(np.max(map_w)),
            "SPO2_roll_mean": rmean(spo2_h), "SPO2_roll_std": rstd(spo2_h),
            "SPO2_roll_min":  rmin(spo2_h),  "SPO2_roll_max": rmax(spo2_h),
            "FiO2_roll_mean": rmean(fio2_h), "FiO2_roll_std": rstd(fio2_h),
            "FiO2_roll_min":  rmin(fio2_h),  "FiO2_roll_max": rmax(fio2_h),
            "UOP_roll_mean":  rmean(uop_h),  "UOP_roll_std":  rstd(uop_h),
            "UOP_roll_min":   rmin(uop_h),   "UOP_roll_max":  rmax(uop_h),
            "Temp_roll_mean": rmean(temp_h), "Temp_roll_std": rstd(temp_h),
            "Temp_roll_min":  rmin(temp_h),  "Temp_roll_max": rmax(temp_h),
            "Lactate_roll_mean": rmean(lact_h), "Lactate_roll_std": rstd(lact_h),
            "Lactate_roll_min":  rmin(lact_h),  "Lactate_roll_max": rmax(lact_h),
            "Arterial_roll_mean":rmean(ph_h),   "Arterial_roll_std":rstd(ph_h),
            "Arterial_roll_min": rmin(ph_h),    "Arterial_roll_max":rmax(ph_h),
            "Serum_roll_mean":   rmean(cre_h),  "Serum_roll_std":   rstd(cre_h),
            "Serum_roll_min":    rmin(cre_h),   "Serum_roll_max":   rmax(cre_h),
            # ── Delta features ─────────────────────────────────
            "hr_delta":      d1(hr_h),    "hr_delta2":       d2(hr_h),
            "rr_delta":      d1(rr_h),    "rr_delta2":       d2(rr_h),
            "sbp_delta":     d1(sbp_h),   "sbp_delta2":      d2(sbp_h),
            "map_delta":     float(maps[i]-maps[i-1]) if i>=1 else 0.0,
            "map_delta2":    float(maps[i]-maps[i-2]) if i>=2 else 0.0,
            "spo2_delta":    d1(spo2_h),  "spo2_delta2":     d2(spo2_h),
            "fio2_delta":    d1(fio2_h),  "fio2_delta2":     d2(fio2_h),
            "uop_delta":     d1(uop_h),   "uop_delta2":      d2(uop_h),
            "lactate_delta": d1(lact_h),  "lactate_delta2":  d2(lact_h),
            "ph_delta":      d1(ph_h),    "ph_delta2":       d2(ph_h),
            "creatinine_delta":d1(cre_h), "creatinine_delta2":d2(cre_h),
            "hco3_delta":    d1(hco3_h),  "hco3_delta2":     d2(hco3_h),
            "temp_delta":    d1(temp_h),  "temp_delta2":     d2(temp_h),
            # ── Slope features (expanding) ─────────────────────
            "SBP_slope":     slope(sbp_h), "MAP_slope":    slope(maps),
            "HR_slope":      slope(hr_h),  "Lactate_slope":slope(lact_h),
            "Serum_slope":   slope(cre_h), "UOP_slope":    slope(uop_h),
            "SPO2_slope":    slope(spo2_h),"FiO2_slope":   slope(fio2_h),
            # ── Composite signals ──────────────────────────────
            "shock_index":       shock,
            "spo2_fio2_ratio":   sf,
            "hco3_lactate_gap":  hl,
            "pp_variation":      pp_var,
            "renal_stress":      renal_stress,
            "map_deficit":       map_def,
            "resp_load":         resp_load,
            # ── Patient aggregates — EXPANDING (no leakage) ───
            "pt_sbp_mean":    float(np.mean(sbp_so)),
            "pt_sbp_min":     float(np.min(sbp_so)),
            "pt_map_mean":    float(np.mean(map_so)),
            "pt_map_min":     float(np.min(map_so)),
            "pt_hr_mean":     float(np.mean(hr_so)),
            "pt_hr_max":      float(np.max(hr_so)),
            "pt_lactate_max": float(np.max(lact_so)),
            "pt_lactate_mean":float(np.mean(lact_so)),
            "pt_creat_max":   float(np.max(cre_so)),
            "pt_avpu_max":    float(np.max(avpu_so)),
            "pt_fio2_max":    float(np.max(fio2_so)),
            "pt_uop_min":     float(np.min(uop_so)),
            "pt_rr_max":      float(np.max(rr_so)),
            "pt_shock_max":   float(np.max(shock_so)),
            "pt_spo2_min":    float(np.min(spo2_so)),
            # ── Static lab features ────────────────────────────
            "lab_Arterial pH":      r.ph,
            "lab_Serum HCO3":       r.hco3,
            "lab_Serum K":          r.k,
            "lab_Serum Na":         r.na,
            "lab_Serum Creatinine": r.cre,
            "lab_HCT":              r.hct,
            "lab_TLC":              r.tlc,
            "lab_Platelets":        r.plt,
            "lab_Total bilirubin":  r.bili,
            "lab_Lactate":          r.lact,
            # ── Composite from static labs ─────────────────────
            "hco3_lactate_gap":      r.hco3 - r.lact * 3,
            "renal_stress_static":   r.cre / max(r.na, 1),
            "acidosis_index":        r.ph - (r.hco3 / 24.0),
            # ── Metadata ──────────────────────────────────────
            "age":                   req.age,
            "gender_encoded":        req.gender,
            "comorbidity_count":     comorbid_count,
            "comorbid_dm":           req.dm,
            "comorbid_htn":          req.htn,
            "comorbid_ckd":          req.ckd,
            "comorbid_ihd":          req.ihd,
            "comorbid_copd":         req.copd,
            "comorbid_no_comorbidity": no_comorbid,
            **diag_enc,
        }

        rows.append([lookup.get(col, 0.0) for col in SELECTED_FEATS])

    return np.array(rows, dtype=np.float32)


def ensemble_predict(X: np.ndarray):
    """Run all three models and return weighted ensemble probabilities."""
    p_xgb = XGB_MODEL.predict_proba(X)[:, 1]
    p_lgb = LGB_MODEL.predict_proba(X)[:, 1]
    p_lr  = LR_MODEL.predict_proba(SCALER.transform(X))[:, 1]
    return W_XGB * p_xgb + W_LGB * p_lgb + W_LR * p_lr


def risk_label(prob: float) -> dict:
    if prob >= 0.70:
        return {"level": "CRITICAL", "color": "#ff4444",
                "message": "Critical risk — immediate haemodynamic intervention indicated"}
    elif prob >= THRESHOLD:
        return {"level": "HIGH",     "color": "#ff6b35",
                "message": "High risk — escalate monitoring, review vasopressor and fluid status"}
    elif prob >= 0.30:
        return {"level": "MODERATE", "color": "#ffc107",
                "message": "Moderate risk — increase observation frequency"}
    else:
        return {"level": "LOW",      "color": "#00d4aa",
                "message": "Low risk — continue standard monitoring protocol"}


def clinical_flags(req: PatientRequest, probs: list) -> list:
    readings = req.readings
    maps     = [(r.sbp + 2*r.dbp)/3 for r in readings]
    sis      = [r.hr / max(r.sbp, 1) for r in readings]
    flags    = []

    min_map   = min(maps);   min_map_slot = maps.index(min_map)
    peak_lact = max(r.lact for r in readings)
    peak_si   = max(sis)
    min_sbp   = min(r.sbp  for r in readings)
    min_spo2  = min(r.spo2 for r in readings)
    min_ph    = min(r.ph   for r in readings)
    max_cre   = max(r.cre  for r in readings)
    max_avpu  = max(r.avpu for r in readings)
    min_uop   = min(r.uop  for r in readings)
    trend     = "deteriorating" if probs[-1] > probs[0] + 0.10 else \
                "improving"     if probs[-1] < probs[0] - 0.10 else "stable"

    if min_map < 65:
        flags.append({"severity":"critical",
            "message":f"MAP nadir {min_map:.1f} mmHg at {min_map_slot*2}h — below critical perfusion threshold (65 mmHg). Vasopressor support and volume assessment indicated."})
    if peak_lact > 4.0:
        flags.append({"severity":"critical",
            "message":f"Lactate {peak_lact:.1f} mmol/L — severe hyperlactataemia. Urgent source control and resuscitation."})
    elif peak_lact > 2.0:
        flags.append({"severity":"warning",
            "message":f"Lactate {peak_lact:.1f} mmol/L — elevated above 2.0 mmol/L. Monitor trend for tissue hypoperfusion."})
    if peak_si > 1.2:
        flags.append({"severity":"critical",
            "message":f"Shock Index {peak_si:.2f} — significant haemodynamic compromise (HR/SBP > 1.2). Consider vasopressor support."})
    elif peak_si > 1.0:
        flags.append({"severity":"warning",
            "message":f"Shock Index {peak_si:.2f} — borderline haemodynamic stress. Increase cardiovascular monitoring."})
    if min_sbp < 90:
        flags.append({"severity":"critical",
            "message":f"SBP nadir {min_sbp:.0f} mmHg — documented hypotension. Reassess intravascular volume and vasomotor tone."})
    if min_spo2 < 90:
        flags.append({"severity":"critical",
            "message":f"SpO2 nadir {min_spo2:.0f}% — severe hypoxaemia. Escalate respiratory support."})
    elif min_spo2 < 94:
        flags.append({"severity":"warning",
            "message":f"SpO2 nadir {min_spo2:.0f}% — hypoxaemia. Review oxygenation strategy."})
    if min_ph < 7.25:
        flags.append({"severity":"critical",
            "message":f"Arterial pH nadir {min_ph:.2f} — significant acidaemia. Assess metabolic vs respiratory aetiology."})
    elif min_ph < 7.32:
        flags.append({"severity":"warning",
            "message":f"Arterial pH {min_ph:.2f} — mild-to-moderate acidosis. Monitor ABG trend."})
    if max_cre > 2.0:
        flags.append({"severity":"warning",
            "message":f"Creatinine peak {max_cre:.1f} mg/dL — AKI criteria met (KDIGO). Nephrology review advised."})
    if max_avpu >= 2:
        flags.append({"severity":"warning",
            "message":"Consciousness impaired (P/S or below). Assess for metabolic encephalopathy or CNS event."})
    if min_uop < 20:
        flags.append({"severity":"warning",
            "message":f"Urine output nadir {min_uop:.0f} ml/hr — oliguria. Consider fluid challenge vs diuretic resistance."})
    if trend == "deteriorating":
        flags.append({"severity":"warning",
            "message":"Progressive haemodynamic deterioration across observation window. Reassess current management plan."})
    return flags


def get_top_shap(shap_vals: np.ndarray, n: int = 10) -> list:
    top_idx = np.argsort(np.abs(shap_vals))[-n:][::-1]
    return [
        {"feature":    SELECTED_FEATS[i],
         "shap":       round(float(shap_vals[i]), 4),
         "direction":  "increases_risk" if shap_vals[i] > 0 else "decreases_risk"}
        for i in top_idx
    ]


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "system":           "ICU-HemoPredict",
        "version":          "3.0.0",
        "model":            "Ensemble (XGBoost + LightGBM + Logistic Regression)",
        "weights":          {"xgb": float(W_XGB), "lgb": float(W_LGB), "lr": float(W_LR)},
        "n_features":       len(SELECTED_FEATS),
        "threshold":        THRESHOLD,
        "validation_auroc": "0.9941 ± 0.0064 (5-fold CV)",
        "sensitivity":      "96.67% ± 1.86% (5-fold CV)",
        "cohort":           "SIUT Karachi — n=253 patients",
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "models_loaded": True,
        "n_features":   len(SELECTED_FEATS),
        "threshold":    THRESHOLD,
    }


@app.post("/predict/stay")
def predict_stay(req: PatientRequest):
    """
    Predict haemodynamic instability risk across a patient's ICU stay.
    Accepts 2 to 24 time-point readings (2-hour intervals).
    Returns per-slot risk scores, SHAP attributions, clinical flags, and derived indices.
    """
    try:
        X          = build_feature_matrix(req)
        probs      = ensemble_predict(X).tolist()
        shap_vals  = EXPLAINER.shap_values(X)

        # Per-slot results
        slots = []
        for i, prob in enumerate(probs):
            risk = risk_label(prob)
            sv   = shap_vals[i]
            slots.append({
                "slot":         i,
                "time_hrs":     i * 2,
                "probability":  round(prob * 100, 1),
                "risk":         risk,
                "top_features": get_top_shap(sv, n=8),
            })

        # Summary
        peak_prob  = max(probs)
        peak_slot  = probs.index(peak_prob)
        high_slots = sum(1 for p in probs if p >= THRESHOLD)
        trend      = ("deteriorating" if probs[-1] > probs[0] + 0.10
                      else "improving" if probs[-1] < probs[0] - 0.10
                      else "stable")

        # Global SHAP over full stay
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_global_idx = np.argsort(mean_abs_shap)[-12:][::-1]
        shap_global = [
            {"feature":       SELECTED_FEATS[j],
             "mean_abs_shap": round(float(mean_abs_shap[j]), 4),
             "direction":     "increases_risk" if shap_vals[:, j].mean() > 0 else "decreases_risk"}
            for j in top_global_idx
        ]

        # Derived haemodynamic indices
        readings = req.readings
        maps     = [(r.sbp + 2*r.dbp)/3 for r in readings]
        sis      = [r.hr / max(r.sbp, 1) for r in readings]
        sfs      = [r.spo2 / max(r.fio2, 1) for r in readings]
        derived  = {
            "mean_map":          round(float(np.mean(maps)), 1),
            "min_map":           round(float(np.min(maps)), 1),
            "peak_shock_index":  round(max(sis), 3),
            "peak_lactate":      round(max(r.lact for r in readings), 2),
            "min_spo2":          round(min(r.spo2 for r in readings), 1),
            "min_sf_ratio":      round(min(sfs), 3),
            "peak_creatinine":   round(max(r.cre for r in readings), 2),
            "min_uop":           round(min(r.uop for r in readings), 1),
        }

        return {
            "summary": {
                "peak_probability":   round(peak_prob * 100, 1),
                "peak_risk":          risk_label(peak_prob),
                "peak_slot":          peak_slot,
                "peak_time_hrs":      peak_slot * 2,
                "high_risk_slots":    high_slots,
                "total_slots":        len(probs),
                "trend":              trend,
                "decision_threshold": round(THRESHOLD * 100, 1),
            },
            "slots":       slots,
            "shap_global": shap_global,
            "derived":     derived,
            "flags":       clinical_flags(req, probs),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
