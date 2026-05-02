import React, { useState, useRef } from 'react';
import { Plus, Trash2, Play, Upload, ChevronDown, ChevronUp, AlertCircle, Info, FlaskConical } from 'lucide-react';
import Papa from 'papaparse';
 
const VITALS = [
  { key:'sbp',  label:'SBP',   unit:'mmHg',      min:40,   max:250,  step:1,   def:120, group:'Haemodynamics' },
  { key:'dbp',  label:'DBP',   unit:'mmHg',      min:20,   max:160,  step:1,   def:75,  group:'Haemodynamics' },
  { key:'hr',   label:'HR',    unit:'bpm',        min:20,   max:250,  step:1,   def:85,  group:'Haemodynamics' },
  { key:'rr',   label:'RR',    unit:'br/min',     min:4,    max:60,   step:1,   def:18,  group:'Respiratory' },
  { key:'temp', label:'Temp',  unit:'°C',         min:32,   max:42,   step:0.1, def:37.0,group:'General' },
  { key:'spo2', label:'SpO₂',  unit:'%',          min:50,   max:100,  step:0.1, def:97,  group:'Respiratory' },
  { key:'fio2', label:'FiO₂',  unit:'%',          min:21,   max:100,  step:1,   def:21,  group:'Respiratory' },
  { key:'uop',  label:'UOP',   unit:'ml/hr',      min:0,    max:500,  step:1,   def:60,  group:'Renal' },
  { key:'avpu', label:'AVPU',  unit:'0-3',        min:0,    max:3,    step:1,   def:0,   group:'Neuro' },
  { key:'lact', label:'Lact',  unit:'mmol/L',     min:0,    max:20,   step:0.1, def:1.0, group:'Metabolic' },
  { key:'ph',   label:'pH',    unit:'',           min:6.8,  max:7.8,  step:0.01,def:7.4, group:'Metabolic' },
  { key:'hco3', label:'HCO₃',  unit:'mEq/L',     min:5,    max:45,   step:0.5, def:24,  group:'Metabolic' },
  { key:'k',    label:'K⁺',    unit:'mEq/L',     min:1.5,  max:8,    step:0.1, def:4.0, group:'Labs' },
  { key:'na',   label:'Na⁺',   unit:'mEq/L',     min:110,  max:170,  step:1,   def:138, group:'Labs' },
  { key:'cre',  label:'Cre',   unit:'mg/dL',     min:0.1,  max:20,   step:0.1, def:0.9, group:'Renal' },
  { key:'hct',  label:'Hct',   unit:'%',          min:10,   max:65,   step:0.5, def:36,  group:'Labs' },
  { key:'tlc',  label:'TLC',   unit:'×10³',       min:1,    max:50,   step:0.5, def:10,  group:'Labs' },
  { key:'plt',  label:'Plt',   unit:'×10³',       min:10,   max:700,  step:5,   def:210, group:'Labs' },
  { key:'bili', label:'Bili',  unit:'mg/dL',     min:0.1,  max:30,   step:0.1, def:0.8, group:'Labs' },
  { key:'crt',  label:'CRT',   unit:'sec',        min:0,    max:10,   step:0.5, def:2,   group:'Haemodynamics' },
];
 
const DIAGNOSES = [
  { value:'sepsis_shock',      label:'Sepsis / Septic Shock' },
  { value:'respiratory',       label:'Respiratory Failure' },
  { value:'renal',             label:'Renal Failure / AKI' },
  { value:'cardiac',           label:'Cardiac Emergency' },
  { value:'gi_liver',          label:'GI / Hepatic' },
  { value:'neuro',             label:'Neurological' },
  { value:'infection_other',   label:'Other Infection' },
  { value:'other',             label:'Other / Unknown' },
];
 
// ── Demo datasets ───────────────────────────────────────────────
// Stable: normal vitals, no deterioration → ~1.2% across all slots
const DEMO_STABLE = {
  meta: { name: 'Demo — Stable Patient', age: 45, gender: 1, dm: 0, htn: 0, ckd: 0, ihd: 0, copd: 0, diagnosis: 'other' },
  slots: [
    { sbp:122, dbp:76, hr:72, rr:14, temp:36.8, spo2:98, fio2:21, uop:65, avpu:0, lact:0.8, ph:7.42, hco3:24, k:4.0, na:138, cre:0.8, hct:38, tlc:7,  plt:220, bili:0.6, crt:1 },
    { sbp:120, dbp:74, hr:74, rr:14, temp:36.7, spo2:98, fio2:21, uop:62, avpu:0, lact:0.7, ph:7.41, hco3:24, k:4.1, na:137, cre:0.8, hct:38, tlc:7,  plt:218, bili:0.6, crt:1 },
    { sbp:121, dbp:75, hr:73, rr:15, temp:36.8, spo2:99, fio2:21, uop:64, avpu:0, lact:0.8, ph:7.42, hco3:25, k:4.0, na:138, cre:0.8, hct:37, tlc:7,  plt:215, bili:0.6, crt:1 },
  ],
};
 
// Early-warning progressive sepsis: looks manageable at T+0h to T+6h,
// model catches instability at T+8h before it becomes clinically obvious.
// Scores: 2.3% → 5.4% → 11.1% → 13.1% → 98.6% → 98.6%
// Key demo narrative: at T+0h–T+6h a doctor may not escalate,
// but the trend (falling MAP, rising lactate, dropping SpO2, rising FiO2 demand)
// shows the system predicting the crash before it happens.
const DEMO_EARLY_WARNING = {
  meta: { name: 'Demo — Early Warning', age: 55, gender: 1, dm: 1, htn: 1, ckd: 0, ihd: 0, copd: 0, diagnosis: 'sepsis_shock' },
  slots: [
    { sbp:112, dbp:70, hr:92,  rr:18, temp:37.8, spo2:95, fio2:21, uop:45, avpu:0, lact:1.9, ph:7.36, hco3:21, k:3.8, na:137, cre:1.1, hct:34, tlc:11, plt:180, bili:0.9, crt:2 },
    { sbp:106, dbp:66, hr:97,  rr:20, temp:38.1, spo2:93, fio2:28, uop:36, avpu:0, lact:2.4, ph:7.33, hco3:19, k:3.6, na:136, cre:1.3, hct:33, tlc:13, plt:165, bili:1.1, crt:2 },
    { sbp:100, dbp:62, hr:103, rr:22, temp:38.4, spo2:91, fio2:35, uop:27, avpu:1, lact:2.9, ph:7.30, hco3:17, k:3.4, na:135, cre:1.6, hct:31, tlc:15, plt:150, bili:1.3, crt:3 },
    { sbp:94,  dbp:58, hr:110, rr:25, temp:38.7, spo2:89, fio2:45, uop:19, avpu:1, lact:3.5, ph:7.27, hco3:15, k:3.3, na:134, cre:2.0, hct:30, tlc:17, plt:135, bili:1.6, crt:4 },
    { sbp:88,  dbp:54, hr:117, rr:28, temp:39.0, spo2:87, fio2:55, uop:13, avpu:2, lact:4.2, ph:7.24, hco3:13, k:3.1, na:132, cre:2.5, hct:28, tlc:19, plt:118, bili:1.9, crt:4 },
    { sbp:82,  dbp:50, hr:124, rr:31, temp:39.2, spo2:85, fio2:65, uop:8,  avpu:2, lact:5.1, ph:7.20, hco3:11, k:2.9, na:130, cre:3.0, hct:27, tlc:21, plt:100, bili:2.3, crt:5 },
  ],
};
 
function makeSlot() {
  return VITALS.reduce((a, v) => ({ ...a, [v.key]: v.def }), {});
}
 
const GROUPS = [...new Set(VITALS.map(v => v.group))];
const GROUP_COLORS = {
  'Haemodynamics': 'var(--accent)',
  'Respiratory':   'var(--green)',
  'Renal':         'var(--purple)',
  'Neuro':         'var(--yellow)',
  'Metabolic':     'var(--orange)',
  'Labs':          'var(--text2)',
  'General':       'var(--text3)',
};
 
export default function PatientInput({ onResult, apiBase }) {
  const [meta, setMeta] = useState({
    name: '', id: '', age: 50, gender: 1,
    dm: 0, htn: 0, ckd: 0, ihd: 0, copd: 0,
    diagnosis: 'other',
  });
  const [slots, setSlots] = useState([makeSlot(), makeSlot(), makeSlot()]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showMeta, setShowMeta] = useState(true);
  const fileRef = useRef();
 
  const loadDemo = (demo) => {
    setMeta(m => ({ ...m, ...demo.meta }));
    setSlots(demo.slots.map(s => ({ ...s })));
    setError('');
  };
 
  const addSlot = () => setSlots(s => [...s, makeSlot()]);
  const removeSlot = (i) => setSlots(s => s.filter((_, idx) => idx !== i));
  const updateSlot = (i, key, val) => setSlots(s => s.map((slot, idx) => idx === i ? { ...slot, [key]: parseFloat(val) } : slot));
 
  const handleCSVImport = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    Papa.parse(file, {
      header: true, skipEmptyLines: true,
      complete: (res) => {
        const rows = res.data;
        const newSlots = rows.map(row => {
          const s = makeSlot();
          VITALS.forEach(v => {
            if (row[v.key] !== undefined) s[v.key] = parseFloat(row[v.key]) || v.def;
          });
          return s;
        });
        if (newSlots.length >= 2) setSlots(newSlots);
        else setError('CSV needs at least 2 rows (time slots)');
      }
    });
  };
 
  const handleSubmit = async () => {
    if (slots.length < 2) { setError('Need at least 2 time slots.'); return; }
    setError('');
    setLoading(true);
    try {
      const payload = {
        readings: slots.map(s => ({
          sbp:  s.sbp,  dbp: s.dbp,  hr:   s.hr,   rr:  s.rr,
          temp: s.temp, spo2:s.spo2, fio2: s.fio2, uop: s.uop,
          avpu: parseInt(s.avpu), lact: s.lact, ph: s.ph, hco3: s.hco3,
          k: s.k, na: s.na, cre: s.cre, hct: s.hct,
          tlc: s.tlc, plt: s.plt, bili: s.bili, crt: s.crt,
        })),
        age:       meta.age,
        gender:    meta.gender,
        dm:        meta.dm,
        htn:       meta.htn,
        ckd:       meta.ckd,
        ihd:       meta.ihd,
        copd:      meta.copd,
        diagnosis: meta.diagnosis,
      };
      const res = await fetch(`${apiBase}/predict/stay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API error');
      }
      const data = await res.json();
      onResult(data, { ...meta });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };
 
  return (
    <div style={{ padding: '28px 36px', maxWidth: 1080 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 22, marginBottom: 4 }}>
          New Patient Analysis
        </h2>
        <p style={{ color: 'var(--text2)', fontSize: 13 }}>
          Enter patient metadata and 2–24 time-point readings at 2-hour intervals.
        </p>
      </div>
 
      {/* Demo load buttons */}
      <div style={{
        display: 'flex', gap: 10, marginBottom: 18,
        padding: '12px 16px',
        background: 'rgba(0,212,255,0.04)',
        border: '1px solid rgba(0,212,255,0.15)',
        borderRadius: 'var(--radius)',
        alignItems: 'center',
        flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 6 }}>
          <FlaskConical size={13} color="var(--accent)" />
          <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', letterSpacing: '0.5px' }}>
            DEMO DATA
          </span>
        </div>
        <button
          onClick={() => loadDemo(DEMO_STABLE)}
          style={{
            padding: '6px 14px', fontSize: 12, fontWeight: 600,
            background: 'rgba(0,212,170,0.10)',
            border: '1px solid rgba(0,212,170,0.35)',
            color: 'var(--green)',
            borderRadius: 20,
            transition: 'var(--transition)',
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(0,212,170,0.18)'}
          onMouseLeave={e => e.currentTarget.style.background = 'rgba(0,212,170,0.10)'}
        >
          ✓ Load Stable Patient
        </button>
        <button
          onClick={() => loadDemo(DEMO_EARLY_WARNING)}
          style={{
            padding: '6px 14px', fontSize: 12, fontWeight: 600,
            background: 'rgba(255,107,53,0.10)',
            border: '1px solid rgba(255,107,53,0.35)',
            color: 'var(--orange)',
            borderRadius: 20,
            transition: 'var(--transition)',
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,107,53,0.18)'}
          onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,107,53,0.10)'}
        >
          ⚠ Load Early-Warning Patient
        </button>
        <span style={{ fontSize: 11, color: 'var(--text3)', marginLeft: 4 }}>
          Pre-filled with validated demo cases — click Run Analysis directly
        </span>
      </div>
 
      {/* Patient metadata panel */}
      <div style={{
        background: 'var(--bg2)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)', marginBottom: 16, overflow: 'hidden',
      }}>
        <button
          onClick={() => setShowMeta(m => !m)}
          style={{
            width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '14px 20px', background: 'transparent', color: 'var(--text1)',
            fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 13,
          }}
        >
          <span>Patient Metadata</span>
          {showMeta ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </button>
 
        {showMeta && (
          <div style={{ padding: '0 20px 20px', borderTop: '1px solid var(--border)' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16, paddingTop: 16 }}>
              <Field label="Patient Name / ID">
                <input
                  value={meta.name}
                  onChange={e => setMeta(m => ({ ...m, name: e.target.value }))}
                  placeholder="e.g. John Doe / PT-0042"
                  style={inputStyle}
                />
              </Field>
              <Field label="Age">
                <input type="number" value={meta.age} min={0} max={120}
                  onChange={e => setMeta(m => ({ ...m, age: +e.target.value }))}
                  style={inputStyle}
                />
              </Field>
              <Field label="Gender">
                <select value={meta.gender} onChange={e => setMeta(m => ({ ...m, gender: +e.target.value }))} style={inputStyle}>
                  <option value={1}>Male</option>
                  <option value={0}>Female</option>
                </select>
              </Field>
              <Field label="Diagnosis Category">
                <select value={meta.diagnosis} onChange={e => setMeta(m => ({ ...m, diagnosis: e.target.value }))} style={inputStyle}>
                  {DIAGNOSES.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                </select>
              </Field>
            </div>
 
            <div style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginBottom: 10, letterSpacing: '0.5px' }}>
              COMORBIDITIES
            </div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {[['dm','Diabetes'],['htn','Hypertension'],['ckd','CKD'],['ihd','IHD'],['copd','COPD']].map(([k,l]) => (
                <button
                  key={k}
                  onClick={() => setMeta(m => ({ ...m, [k]: m[k] ? 0 : 1 }))}
                  style={{
                    padding: '6px 14px',
                    borderRadius: 20,
                    fontSize: 12, fontWeight: 600,
                    background: meta[k] ? 'rgba(0,212,255,0.12)' : 'var(--bg3)',
                    border: `1px solid ${meta[k] ? 'rgba(0,212,255,0.4)' : 'var(--border)'}`,
                    color: meta[k] ? 'var(--accent)' : 'var(--text2)',
                    transition: 'var(--transition)',
                  }}
                >
                  {l} {meta[k] ? '✓' : ''}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
 
      {/* Time slot data entry */}
      <div style={{
        background: 'var(--bg2)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)', marginBottom: 16, overflow: 'hidden',
      }}>
        <div style={{
          padding: '14px 20px',
          borderBottom: '1px solid var(--border)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <div>
            <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 13 }}>
              Time-Series Readings
            </span>
            <span style={{ marginLeft: 10, fontSize: 11, color: 'var(--text3)' }}>
              {slots.length} slots · {slots.length * 2}h observation window
            </span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <input type="file" ref={fileRef} accept=".csv" style={{ display:'none' }} onChange={handleCSVImport} />
            <button
              onClick={() => fileRef.current.click()}
              style={{
                padding: '6px 13px', fontSize: 12,
                background: 'var(--bg3)', color: 'var(--text2)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius)',
                display:'flex', alignItems:'center', gap:5,
              }}
            >
              <Upload size={12} /> Import CSV
            </button>
            <button
              onClick={addSlot}
              disabled={slots.length >= 24}
              style={{
                padding: '6px 13px', fontSize: 12,
                background: 'var(--bg3)', color: 'var(--text2)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius)',
                display:'flex', alignItems:'center', gap:5,
                opacity: slots.length >= 24 ? 0.4 : 1,
              }}
            >
              <Plus size={12} /> Add Slot
            </button>
          </div>
        </div>
 
        {/* Group legend */}
        <div style={{ padding: '10px 20px', borderBottom: '1px solid var(--border)', display: 'flex', gap: 14, flexWrap: 'wrap' }}>
          {GROUPS.map(g => (
            <div key={g} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 7, height: 7, borderRadius: 2, background: GROUP_COLORS[g] }} />
              <span style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>{g}</span>
            </div>
          ))}
        </div>
 
        {/* Table */}
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', minWidth: Math.max(900, slots.length * 110 + 160) }}>
            <thead>
              <tr>
                <th style={{ ...thStyle, width: 130, textAlign: 'left', position: 'sticky', left: 0, background: 'var(--bg3)', zIndex: 1 }}>
                  Parameter
                </th>
                {slots.map((_, si) => (
                  <th key={si} style={{ ...thStyle, minWidth: 96 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'center' }}>
                      <span style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: 10 }}>
                        T+{si * 2}h
                      </span>
                      {slots.length > 2 && (
                        <button
                          onClick={() => removeSlot(si)}
                          style={{ background: 'transparent', color: 'var(--text3)', padding: 2 }}
                          title="Remove slot"
                        >
                          <Trash2 size={10} />
                        </button>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {VITALS.map((v, vi) => (
                <tr key={v.key} style={{ borderTop: '1px solid var(--border)' }}>
                  <td style={{
                    padding: '5px 12px',
                    position: 'sticky', left: 0,
                    background: vi % 2 === 0 ? 'var(--bg3)' : 'var(--bg2)',
                    zIndex: 1,
                    borderRight: '1px solid var(--border)',
                  }}>
                    <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                      <div style={{ width:5, height:5, borderRadius:'50%', background: GROUP_COLORS[v.group], flexShrink:0 }} />
                      <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text1)', fontFamily: 'var(--font-mono)' }}>{v.label}</span>
                      <span style={{ fontSize: 10, color: 'var(--text3)' }}>{v.unit}</span>
                    </div>
                  </td>
                  {slots.map((slot, si) => (
                    <td key={si} style={{
                      padding: '3px 5px',
                      background: vi % 2 === 0 ? (si % 2 === 0 ? '#0c1219' : '#0e151f') : (si % 2 === 0 ? '#0a1018' : '#0c1220'),
                    }}>
                      <input
                        type="number"
                        value={slot[v.key]}
                        min={v.min} max={v.max} step={v.step}
                        onChange={e => updateSlot(si, v.key, e.target.value)}
                        style={{
                          width: '100%',
                          padding: '5px 6px',
                          background: 'var(--bg)',
                          border: '1px solid var(--border)',
                          borderRadius: 5,
                          color: 'var(--text1)',
                          fontSize: 12,
                          fontFamily: 'var(--font-mono)',
                          textAlign: 'center',
                          transition: 'var(--transition)',
                        }}
                        onFocus={e => e.target.style.borderColor = 'var(--accent)'}
                        onBlur={e => e.target.style.borderColor = 'var(--border)'}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
 
      {/* CSV template note */}
      <div style={{
        display: 'flex', alignItems: 'flex-start', gap: 8,
        padding: '10px 14px',
        background: 'rgba(0,212,255,0.05)',
        border: '1px solid rgba(0,212,255,0.15)',
        borderRadius: 'var(--radius)',
        marginBottom: 16,
        fontSize: 11, color: 'var(--text2)',
      }}>
        <Info size={13} color="var(--accent)" style={{ flexShrink: 0, marginTop: 1 }} />
        <span>
          CSV import: one row per time slot with columns:
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text1)', marginLeft: 4 }}>
            sbp, dbp, hr, rr, temp, spo2, fio2, uop, avpu, lact, ph, hco3, k, na, cre, hct, tlc, plt, bili, crt
          </span>
        </span>
      </div>
 
      {/* Error */}
      {error && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '10px 14px',
          background: 'rgba(255,61,90,0.08)',
          border: '1px solid rgba(255,61,90,0.3)',
          borderRadius: 'var(--radius)',
          marginBottom: 14,
          fontSize: 12, color: 'var(--red)',
        }}>
          <AlertCircle size={14} />
          {error}
        </div>
      )}
 
      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={loading}
        style={{
          padding: '12px 32px',
          background: loading ? 'var(--bg3)' : 'var(--accent)',
          color: loading ? 'var(--text3)' : 'var(--bg)',
          borderRadius: 'var(--radius)',
          fontFamily: 'var(--font-display)',
          fontWeight: 800, fontSize: 14,
          display: 'flex', alignItems: 'center', gap: 10,
          transition: 'var(--transition)',
          opacity: loading ? 0.7 : 1,
        }}
      >
        {loading ? (
          <>
            <div className="spinner" style={{ borderTopColor: 'var(--text3)' }} />
            Analysing...
          </>
        ) : (
          <>
            <Play size={16} strokeWidth={2.5} />
            Run Haemodynamic Analysis
          </>
        )}
      </button>
    </div>
  );
}
 
const thStyle = {
  padding: '8px 6px',
  fontSize: 9,
  fontFamily: 'var(--font-mono)',
  color: 'var(--text3)',
  fontWeight: 600,
  letterSpacing: '0.5px',
  textAlign: 'center',
  background: 'var(--bg3)',
  whiteSpace: 'nowrap',
};
 
const inputStyle = {
  width: '100%',
  padding: '8px 10px',
  background: 'var(--bg3)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius)',
  color: 'var(--text1)',
  fontSize: 13,
};
 
function Field({ label, children }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      <label style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)', letterSpacing: '0.5px' }}>
        {label.toUpperCase()}
      </label>
      {children}
    </div>
  );
}