//BatchAnalysis.jsx
import React, { useState, useRef } from 'react';
import { Upload, Play, FileSpreadsheet, AlertCircle, TrendingUp, TrendingDown, Minus, CheckCircle } from 'lucide-react';

const RISK_COLORS = {
  CRITICAL: 'var(--red)',
  HIGH:     'var(--orange)',
  MODERATE: 'var(--yellow)',
  LOW:      'var(--green)',
};

const CSV_TEMPLATE = `patient_id,slot,sbp,dbp,hr,rr,temp,spo2,fio2,uop,avpu,lact,ph,hco3,k,na,cre,hct,tlc,plt,bili,crt,age,gender,dm,htn,ckd,ihd,copd,diagnosis
PT-001,0,115,72,88,18,37.2,97,21,65,0,1.2,7.40,24,4.0,137,0.9,34,10,210,0.9,2,62,1,1,1,1,0,0,sepsis_shock
PT-001,1,108,68,95,20,37.5,95,21,55,0,1.8,7.38,22,4.2,136,1.0,33,11,205,1.0,2,62,1,1,1,1,0,0,sepsis_shock
PT-001,2,100,65,102,22,37.8,93,28,40,1,2.4,7.35,20,4.5,135,1.1,32,12,198,1.1,2,62,1,1,1,1,0,0,sepsis_shock
PT-002,0,130,82,76,16,36.8,98,21,70,0,0.9,7.42,25,3.8,139,0.8,38,8,245,0.7,1,45,0,0,1,0,0,0,cardiac
PT-002,1,125,80,79,17,36.9,98,21,68,0,1.0,7.41,24,3.9,139,0.8,37,8,240,0.7,1,45,0,0,1,0,0,0,cardiac
PT-002,2,128,81,78,16,36.9,99,21,72,0,0.8,7.42,24,3.8,138,0.8,38,8,242,0.7,1,45,0,0,1,0,0,0,cardiac`;

export default function BatchAnalysis({ apiBase, onResult }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const fileRef = useRef();

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (f) setFile(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.name.endsWith('.csv')) setFile(f);
  };

  const downloadTemplate = () => {
    const blob = new Blob([CSV_TEMPLATE], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'hemopredict_batch_template.csv'; a.click();
    URL.revokeObjectURL(url);
  };

  const runBatch = async () => {
    if (!file) { setError('Please select a CSV file.'); return; }
    setError(''); setLoading(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${apiBase}/predict/batch-csv`, { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API error');
      }
      const data = await res.json();
      setResults(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '28px 36px', maxWidth: 900 }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 22, marginBottom: 4 }}>
          Batch CSV Analysis
        </h2>
        <p style={{ color: 'var(--text2)', fontSize: 13 }}>
          Upload a CSV with multiple patients. Each row is one time slot. Requires patient_id column to group patients.
        </p>
      </div>

      {/* Upload zone */}
      <div
        onDragOver={e => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileRef.current.click()}
        style={{
          border: `2px dashed ${file ? 'var(--accent)' : 'var(--border2)'}`,
          borderRadius: 'var(--radius-lg)',
          padding: '40px',
          textAlign: 'center',
          cursor: 'pointer',
          background: file ? 'rgba(0,212,255,0.04)' : 'var(--bg2)',
          marginBottom: 16,
          transition: 'var(--transition)',
        }}
      >
        <input type="file" ref={fileRef} accept=".csv" style={{ display:'none' }} onChange={handleFile} />
        <FileSpreadsheet size={36} color={file ? 'var(--accent)' : 'var(--text3)'} style={{ marginBottom: 10 }} />
        {file ? (
          <>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--accent)', marginBottom: 4 }}>
              {file.name}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text3)' }}>
              {(file.size / 1024).toFixed(1)} KB · Click to change
            </div>
          </>
        ) : (
          <>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--text2)', marginBottom: 4 }}>
              Drop CSV here or click to upload
            </div>
            <div style={{ fontSize: 12, color: 'var(--text3)' }}>
              Multi-patient time-series data · .csv only
            </div>
          </>
        )}
      </div>

      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        <button
          onClick={runBatch}
          disabled={loading || !file}
          style={{
            padding: '11px 24px',
            background: loading || !file ? 'var(--bg3)' : 'var(--accent)',
            color: loading || !file ? 'var(--text3)' : 'var(--bg)',
            borderRadius: 'var(--radius)',
            fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 13,
            display: 'flex', alignItems: 'center', gap: 8,
            transition: 'var(--transition)',
            opacity: !file ? 0.5 : 1,
          }}
        >
          {loading ? <><div className="spinner" style={{ borderTopColor: 'var(--text3)' }} /> Analysing...</> : <><Play size={14} strokeWidth={2.5} /> Run Batch Analysis</>}
        </button>
        <button
          onClick={downloadTemplate}
          style={{
            padding: '11px 20px',
            background: 'var(--bg2)',
            color: 'var(--text2)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            fontSize: 12,
            display: 'flex', alignItems: 'center', gap: 6,
          }}
        >
          <Upload size={13} /> Download Template CSV
        </button>
      </div>

      {error && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '10px 14px', marginBottom: 14,
          background: 'rgba(255,61,90,0.08)', border: '1px solid rgba(255,61,90,0.3)',
          borderRadius: 'var(--radius)', fontSize: 12, color: 'var(--red)',
        }}>
          <AlertCircle size={14} /> {error}
        </div>
      )}

      {/* CSV format spec */}
      <div style={{
        background: 'var(--bg2)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)', padding: '16px 20px', marginBottom: 16,
      }}>
        <div style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)', letterSpacing: '1px', marginBottom: 10 }}>
          CSV FORMAT
        </div>
        <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.8 }}>
          Required columns: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)', fontSize: 11 }}>
            patient_id, slot, sbp, dbp, hr, rr, temp, spo2, fio2, uop, avpu, lact, ph, hco3, k, na, cre, hct, tlc, plt, bili, crt
          </span>
          <br/>
          Optional: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text2)', fontSize: 11 }}>
            age, gender, dm, htn, ckd, ihd, copd, diagnosis
          </span>
          <br/>
          One row = one time slot. Group by patient_id. Sort by slot (0-based). Min 2 slots per patient.
        </div>
      </div>

      {/* Batch results */}
      {results && (
        <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', overflow: 'hidden' }}>
          <div style={{
            padding: '14px 20px', borderBottom: '1px solid var(--border)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 14 }}>
              Batch Results
            </span>
            <span style={{ fontSize: 12, color: 'var(--text3)' }}>
              {results.total} patients analysed
            </span>
          </div>

          {/* Summary row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', borderBottom: '1px solid var(--border)' }}>
            {[
              { label:'CRITICAL', count: results.patients.filter(p=>p.peak_risk?.level==='CRITICAL').length, color:'var(--red)' },
              { label:'HIGH',     count: results.patients.filter(p=>p.peak_risk?.level==='HIGH').length,     color:'var(--orange)' },
              { label:'MODERATE', count: results.patients.filter(p=>p.peak_risk?.level==='MODERATE').length,color:'var(--yellow)' },
              { label:'LOW',      count: results.patients.filter(p=>p.peak_risk?.level==='LOW').length,      color:'var(--green)' },
            ].map((s, i) => (
              <div key={i} style={{
                padding: '14px 20px', textAlign: 'center',
                borderRight: i < 3 ? '1px solid var(--border)' : 'none',
              }}>
                <div style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginBottom: 4 }}>{s.label}</div>
                <div style={{ fontSize: 24, fontFamily: 'var(--font-display)', fontWeight: 800, color: s.color }}>{s.count}</div>
              </div>
            ))}
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: 'var(--bg3)' }}>
                {['Patient ID','Slots','Peak Risk','Peak Prob.','Trend','High-Risk Slots'].map(h => (
                  <th key={h} style={{ padding:'8px 16px', textAlign:'left', fontSize:10, color:'var(--text3)', fontFamily:'var(--font-mono)', fontWeight:500 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.patients.map((p, i) => {
                const risk = p.peak_risk?.level || 'LOW';
                return (
                  <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
                    <td style={{ padding:'10px 16px', fontSize:13, fontWeight:500, fontFamily:'var(--font-mono)' }}>{p.patient_id}</td>
                    <td style={{ padding:'10px 16px', fontSize:12, color:'var(--text2)' }}>{p.n_slots}</td>
                    <td style={{ padding:'10px 16px' }}>
                      <span style={{
                        padding:'3px 9px', borderRadius:20,
                        fontSize:10, fontWeight:700, fontFamily:'var(--font-mono)',
                        color:RISK_COLORS[risk], background:`${RISK_COLORS[risk]}18`,
                        border:`1px solid ${RISK_COLORS[risk]}40`,
                      }}>{risk}</span>
                    </td>
                    <td style={{ padding:'10px 16px', fontSize:13, fontFamily:'var(--font-mono)', color:RISK_COLORS[risk] }}>
                      {p.peak_probability}%
                    </td>
                    <td style={{ padding:'10px 16px', fontSize:12, color:'var(--text2)', display:'flex', alignItems:'center', gap:4 }}>
                      {p.trend==='deteriorating' && <TrendingDown size={13} color="var(--red)" />}
                      {p.trend==='improving'     && <TrendingUp size={13} color="var(--green)" />}
                      {p.trend==='stable'        && <Minus size={13} color="var(--yellow)" />}
                      {p.trend}
                    </td>
                    <td style={{ padding:'10px 16px', fontSize:12, color:'var(--text2)', fontFamily:'var(--font-mono)' }}>
                      {p.high_risk_slots}/{p.n_slots}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
