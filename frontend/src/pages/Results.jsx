import React, { useState } from 'react';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Cell,
} from 'recharts';
import {
  AlertTriangle, TrendingUp, TrendingDown, Minus,
  Activity, Heart, Droplet, Brain, Wind, FlaskConical,
  ChevronDown, ChevronUp, Download
} from 'lucide-react';
 
const RISK_COLORS = {
  CRITICAL: '#ff3d5a',
  HIGH:     '#ff7a2f',
  MODERATE: '#f5c842',
  LOW:      '#00e5a0',
};
 
const RISK_BG = {
  CRITICAL: 'rgba(255,61,90,0.1)',
  HIGH:     'rgba(255,122,47,0.1)',
  MODERATE: 'rgba(245,200,66,0.1)',
  LOW:      'rgba(0,229,160,0.1)',
};
 
function RiskBadge({ level, large }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: large ? '6px 16px' : '3px 10px',
      borderRadius: 20,
      fontSize: large ? 13 : 10,
      fontWeight: 700,
      fontFamily: 'var(--font-mono)',
      color: RISK_COLORS[level] || 'var(--text3)',
      background: RISK_BG[level] || 'var(--bg3)',
      border: `1px solid ${RISK_COLORS[level] || 'var(--border)'}40`,
      letterSpacing: large ? '0' : '0.5px',
    }}>
      {(level === 'CRITICAL' || level === 'HIGH') && <AlertTriangle size={large ? 13 : 9} />}
      {level}
    </span>
  );
}
 
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--bg2)', border: '1px solid var(--border2)',
      borderRadius: 8, padding: '10px 14px', fontSize: 12,
    }}>
      <div style={{ color: 'var(--text3)', marginBottom: 4, fontFamily: 'var(--font-mono)', fontSize: 10 }}>
        T+{label}h
      </div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || 'var(--text1)', fontFamily: 'var(--font-mono)' }}>
          {p.name}: <b>{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}</b>
          {p.name === 'Risk %' && '%'}
        </div>
      ))}
    </div>
  );
};
 
function SectionCard({ title, icon: Icon, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{
      background: 'var(--bg2)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', overflow: 'hidden', marginBottom: 14,
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', padding: '14px 20px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: 'transparent', color: 'var(--text1)',
          fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 14,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {Icon && <Icon size={15} color="var(--accent)" />}
          {title}
        </div>
        {open ? <ChevronUp size={14} color="var(--text3)" /> : <ChevronDown size={14} color="var(--text3)" />}
      </button>
      {open && (
        <div style={{ borderTop: '1px solid var(--border)', padding: 20 }}>
          {children}
        </div>
      )}
    </div>
  );
}
 
export default function Results({ result }) {
  const { summary, slots, shap_global, derived, flags, meta } = result;
  const risk = summary?.peak_risk?.level || 'LOW';
 
  // Build chart data
  const chartData = slots.map(s => ({
    time:  s.time_hrs,
    prob:  s.probability,
    risk:  s.probability,
    level: s.risk?.level,
  }));
 
  // SHAP chart data
  const shapData = (shap_global || []).slice(0, 12).map(f => ({
    name: f.feature.replace(/_/g,'_').substring(0, 22),
    full: f.feature,
    value: Math.abs(f.mean_abs_shap),
    direction: f.direction,
  })).sort((a, b) => b.value - a.value);
 
  const maxShap = Math.max(...shapData.map(d => d.value), 0.001);
 
  // Vitals trend from slots
  const vitalsTrend = slots.map(s => {
    const r = result.meta?._readings?.[s.slot];
    return { time: s.time_hrs, prob: s.probability };
  });
 
  const handleExport = () => {
    const data = JSON.stringify(result, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hemopredict_${meta?.name || 'patient'}_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
 
  return (
    <div style={{ padding: '28px 36px', maxWidth: 1060, animation: 'fadeIn 0.35s ease' }}>
      {/* Patient header */}
      <div style={{
        background: `linear-gradient(135deg, var(--bg2) 0%, ${RISK_BG[risk]} 100%)`,
        border: `1px solid ${RISK_COLORS[risk]}35`,
        borderRadius: 'var(--radius-lg)',
        padding: '22px 26px',
        marginBottom: 16,
        display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 20 }}>
              {meta?.name || 'Patient Analysis'}
            </h2>
            <RiskBadge level={risk} large />
          </div>
          <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
            {[
              { label: 'Age', val: meta?.age ? `${meta.age}y` : '—' },
              { label: 'Gender', val: meta?.gender === 1 ? 'Male' : 'Female' },
              { label: 'Diagnosis', val: meta?.diagnosis?.replace(/_/g,' ') || '—' },
              { label: 'Slots', val: summary?.total_slots },
              { label: 'Obs. Window', val: `${(summary?.total_slots - 1) * 2}h` },
            ].map(({ label, val }) => (
              <div key={label}>
                <span style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>{label}: </span>
                <span style={{ fontSize: 13, color: 'var(--text1)', fontWeight: 500 }}>{val}</span>
              </div>
            ))}
          </div>
          {meta?.dm || meta?.htn || meta?.ckd || meta?.ihd || meta?.copd ? (
            <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {[['dm','DM'],['htn','HTN'],['ckd','CKD'],['ihd','IHD'],['copd','COPD']].filter(([k]) => meta[k]).map(([k,l]) => (
                <span key={k} style={{
                  padding: '2px 8px', borderRadius: 10, fontSize: 10,
                  background: 'rgba(255,255,255,0.06)', color: 'var(--text2)',
                  border: '1px solid var(--border)',
                }}>{l}</span>
              ))}
            </div>
          ) : null}
        </div>
 
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <div style={{ fontSize: 48, fontFamily: 'var(--font-display)', fontWeight: 800, lineHeight: 1, color: RISK_COLORS[risk] }}>
            {summary?.peak_probability}%
          </div>
          <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 10 }}>peak instability risk</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'flex-end' }}>
            {summary?.trend === 'deteriorating' && <><TrendingDown size={14} color="var(--red)" /><span style={{ fontSize: 12, color: 'var(--red)' }}>Deteriorating</span></>}
            {summary?.trend === 'improving'     && <><TrendingUp size={14} color="var(--green)" /><span style={{ fontSize: 12, color: 'var(--green)' }}>Improving</span></>}
            {summary?.trend === 'stable'        && <><Minus size={14} color="var(--yellow)" /><span style={{ fontSize: 12, color: 'var(--yellow)' }}>Stable</span></>}
          </div>
          <button
            onClick={handleExport}
            style={{
              marginTop: 12,
              padding: '6px 14px', fontSize: 11,
              background: 'var(--bg3)', color: 'var(--text2)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius)',
              display: 'flex', alignItems: 'center', gap: 5,
              cursor: 'pointer',
            }}
          >
            <Download size={11} /> Export JSON
          </button>
        </div>
      </div>
 
      {/* Summary KPIs */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 14 }}>
        {[
          { label: 'PEAK AT', val: `${summary?.peak_time_hrs}h`, sub: `Slot ${summary?.peak_slot}` },
          { label: 'HIGH-RISK SLOTS', val: `${summary?.high_risk_slots}/${summary?.total_slots}`, sub: `≥${summary?.decision_threshold}% threshold` },
          { label: 'MEAN MAP', val: `${derived?.mean_map} mmHg`, sub: derived?.min_map < 65 ? '⚠ Nadir <65' : 'Adequate', color: derived?.min_map < 65 ? 'var(--red)' : 'var(--green)' },
          { label: 'PEAK LACTATE', val: `${derived?.peak_lactate}`, sub: 'mmol/L', color: derived?.peak_lactate > 4 ? 'var(--red)' : derived?.peak_lactate > 2 ? 'var(--orange)' : 'var(--green)' },
        ].map((k, i) => (
          <div key={i} style={{
            background: 'var(--bg2)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius)', padding: '14px 16px',
          }}>
            <div style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'var(--font-mono)', letterSpacing: '0.5px', marginBottom: 6 }}>{k.label}</div>
            <div style={{ fontSize: 22, fontFamily: 'var(--font-display)', fontWeight: 800, color: k.color || 'var(--text1)' }}>{k.val}</div>
            <div style={{ fontSize: 10, color: 'var(--text3)', marginTop: 3 }}>{k.sub}</div>
          </div>
        ))}
      </div>
 
      {/* Risk trajectory chart */}
      <SectionCard title="Risk Probability Trajectory" icon={Activity}>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={chartData} margin={{ top: 8, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={RISK_COLORS[risk]} stopOpacity={0.3} />
                <stop offset="95%" stopColor={RISK_COLORS[risk]} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="time"
              tickFormatter={v => `${v}h`}
              tick={{ fill: 'var(--text3)', fontSize: 10, fontFamily: 'DM Mono' }}
              axisLine={{ stroke: 'var(--border)' }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tickFormatter={v => `${v}%`}
              tick={{ fill: 'var(--text3)', fontSize: 10, fontFamily: 'DM Mono' }}
              axisLine={{ stroke: 'var(--border)' }}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={summary?.decision_threshold || 35} stroke="rgba(255,255,255,0.2)" strokeDasharray="5 5"
              label={{ value: `Threshold ${summary?.decision_threshold}%`, fill: 'var(--text3)', fontSize: 9, fontFamily: 'DM Mono' }}
            />
            <ReferenceLine y={75} stroke="rgba(255,61,90,0.3)" strokeDasharray="3 3" />
            <Area
              type="monotone"
              dataKey="prob"
              name="Risk %"
              stroke={RISK_COLORS[risk]}
              strokeWidth={2.5}
              fill="url(#riskGrad)"
              dot={{ fill: RISK_COLORS[risk], strokeWidth: 0, r: 4 }}
              activeDot={{ r: 6, strokeWidth: 0 }}
            />
          </AreaChart>
        </ResponsiveContainer>
        <div style={{ marginTop: 12, display: 'flex', gap: 16, fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
          <span style={{ display:'flex', alignItems:'center', gap:4 }}>
            <span style={{ width:20, height:1, background:'rgba(255,255,255,0.2)', display:'inline-block' }} />
            Decision threshold
          </span>
          <span style={{ display:'flex', alignItems:'center', gap:4 }}>
            <span style={{ width:20, height:1, background:'rgba(255,61,90,0.4)', display:'inline-block' }} />
            Critical zone (75%)
          </span>
        </div>
      </SectionCard>
 
      {/* Slot-by-slot timeline */}
      <SectionCard title="Per-Slot Risk Timeline" icon={Activity} defaultOpen={false}>
        <div style={{ overflowX: 'auto' }}>
          <div style={{ display: 'flex', gap: 6, minWidth: 'max-content', paddingBottom: 4 }}>
            {slots.map((s, i) => {
              const c = RISK_COLORS[s.risk?.level] || 'var(--text3)';
              return (
                <div key={i} style={{
                  width: 76,
                  background: 'var(--bg3)',
                  border: `1px solid ${c}40`,
                  borderRadius: 8,
                  padding: '10px 8px',
                  textAlign: 'center',
                  flexShrink: 0,
                }}>
                  <div style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginBottom: 4 }}>
                    T+{s.time_hrs}h
                  </div>
                  <div style={{ fontSize: 16, fontFamily: 'var(--font-display)', fontWeight: 800, color: c, lineHeight: 1, marginBottom: 4 }}>
                    {s.probability}%
                  </div>
                  <div style={{
                    fontSize: 8, padding: '2px 5px',
                    background: `${c}18`, color: c,
                    borderRadius: 10, fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                  }}>
                    {s.risk?.level}
                  </div>
                  <div style={{
                    marginTop: 6, height: 3, borderRadius: 2,
                    background: `${c}35`,
                    position: 'relative', overflow: 'hidden',
                  }}>
                    <div style={{
                      position: 'absolute', left: 0, top: 0, bottom: 0,
                      width: `${s.probability}%`,
                      background: c, borderRadius: 2,
                    }} />
                  </div>
                  {s.clinical_score !== undefined && (
                    <div style={{ fontSize: 8, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginTop: 5 }}>
                      clin {s.clinical_score}%
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </SectionCard>
 
      {/* Derived haemodynamic indices */}
      <SectionCard title="Derived Haemodynamic Indices" icon={Heart}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
          {[
            { label: 'Mean MAP', val: `${derived?.mean_map} mmHg`, warn: derived?.mean_map < 70, crit: derived?.mean_map < 65 },
            { label: 'MAP Nadir', val: `${derived?.min_map} mmHg`, warn: derived?.min_map < 75, crit: derived?.min_map < 65 },
            { label: 'Peak Shock Index', val: derived?.peak_shock_index?.toFixed(3), warn: derived?.peak_shock_index > 1.0, crit: derived?.peak_shock_index > 1.2 },
            { label: 'Peak Lactate', val: `${derived?.peak_lactate} mmol/L`, warn: derived?.peak_lactate > 2, crit: derived?.peak_lactate > 4 },
            { label: 'Min SpO₂', val: `${derived?.min_spo2}%`, warn: derived?.min_spo2 < 94, crit: derived?.min_spo2 < 90 },
            { label: 'Min SF Ratio', val: derived?.min_sf_ratio?.toFixed(3), warn: derived?.min_sf_ratio < 3.0, crit: derived?.min_sf_ratio < 2.0 },
            { label: 'Peak Creatinine', val: `${derived?.peak_creatinine} mg/dL`, warn: derived?.peak_creatinine > 1.5, crit: derived?.peak_creatinine > 2.0 },
            { label: 'Min UOP', val: `${derived?.min_uop} ml/hr`, warn: derived?.min_uop < 30, crit: derived?.min_uop < 20 },
          ].map((d, i) => {
            const col = d.crit ? 'var(--red)' : d.warn ? 'var(--orange)' : 'var(--green)';
            return (
              <div key={i} style={{
                background: 'var(--bg3)', border: `1px solid ${d.crit ? 'rgba(255,61,90,0.25)' : d.warn ? 'rgba(255,122,47,0.2)' : 'var(--border)'}`,
                borderRadius: 8, padding: '12px 14px',
              }}>
                <div style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginBottom: 5 }}>{d.label}</div>
                <div style={{ fontSize: 18, fontFamily: 'var(--font-display)', fontWeight: 700, color: col }}>{d.val}</div>
                {(d.crit || d.warn) && (
                  <div style={{ fontSize: 9, color: col, marginTop: 3 }}>
                    {d.crit ? '⚠ Critical' : '↑ Elevated'}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </SectionCard>
 
      {/* Clinical flags */}
      {flags && flags.length > 0 && (
        <SectionCard title={`Clinical Alerts (${flags.length})`} icon={AlertTriangle}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {flags.map((f, i) => {
              const isCrit = f.severity === 'critical';
              const col = isCrit ? 'var(--red)' : 'var(--orange)';
              return (
                <div key={i} style={{
                  display: 'flex', gap: 10,
                  padding: '10px 14px',
                  background: isCrit ? 'rgba(255,61,90,0.07)' : 'rgba(255,122,47,0.07)',
                  border: `1px solid ${col}30`,
                  borderRadius: 8,
                  borderLeft: `3px solid ${col}`,
                }}>
                  <AlertTriangle size={14} color={col} style={{ flexShrink: 0, marginTop: 1 }} />
                  <div>
                    <div style={{ fontSize: 10, fontWeight: 700, color: col, fontFamily: 'var(--font-mono)', marginBottom: 3, letterSpacing: '0.5px' }}>
                      {f.severity.toUpperCase()}
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text1)', lineHeight: 1.6 }}>{f.message}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </SectionCard>
      )}
 
      {/* SHAP feature attribution */}
      <SectionCard title="SHAP Feature Attribution" icon={FlaskConical}>
        <p style={{ fontSize: 12, color: 'var(--text3)', marginBottom: 16, lineHeight: 1.6 }}>
          Mean absolute SHAP values across all time slots (XGBoost TreeSHAP).
          Bar length = contribution magnitude. Red = increases instability risk. Blue = decreases risk.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {shapData.map((f, i) => {
            const pct = (f.value / maxShap) * 100;
            const isPos = f.direction === 'increases_risk';
            const col = isPos ? '#ff3d5a' : '#00d4ff';
            return (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{
                  width: 190, flexShrink: 0,
                  fontSize: 11, color: 'var(--text2)',
                  fontFamily: 'var(--font-mono)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  textAlign: 'right',
                }} title={f.full}>
                  {f.name}
                </div>
                <div style={{
                  flex: 1, height: 14, background: 'var(--bg3)',
                  borderRadius: 4, position: 'relative', overflow: 'hidden',
                }}>
                  <div style={{
                    position: 'absolute',
                    left: isPos ? '50%' : `${50 - pct / 2}%`,
                    width: `${pct / 2}%`,
                    height: '100%',
                    background: col,
                    borderRadius: isPos ? '0 4px 4px 0' : '4px 0 0 4px',
                    opacity: 0.8,
                  }} />
                  <div style={{
                    position: 'absolute', left: '50%',
                    top: 0, width: 1, height: '100%',
                    background: 'var(--border2)',
                  }} />
                </div>
                <div style={{
                  width: 60, flexShrink: 0,
                  fontSize: 10, color: col,
                  fontFamily: 'var(--font-mono)', textAlign: 'right',
                }}>
                  {f.value.toFixed(4)}
                </div>
              </div>
            );
          })}
        </div>
      </SectionCard>
 
      {/* Clinical reasoning */}
      <SectionCard title="Clinical Reasoning" icon={Brain}>
        <ReasoningPanel summary={summary} derived={derived} shap={shap_global} flags={flags} meta={meta} />
      </SectionCard>
 
      {/* Disclaimer */}
      <div style={{
        padding: '12px 16px',
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        fontSize: 10, color: 'var(--text3)',
        lineHeight: 1.7,
      }}>
        <strong style={{ color: 'var(--text2)' }}>Research Disclaimer:</strong> ICU HemoPredict is a clinical decision support tool validated on 253 patients
        at SIUT Karachi (AUROC 0.9941). It does not replace clinical judgment. Feature attributions represent statistical model contributions,
        not causal clinical mechanisms. Prospective multi-centre validation required before clinical deployment. NEDUET CS Batch 2022 — Group 17.
      </div>
    </div>
  );
}
 
function ReasoningPanel({ summary, derived, shap, flags, meta }) {
  const risk = summary?.peak_risk?.level;
  const topFeatures = (shap || []).slice(0, 3).map(f => f.feature);
 
  const blocks = [];
 
  // Block 1: overall verdict
  blocks.push({
    icon: Activity,
    color: RISK_COLORS[risk],
    title: 'Risk Classification',
    text: `The ensemble model assigns a peak haemodynamic instability probability of ${summary?.peak_probability}%, 
    classified as ${risk} risk. ${summary?.high_risk_slots} of ${summary?.total_slots} time slots exceeded the 
    ${summary?.decision_threshold}% decision threshold. The clinical trajectory is ${summary?.trend}.`,
  });
 
  // Block 2: haemodynamic interpretation
  if (derived?.min_map < 65) {
    blocks.push({
      icon: Heart,
      color: 'var(--red)',
      title: 'Haemodynamic Compromise',
      text: `MAP nadir of ${derived?.min_map} mmHg falls below the critical perfusion threshold of 65 mmHg. 
      This, combined with a peak Shock Index of ${derived?.peak_shock_index?.toFixed(2)}, indicates significant 
      cardiovascular decompensation. Vasopressor support and volume status reassessment are indicated.`,
    });
  } else if (derived?.min_map < 75) {
    blocks.push({
      icon: Heart,
      color: 'var(--orange)',
      title: 'Marginal Perfusion Pressure',
      text: `MAP nadir of ${derived?.min_map} mmHg approaches but does not breach the 65 mmHg critical threshold. 
      Close monitoring of cardiovascular status is warranted. Shock Index peak of ${derived?.peak_shock_index?.toFixed(2)} 
      warrants continued observation.`,
    });
  }
 
  // Block 3: metabolic
  if (derived?.peak_lactate > 2) {
    blocks.push({
      icon: FlaskConical,
      color: derived?.peak_lactate > 4 ? 'var(--red)' : 'var(--orange)',
      title: 'Metabolic Assessment',
      text: `Peak lactate of ${derived?.peak_lactate} mmol/L indicates ${derived?.peak_lactate > 4 ? 'severe hyperlactataemia consistent with tissue hypoperfusion. Urgent source control and resuscitation are required' : 'moderate lactate elevation above the 2.0 mmol/L threshold. Serial trending is recommended to assess for evolving hypoperfusion'}. 
      This is a key driver in the model's risk attribution.`,
    });
  }
 
  // Block 4: SHAP drivers
  if (topFeatures.length > 0) {
    blocks.push({
      icon: Brain,
      color: 'var(--accent)',
      title: 'Model Feature Drivers',
      text: `The dominant SHAP predictors for this patient are: ${topFeatures.slice(0,3).join(', ')}. 
      These features collectively account for the largest share of the model's instability prediction. 
      The LightGBM component (weight 0.7) drives the ensemble; XGBoost (0.2) and Logistic Regression (0.1) contribute as corrective learners. 
      50 SHAP-selected features were used from 125 engineered candidates.`,
    });
  }
 
  // Block 5: respiratory
  if (derived?.min_spo2 < 94) {
    blocks.push({
      icon: Wind,
      color: derived?.min_spo2 < 90 ? 'var(--red)' : 'var(--orange)',
      title: 'Oxygenation',
      text: `SpO₂ nadir of ${derived?.min_spo2}% indicates ${derived?.min_spo2 < 90 ? 'severe hypoxaemia' : 'borderline hypoxaemia'}. 
      SpO₂/FiO₂ ratio minimum of ${derived?.min_sf_ratio?.toFixed(3)} provides additional context on respiratory reserve. 
      Escalation of respiratory support should be considered.`,
    });
  }
 
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {blocks.map((b, i) => {
        const Icon = b.icon;
        return (
          <div key={i} style={{
            display: 'flex', gap: 12,
            padding: '12px 16px',
            background: 'var(--bg3)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            borderLeft: `3px solid ${b.color}`,
          }}>
            <Icon size={15} color={b.color} style={{ flexShrink: 0, marginTop: 2 }} />
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: b.color, marginBottom: 5, fontFamily: 'var(--font-display)' }}>
                {b.title}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.7 }}>{b.text}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}