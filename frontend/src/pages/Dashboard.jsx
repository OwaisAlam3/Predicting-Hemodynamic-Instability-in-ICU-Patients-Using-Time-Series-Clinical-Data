//Dashboard.jsx
import React from 'react';
import { Activity, TrendingUp, TrendingDown, Minus, Users, AlertTriangle, CheckCircle, Zap } from 'lucide-react';

const RISK_COLORS = {
  CRITICAL: 'var(--red)',
  HIGH:     'var(--orange)',
  MODERATE: 'var(--yellow)',
  LOW:      'var(--green)',
};

function StatCard({ label, value, sub, color, icon: Icon }) {
  return (
    <div style={{
      background: 'var(--bg2)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      padding: '20px 22px',
      display: 'flex', flexDirection: 'column', gap: 8,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', letterSpacing: '0.5px' }}>
          {label}
        </span>
        {Icon && <Icon size={15} color={color || 'var(--text3)'} />}
      </div>
      <div style={{ fontSize: 28, fontFamily: 'var(--font-display)', fontWeight: 800, color: color || 'var(--text1)' }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text3)' }}>{sub}</div>}
    </div>
  );
}

export default function Dashboard({ patients, setPage, apiStatus }) {
  const total = patients.length;
  const critical = patients.filter(p => p.summary?.peak_risk?.level === 'CRITICAL').length;
  const high     = patients.filter(p => p.summary?.peak_risk?.level === 'HIGH').length;
  const low      = patients.filter(p => ['LOW','MODERATE'].includes(p.summary?.peak_risk?.level)).length;

  const recentFive = patients.slice(0, 5);

  return (
    <div style={{ padding: '32px 36px', maxWidth: 1100, animation: 'fadeIn 0.3s ease' }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <div style={{
            padding: '3px 10px',
            background: 'rgba(0,212,255,0.08)',
            border: '1px solid rgba(0,212,255,0.2)',
            borderRadius: 20,
            fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--accent)',
            letterSpacing: '1px',
          }}>
            CLINICAL DECISION SUPPORT
          </div>
          <div style={{
            width: 7, height: 7, borderRadius: '50%',
            background: apiStatus === 'online' ? 'var(--green)' : 'var(--red)',
            animation: apiStatus === 'online' ? 'pulse 2s infinite' : 'none',
          }} />
        </div>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 800, fontSize: 32, letterSpacing: '-1px',
          color: 'var(--text1)', lineHeight: 1.1,
        }}>
          ICU Haemodynamic<br />
          <span style={{ color: 'var(--accent)' }}>Instability</span> Monitor
        </h1>
        <p style={{ marginTop: 10, color: 'var(--text2)', fontSize: 13, maxWidth: 520 }}>
          Ensemble ML model (XGBoost + LightGBM + LR) validated on 253 ICU patients at SIUT Karachi.
          AUROC 0.9941 · Sensitivity 96.67% · Specificity 95.59%
        </p>
      </div>

      {/* Quick actions */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 32 }}>
        <button
          onClick={() => setPage('input')}
          style={{
            padding: '11px 22px',
            background: 'var(--accent)',
            color: 'var(--bg)',
            borderRadius: 'var(--radius)',
            fontFamily: 'var(--font-display)',
            fontWeight: 700, fontSize: 13,
            display: 'flex', alignItems: 'center', gap: 7,
            transition: 'var(--transition)',
          }}
          onMouseEnter={e => e.currentTarget.style.opacity = '0.88'}
          onMouseLeave={e => e.currentTarget.style.opacity = '1'}
        >
          <Zap size={14} strokeWidth={2.5} />
          New Patient Analysis
        </button>
        <button
          onClick={() => setPage('batch')}
          style={{
            padding: '11px 22px',
            background: 'var(--bg2)',
            color: 'var(--text1)',
            borderRadius: 'var(--radius)',
            border: '1px solid var(--border)',
            fontFamily: 'var(--font-display)',
            fontWeight: 600, fontSize: 13,
            display: 'flex', alignItems: 'center', gap: 7,
            transition: 'var(--transition)',
          }}
          onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border2)'}
          onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
        >
          Import CSV Batch
        </button>
      </div>

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 28 }}>
        <StatCard label="TOTAL ANALYSES" value={total || '—'} sub="This session" icon={Users} />
        <StatCard label="CRITICAL RISK" value={critical || '0'} sub="Immediate intervention" color="var(--red)" icon={AlertTriangle} />
        <StatCard label="HIGH RISK" value={high || '0'} sub="Escalate monitoring" color="var(--orange)" icon={Activity} />
        <StatCard label="STABLE / LOW" value={low || '0'} sub="Standard protocol" color="var(--green)" icon={CheckCircle} />
      </div>

      {/* Model performance banner */}
      <div style={{
        background: 'var(--bg2)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        padding: '20px 24px',
        marginBottom: 28,
      }}>
        <div style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)', letterSpacing: '1px', marginBottom: 14 }}>
          VALIDATED MODEL PERFORMANCE — SIUT KARACHI COHORT (n=253)
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 0 }}>
          {[
            { label: 'AUROC', val: '0.9941', sub: '±0.0064 CV' },
            { label: 'SENSITIVITY', val: '96.67%', sub: '±1.86% CV' },
            { label: 'SPECIFICITY', val: '95.59%', sub: '±4.87% CV' },
            { label: 'F1 SCORE', val: '0.9589', sub: 'held-out test' },
            { label: 'BRIER SCORE', val: '0.0346', sub: 'well-calibrated' },
          ].map((m, i) => (
            <div key={i} style={{
              textAlign: 'center',
              borderRight: i < 4 ? '1px solid var(--border)' : 'none',
              padding: '0 16px',
            }}>
              <div style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginBottom: 4 }}>{m.label}</div>
              <div style={{ fontSize: 22, fontFamily: 'var(--font-display)', fontWeight: 800, color: 'var(--accent)' }}>{m.val}</div>
              <div style={{ fontSize: 10, color: 'var(--text3)', marginTop: 2 }}>{m.sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent analyses */}
      {recentFive.length > 0 && (
        <div style={{
          background: 'var(--bg2)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          overflow: 'hidden',
        }}>
          <div style={{ padding: '16px 22px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 14 }}>Recent Analyses</span>
            <span style={{ fontSize: 11, color: 'var(--text3)' }}>Session only · not persisted</span>
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: 'var(--bg3)' }}>
                {['Patient', 'Diagnosis', 'Slots', 'Peak Risk', 'Probability', 'Trend'].map(h => (
                  <th key={h} style={{
                    padding: '8px 16px', textAlign: 'left',
                    fontSize: 10, color: 'var(--text3)',
                    fontFamily: 'var(--font-mono)', letterSpacing: '0.5px',
                    fontWeight: 500,
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {recentFive.map((p, i) => {
                const risk = p.summary?.peak_risk?.level || '—';
                const trend = p.summary?.trend;
                return (
                  <tr
                    key={p.id}
                    style={{
                      borderTop: '1px solid var(--border)',
                      cursor: 'pointer',
                      transition: 'var(--transition)',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--bg3)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  >
                    <td style={{ padding: '10px 16px', fontSize: 13, fontWeight: 500 }}>
                      {p.meta?.name || `Patient ${p.meta?.id || i + 1}`}
                    </td>
                    <td style={{ padding: '10px 16px', fontSize: 12, color: 'var(--text2)' }}>
                      {p.meta?.diagnosis?.replace(/_/g, ' ') || '—'}
                    </td>
                    <td style={{ padding: '10px 16px', fontSize: 12, color: 'var(--text2)', fontFamily: 'var(--font-mono)' }}>
                      {p.summary?.total_slots}
                    </td>
                    <td style={{ padding: '10px 16px' }}>
                      <span style={{
                        padding: '3px 9px',
                        borderRadius: 20,
                        fontSize: 10, fontWeight: 700,
                        fontFamily: 'var(--font-mono)',
                        color: RISK_COLORS[risk],
                        background: `${RISK_COLORS[risk]}18`,
                        border: `1px solid ${RISK_COLORS[risk]}40`,
                      }}>{risk}</span>
                    </td>
                    <td style={{ padding: '10px 16px', fontSize: 13, fontFamily: 'var(--font-mono)', color: RISK_COLORS[risk] }}>
                      {p.summary?.peak_probability}%
                    </td>
                    <td style={{ padding: '10px 16px', fontSize: 12, color: 'var(--text2)', display:'flex', alignItems:'center', gap:4 }}>
                      {trend === 'deteriorating' && <TrendingDown size={13} color="var(--red)" />}
                      {trend === 'improving'     && <TrendingUp size={13} color="var(--green)" />}
                      {trend === 'stable'        && <Minus size={13} color="var(--yellow)" />}
                      {trend}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Empty state */}
      {patients.length === 0 && (
        <div style={{
          background: 'var(--bg2)',
          border: '1px dashed var(--border2)',
          borderRadius: 'var(--radius-lg)',
          padding: '48px',
          textAlign: 'center',
        }}>
          <Activity size={32} color="var(--text3)" style={{ marginBottom: 12 }} />
          <p style={{ color: 'var(--text2)', fontSize: 14, marginBottom: 6 }}>No analyses in this session</p>
          <p style={{ color: 'var(--text3)', fontSize: 12 }}>
            Start with a new patient analysis or import a CSV batch to see results here.
          </p>
        </div>
      )}
    </div>
  );
}
