import React, { useState } from 'react';
import {
  Activity, LayoutDashboard, UserPlus, FileSpreadsheet,
  BarChart3, ChevronRight, ChevronDown, Clock, AlertTriangle,
  CheckCircle, WifiOff, Wifi, Database, FlaskConical
} from 'lucide-react';

const NAV = [
  { id: 'dashboard', icon: LayoutDashboard, label: 'Overview' },
  { id: 'input',     icon: UserPlus,        label: 'New Patient' },
  { id: 'batch',     icon: FileSpreadsheet, label: 'Batch CSV' },
  { id: 'results',   icon: BarChart3,       label: 'Results' },
];

const RISK_COLORS = {
  CRITICAL: 'var(--red)',
  HIGH:     'var(--orange)',
  MODERATE: 'var(--yellow)',
  LOW:      'var(--green)',
};

export default function Sidebar({ page, setPage, apiStatus, patients, onSelectPatient }) {
  const [historyOpen, setHistoryOpen] = useState(true);

  return (
    <aside style={{
      width: 240,
      minWidth: 240,
      height: '100vh',
      background: 'var(--bg2)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Logo */}
      <div style={{ padding: '20px 20px 16px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 34, height: 34,
            background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
            borderRadius: 8,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Activity size={18} color="var(--bg)" strokeWidth={2.5} />
          </div>
          <div>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 15, letterSpacing: '-0.3px', color: 'var(--text1)' }}>
              HemoPredict
            </div>
            <div style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)', letterSpacing: '0.5px' }}>
              ICU · v3.0 · NEDUET
            </div>
          </div>
        </div>

        {/* API Status */}
        <div style={{
          marginTop: 12,
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '5px 10px',
          background: 'var(--bg3)',
          borderRadius: 6,
          border: `1px solid ${apiStatus === 'online' ? 'rgba(0,229,160,0.25)' : 'rgba(255,61,90,0.25)'}`,
        }}>
          {apiStatus === 'online'
            ? <Wifi size={11} color="var(--green)" />
            : apiStatus === 'offline'
            ? <WifiOff size={11} color="var(--red)" />
            : <div className="spinner" style={{ width: 11, height: 11, borderWidth: 1.5 }} />
          }
          <span style={{
            fontSize: 10, fontFamily: 'var(--font-mono)',
            color: apiStatus === 'online' ? 'var(--green)' : apiStatus === 'offline' ? 'var(--red)' : 'var(--text3)',
          }}>
            API {apiStatus === 'checking' ? 'connecting...' : apiStatus}
          </span>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ padding: '12px 10px', flex: 'none' }}>
        <div style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text3)', letterSpacing: '1px', padding: '0 8px', marginBottom: 6 }}>
          NAVIGATION
        </div>
        {NAV.map(({ id, icon: Icon, label }) => {
          const active = page === id;
          return (
            <button
              key={id}
              onClick={() => setPage(id)}
              style={{
                width: '100%',
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '8px 10px',
                borderRadius: 7,
                marginBottom: 2,
                background: active ? 'rgba(0,212,255,0.1)' : 'transparent',
                border: active ? '1px solid rgba(0,212,255,0.2)' : '1px solid transparent',
                color: active ? 'var(--accent)' : 'var(--text2)',
                fontFamily: 'var(--font-body)',
                fontWeight: active ? 600 : 400,
                fontSize: 13,
                transition: 'var(--transition)',
                cursor: 'pointer',
              }}
            >
              <Icon size={15} strokeWidth={active ? 2.5 : 2} />
              {label}
              {active && <ChevronRight size={12} style={{ marginLeft: 'auto' }} />}
            </button>
          );
        })}
      </nav>

      {/* Session history */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', borderTop: '1px solid var(--border)' }}>
        <button
          onClick={() => setHistoryOpen(h => !h)}
          style={{
            width: '100%', display: 'flex', alignItems: 'center', gap: 8,
            padding: '10px 18px',
            background: 'transparent', color: 'var(--text3)',
            fontSize: 9, fontFamily: 'var(--font-mono)', letterSpacing: '1px',
          }}
        >
          <Database size={10} />
          SESSION HISTORY
          <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--text3)' }}>
            {patients.length}
          </span>
          {historyOpen
            ? <ChevronDown size={11} />
            : <ChevronRight size={11} />
          }
        </button>

        {historyOpen && (
          <div style={{ overflow: 'auto', flex: 1, padding: '0 10px 10px' }}>
            {patients.length === 0 && (
              <div style={{ padding: '12px 8px', color: 'var(--text3)', fontSize: 11, textAlign: 'center' }}>
                No analyses yet
              </div>
            )}
            {patients.map((p) => {
              const risk = p.summary?.peak_risk?.level || 'LOW';
              const name = p.meta?.name || `Patient ${p.meta?.id || '—'}`;
              return (
                <button
                  key={p.id}
                  onClick={() => onSelectPatient(p)}
                  style={{
                    width: '100%', textAlign: 'left',
                    padding: '8px 10px',
                    borderRadius: 6, marginBottom: 2,
                    background: 'var(--bg3)',
                    border: '1px solid var(--border)',
                    cursor: 'pointer',
                    transition: 'var(--transition)',
                  }}
                  onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border2)'}
                  onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                    <div style={{
                      width: 7, height: 7, borderRadius: '50%',
                      background: RISK_COLORS[risk] || 'var(--text3)',
                      flexShrink: 0,
                    }} />
                    <span style={{ fontSize: 12, color: 'var(--text1)', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {name}
                    </span>
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)', display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: RISK_COLORS[risk] }}>{risk}</span>
                    <span>{p.summary?.peak_probability}%</span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{
        padding: '12px 18px',
        borderTop: '1px solid var(--border)',
        fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)',
        lineHeight: 1.6,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 3 }}>
          <FlaskConical size={10} />
          <span>NEDUET × SIUT, Karachi</span>
        </div>
        <div>AUROC 0.9941 · n=253 patients</div>
        <div style={{ marginTop: 4, color: 'rgba(255,255,255,0.15)' }}>
          Research prototype · CS Batch 2022
        </div>
      </div>
    </aside>
  );
}
