import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import PatientInput from './pages/PatientInput';
import BatchAnalysis from './pages/BatchAnalysis';
import Results from './pages/Results';

const API_BASE = (
  process.env.REACT_APP_API_URL || 'http://localhost:8000'
).trim();

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [result, setResult] = useState(null);
  const [patients, setPatients] = useState([]); // session history
  const [apiStatus, setApiStatus] = useState('checking');

  React.useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.ok ? setApiStatus('online') : setApiStatus('offline'))
      .catch(() => setApiStatus('offline'));
  }, []);

  const handleResult = (data, patientMeta) => {
    const entry = { ...data, meta: patientMeta, id: Date.now(), ts: new Date().toISOString() };
    setPatients(prev => [entry, ...prev.slice(0, 19)]);
    setResult(entry);
    setPage('results');
  };

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar
        page={page}
        setPage={setPage}
        apiStatus={apiStatus}
        patients={patients}
        onSelectPatient={(p) => { setResult(p); setPage('results'); }}
      />
      <main style={{
        flex: 1,
        overflow: 'auto',
        background: 'var(--bg)',
        position: 'relative',
      }}>
        {page === 'dashboard' && (
          <Dashboard
            patients={patients}
            setPage={setPage}
            apiStatus={apiStatus}
          />
        )}
        {page === 'input' && (
          <PatientInput
            onResult={handleResult}
            apiBase={API_BASE}
          />
        )}
        {page === 'batch' && (
          <BatchAnalysis
            apiBase={API_BASE}
            onResult={handleResult}
          />
        )}
        {page === 'results' && result && (
          <Results result={result} />
        )}
        {page === 'results' && !result && (
          <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100%', flexDirection:'column', gap:16 }}>
            <span style={{ fontSize:48 }}>🔬</span>
            <p style={{ color:'var(--text3)', fontFamily:'var(--font-display)', fontSize:18 }}>No results yet. Run an analysis first.</p>
            <button
              onClick={() => setPage('input')}
              style={{ padding:'10px 24px', background:'var(--accent)', color:'var(--bg)', borderRadius:'var(--radius)', fontFamily:'var(--font-display)', fontWeight:700, fontSize:13 }}
            >
              New Patient
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
