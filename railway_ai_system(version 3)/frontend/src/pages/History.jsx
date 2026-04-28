import React, { useEffect, useState } from 'react';
import { useApi } from '../hooks/useApi';

export default function History() {
  const { request, loading } = useApi();
  const [runs, setRuns] = useState([]);

  useEffect(() => {
    fetchHistory();
  }, [request]);

  const fetchHistory = async () => {
    try {
      const data = await request('/history');
      if (data) setRuns(data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '2rem auto', padding: '0 2rem' }}>
      <h1>Operational History Ledger</h1>
      <p style={{ color: 'var(--text-secondary)' }}>Persistent tracking of all AI evaluations derived from the SQLite database.</p>
      
      {loading ? (
        <div style={{ marginTop: '2rem', padding: '2rem', backgroundColor: 'var(--bg-card)', borderRadius: '8px', textAlign: 'center' }}>
          Querying Database Ledger...
        </div>
      ) : (
        <table style={{ width: '100%', marginTop: '2rem', borderCollapse: 'collapse', backgroundColor: 'var(--bg-card)', borderRadius: '8px', overflow: 'hidden' }}>
          <thead style={{ backgroundColor: '#1f2937' }}>
            <tr>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '1px solid #374151' }}>Run ID</th>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '1px solid #374151' }}>Timestamp</th>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '1px solid #374151' }}>Video Input</th>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '1px solid #374151' }}>Status</th>
              <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '1px solid #374151' }}>Risk Score</th>
            </tr>
          </thead>
          <tbody>
            {runs.length === 0 ? (
              <tr><td colSpan="5" style={{ padding: '1rem', textAlign: 'center' }}>No historical operations found in the database.</td></tr>
            ) : (
              runs.map(run => (
                <tr key={run.id} style={{ borderBottom: '1px solid #374151' }}>
                  <td style={{ padding: '1rem', fontFamily: 'monospace', color: 'var(--accent-blue)' }}>{run.id}</td>
                  <td style={{ padding: '1rem' }}>{new Date(run.timestamp).toLocaleString()}</td>
                  <td style={{ padding: '1rem' }}>{run.video_name}</td>
                  <td style={{ padding: '1rem', textTransform: 'uppercase', fontSize: '0.9rem', fontWeight: 'bold' }}>
                    <span style={{ color: run.status === 'complete' ? 'var(--accent-green)' : 'var(--accent-blue)' }}>{run.status}</span>
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <div style={{ width: '100px', height: '8px', backgroundColor: '#374151', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{ width: `${run.risk_score * 100}%`, height: '100%', backgroundColor: run.risk_score > 0.7 ? 'var(--accent-red)' : 'var(--accent-green)', transition: 'width 0.5s ease' }}></div>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      )}
    </div>
  );
}
