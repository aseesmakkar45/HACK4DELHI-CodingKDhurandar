import React, { useEffect, useState } from 'react';
import { useApi } from '../hooks/useApi';
import { AlertTriangle, CheckCircle } from 'lucide-react';

export default function AlertCenter() {
  const { request, loading } = useApi();
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    fetchAlerts();
  }, [request]);

  const fetchAlerts = async () => {
    try {
      const data = await request('/alerts');
      if (data) setAlerts(data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleAcknowledge = async (id) => {
    try {
      await request(`/alerts/${id}/acknowledge`, { method: 'POST' });
      fetchAlerts(); // Refresh the ledger immediately
    } catch (err) {
      console.error("Acknowledgment failed", err);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '2rem auto', padding: '0 2rem' }}>
      <h1>Critical Threat Alerts</h1>
      <p style={{ color: 'var(--text-secondary)' }}>Review physical security breaches natively filtered and stored by SQLAlchemy logic.</p>
      
      {loading ? (
        <div style={{ marginTop: '2rem', padding: '2rem', backgroundColor: 'var(--bg-card)', borderRadius: '8px', textAlign: 'center' }}>
          Querying Database Ledger...
        </div>
      ) : (
        <div style={{ marginTop: '2rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {alerts.length === 0 ? (
            <div style={{ padding: '2rem', backgroundColor: 'var(--bg-card)', borderRadius: '8px', textAlign: 'center', border: '1px dashed #374151' }}>
              No critical threats actively tracked in the SQL Ledger.
            </div>
          ) : (
            alerts.map(a => (
              <div key={a.id} style={{ 
                padding: '1.5rem', 
                backgroundColor: 'var(--bg-card)', 
                borderRadius: '8px', 
                borderLeft: `4px solid ${a.acknowledged ? 'var(--accent-blue)' : 'var(--accent-red)'}`,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                opacity: a.acknowledged ? 0.6 : 1,
                transition: 'all 0.3s ease'
              }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', marginBottom: '0.5rem' }}>
                    <AlertTriangle size={20} color={a.acknowledged ? 'var(--accent-blue)' : 'var(--accent-red)'} />
                    <strong style={{ fontFamily: 'monospace' }}>{a.id}</strong>
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{new Date(a.timestamp).toLocaleString()}</span>
                    <span style={{ fontSize: '0.85rem', padding: '3px 8px', borderRadius: '12px', backgroundColor: '#374151', fontFamily: 'monospace' }}>Parent: {a.run_id}</span>
                  </div>
                  <div style={{ fontSize: '1.1rem', margin: '0.7rem 0', color: a.acknowledged ? 'var(--text-muted)' : 'white' }}>
                    {a.message}
                  </div>
                </div>
                
                <button 
                  onClick={() => handleAcknowledge(a.id)}
                  disabled={a.acknowledged}
                  style={{ 
                    padding: '0.6rem 1.2rem', 
                    backgroundColor: a.acknowledged ? '#374151' : 'var(--accent-blue)', 
                    color: a.acknowledged ? 'var(--text-muted)' : 'white', 
                    border: 'none', 
                    borderRadius: '6px', 
                    cursor: a.acknowledged ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    fontWeight: 'bold',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <CheckCircle size={18} />
                  {a.acknowledged ? 'Secured' : 'Acknowledge Event'}
                </button>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
