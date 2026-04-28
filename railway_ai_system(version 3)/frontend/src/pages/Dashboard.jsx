import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import UploadPanel from '../components/dashboard/UploadPanel';
import ProgressBar from '../components/dashboard/ProgressBar';
import AnalysisResults from '../components/dashboard/AnalysisResults';

export default function Dashboard() {
  const { runId } = useParams();
  const navigate = useNavigate();

  const [activeRunId, setActiveRunId] = useState(() => runId || localStorage.getItem('railguard_last_run') || null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isComplete, setIsComplete] = useState(() => !!runId || !!localStorage.getItem('railguard_last_run'));

  useEffect(() => {
    if (runId) {
      setActiveRunId(runId);
      setIsComplete(true);
      setIsProcessing(false);
      localStorage.setItem('railguard_last_run', runId);
    }
  }, [runId]);

  const handleUploadStart = () => {
    setIsProcessing(true);
    setIsComplete(false);
  };

  const handleUploadSuccess = (runId) => {
    setActiveRunId(runId);
  };

  const handleAnalysisComplete = () => {
    setIsComplete(true);
    setIsProcessing(false);
    localStorage.setItem('railguard_last_run', activeRunId);
    navigate(`/results/${activeRunId}`);
  };

  const handleReset = () => {
    setActiveRunId(null);
    setIsProcessing(false);
    setIsComplete(false);
    localStorage.removeItem('railguard_last_run');
    navigate('/');
  };

  return (
    <div className="dashboard-container" style={{ maxWidth: '1200px', margin: '3rem auto', padding: '0 2rem' }}>
      <header className="card-glass" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3rem', padding: '2rem 2.5rem', borderRadius: 'var(--radius-xl)' }}>
        <div>
          <h1 className="text-gradient" style={{ fontSize: '2.5rem', marginBottom: '0.25rem' }}>RailGuard AI</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', fontWeight: 500 }}>Advanced Telemetry & Visual Tampering Assessment</p>
        </div>
        {activeRunId && (
          <button className="btn btn-outline" onClick={handleReset}>
            Start New Run
          </button>
        )}
      </header>
      
      {!activeRunId && !isProcessing && (
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '4rem' }}>
          <UploadPanel onUploadStart={handleUploadStart} onUploadSuccess={handleUploadSuccess} />
        </div>
      )}

      {isProcessing && !activeRunId && (
        <div className="card-glass animate-pulse-soft" style={{ textAlign: 'center', padding: '5rem', backgroundColor: 'var(--bg-secondary)' }}>
          <h3 className="text-gradient" style={{ fontSize: '1.8rem' }}>Initializing AI Neural Core...</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>Securing weights and instantiating YOLO Engine</p>
        </div>
      )}

      {activeRunId && !isComplete && (
        <div className="card-glass" style={{ marginTop: '2rem', padding: '2.5rem' }}>
          <ProgressBar runId={activeRunId} onComplete={handleAnalysisComplete} />
        </div>
      )}

      {isComplete && activeRunId && (
        <div style={{ marginTop: '2rem' }}>
          <AnalysisResults runId={activeRunId} />
        </div>
      )}
    </div>
  );
}
