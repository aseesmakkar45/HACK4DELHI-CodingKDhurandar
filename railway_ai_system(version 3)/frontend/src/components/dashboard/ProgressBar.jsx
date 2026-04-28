import React, { useEffect, useState } from 'react';
import { useApi } from '../../hooks/useApi';
import { Loader, CheckCircle } from 'lucide-react';
import './ProgressBar.css';

export default function ProgressBar({ runId, onComplete }) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('initializing');
  const { request } = useApi();

  useEffect(() => {
    let interval;
    
    const pollProgress = async () => {
      try {
        const data = await request(`/analyze/${runId}/progress`);
        
        if (data.status === 'complete') {
          setProgress(100);
          setStatus('complete');
          clearInterval(interval);
          setTimeout(() => onComplete(), 800); // give time to show 100%
        } else {
          // Visual cosmetic loop bounding at 95% until python actually finishes
          setProgress((prev) => {
            const newProg = prev + Math.floor(Math.random() * 8) + 2;
            return Math.min(newProg, 95);
          });
          setStatus('processing');
        }
      } catch (e) {
        console.error("Polling error", e);
      }
    };

    interval = setInterval(pollProgress, 1000);
    return () => clearInterval(interval);
  }, [runId, onComplete, request]);

  return (
    <div className="progress-container card">
      <div className="progress-header">
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {status === 'complete' ? <CheckCircle color="var(--accent-green)" /> : <Loader className="animate-spin" color="var(--accent-blue)" />}
          {status === 'complete' ? 'Analysis Complete' : 'Processing Video...'}
        </h3>
        <span className="progress-percent">{progress}%</span>
      </div>
      
      <div className="progress-track">
        <div 
          className={`progress-fill ${status === 'complete' ? 'complete' : ''}`}
          style={{ width: `${progress}%` }}
        ></div>
      </div>
      
      <p className="progress-subtext text-muted">Run ID: {runId}</p>
    </div>
  );
}
