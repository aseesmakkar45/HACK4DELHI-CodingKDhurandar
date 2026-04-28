import React, { useEffect, useState } from 'react';
import { useApi } from '../../hooks/useApi';
import { ShieldCheck, ShieldAlert, Cpu, Activity } from 'lucide-react';
import DetectionGallery from './DetectionGallery';
import './AnalysisResults.css';

export default function AnalysisResults({ runId }) {
  const [results, setResults] = useState(null);
  const [featureWeights, setFeatureWeights] = useState([]);
  const { request, loading } = useApi();

  useEffect(() => {
    const fetchResults = async () => {
      try {
        // Fetching from FastAPI backend
        const response = await request(`/results/${runId}`);
        if (!response.error) {
            setResults(response);
        } else {
            console.error("Backend Error:", response.error);
        }
        
        // Fetch global feature weights
        const weights = await request(`/results/ml/feature-importance`);
        if (weights && weights.length > 0 && !weights.error) {
            setFeatureWeights(weights);
        }
      } catch (e) {
        console.error("Failed to load results", e);
      }
    };
    fetchResults();
  }, [runId, request]);

  if (!results) return <div className="card-glass animate-pulse-soft">Loading Results...</div>;

  const algoDescriptions = {
    "LogReg": "Base threshold check for standard security violations.",
    "DTree": "Strict rule logic to verify physical threat actions.",
    "RForest": "Eliminates false alarms via team consensus voting.",
    "KNN": "Matches ongoing activity against historical sabotage signatures.",
    "BehaviorEngine": "Skeletal pose analysis detecting crouching, track contact & persistence."
  };

  const operatorModelNames = {
    "LogReg": "Baseline Filter",
    "DTree": "Logic Engine",
    "RForest": "Consensus Engine",
    "KNN": "Pattern Matcher",
    "BehaviorEngine": "Pose Intelligence"
  };

  const operatorFeatureNames = {
    "Dominant Class": "Primary Threat Profile",
    "Total Objects": "Crowd Density",
    "Conf Std": "Movement Unpredictability",
    "Conf Stddev": "Movement Unpredictability",
    "Avg Confidence": "Average Threat Certainty",
    "Avg Area": "Average Object Size",
    "Max Confidence": "Highest Threat Certainty",
    "Anomaly Ratio": "Suspicious Activity Ratio",
    "Max Area": "Largest Target Size",
    "Human Count": "People Detected",
    "Machinery Count": "Equipment Detected"
  };

  const isCritical = results.risk_score > 0.75;

  return (
    <div className="analysis-results">
      <div className="metrics-grid">
        <div className={`card-glass risk-card ${isCritical ? 'critical' : 'safe'}`}>
          <div className="card-header">
            <h3 className="text-gradient">Fused Risk Score</h3>
            {isCritical ? <ShieldAlert color="var(--accent-red)" size={28} /> : <ShieldCheck color="var(--accent-green)" size={28}/>}
          </div>
          <div className="risk-value">{(results.risk_score * 100).toFixed(1)}%</div>
          <p className="risk-label">{isCritical ? 'TAMPERING DETECTED' : 'TRACK CLEAR'}</p>
        </div>
        
        <div className="card-glass" style={{ borderTop: "4px solid var(--accent-purple)", display: "flex", flexDirection: "column" }}>
          <div className="card-header">
            <h3 style={{ color: "var(--text-primary)" }}>Behavioral Diagnostics</h3>
            <Activity color="var(--accent-purple)" size={24} />
          </div>
          <div style={{ flexGrow: 1, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", textAlign: "center", padding: "1rem" }}>
            <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", letterSpacing: "1px", textTransform: "uppercase", fontWeight: 600 }}>Primary Intent Detected</span>
            <h2 style={{ fontSize: "1.4rem", color: "var(--text-primary)", marginTop: "0.5rem", fontWeight: 800 }}>
              {results.primary_behavior ? results.primary_behavior.replace(/_/g, " ") : "NO BEHAVIOR DATA"}
            </h2>
            <p style={{marginTop: "1rem", fontSize: "0.9rem", color: "var(--text-muted)", fontStyle: "italic"}}>Evaluated via Temporal Behavior Engine tracking structural object interactions.</p>
          </div>
        </div>
        
        <div className="card-glass">
          <div className="card-header">
            <h3 style={{ color: "var(--text-primary)" }}>Pipeline Stats</h3>
            <Activity color="var(--accent-blue)" size={24} />
          </div>
          <div className="stat-row">
            <span>Frames Analyzed</span>
            <strong>{results.total_frames_analyzed}</strong>
          </div>
          <div className="stat-row">
            <span>Anomalies Flagged</span>
            <strong style={{ color: 'var(--accent-amber)' }}>{results.anomalies_detected}</strong>
          </div>
          <div className="stat-row">
            <span>Human Tampering</span>
            <strong style={{ color: results.humans_detected > 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
              {results.humans_detected}
            </strong>
          </div>
          <div className="stat-row">
            <span>Dominant Threat</span>
            <strong style={{ textTransform: 'capitalize', color: 'var(--text-primary)' }}>
              {results.dominant_class ? results.dominant_class.replace(/_/g, " ") : "None"}
            </strong>
          </div>
        </div>
        
        <div className="card-glass" style={{ gridColumn: '1 / -1', borderTop: '4px solid var(--accent-purple)' }}>
          <div className="card-header">
            <h3 style={{ color: "var(--text-primary)" }}>Model Breakdown</h3>
            <Cpu color="var(--accent-purple)" size={24} />
          </div>
          <div className="model-bars" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
            {results.models.map(m => (
              <div key={m.name} className="model-bar-container">
                <div className="model-bar-row">
                  <span className="model-name" style={{ width: '170px', fontSize: '0.8rem' }} title={m.name}>
                    {operatorModelNames[m.name] ? `${m.name} (${operatorModelNames[m.name]})` : m.name}
                  </span>
                  <div className="model-bar-bg">
                    <div className="model-bar-fill" style={{ width: `${m.score * 100}%`, backgroundColor: m.score > 0.5 ? 'var(--accent-red)' : 'var(--accent-blue)' }}></div>
                  </div>
                  <span className="model-score">{(m.score * 100).toFixed(0)}%</span>
                </div>
                <div className="model-description" style={{ paddingLeft: '0', marginTop: '0.3rem' }}>
                  {algoDescriptions[m.name] || "Evaluates fused intelligence."}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* --- NEW INTERPRETABILITY CARD --- */}
        {featureWeights.length > 0 && (
          <div className="card-glass" style={{ gridColumn: '1 / -1', borderTop: "4px solid var(--accent-amber)" }}>
            <div className="card-header">
              <h3 style={{ color: "var(--text-primary)" }}>AI Reasoning Factors</h3>
              <Activity color="var(--accent-amber)" size={24} />
            </div>
            <p style={{ fontSize: '0.85rem', color: "var(--text-muted)", marginBottom: "1.5rem" }}>
              Displays how the AI distributes its "thinking capacity". The percentages show how much weight the system places on each factor when calculating the final threat score (totaling 100%).
            </p>
            <div className="model-bars" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
              {featureWeights.slice(0, 6).map(feat => (
                <div key={feat.name} className="model-bar-container">
                  <div className="model-bar-row">
                     <span className="model-name" style={{ width: '150px' }} title={feat.name}>
                        {operatorFeatureNames[feat.name] || feat.name}
                     </span>
                    <div className="model-bar-bg">
                      <div className="model-bar-fill" style={{ width: `${feat.value * 2}%`, backgroundColor: 'var(--accent-amber)' }}></div>
                    </div>
                    <span className="model-score" style={{ width: '50px' }}>{feat.value}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
      
      <div style={{ marginTop: '2rem' }}>
        <DetectionGallery timeline={results.incident_events || []} />
      </div>
    </div>
  );
}
