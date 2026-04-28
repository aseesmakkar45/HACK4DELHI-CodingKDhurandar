import React, { useState } from 'react';
import { Maximize2, AlertTriangle, Clock } from 'lucide-react';
import './DetectionGallery.css';

export default function DetectionGallery({ timeline = [] }) {
  // NOTE: timeline property now receives the clustered incident_events array.
  
  if (!timeline || timeline.length === 0) {
    return (
      <div className="card-glass" style={{ padding: '3rem', textAlign: 'center' }}>
        <h3 style={{ color: 'var(--accent-green)' }}>No Incidents Detected</h3>
        <p style={{ color: 'var(--text-muted)' }}>The track is completely clear.</p>
      </div>
    );
  }

  return (
    <div className="card-glass detection-gallery">
      <div className="card-header" style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '1.4rem', color: 'var(--text-primary)' }}>Forensic Incident Ledger</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>AI clustered warning blocks</p>
        </div>
        <span style={{ color: 'var(--accent-blue)', fontWeight: 600, fontSize: '1rem', backgroundColor: 'var(--accent-blue-light)', padding: '0.4rem 1rem', borderRadius: 'var(--radius-xl)' }}>
          {timeline.length} Incidents Logged
        </span>
      </div>
      
      <div className="ledger-stack">
        {timeline.map((incident, index) => {
          // Duration Calculation
          const durationSec = Math.max(1, (incident.duration_frames / 1)).toFixed(1); // Assuming 1 FPS Extraction
          const startMin = Math.floor(incident.start_frame_idx / 60).toString().padStart(2, '0');
          const startSec = (incident.start_frame_idx % 60).toString().padStart(2, '0');
          const timestampStr = `${startMin}:${startSec}`;

          const frameData = incident.representative_frame;
          if (!frameData) return null;

          const imageSrc = frameData.frame ? `/api/${frameData.frame.replace(/\\/g, '/')}` : '';
          const peakRisk = incident.peak_risk || 0;
          
          let displayTitle = incident.event_type ? incident.event_type.replace(/_/g, ' ') : "UNKNOWN EVENT";
          let displayColor = "var(--text-secondary)";
          if (peakRisk >= 0.75) {
             displayColor = "var(--accent-red)";
          } else if (peakRisk >= 0.4) {
             displayColor = "var(--accent-amber)";
          } else {
             displayColor = "var(--accent-blue)";
          }

          return (
            <div key={index} className="ledger-item">
              <div className="ledger-frame-container">
                <div className="frame-placeholder" style={{ backgroundImage: `url(${imageSrc})` }}>
                  {frameData.yolo_raw && frameData.yolo_raw.map((bbox, bIdx) => {
                    if (bbox.confidence < 0.25) return null; 
                    let w_pct, h_pct, top_pct, left_pct;
                    const realW = frameData.img_width || 1920;
                    const realH = frameData.img_height || 1080;
                    
                    if (bbox.obb_xywhr) {
                      w_pct = (bbox.obb_xywhr[2] / realW) * 100;
                      h_pct = (bbox.obb_xywhr[3] / realH) * 100;
                      top_pct = ((bbox.obb_xywhr[1] / realH) * 100) - (h_pct / 2);
                      left_pct = ((bbox.obb_xywhr[0] / realW) * 100) - (w_pct / 2);
                    } else if (bbox.bbox) {
                      const x1 = bbox.bbox[0]; const y1 = bbox.bbox[1];
                      const x2 = bbox.bbox[2]; const y2 = bbox.bbox[3];
                      w_pct = ((x2 - x1) / realW) * 100;
                      h_pct = ((y2 - y1) / realH) * 100;
                      top_pct = (y1 / realH) * 100;
                      left_pct = (x1 / realW) * 100;
                    } else return null;

                    // Keypoint colors: head=cyan, torso=green, arms=yellow, legs=magenta
                    const kpColors = [
                      '#00ffff','#00ffff','#00ffff','#00ffff','#00ffff', // nose, eyes, ears (0-4)
                      '#00ff88','#00ff88',  // shoulders (5-6)
                      '#ffdd00','#ffdd00',  // elbows (7-8)
                      '#ff6600','#ff6600',  // wrists (9-10)
                      '#00ff88','#00ff88',  // hips (11-12)
                      '#ff00ff','#ff00ff',  // knees (13-14)
                      '#ff0066','#ff0066'   // ankles (15-16)
                    ];

                    // Skeleton connections: pairs of keypoint indices to draw lines between
                    const skeleton = [
                      [5,6],[5,7],[7,9],[6,8],[8,10], // shoulders-elbows-wrists
                      [5,11],[6,12],[11,12],           // torso
                      [11,13],[13,15],[12,14],[14,16]  // hips-knees-ankles
                    ];

                    return (
                      <React.Fragment key={bIdx}>
                        <div className="mock-bbox" style={{
                          top: `${top_pct}%`, left: `${left_pct}%`, width: `${w_pct}%`, height: `${h_pct}%`
                        }}></div>
                        {/* Render 17 YOLO-Pose Keypoints */}
                        {bbox.keypoints && bbox.keypoints.length > 0 && bbox.keypoints.map((kp, kpIdx) => {
                          if (!kp || kp[0] === 0 && kp[1] === 0) return null;
                          const kpLeft = (kp[0] / realW) * 100;
                          const kpTop = (kp[1] / realH) * 100;
                          return (
                            <div key={`kp-${bIdx}-${kpIdx}`} className="pose-keypoint" style={{
                              position: 'absolute',
                              left: `${kpLeft}%`,
                              top: `${kpTop}%`,
                              width: '8px',
                              height: '8px',
                              borderRadius: '50%',
                              backgroundColor: kpColors[kpIdx] || '#fff',
                              border: '2px solid rgba(0,0,0,0.6)',
                              transform: 'translate(-50%, -50%)',
                              zIndex: 15,
                              boxShadow: `0 0 6px ${kpColors[kpIdx] || '#fff'}`
                            }}></div>
                          );
                        })}
                        {/* Render Skeleton Lines via SVG */}
                        {bbox.keypoints && bbox.keypoints.length >= 15 && (
                          <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 14, pointerEvents: 'none' }}>
                            {skeleton.map(([a, b], sIdx) => {
                              const kpA = bbox.keypoints[a];
                              const kpB = bbox.keypoints[b];
                              if (!kpA || !kpB || (kpA[0]===0 && kpA[1]===0) || (kpB[0]===0 && kpB[1]===0)) return null;
                              return (
                                <line key={`sk-${bIdx}-${sIdx}`}
                                  x1={`${(kpA[0]/realW)*100}%`} y1={`${(kpA[1]/realH)*100}%`}
                                  x2={`${(kpB[0]/realW)*100}%`} y2={`${(kpB[1]/realH)*100}%`}
                                  stroke="#00ff88" strokeWidth="2" strokeOpacity="0.7"
                                />
                              );
                            })}
                          </svg>
                        )}
                      </React.Fragment>
                    );
                  })}
                  <div className="frame-overlay"><button className="expand-btn"><Maximize2 size={16} /></button></div>
                </div>
              </div>
              
              <div className="ledger-details">
                <div className="ledger-header-row">
                  <span className="incident-badge">INCIDENT #{index + 1}</span>
                  <div className="timestamp-block">
                    <Clock size={16} color="var(--text-muted)"/>
                    <span>T+{timestampStr}</span>
                  </div>
                </div>
                
                <h4 className="event-title" style={{ color: displayColor }}>
                  <AlertTriangle size={20} />
                  {displayTitle}
                </h4>
                
                <div className="event-metrics">
                  <div className="metric-box">
                    <span className="metric-label">Peak Fusion Risk</span>
                    <span className="metric-value" style={{ color: peakRisk > 0.75 ? "var(--accent-red)" : "var(--accent-amber)" }}>
                      {(peakRisk * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-label">Incident Duration</span>
                    <span className="metric-value">{durationSec} Seconds</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
