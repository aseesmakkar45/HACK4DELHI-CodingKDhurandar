import { useState, useEffect, useRef } from 'react';
import './App.css';
import './legacy.css';

const REGIONS = ['Northern India', 'Southern India', 'Western India', 'Eastern India', 'Central India'];

const CITIES = {
  'Northern India': ['Delhi', 'Chandigarh', 'Jaipur', 'Lucknow', 'Kanpur'],
  'Southern India': ['Chennai', 'Bengaluru', 'Hyderabad', 'Kochi', 'Madurai'],
  'Western India': ['Mumbai', 'Ahmedabad', 'Pune', 'Surat', 'Nagpur'],
  'Eastern India': ['Kolkata', 'Patna', 'Bhubaneswar', 'Guwahati', 'Ranchi'],
  'Central India': ['Bhopal', 'Indore', 'Raipur', 'Jabalpur', 'Gwalior']
};

const DUMMY_IMAGES = [
  "https://images.unsplash.com/photo-1541427468627-a475267a57a8?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1563206767-5b18f21f7a40?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1474487548417-781cb71495f3?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?auto=format&fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1517524008697-84bbe3c3fd98?auto=format&fit=crop&w=400&q=80"
];

function App() {
  const [loggedOut, setLoggedOut] = useState(true);
  const [accessLevel, setAccessLevel] = useState('National Admin');
  const [region, setRegion] = useState('Northern India');
  const [city, setCity] = useState('Delhi');
  const [controlRoom, setControlRoom] = useState('CR-1');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  
  const [mainTab, setMainTab] = useState('home');
  const [activeScope, setActiveScope] = useState('All India');
  const [alerts, setAlerts] = useState([]);
  const [viewEvidence, setViewEvidence] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [maintenanceMode, setMaintenanceMode] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const audioRef = useRef(null);

  const handleLogin = (e) => {
    e.preventDefault();
    if (password === "123" || password === "admin") {
      setLoggedOut(false);
      let scopeStr = "All India";
      if (accessLevel === "Regional Admin") scopeStr = region;
      if (accessLevel === "City Admin") scopeStr = city;
      if (accessLevel === "Control Room") scopeStr = `${city} | ${controlRoom}`;
      
      setActiveScope(scopeStr);
      setMainTab('home');
      setLoginError('');
      setCurrentPage(1);
      
      const zoneStr = scopeStr === "All India" ? "" : `?zone=${encodeURIComponent(city)}`;
      fetch(`/api/incidents${zoneStr}`)
        .then(res => res.json())
        .then(data => setAlerts(data))
        .catch(e => console.log(e));
    } else {
      setLoginError('Invalid Administrator credentials. Try 123');
    }
  };

  const handleAcknowledge = async (id) => {
    const comment = prompt("Enter resolution comment / operator action taken:", "Track inspected. Resolved.");
    if (comment === null) return;
    try {
      await fetch(`/api/incidents/${id}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comment })
      });
      setAlerts(alerts.map(a => a.id === id ? { ...a, status: 'resolved', operator_comment: comment, resolved_at: Date.now() / 1000 } : a));
    } catch(e) {}
  };

  const toggleMaintenance = async () => {
    const endpoint = maintenanceMode ? '/api/maintenance/disable' : '/api/maintenance/enable';
    const body = maintenanceMode ? {} : { duration_minutes: 120 };
    try {
      await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      setMaintenanceMode(!maintenanceMode);
    } catch(e) { console.log(e); }
  };

  const toggleDemo = async () => {
    const endpoint = demoMode ? '/api/demo/stop' : '/api/demo/start';
    try {
      await fetch(endpoint, { method: 'POST' });
      setDemoMode(!demoMode);
    } catch(e) { console.log(e); }
  };

  const clearHistory = async () => {
    if (!confirm('Clear ALL incident history from the database? This cannot be undone.')) return;
    try {
      await fetch('/api/incidents/clear', { method: 'DELETE' });
      setAlerts([]);
    } catch(e) { console.log(e); }
  };

  useEffect(() => {
    if (loggedOut) return;
    const wsUrl = `ws://${window.location.host}/ws/alarms`;
    let ws = new WebSocket(wsUrl);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'alert') {
          setAlerts(prev => {
            if (prev.find(a => a.id === data.id)) return prev;
            return [data, ...prev];
          });
          setViewEvidence(data.id);
          if (data.level === 'critical' && audioRef.current) {
             audioRef.current.play().catch(e => console.log(e));
          }
        }
      } catch(e) {}
    };
    return () => ws.close();
  }, [loggedOut]);


  if (loggedOut) {
    return (
      <div className="login-wrapper">
        <div className="login-card" style={{width: 500}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 15, marginBottom: 25}}>
             <svg width="48" height="48" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 32 C 16 12, 48 12, 60 32 C 48 52, 16 52, 4 32 Z" stroke="var(--accent-gold)" strokeWidth="4" fill="transparent"/>
                <circle cx="32" cy="32" r="12" fill="var(--bg-secondary)" stroke="var(--accent-gold)" strokeWidth="3"/>
                <path d="M28 20 L24 44 M36 20 L40 44" stroke="#3b82f6" strokeWidth="3"/>
                <path d="M26 26 L38 26 M25 32 L39 32 M24 38 L40 38" stroke="#3b82f6" strokeWidth="2"/>
             </svg>
             <h2 style={{fontSize: 22, fontWeight: 500, margin: 0}}>Ministry of Railways<br/><span style={{fontSize: 14, color: 'var(--text-secondary)'}}>RailDrishti AI Platform</span></h2>
          </div>
          <form onSubmit={handleLogin}>
            <div className="form-group">
              <label>Role / Access Level</label>
              <select value={accessLevel} onChange={e => {
                  setAccessLevel(e.target.value);
                  setRegion('Northern India');
                  setCity('Delhi');
              }}>
                <option value="National Admin">Railway Board (National)</option>
                <option value="Regional Admin">Zonal Headquarters</option>
                <option value="City Admin">Divisional Office</option>
                <option value="Control Room">Station Control Room</option>
              </select>
            </div>
            {accessLevel !== "National Admin" && (
                <div className="form-group">
                  <label>Select Railway Zone</label>
                  <select value={region} onChange={e => {setRegion(e.target.value); setCity(CITIES[e.target.value][0]);}}>
                    {REGIONS.map(r => <option key={r} value={r}>{r}</option>)}
                  </select>
                </div>
            )}
            {(accessLevel === "City Admin" || accessLevel === "Control Room") && (
                <div className="form-group">
                  <label>Select Division</label>
                  <select value={city} onChange={e => setCity(e.target.value)}>
                    {CITIES[region].map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
            )}
            {accessLevel === "Control Room" && (
                <div className="form-group">
                  <label>Assigned Desk</label>
                  <select value={controlRoom} onChange={e => setControlRoom(e.target.value)}>
                    {[1,2,3,4,5].map(n => <option key={n} value={`CR-${n}`}>{city} Control Desk {n}</option>)}
                  </select>
                </div>
            )}
            <div style={{display: 'flex', gap: 10, marginTop: 25}}>
                <div className="form-group" style={{flex: 1, marginBottom: 0}}>
                  <label>Operator ID</label>
                  <input type="text" value={username} onChange={e => setUsername(e.target.value)} placeholder="Employee ID" required/>
                </div>
                <div className="form-group" style={{flex: 1, marginBottom: 0}}>
                  <label>Passcode</label>
                  <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="***" required/>
                </div>
            </div>
            {loginError && <p style={{color: '#ef4444', fontSize: '13px', marginTop: 15}}>{loginError}</p>}
            <button type="submit" className="btn-primary" style={{marginTop: 25}}>Login to Operator Dashboard</button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container" style={{flexDirection: 'column'}}>
      <audio ref={audioRef} src="https://assets.mixkit.co/active_storage/sfx/995/995-preview.mp3" preload="auto"></audio>
      <header className="top-bar">
        <div className="title-section" style={{display: 'flex', alignItems: 'center', gap: 15}}>
          <svg width="32" height="32" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4 32 C 16 12, 48 12, 60 32 C 48 52, 16 52, 4 32 Z" stroke="var(--accent-gold)" strokeWidth="4" fill="transparent"/>
            <circle cx="32" cy="32" r="12" fill="var(--bg-secondary)" stroke="var(--accent-gold)" strokeWidth="3"/>
            <path d="M28 20 L24 44 M36 20 L40 44" stroke="#3b82f6" strokeWidth="3"/>
            <path d="M26 26 L38 26 M25 32 L39 32 M24 38 L40 38" stroke="#3b82f6" strokeWidth="2"/>
          </svg>
          <div>
            <h1 style={{color: 'var(--text-primary)'}}>Ministry of Railways | <span style={{color: 'var(--accent-gold)'}}>RailDrishti AI</span></h1>
            <p>Operator: {accessLevel} | Scope: {activeScope}</p>
          </div>
        </div>
        <button onClick={() => {setLoggedOut(true); setAlerts([])}} className="upload-button" style={{background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-secondary)'}}>Log Out</button>
        <button onClick={toggleMaintenance} style={{padding: '6px 14px', fontSize: 12, borderRadius: 2, background: maintenanceMode ? '#f59e0b' : 'var(--bg-tertiary)', border: `1px solid ${maintenanceMode ? '#f59e0b' : 'var(--border-color)'}`, color: maintenanceMode ? '#000' : 'var(--text-secondary)', cursor: 'pointer', fontWeight: maintenanceMode ? 600 : 400}}>
          {maintenanceMode ? '🔧 MAINTENANCE ON' : '🔧 Maintenance Mode'}
        </button>
        <button onClick={toggleDemo} style={{padding: '6px 14px', fontSize: 12, borderRadius: 2, background: demoMode ? '#10b981' : 'var(--bg-tertiary)', border: `1px solid ${demoMode ? '#10b981' : 'var(--border-color)'}`, color: demoMode ? '#fff' : 'var(--text-secondary)', cursor: 'pointer', fontWeight: demoMode ? 600 : 400}}>
          {demoMode ? '▶ DEMO LIVE' : '▶ Demo Mode'}
        </button>
        <button onClick={clearHistory} style={{padding: '6px 14px', fontSize: 12, borderRadius: 2, background: 'var(--bg-tertiary)', border: '1px solid #ef4444', color: '#ef4444', cursor: 'pointer'}}>
          🗑 Clear History
        </button>
      </header>
      <nav className="top-nav">
        <div className={`top-nav-item ${mainTab === 'home' ? 'active' : ''}`} onClick={() => setMainTab('home')}>Overview</div>
        <div className={`top-nav-item ${mainTab === 'run' ? 'active' : ''}`} onClick={() => setMainTab('run')}>Live Monitoring</div>
        <div className={`top-nav-item ${mainTab === 'trains' ? 'active' : ''}`} onClick={() => setMainTab('trains')}>Train Tracking</div>
        <div className={`top-nav-item ${mainTab === 'history' ? 'active' : ''}`} onClick={() => setMainTab('history')}>Incident Database</div>
        <div className={`top-nav-item ${mainTab === 'maintenance' ? 'active' : ''}`} onClick={() => setMainTab('maintenance')}>Track & Weather Log</div>
      </nav>

      {mainTab === 'history' && (
        <div className="page-container" style={{maxWidth: 1200}}>
           <h1 style={{fontSize: 20, marginBottom: 5}}>Incident Database</h1>
           <p style={{color: 'var(--text-secondary)', marginBottom: 20, fontSize: 13}}>Historical review and forensics of all logged security events within {activeScope}.</p>
           
           <div style={{display: 'flex', flexDirection: 'column', gap: 20}}>
             {alerts.map(alert => (
               <div key={alert.id} className="db-panel" style={{background: 'var(--bg-secondary)', padding: 20, borderRadius: 4, border: '1px solid var(--border-color)', display: 'flex', gap: 20}}>
                 <div style={{width: 250, flexShrink: 0}}>
                    <h3 style={{fontSize: 13, marginBottom: 10, color: 'var(--text-primary)', textTransform: 'none'}}>Forensic Intel & GPS</h3>
                    <div style={{width: '100%', height: 100, background: '#000', borderRadius: '4px 4px 0 0', overflow: 'hidden'}}>
                       <iframe src="https://www.youtube.com/embed/y2m5A6b7w6k?autoplay=0&controls=0&mute=1" style={{width: '100%', height: '100%', pointerEvents: 'none'}} frameBorder="0"></iframe>
                    </div>
                    <div style={{width: '100%', height: 80, overflow: 'hidden', borderRadius: '0 0 4px 4px'}}>
                       <iframe src={`https://maps.google.com/maps?q=${alert.lat||'28.6139'},${alert.lng||'77.2090'}&t=&z=15&ie=UTF8&iwloc=&output=embed`} width="100%" height="80" style={{border: 0}} loading="lazy"></iframe>
                    </div>
                 </div>
                 <div style={{flex: 1, display: 'flex', flexDirection: 'column', gap: 15, justifyContent: 'center'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed var(--border-color)', paddingBottom: 10}}>
                       <div>
                          <strong style={{color: 'var(--accent-gold)'}}>Incident #{alert.id}</strong><br/>
                          <span style={{fontSize: 12, color: 'var(--text-secondary)'}}>Hierarchy: {alert.region || 'Region'} &gt; {alert.city || alert.zone} &gt; {alert.control_room ? alert.control_room.replace('CR-', 'Assigned Desk ') : 'Assigned Desk 1'} &gt; {alert.sensor}</span><br/>
                          <span style={{fontSize: 11, color: '#3b82f6', fontFamily: 'monospace'}}>GPS Mapping: {alert.lat || '28.61'}, {alert.lng || '77.20'}</span>
                       </div>
                       <div style={{textAlign: 'right'}}>
                          <span style={{fontSize: 11, padding: '4px 10px', borderRadius: 2, background: alert.status === 'resolved' ? '#10b981' : 'var(--alert-red)', color: 'white', fontWeight: 600}}>{alert.status === 'resolved' ? 'RESOLVED' : 'ACTIVE'}</span>
                       </div>
                    </div>
                    <div>
                       <strong style={{color: 'var(--text-primary)', fontSize: 13}}>Threat Signature:</strong><br/>
                       <span style={{fontSize: 13, color: 'var(--text-secondary)'}}>{alert.message}</span>
                    </div>
                    <div style={{display: 'flex', gap: 30, fontSize: 12, color: 'var(--text-secondary)'}}>
                       <div><strong>Logged:</strong> {new Date(alert.timestamp * 1000).toLocaleString()}</div>
                       {alert.resolved_at && (
                          <div><strong>Resolved At:</strong> {new Date(alert.resolved_at * 1000).toLocaleString()}</div>
                       )}
                    </div>
                    {alert.operator_comment && (
                       <div style={{background: 'var(--bg-tertiary)', padding: 12, borderRadius: 4, borderLeft: '3px solid #10b981', fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.5}}>
                          <strong>Operator Comment:</strong> {alert.operator_comment}
                       </div>
                    )}
                 </div>
               </div>
             ))}
             
             {alerts.length === 0 && (
                <div style={{padding: 40, textAlign: 'center', color: 'var(--text-secondary)', border: '1px dashed var(--border-color)', borderRadius: 4}}>
                   No incidents logged in the system.
                </div>
             )}
           </div>
        </div>
      )}

      {mainTab === 'home' && (() => {
        let totalFeeds = 12;
        if (accessLevel === "City Admin") totalFeeds = 60;
        if (accessLevel === "Regional Admin") totalFeeds = 300;
        if (accessLevel === "National Admin") totalFeeds = 1500;
        return (
        <div className="page-container govt-dashboard">
          <div className="govt-header" style={{borderBottom: '1px solid var(--border-color)', paddingBottom: 15, marginBottom: 25}}>
            <h1 style={{margin: '0 0 5px 0', fontSize: 20, fontWeight: 500, color: 'var(--text-primary)'}}>RailDrishti AI: Central Operations Dashboard</h1>
            <div style={{fontSize: 13, color: 'var(--text-secondary)'}}>System Time: {new Date().toUTCString()}</div>
          </div>

          <div className="govt-grid">
            <div className="govt-left-column" style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignContent: 'start'}}>
                <div className="govt-panel widget-intel">
                   <h3 style={{textTransform: 'none'}}>System Status</h3>
                   <div className="intel-data">
                     <div className="data-row"><span>Region/Scope:</span> <strong>{activeScope}</strong></div>
                     <div className="data-row"><span>Active Feeds:</span> <strong>{totalFeeds} Cameras Online</strong></div>
                     <div className="data-row"><span>Server Connection:</span> <strong style={{color:'#10b981'}}>Online (Latency: 12ms)</strong></div>
                     <div className="data-row"><span>Analytics Module:</span> <strong style={{color:'#10b981'}}>Running</strong></div>
                   </div>
                </div>

                <div className="govt-panel widget-threat">
                   <h3 style={{textTransform: 'none'}}>Pending Alerts</h3>
                   <div className="threat-status">
                     <div className="status-circle" style={{borderColor: alerts.filter(a=>a.status==='active').length > 0 ? 'var(--alert-red)' : '#10b981'}}>
                        <span className="count">{alerts.filter(a=>a.status==='active').length}</span>
                        <span className="label">Unresolved</span>
                     </div>
                     <div className="status-meta">
                       <div>Total Events Logged: {alerts.length}</div>
                       <div>Last Event: {alerts.length > 0 ? new Date(alerts[0].timestamp*1000).toLocaleTimeString() : 'None'}</div>
                       <div>Resolution Rate: {alerts.length > 0 ? Math.round((alerts.filter(a=>a.status==='resolved').length / alerts.length)*100) : 100}%</div>
                     </div>
                   </div>
                </div>

                {accessLevel === "National Admin" && (
                    <div className="govt-panel widget-map" style={{gridColumn: 'span 2'}}>
                       <h3 style={{textTransform: 'none'}}>Zonal Network Status</h3>
                       <div style={{height: 180, width: '100%', background: 'var(--bg-tertiary)', borderRadius: 4, display: 'flex', alignItems: 'flex-end', justifyContent: 'space-around', paddingBottom: 10, marginTop: 15}}>
                          {REGIONS.map((r, i) => {
                             const health = 70 + Math.random()*30;
                             return (
                             <div key={r} style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                                <div style={{height: health + '%', width: 40, background: health > 85 ? '#3b82f6' : 'var(--alert-red)', borderTop: '2px solid #60a5fa', opacity: 0.8}}></div>
                                <span style={{fontSize: 10, color: 'var(--text-secondary)', marginTop: 8}}>{r.split(' ')[0]} Rly</span>
                             </div>
                          )})}
                       </div>
                    </div>
                )}
                
                {(accessLevel === "City Admin" || accessLevel === "Regional Admin") && (
                    <div className="govt-panel widget-map" style={{gridColumn: 'span 2'}}>
                       <h3 style={{textTransform: 'none'}}>Divisional Summary: {activeScope.toUpperCase()}</h3>
                       <p style={{fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, marginTop: 15}}>System is functioning normally. Local RPF response teams are available. Camera feeds are routing correctly to the divisional headquarters without latency issues.</p>
                       <div style={{display: 'flex', gap: 15, marginTop: 20}}>
                          <div style={{flex: 1, padding: 15, background: 'var(--bg-tertiary)', borderLeft: '4px solid #10b981', fontSize: 13}}>Maintenance Status: <br/><strong style={{fontSize: 15, color: 'white', fontWeight: 500}}>Routine Checkups Only</strong></div>
                          <div style={{flex: 1, padding: 15, background: 'var(--bg-tertiary)', borderLeft: '4px solid #3b82f6', fontSize: 13}}>Weather Advisory: <br/><strong style={{fontSize: 15, color: 'white', fontWeight: 500}}>No Alerts Issued</strong></div>
                       </div>
                    </div>
                )}

                {accessLevel === "Control Room" && (
                    <div className="govt-panel widget-map" style={{gridColumn: 'span 2'}}>
                       <h3 style={{textTransform: 'none'}}>Local Station Overview</h3>
                       <p style={{fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, marginTop: 15}}>Operator shift is active. Live video analytics are routing to your dashboard for monitoring. Please review all incoming alerts marked in the Incident Log.</p>
                       <div style={{display: 'flex', gap: 15, marginTop: 20}}>
                          <div style={{flex: 1, padding: 15, background: 'var(--bg-tertiary)', borderLeft: '4px solid #3b82f6', fontSize: 13}}>Current Shift: <br/><strong style={{fontSize: 15, color: 'white', fontWeight: 500}}>08:00 - 16:00 (Standard)</strong></div>
                          <div style={{flex: 1, padding: 15, background: 'var(--bg-tertiary)', borderLeft: '4px solid #10b981', fontSize: 13}}>System Health: <br/><strong style={{fontSize: 15, color: 'white', fontWeight: 500}}>Optimal</strong></div>
                       </div>
                    </div>
                )}
            </div>
            
            <div className="govt-right-column">
               <div className="govt-panel" style={{height: '100%'}}>
                  <img src="https://images.unsplash.com/photo-1474487548417-781cb71495f3?auto=format&fit=crop&w=800&q=80" style={{width: '100%', height: 160, objectFit: 'cover', borderRadius: 4, marginBottom: 15, opacity: 0.85, filter: 'grayscale(20%)'}} alt="Railway Infrastructure"/>
                  <h3 style={{textTransform: 'none', color: 'var(--accent-gold)'}}>About RailDrishti AI Architecture</h3>
                  <div style={{fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7}}>
                     <p><strong>RailDrishti AI</strong> is the next-generation optical safety envelope deployed across the Indian Railways network. Harmonizing securely with the KAVACH anti-collision protocol, RailDrishti provides real-time, edge-computed video analytics for instantaneous threat identification.</p>
                     
                     <p style={{marginTop: 15, color: 'var(--text-primary)'}}><strong>Core Operational Flow:</strong></p>
                     <ul style={{marginTop: 5, paddingLeft: 20, marginBottom: 15}}>
                        <li style={{marginBottom: 8}}><strong>Edge Processing:</strong> Live CCTV feeds are analyzed locally at Station Nodes utilizing deeply quantized YOLOv8 neural networks.</li>
                        <li style={{marginBottom: 8}}><strong>Bandwidth Optimization:</strong> Only active anomaly metadata and highly compressed telemetry loops are uplinked to Zonal servers over RailTel lines, preserving 98% network stability.</li>
                        <li style={{marginBottom: 8}}><strong>Sensor Fusion Pipeline:</strong> Optical intelligence streams natively interface with Track Vibration Monitors ensuring near-zero false-positive emergency alerts.</li>
                     </ul>
                     
                     <p>The system autonomously pinpoints track tampering, unauthorized foot-traffic, and obstacle incursions milliseconds before Locopilots make visual contact, facilitating rapid-response deceleration advisories directly to the locomotive cab.</p>
                  </div>
               </div>
            </div>

          </div>
          <div style={{marginTop: 40, borderTop: '1px solid var(--border-color)', paddingTop: 15, fontSize: 11, color: 'var(--text-secondary)', textAlign: 'center'}}>
            Note: Information displayed on this portal is strictly for authorized Indian Railways and RPF personnel.
          </div>
        </div>
        );
      })()}

      {mainTab === 'trains' && (
        <div className="page-container" style={{maxWidth: 1000}}>
           <h1 style={{fontSize: 20, marginBottom: 5}}>Active Train Schedule</h1>
           <p style={{color: 'var(--text-secondary)', marginBottom: 20, fontSize: 13}}>Live telemetry for scheduled departures, arrivals, and operational statuses within {activeScope}.</p>
           <div className="db-panel" style={{background: 'var(--bg-secondary)', padding: 20, borderRadius: 4, border: '1px solid var(--border-color)'}}>
               <table style={{width: '100%', fontSize: 13, textAlign: 'left', color: 'var(--text-secondary)', borderCollapse: 'collapse'}}>
                 <thead>
                   <tr style={{color: 'var(--text-primary)', borderBottom: '1px solid var(--border-color)'}}>
                     <th style={{padding: '12px 0'}}>Train No. / Name</th>
                     <th>Route</th>
                     <th>Scheduled Dep.</th>
                     <th>Expected Arr.</th>
                     <th>Live Status</th>
                   </tr>
                 </thead>
                 <tbody>
                   <tr style={{borderBottom: '1px solid var(--border-color)'}}>
                     <td style={{padding: '12px 0'}}>12004 SHATABDI EXP</td><td>New Delhi - Lucknow Jn</td><td>06:10</td><td>12:40</td><td style={{color: '#10b981'}}>On Time</td>
                   </tr>
                   <tr style={{borderBottom: '1px solid var(--border-color)'}}>
                     <td style={{padding: '12px 0'}}>12951 RAJDHANI EXP</td><td>Mumbai Central - New Delhi</td><td>17:00</td><td>08:32</td><td style={{color: 'var(--accent-gold)'}}>Delayed 20 min</td>
                   </tr>
                   <tr style={{borderBottom: '1px solid var(--border-color)'}}>
                     <td style={{padding: '12px 0'}}>12011 KALKA SHTBDI</td><td>New Delhi - Kalka</td><td>07:40</td><td>11:45</td><td style={{color: '#10b981'}}>On Time</td>
                   </tr>
                 </tbody>
               </table>
           </div>
        </div>
      )}

      {mainTab === 'maintenance' && (
        <div className="page-container" style={{maxWidth: 1000}}>
           <h1 style={{fontSize: 20, marginBottom: 5}}>Maintenance & Weather Log</h1>
           <p style={{color: 'var(--text-secondary)', marginBottom: 20, fontSize: 13}}>Track repair blocks, speed restrictions, and weather reports across {activeScope}.</p>
           <div style={{display: 'flex', gap: 20}}>
             <div className="db-panel" style={{flex: 1, background: 'var(--bg-secondary)', padding: 20, borderRadius: 4, border: '1px solid var(--border-color)'}}>
                 <h2 style={{fontSize: 15, marginTop: 0, fontWeight: 500}}>Active Track Blocks</h2>
                 <div style={{fontSize: 13, color: 'var(--text-secondary)', marginTop: 15, lineHeight: 1.6}}>
                   <div style={{marginBottom: 15, borderBottom: '1px solid var(--border-color)', paddingBottom: 10}}>
                     <strong style={{color: 'var(--text-primary)'}}>Block #1X-NDLS:</strong> Main Platform 3 (<span style={{color: 'var(--alert-red)'}}>Traffic Block Applied</span>)<br/>
                     <span style={{color: 'var(--text-secondary)'}}>Department: Engineering | Duration: 02:00 - 05:00 hrs</span>
                   </div>
                   <div style={{marginBottom: 15, borderBottom: '1px solid var(--border-color)', paddingBottom: 10}}>
                     <strong style={{color: 'var(--text-primary)'}}>Block #2Y-SUB:</strong> Suburban Line Welding (<span style={{color: 'var(--accent-gold)'}}>Caution Order: 30km/h</span>)<br/>
                     <span style={{color: 'var(--text-secondary)'}}>Department: Track Maintenance | Duration: 09:00 - 11:00 hrs</span>
                   </div>
                 </div>
             </div>
             <div className="db-panel" style={{width: 300, background: 'var(--bg-secondary)', padding: 20, borderRadius: 4, border: '1px solid var(--border-color)'}}>
                 <h2 style={{fontSize: 15, marginTop: 0, fontWeight: 500}}>Local Weather Report</h2>
                 <div style={{fontSize: 13, color: 'var(--text-secondary)', marginTop: 15, lineHeight: 1.8}}>
                   <p style={{margin: '5px 0'}}>Visibility: <strong>Clear (4000m)</strong></p>
                   <p style={{margin: '5px 0'}}>Temperature: <strong>24°C</strong></p>
                   <p style={{margin: '5px 0'}}>Rainfall: <strong>None</strong></p>
                   <p style={{margin: '5px 0'}}>Fog Probability: <strong>Low</strong></p>
                 </div>
             </div>
           </div>
        </div>
      )}

      {mainTab === 'run' && (() => {
        let totalFeeds = 12;
        let feedsPerPage = 12;
        let gridCols = 'repeat(4, 1fr)';
        let imgHeight = 110;

        if (accessLevel === "City Admin") { 
           totalFeeds = 60; feedsPerPage = 60; gridCols = 'repeat(10, 1fr)'; imgHeight = 50;
        } else if (accessLevel === "Regional Admin") { 
           totalFeeds = 300; feedsPerPage = 300; gridCols = 'repeat(15, 1fr)'; imgHeight = 30;
        } else if (accessLevel === "National Admin") { 
           totalFeeds = 1500; feedsPerPage = 300; gridCols = 'repeat(15, 1fr)'; imgHeight = 40;
        }

        const totalPages = Math.ceil(totalFeeds / feedsPerPage);
        const startIndex = (currentPage - 1) * feedsPerPage;
        const endIndex = Math.min(startIndex + feedsPerPage, totalFeeds);
        const renderedFeeds = endIndex - startIndex;

        return (
        <div className="main-content">
          <main className="content-area" style={{display: 'flex', gap: 20}}>
            <div style={{flex: 1, display: 'flex', flexDirection: 'column'}}>
              
              <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15, background: 'var(--bg-secondary)', padding: '10px 15px', borderRadius: 4, border: '1px solid var(--border-color)'}}>
                 <h2 style={{color: 'var(--text-primary)', margin: 0, fontSize: 13, fontWeight: 500}}>
                   {activeScope.toUpperCase()} | CAMERA FEEDS ({renderedFeeds} / {totalFeeds} ACTIVE)
                 </h2>
                 {totalPages > 1 && (
                    <div style={{display: 'flex', gap: 5}}>
                       {Array.from({length: totalPages}).map((_, i) => (
                          <button key={i} onClick={() => setCurrentPage(i+1)} style={{padding: '4px 10px', fontSize: 12, background: currentPage === i+1 ? 'var(--text-primary)' : 'var(--bg-tertiary)', color: currentPage === i+1 ? '#000' : 'var(--text-secondary)', border: '1px solid var(--border-color)', borderRadius: 2, cursor: 'pointer'}}>
                             Page {i+1}
                          </button>
                       ))}
                    </div>
                 )}
              </div>

              {viewEvidence && (
                <div style={{marginBottom: 20, background: 'var(--bg-secondary)', padding: 15, borderRadius: 4, border: '2px solid var(--alert-red)', boxShadow: '0 0 15px rgba(220,38,38,0.1)'}}>
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: 10}}>
                    <h3 style={{margin: 0, color: 'var(--text-primary)', fontSize: 14, fontWeight: 500}}>Incident Details (Review ID: {viewEvidence})</h3>
                    <button onClick={() => setViewEvidence(null)} style={{background: 'var(--alert-red)', border: 'none', color: 'white', cursor: 'pointer', fontSize: 11, padding: '4px 10px', borderRadius: 2}}>Close Pane</button>
                  </div>
                  <div style={{display: 'flex', gap: 15, marginTop: 15}}>
                    <div style={{flex: 1}}>
                       <h4 style={{fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 500}}>Video Playback</h4>
                       <iframe src="https://www.youtube.com/embed/y2m5A6b7w6k?autoplay=1&mute=1&loop=1&playlist=y2m5A6b7w6k&controls=0" style={{width: '100%', height: 100, borderRadius: 2, pointerEvents: 'none'}} frameBorder="0"></iframe>
                    </div>
                    <div style={{flex: 1}}>
                       <h4 style={{fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 500}}>Track Sensor Data</h4>
                       <div style={{height: 100, width: '100%', borderBottom: '1px solid var(--border-color)', borderLeft: '1px solid var(--border-color)', display: 'flex', alignItems: 'flex-end', gap: 2, padding: 5}}>
                          {Array.from({length: 30}).map((_, i) => <div key={i} style={{flex: 1, backgroundColor: (i > 12 && i < 18) ? 'var(--alert-red)' : '#3b82f6', height: `${(i > 12 && i < 18) ? (60 + Math.random()*40) : (10 + Math.random()*20)}%`}}></div>)}
                       </div>
                    </div>
                    <div style={{flex: 1}}>
                       <h4 style={{fontSize: 11, color: 'var(--text-secondary)', marginBottom: 5, fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>
                          Loc: {alerts.find(a=>a.id===viewEvidence)?.region || 'N/A'} &gt; {alerts.find(a=>a.id===viewEvidence)?.city || 'N/A'} &gt; {alerts.find(a=>a.id===viewEvidence)?.control_room ? alerts.find(a=>a.id===viewEvidence).control_room.replace('CR-', 'Desk ') : 'Desk 1'}
                       </h4>
                       <iframe src={`https://maps.google.com/maps?q=${alerts.find(a=>a.id===viewEvidence)?.lat||'28.6'},${alerts.find(a=>a.id===viewEvidence)?.lng||'77.2'}&t=&z=15&ie=UTF8&iwloc=&output=embed`} width="100%" height="100" style={{border: 0, borderRadius: 2}} loading="lazy"></iframe>
                    </div>
                  </div>
                </div>
              )}

              <div className="feeds-grid" style={{ display: 'grid', gridTemplateColumns: gridCols, gap: accessLevel === "Control Room" ? 8 : 2, width: '100%', opacity: viewEvidence ? 0.3 : 1, transition: '0.2s' }}>
                {currentPage === 1 && (
                  <div className="feed-container" style={{ margin: 0 }}>
                    {accessLevel === "Control Room" && (
                      <div className="feed-header" style={{padding: '4px 6px', background: 'var(--bg-tertiary)'}}>
                        <h2 className="feed-title" style={{fontSize: 10, color: 'var(--text-primary)', borderBottom: 'none', padding: 0, margin: 0}}>Cam 01 (Live Analysis)</h2>
                      </div>
                    )}
                    <div className="video-wrapper">
                      <img className="video-stream" src="/video_feed" alt="Live AI Feed" style={{width: '100%', height: imgHeight, objectFit: 'cover'}}/>
                    </div>
                  </div>
                )}
                {Array.from({length: currentPage === 1 ? renderedFeeds - 1 : renderedFeeds}).map((_, i) => {
                  const camNumber = startIndex + (currentPage === 1 ? i + 2 : i + 1);
                  return (
                  <div key={i} className="feed-container" style={{ margin: 0 }}>
                    {accessLevel === "Control Room" && (
                      <div className="feed-header" style={{padding: '4px 6px', background: 'var(--bg-tertiary)'}}>
                        <h2 className="feed-title" style={{fontSize: 10, color: 'var(--text-secondary)', borderBottom: 'none', padding: 0, margin: 0}}>Cam {camNumber.toString().padStart(4, '0')}</h2>
                      </div>
                    )}
                    <div className="video-wrapper">
                      <img className="video-stream" src={DUMMY_IMAGES[camNumber % DUMMY_IMAGES.length]} alt={`CCTV ${camNumber}`} style={{width: '100%', height: imgHeight, objectFit: 'cover', opacity: 0.8}}/>
                    </div>
                  </div>
                )})}
              </div>
            </div>

            <div className="alerts-panel" style={{width: 300, borderRadius: 4}}>
              <div className="alerts-header" style={{background: 'var(--bg-tertiary)'}}>
                <span style={{fontSize: 13, textTransform: 'none', fontWeight: 500}}>Incident Log</span>
                <span style={{fontSize: 10, color: 'var(--text-secondary)', fontWeight: 'normal'}}>{activeScope}</span>
              </div>
              <div className="alerts-list" style={{maxHeight: 'calc(100vh - 120px)', overflowY: 'auto', padding: '10px 15px'}}>
                {alerts.length === 0 ? (
                  <div style={{color: 'var(--text-secondary)', fontSize: 13, textAlign: 'center', marginTop: 30}}>No pending incidents.</div>
                ) : alerts.map((alert) => (
                  <div key={alert.id} className={`alert-card ${alert.level === 'critical' ? 'critical' : 'warning'}`} style={{opacity: alert.status === 'resolved' ? 0.6 : 1, padding: 12, marginBottom: 10, borderRadius: 4, background: 'var(--bg-secondary)', borderLeft: `3px solid ${alert.level === 'critical' ? 'var(--alert-red)' : 'var(--accent-gold)'}`, borderRight: '1px solid var(--border-color)', borderTop: '1px solid var(--border-color)', borderBottom: '1px solid var(--border-color)'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}>
                       <div className="alert-time" style={{fontSize: 10, color: 'var(--text-secondary)'}}>#{alert.id} • {new Date(alert.timestamp * 1000).toLocaleTimeString()}</div>
                       <span style={{fontSize: 9, padding: '2px 6px', borderRadius: 2, background: alert.status === 'resolved' ? '#10b981' : (alert.level === 'critical' ? 'var(--alert-red)' : 'var(--accent-gold)'), color: alert.status === 'resolved' ? 'white' : 'black', fontWeight: 500}}>{alert.status === 'resolved' ? 'Resolved' : 'Review Req.'}</span>
                    </div>
                    <div className="alert-msg" style={{margin: '0 0 10px 0', fontSize: 12, lineHeight: 1.4, color: 'var(--text-primary)'}}>{alert.message}</div>
                     {alert.tamper_confidence && (
                       <div style={{fontSize: 11, marginBottom: 8, color: 'var(--text-secondary)'}}>
                         TBE Confidence: <strong style={{color: alert.tamper_confidence >= 85 ? 'var(--alert-red)' : '#f59e0b'}}>{alert.tamper_confidence}%</strong>
                       </div>
                     )}
                    
                    <div style={{display: 'flex', gap: 6}}>
                      <button onClick={() => setViewEvidence(alert.id)} style={{flex: 1, padding: '5px 0', background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'var(--text-primary)', cursor: 'pointer', fontSize: 11, borderRadius: 2}}>
                         View Details
                      </button>
                      {alert.status === 'active' && (
                        <button onClick={() => handleAcknowledge(alert.id)} style={{flex: 1, padding: '5px 0', background: '#3b82f6', border: 'none', color: 'white', fontWeight: 500, cursor: 'pointer', fontSize: 11, borderRadius: 2}}>
                           Mark Verified
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </main>
        </div>
        );
      })()}
    </div>
  );
}

export default App;
