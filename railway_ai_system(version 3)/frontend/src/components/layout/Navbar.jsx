import { Link, useLocation } from 'react-router-dom';
import { Activity, LayoutDashboard, History, Settings, ShieldAlert } from 'lucide-react';
import './Navbar.css';

export default function Navbar() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/alerts', label: 'Alerts', icon: ShieldAlert },
    { path: '/history', label: 'History', icon: History },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          <Activity className="logo-icon" size={28} />
          <span className="logo-text">RailGuard</span>
        </div>
        
        <div className="navbar-links">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-link ${isActive ? 'active' : ''}`}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>

        <div className="navbar-actions">
          <button className="settings-btn" title="Settings">
            <Settings size={20} />
          </button>
        </div>
      </div>
    </nav>
  );
}
